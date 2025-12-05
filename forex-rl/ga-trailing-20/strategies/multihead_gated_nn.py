"""Multi-head gated neural strategy for multi-instrument entrances.

This strategy emits per-instrument discrete signals {-1, 0, +1} by:
- Training a shared MLP that outputs one continuous score per instrument.
- Calibrating per-instrument high/low quantile thresholds on the training
  predictions so that only the most confident scores open positions.
- Mapping scores to {-1, 0, +1} using these thresholds at inference time.

This approximates a "direction + gate" architecture:
- The score sign encodes direction.
- Quantile thresholds implement a gate that targets a small active fraction
  (e.g. ~10% time-in-market) instead of always trading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import (
    CrossValidationConfig,
    DatasetSplit,
    RegularizationConfig,
    Strategy,
    StrategyArtifact,
    TrailingStopSimulator,
)
from simulator.costs import pip_size  # type: ignore[import]


@dataclass
class MultiHeadGatedNNConfig:
    """Training hyper-parameters for the multi-instrument gated model.

    Architecture:
    - Shared torso: two hidden layers of size 64 (ReLU).
    - Per-instrument head: 6-unit hidden layer + 2 outputs:
      * head[..., 0] = gate logit (enter vs no-enter).
      * head[..., 1] = direction/magnitude score (unbounded; squashed via tanh).
    """

    # SGD / optimization
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 20
    patience: int = 5  # early stopping
    checkpoint_every: int = 0  # Optional checkpointing: 0 = disabled

    # Regularization
    dropout: float = 0.1

    # Activity / gating
    active_frac: float = 0.10  # Target fraction of non-flat signals (~time-in-market)

    # Label shaping
    ret_scale: float = 0.005  # Scale for direction target before tanh
    # Only label gate=1 for relatively strong trailing-stop episodes; higher
    # thresholds encourage more conservative trading. We tighten this so that
    # only higher-edge episodes are considered trade-worthy.
    gate_ret_threshold: float = 0.0075  # |ret| > threshold => gate target = 1
    # Trailing-stop label simulation
    max_trail_pips: float = 20.0  # Max trail distance used when simulating label trades

    # Gate sparsity regularization: penalize high average gate probabilities so
    # that the network learns to be selective about when to enter.
    gate_sparsity_weight: float = 0.05

    # Explicit "flat reward": small asymmetric reward for keeping the gate
    # closed on bars where we do NOT label a strong episode (gate_tgt == 0).
    # Implemented as a negative loss term, i.e. we *reward* low gate_probs
    # on non-episode bars. This is intentionally small compared to the main
    # losses so it nudges behaviour rather than dominating training.
    flat_reward_weight: float = 0.01

    # Optional explicit feature lags (in bars). For example, (0, 7, 49)
    # concatenates features at t, t-7 and t-49 into the input, giving the
    # network direct access to short- and medium-term temporal structure
    # without requiring a full sequence model.
    feature_lags: Tuple[int, ...] = (0,)


class MultiHeadGatedNNStrategy(Strategy):
    """Multi-instrument neural strategy with per-instrument gated heads."""

    def __init__(
        self,
        instruments: Sequence[str],
        config: Optional[MultiHeadGatedNNConfig] = None,
        *,
        regularization: Optional[RegularizationConfig] = None,
        cv: Optional[CrossValidationConfig] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(regularization=regularization, cv=cv)
        self.config = config or MultiHeadGatedNNConfig()
        self.instruments: Tuple[str, ...] = tuple(instruments)
        self.num_instruments: int = len(self.instruments)
        if self.num_instruments <= 0:
            raise ValueError("MultiHeadGatedNNStrategy requires at least one instrument")

        self.device_str = device
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        self._model: Optional[nn.Module] = None
        self._input_dim: Optional[int] = None  # effective (possibly lag-augmented) input dim
        self._raw_input_dim: Optional[int] = None  # raw feature dim before lag concatenation
        self.checkpoint_every: int = int(self.config.checkpoint_every)
        self.checkpoint_base: Optional[str] = None

        # When True, run a full simulator backtest on the validation split
        # after each epoch and log high-level metrics. This is wired up by the
        # training driver and defaults to False at the strategy level.
        self.eval_every_epoch: bool = False

        # Per-instrument thresholds for mapping gate probabilities to {-1,0,+1}
        # decisions. Calibrated after training from the distribution of
        # training gate outputs so that only high-confidence signals open
        # positions.
        self.gate_thresholds: Optional[np.ndarray] = None

        # Cached lag configuration and a small buffer of recent raw feature
        # vectors so that, at inference time, we can reconstruct the same
        # [X_t, X_{t-l1}, ...] layout that was used during training.
        self._lags_seq: Tuple[int, ...] = tuple(self.config.feature_lags or (0,))
        self._max_lag: int = max(self._lags_seq) if self._lags_seq else 0
        self._lag_buffer: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_split: DatasetSplit,
        val_split: Optional[DatasetSplit],
        simulator: TrailingStopSimulator,
    ) -> None:
        # Materialize records and build supervised targets from next-bar returns
        # for each instrument.
        train_records = list(train_split)
        if len(train_records) < 2:
            return
        X_train, Y_train = self._build_xy(train_records)

        X_val: Optional[np.ndarray] = None
        Y_val: Optional[np.ndarray] = None
        if val_split is not None:
            val_records = list(val_split)
            if len(val_records) > 1:
                X_val, Y_val = self._build_xy(val_records)

        # Optional temporal lag expansion of features. This mirrors the edge
        # check script: we build lagged copies of X (and trim Y accordingly)
        # so that each sample sees [X_t, X_{t-l1}, X_{t-l2}, ...].
        self._raw_input_dim = X_train.shape[1]
        raw_lags = self.config.feature_lags or (0,)
        clean_lags = sorted({int(l) for l in raw_lags if int(l) >= 0})
        if not clean_lags:
            clean_lags = [0]
        if 0 not in clean_lags:
            clean_lags.insert(0, 0)
        self._lags_seq = tuple(clean_lags)
        self._max_lag = max(self._lags_seq) if self._lags_seq else 0
        self._lag_buffer = []

        if any(l > 0 for l in self._lags_seq) and X_train.shape[0] > 0:
            X_train, Y_train = self._apply_lags_to_xy(X_train, Y_train, self._lags_seq)
            if X_val is not None and Y_val is not None and X_val.shape[0] > 0:
                X_val, Y_val = self._apply_lags_to_xy(X_val, Y_val, self._lags_seq)

        self._input_dim = X_train.shape[1]
        self._model = self._build_model(self._input_dim).to(self.device)
        cfg = self.config
        opt = optim.Adam(self._model.parameters(), lr=cfg.learning_rate)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        def _iter_minibatches(X: np.ndarray, Y: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
            n = X.shape[0]
            idx = np.arange(n)
            np.random.shuffle(idx)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                sel = idx[start:end]
                yield X[sel], Y[sel]

        best_val = float("inf")
        epochs_no_improve = 0
        for epoch in range(cfg.epochs):
            self._model.train()
            total_loss = 0.0
            n_obs = 0
            for xb, yb in _iter_minibatches(X_train, Y_train, cfg.batch_size):
                xt = torch.from_numpy(xb).to(self.device)
                y_ret = torch.from_numpy(yb).to(self.device)  # [B, num_inst]
                opt.zero_grad()
                preds = self._model(xt)  # [B, num_inst, 4]
                gate_logits = preds[..., 0]
                dir_raw = preds[..., 1]
                pos_raw = preds[..., 2]
                trail_raw = preds[..., 3]

                # Direction target: rescaled returns squashed into [-1, 1].
                scale = max(1e-8, float(cfg.ret_scale))
                dir_tgt = torch.tanh(y_ret / scale)

                # Gate target: soft label based on absolute episode return.
                # Larger |ret| push gate_tgt toward 1, small/noisy returns
                # push it toward 0. This avoids a brittle hard threshold.
                gate_tgt = torch.clamp(torch.abs(y_ret) / scale, min=0.0, max=1.0)

                # Loss components
                loss_dir = mse_loss(torch.tanh(dir_raw), dir_tgt)
                loss_gate = bce_loss(gate_logits, gate_tgt)

                # Position size target: encourage larger positions for larger
                # positive episode returns, and small size otherwise.
                pos_tgt = torch.clamp(y_ret / 0.01, min=0.0, max=1.0)
                loss_pos = mse_loss(torch.sigmoid(pos_raw), pos_tgt)

                # Trailing distance target: map absolute episode return to
                # [0,1] so that larger absolute returns encourage wider stops.
                trail_tgt = torch.clamp(torch.abs(y_ret) / 0.01, min=0.0, max=1.0)
                loss_trail = mse_loss(torch.sigmoid(trail_raw), trail_tgt)

                loss = loss_dir + loss_gate + 0.5 * loss_pos + 0.5 * loss_trail

                # Gate sparsity regularization: penalize high average gate
                # probabilities so that the model learns to be selective about
                # entering trades.
                gate_probs = torch.sigmoid(gate_logits)
                if self.config.gate_sparsity_weight > 0.0:
                    loss = loss + float(self.config.gate_sparsity_weight) * gate_probs.mean()

                # Explicit flat reward: on bars where we do NOT label a strong
                # trailing-stop episode (gate_tgt == 0), mildly *reward* the
                # model for keeping the gate closed (low gate_probs). Since we
                # minimise loss, this is implemented as a negative loss term.
                flat_w = float(getattr(self.config, "flat_reward_weight", 0.0))
                if flat_w != 0.0:
                    flat_mask = gate_tgt < 0.5
                    if flat_mask.any():
                        flat_probs = gate_probs[flat_mask]
                        flat_reward = (1.0 - flat_probs).mean()
                        loss = loss - flat_w * flat_reward

                # L2 regularization
                if self.regularization.l2 > 0.0:
                    l2 = sum((p ** 2).sum() for p in self._model.parameters())
                    loss = loss + float(self.regularization.l2) * l2

                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * xb.shape[0]
                n_obs += xb.shape[0]
            _ = total_loss / max(1, n_obs)

            if X_val is not None and Y_val is not None:
                self._model.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(X_val).to(self.device)
                    yv_ret = torch.from_numpy(Y_val).to(self.device)
                    preds_v = self._model(xv)
                    gate_logits_v = preds_v[..., 0]
                    dir_raw_v = preds_v[..., 1]
                    pos_raw_v = preds_v[..., 2]
                    trail_raw_v = preds_v[..., 3]
                    scale_v = max(1e-8, float(cfg.ret_scale))
                    dir_tgt_v = torch.tanh(yv_ret / scale_v)
                    gate_tgt_v = torch.clamp(torch.abs(yv_ret) / scale_v, min=0.0, max=1.0)
                    pos_tgt_v = torch.clamp(yv_ret / 0.01, min=0.0, max=1.0)
                    trail_tgt_v = torch.clamp(torch.abs(yv_ret) / 0.01, min=0.0, max=1.0)
                    val_loss = float(
                        mse_loss(torch.tanh(dir_raw_v), dir_tgt_v)
                        + bce_loss(gate_logits_v, gate_tgt_v)
                        + 0.5 * mse_loss(torch.sigmoid(pos_raw_v), pos_tgt_v)
                        + 0.5 * mse_loss(torch.sigmoid(trail_raw_v), trail_tgt_v)
                    )
                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= cfg.patience:
                        break

            # Optional per-epoch simulator evaluation on validation split to
            # track trading behavior (Sharpe, PF, trades, time-in-market).
            if self.eval_every_epoch and val_split is not None:
                try:
                    result = simulator.evaluate(self, val_split, record_equity=False, return_trades=True)
                    m = getattr(result, "metrics", None)
                    if m is not None:
                        print(
                            {
                                "epoch": epoch,
                                "segment": "val_epoch",
                                "cum_return": m.cum_return,
                                "sharpe": m.sharpe,
                                "profit_factor": m.profit_factor,
                                "trades": len(result.trades),
                                "per_instrument_tim_mean": m.per_instrument_tim_mean,
                                "per_instrument_tim_std": m.per_instrument_tim_std,
                            }
                        )
                except Exception:
                    # Evaluation errors shouldn't crash training.
                    pass

            # Optional checkpointing after each epoch (or every k epochs)
            if self.checkpoint_base and self.checkpoint_every > 0:
                if (epoch + 1) % self.checkpoint_every == 0:
                    ckpt_path = f"{self.checkpoint_base}_ep{epoch+1}.pt"
                    try:
                        self.save(ckpt_path)
                    except Exception:
                        # Checkpoint failures should not crash training.
                        pass

        # Calibrate per-instrument gate thresholds from training predictions so
        # that only high-confidence gate activations open trades. We use the
        # (1 - active_frac) quantile of gate probabilities, per instrument.
        try:
            self._model.eval()
            with torch.no_grad():
                xt_all = torch.from_numpy(X_train).to(self.device)
                preds_all = self._model(xt_all).cpu().numpy()  # [N, num_inst, 2]
            if preds_all.ndim == 2:
                preds_all = preds_all[:, :, None]
            gate_logits_all = preds_all[..., 0]
            gate_probs_all = 1.0 / (1.0 + np.exp(-gate_logits_all))
            alpha = float(np.clip(self.config.active_frac, 0.01, 0.9))
            thr_list: List[float] = []
            for j in range(self.num_instruments):
                col = gate_probs_all[:, j]
                thr_list.append(float(np.quantile(col, 1.0 - alpha)))
            self.gate_thresholds = np.array(thr_list, dtype=np.float32)
        except Exception:
            # If calibration fails, fall back to a neutral threshold.
            self.gate_thresholds = np.full(self.num_instruments, 0.5, dtype=np.float32)

    def cross_validate(
        self,
        data: DatasetSplit,
        simulator: TrailingStopSimulator,
    ) -> Dict[str, Any]:
        result = simulator.evaluate(self, data)
        return {"metrics": getattr(result, "metrics", None)}

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(self.instruments), dtype=float)
        # We always receive the *raw* feature vector for the current bar. If
        # training used temporal lags, we reconstruct the lag-augmented input
        # by maintaining a small buffer of recent raw feature vectors and
        # concatenating them in the same [X_t, X_{t-l1}, ...] order.
        raw = features
        if raw.ndim > 1:
            raw = raw[0]
        raw = np.asarray(raw, dtype=np.float32)

        use_lags = self._raw_input_dim is not None and any(l > 0 for l in self._lags_seq)
        if not use_lags:
            feats = raw[None, :]
        else:
            # Initialise raw_input_dim lazily if needed.
            if self._raw_input_dim is None:
                self._raw_input_dim = int(raw.shape[0])
            # Maintain buffer of last (max_lag + 1) raw feature vectors.
            self._lag_buffer.append(raw.copy())
            if len(self._lag_buffer) < self._max_lag + 1:
                # Pad the history at the start so early bars still have lags.
                while len(self._lag_buffer) < self._max_lag + 1:
                    self._lag_buffer.insert(0, self._lag_buffer[0])
            # Keep only the most recent window.
            if len(self._lag_buffer) > self._max_lag + 1:
                self._lag_buffer = self._lag_buffer[-(self._max_lag + 1) :]

            feats_per_lag: List[np.ndarray] = []
            for lag in self._lags_seq:
                idx = self._max_lag - int(lag)
                idx = max(0, min(idx, len(self._lag_buffer) - 1))
                feats_per_lag.append(self._lag_buffer[idx])
            feats = np.concatenate(feats_per_lag, axis=-1)[None, :]

        xt = torch.from_numpy(feats.astype(np.float32)).to(self.device)
        self._model.eval()
        with torch.no_grad():
            out = self._model(xt).cpu().numpy()  # [B, num_inst, 4]
        if out.ndim == 2:
            out = out[:, :, None]
        logits = out[0, :, 0]
        dir_raw = out[0, :, 1]
        pos_raw = out[0, :, 2]
        trail_raw = out[0, :, 3]

        gate_probs = 1.0 / (1.0 + np.exp(-logits))
        dir_scores = np.tanh(dir_raw)
        pos_frac = 1.0 / (1.0 + np.exp(-pos_raw))
        trail_frac = 1.0 / (1.0 + np.exp(-trail_raw))

        # Map continuous outputs to:
        # - enter_flags in {0,1} (explicit gate)
        # - dir_sig in {-1,0,+1}
        # - pos_frac in [0,1]
        # - trail_frac in [0,1]
        dir_sig = np.zeros(self.num_instruments, dtype=np.float32)
        enter_flags = np.zeros(self.num_instruments, dtype=np.float32)

        if self.gate_thresholds is None:
            # Fallback: no calibrated thresholds. Enter whenever we have a
            # non-zero directional opinion.
            dir_sig[dir_scores > 0.0] = 1.0
            dir_sig[dir_scores < 0.0] = -1.0
            enter_flags[dir_sig != 0.0] = 1.0
        else:
            # Use calibrated per-instrument gate thresholds on the gate
            # probabilities, and only emit a non-zero direction when the gate
            # is open.
            for j in range(self.num_instruments):
                if gate_probs[j] >= float(self.gate_thresholds[j]):
                    enter_flags[j] = 1.0
                    if dir_scores[j] > 0.0:
                        dir_sig[j] = 1.0
                    elif dir_scores[j] < 0.0:
                        dir_sig[j] = -1.0

        # Concatenate explicit entrance flags, direction, position-size
        # fraction, and trailing-distance fraction so that the simulator can
        # use all four.
        return np.concatenate(
            [
                enter_flags.astype(np.float32),
                dir_sig.astype(np.float32),
                pos_frac.astype(np.float32),
                trail_frac.astype(np.float32),
            ],
            axis=0,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> StrategyArtifact:
        if self._model is None or self._input_dim is None:
            raise RuntimeError("Model not trained; cannot save")
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "input_dim": self._input_dim,
                "config": self.config,
                "instruments": list(self.instruments),
                "gate_thresholds": self.gate_thresholds.tolist() if self.gate_thresholds is not None else None,
            },
            path,
        )
        art = StrategyArtifact(
            name="multihead_gated_nn",
            version="v0",
            feature_names=None,
            scaler_state=None,
            extra={"path": path},
        )
        self.set_artifact(art)
        return art

    @classmethod
    def load(cls, path: str) -> "MultiHeadGatedNNStrategy":
        chk = torch.load(path, map_location="cpu")
        cfg = chk.get("config", MultiHeadGatedNNConfig())
        instruments = chk.get("instruments") or []
        strat = cls(instruments=instruments, config=cfg)
        input_dim = int(chk.get("input_dim"))
        strat._input_dim = input_dim
        strat._model = strat._build_model(input_dim)
        strat._model.load_state_dict(chk["state_dict"])
        strat._model.to(strat.device)
        gt = chk.get("gate_thresholds")
        if gt is not None:
            strat.gate_thresholds = np.array(gt, dtype=np.float32)
        return strat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, input_dim: int) -> nn.Module:
        """Shared torso (64,64) feeding per-instrument 6->2 heads."""

        class _Net(nn.Module):
            def __init__(self, in_dim: int, num_inst: int, dropout: float) -> None:
                super().__init__()
                self.num_inst = num_inst
                self.torso = nn.Sequential(
                    nn.Linear(in_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                )
                # Simple channel-wise attention/gating over the shared torso.
                self.attn = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Sigmoid(),
                )
                self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                # Per-instrument heads: 6-unit hidden -> 4 outputs:
                # [gate_logit, dir_raw, pos_raw, trail_raw]
                self.heads = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(64, 6),
                            nn.ReLU(),
                            nn.Linear(6, 4),
                        )
                        for _ in range(num_inst)
                    ]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [B, in_dim] -> [B, num_inst, 4]
                h = self.torso(x)
                w = self.attn(h)
                h = h * w  # channel-wise attention
                h = self.dropout(h)
                outs = []
                for head in self.heads:
                    outs.append(head(h))  # [B, 4]
                return torch.stack(outs, dim=1)

        return _Net(input_dim, self.num_instruments, float(self.config.dropout))

    def _apply_lags_to_xy(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lags: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply simple time lags to already-built (X, Y).

        X is assumed to be ordered in time with shape [T, F]. For each time
        index t >= max(lags) we build a new feature vector by concatenating
        X[t - lag] across all lags. Y is shifted accordingly to keep alignment.
        """
        clean_lags = sorted({int(l) for l in lags if int(l) >= 0})
        if not clean_lags or clean_lags == [0]:
            return X, Y
        if 0 not in clean_lags:
            clean_lags.insert(0, 0)
        max_lag = max(clean_lags)

        T, _ = X.shape
        if T <= max_lag + 1:
            return X, Y

        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        for t in range(max_lag, T):
            feats_per_lag: List[np.ndarray] = []
            for lag in clean_lags:
                feats_per_lag.append(X[t - lag])
            X_list.append(np.concatenate(feats_per_lag, axis=-1))
            Y_list.append(Y[t])

        X_new = np.stack(X_list, axis=0)
        Y_new = np.stack(Y_list, axis=0)
        return X_new, Y_new

    def _build_xy(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, Y) where Y has one trailing-stop episode return per instrument.

        For each time index i and instrument j, we simulate a hypothetical trade
        that enters at bar i and exits only when its trailing stop is hit
        (long-only for label construction). The resulting realized return
        (exit_px / entry_px - 1) is used as the supervision signal.
        """
        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        insts = list(self.instruments)
        n = len(records)
        if n < 2:
            raise ValueError("Not enough records to build supervised targets")

        for i in range(n - 1):
            rec_t = records[i]
            feats = np.asarray(rec_t["features"], dtype=np.float32)
            bars_t = rec_t["bars"]
            if not bars_t:
                continue
            y_vec: List[float] = []
            for inst in insts:
                ret = self._simulate_trailing_episode(records, i, inst)
                y_vec.append(float(ret))
            X_list.append(feats)
            Y_list.append(np.array(y_vec, dtype=np.float32))

        if not X_list:
            raise ValueError("Not enough records to build trailing-stop labels")
        X = np.stack(X_list, axis=0)
        Y = np.stack(Y_list, axis=0)
        return X, Y

    def _simulate_trailing_episode(
        self,
        records: List[Dict[str, Any]],
        start_idx: int,
        inst: str,
        max_horizon_bars: int = 500,
    ) -> float:
        """Simulate a single long trade with a trailing stop from `start_idx`.

        - Entry at close price of instrument `inst` at bar `start_idx`.
        - Trailing stop distance = min_distance_pips (from TrailingConfig)
          but capped at `config.max_trail_pips` and expressed in instrument pips.
        - We walk forward until the trailing stop is hit or we reach `max_horizon_bars`,
          then compute (exit_px / entry_px - 1).
        """
        n = len(records)
        if start_idx >= n - 1:
            return 0.0
        try:
            bars0 = records[start_idx]["bars"].get(inst)
        except Exception:
            return 0.0
        if not bars0:
            return 0.0
        try:
            entry_price = float(bars0.get("close", 0.0))
        except Exception:
            entry_price = 0.0
        if entry_price <= 0.0:
            return 0.0

        # Simple pip-based trailing distance, capped by max_trail_pips.
        psize = pip_size(inst)
        trail_pips = float(self.config.max_trail_pips)
        dist = max(psize, trail_pips * psize)
        trail = entry_price - dist
        favorable = entry_price

        last_idx = min(n - 1, start_idx + max_horizon_bars)
        exit_price = entry_price
        hit = False
        for k in range(start_idx + 1, last_idx + 1):
            try:
                bar = records[k]["bars"].get(inst)
            except Exception:
                bar = None
            if not bar:
                continue
            high = float(bar.get("high", entry_price))
            low = float(bar.get("low", entry_price))
            open_px = float(bar.get("open", entry_price))

            # Long trailing-stop logic: move trail up as new highs are made.
            favorable = max(favorable, high)
            candidate = favorable - dist
            trail = max(trail, candidate)
            if low <= trail:
                # Gap-aware exit: if price gapped through the stop, we assume
                # worst between bar open and stop level.
                exit_price = min(open_px, trail)
                hit = True
                break
            exit_price = float(bar.get("close", exit_price))

        if not hit:
            # If no trailing stop hit within horizon, exit at last observed close.
            try:
                last_bar = records[last_idx]["bars"].get(inst)
                if last_bar:
                    exit_price = float(last_bar.get("close", exit_price))
            except Exception:
                pass

        if entry_price <= 0.0 or exit_price <= 0.0:
            return 0.0
        return (exit_price / entry_price) - 1.0


