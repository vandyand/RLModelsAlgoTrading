"""Gradient-based neural strategy scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class GradientNNConfig:
    """Training hyper-parameters for the supervised/actor-critic model."""

    hidden_dims: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 20
    patience: int = 5  # early stopping
     # Optional checkpointing: save model every N epochs (0 = disabled)
    checkpoint_every: int = 0


class GradientNNStrategy(Strategy):
    """Standard neural network trained via backprop on simulator rewards."""

    def __init__(
        self,
        config: Optional[GradientNNConfig] = None,
        *,
        regularization: Optional[RegularizationConfig] = None,
        cv: Optional[CrossValidationConfig] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(regularization=regularization, cv=cv)
        self.config = config or GradientNNConfig()
        self.device_str = device
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        self._model: Optional[nn.Module] = None
        self._input_dim: Optional[int] = None
        # Optional per-epoch checkpointing, configured via config or externally.
        self.checkpoint_every: int = int(self.config.checkpoint_every)
        self.checkpoint_base: Optional[str] = None
        # Learned decision thresholds for mapping continuous outputs to
        # discrete {-1, 0, +1} signals. If not set, we fall back to zero.
        self.long_threshold: float = 0.0
        self.short_threshold: float = 0.0

    def fit(
        self,
        train_split: DatasetSplit,
        val_split: Optional[DatasetSplit],
        simulator: TrailingStopSimulator,
    ) -> None:
        # Materialize records and build supervised targets from next-bar returns (first instrument only).
        train_records = list(train_split)
        if len(train_records) < 2:
            return
        X_train, y_train = self._build_xy(train_records)
        X_val: Optional[np.ndarray] = None
        y_val: Optional[np.ndarray] = None
        if val_split is not None:
            val_records = list(val_split)
            if len(val_records) > 1:
                X_val, y_val = self._build_xy(val_records)

        self._input_dim = X_train.shape[1]
        self._model = self._build_model(self._input_dim).to(self.device)
        cfg = self.config
        opt = optim.Adam(self._model.parameters(), lr=cfg.learning_rate)
        loss_fn = nn.MSELoss()

        def _iter_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
            n = X.shape[0]
            idx = np.arange(n)
            np.random.shuffle(idx)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                sel = idx[start:end]
                yield X[sel], y[sel]

        best_val = float("inf")
        epochs_no_improve = 0
        for epoch in range(cfg.epochs):
            self._model.train()
            total_loss = 0.0
            n_obs = 0
            for xb, yb in _iter_minibatches(X_train, y_train, cfg.batch_size):
                xt = torch.from_numpy(xb).to(self.device)
                yt = torch.from_numpy(yb).to(self.device)
                opt.zero_grad()
                preds = self._model(xt).squeeze(-1)
                loss = loss_fn(preds, yt)
                # L2 regularization
                if self.regularization.l2 > 0.0:
                    l2 = sum((p ** 2).sum() for p in self._model.parameters())
                    loss = loss + float(self.regularization.l2) * l2
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * xb.shape[0]
                n_obs += xb.shape[0]
            avg_loss = total_loss / max(1, n_obs)

            if X_val is not None and y_val is not None:
                self._model.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(X_val).to(self.device)
                    yv = torch.from_numpy(y_val).to(self.device)
                    preds_v = self._model(xv).squeeze(-1)
                    val_loss = float(loss_fn(preds_v, yv).item())
                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= cfg.patience:
                        break
            else:
                # No validation split; just run full number of epochs.
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

        # After training, calibrate entry/exit thresholds from the training
        # distribution so that only extreme predictions trigger trades. We use
        # symmetric quantiles to target a small active fraction (e.g. ~10%).
        try:
            self._model.eval()
            with torch.no_grad():
                xt_all = torch.from_numpy(X_train).to(self.device)
                preds_all = self._model(xt_all).squeeze(-1).cpu().numpy()
            if preds_all.ndim == 0:
                preds_all = np.array([float(preds_all)], dtype=np.float32)
            lo_q = np.quantile(preds_all, 0.05)
            hi_q = np.quantile(preds_all, 0.95)
            self.short_threshold = float(lo_q)
            self.long_threshold = float(hi_q)
        except Exception:
            # If calibration fails for any reason, leave thresholds at zero.
            self.long_threshold = 0.0
            self.short_threshold = 0.0

    def cross_validate(
        self,
        data: DatasetSplit,
        simulator: TrailingStopSimulator,
    ) -> Dict[str, Any]:
        # Simple wrapper: evaluate current strategy on provided data.
        result = simulator.evaluate(self, data)
        return {"metrics": getattr(result, "metrics", None)}

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            # Fallback to flat signal if not trained.
            return np.zeros(1, dtype=float)
        if features.ndim == 1:
            feats = features[None, :]
        else:
            feats = features
        xt = torch.from_numpy(feats.astype(np.float32)).to(self.device)
        self._model.eval()
        with torch.no_grad():
            out = self._model(xt).squeeze(-1).cpu().numpy()
        # Expect a single scalar score for now (multi-instrument features are
        # treated as a single state vector). Map this continuous score to a
        # discrete {-1, 0, +1} signal using calibrated quantile thresholds so
        # that only the most confident predictions open positions.
        if out.ndim == 0:
            score = float(out)
        else:
            score = float(out[0])
        if score > self.long_threshold:
            sig = 1.0
        elif score < self.short_threshold:
            sig = -1.0
        else:
            sig = 0.0
        return np.array([sig], dtype=float)

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
            },
            path,
        )
        art = StrategyArtifact(
            name="gradient_nn",
            version="v0",
            feature_names=None,
            scaler_state=None,
            extra={"path": path},
        )
        self.set_artifact(art)
        return art

    @classmethod
    def load(cls, path: str) -> "GradientNNStrategy":
        chk = torch.load(path, map_location="cpu")
        cfg = chk.get("config", GradientNNConfig())
        strat = cls(config=cfg)
        input_dim = int(chk.get("input_dim"))
        strat._input_dim = input_dim
        strat._model = strat._build_model(input_dim)
        strat._model.load_state_dict(chk["state_dict"])
        strat._model.to(strat.device)
        return strat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, input_dim: int) -> nn.Module:
        dims = [input_dim] + list(self.config.hidden_dims) + [1]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)

    def _build_xy(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) where y is next-bar return of first instrument."""
        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        # Need at least two records to form a target.
        for i in range(len(records) - 1):
            rec_t = records[i]
            rec_tp1 = records[i + 1]
            feats = np.asarray(rec_t["features"], dtype=np.float32)
            bars_t = rec_t["bars"]
            bars_tp1 = rec_tp1["bars"]
            if not bars_t:
                continue
            first_inst = sorted(bars_t.keys())[0]
            close_t = float(bars_t[first_inst]["close"])
            close_tp1 = float(bars_tp1[first_inst]["close"])
            if close_t <= 0.0:
                ret = 0.0
            else:
                ret = (close_tp1 / close_t) - 1.0
            X_list.append(feats)
            y_list.append(float(ret))
        if not X_list:
            raise ValueError("Not enough records to build supervised targets")
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.float32)
        return X, y
