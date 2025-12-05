nohup python walk_forward_gp.py \
--start 2020-01-01 \
--end 2025-07-01 \
--train-weeks 4 \
--val-weeks 1 \
--step-weeks 1 \
--population 100 \
--generations 15 \
--subsample 1 \
--select-top-k 13 \
--select-by ff \
--select-train-sharpe-min 0.0 \
--strategy-type nn \
--nn-arch 16,8 \
--input-norm rolling \
--nn-mutation-prob 0.25 \
--nn-mutation-sigma 0.025 \
--ff-enable \
--ff-mode nn \
--per-window-build \
--warmup-days 20 &

<<COMMENT
usage: walk_forward_gp.py [-h] [--data-dir DATA_DIR] --start START --end END
                          [--train-months TRAIN_MONTHS]
                          [--val-months VAL_MONTHS]
                          [--step-months STEP_MONTHS]
                          [--train-weeks TRAIN_WEEKS] [--val-weeks VAL_WEEKS]
                          [--step-weeks STEP_WEEKS] [--population POPULATION]
                          [--generations GENERATIONS] [--max-depth MAX_DEPTH]
                          [--cost-bps COST_BPS] [--min-hold MIN_HOLD]
                          [--cooldown COOLDOWN] [--subsample SUBSAMPLE]
                          [--seed SEED] [--windows-limit WINDOWS_LIMIT]
                          [--checkpoint-dir CHECKPOINT_DIR] [--no-gen-logs]
                          [--final-top-k FINAL_TOP_K]
                          [--select-top-k SELECT_TOP_K]
                          [--select-by {ff,fitness}]
                          [--select-ff-threshold SELECT_FF_THRESHOLD]
                          [--select-train-sharpe-min SELECT_TRAIN_SHARPE_MIN]
                          [--select-train-trades-min SELECT_TRAIN_TRADES_MIN]
                          [--strategy-type {logic,nn}] [--nn-arch NN_ARCH]
                          [--input-norm {rolling,affine}]
                          [--nn-mutation-prob NN_MUTATION_PROB]
                          [--nn-mutation-sigma NN_MUTATION_SIGMA]
                          [--nn-affine-sigma NN_AFFINE_SIGMA] [--ff-enable]
                          [--ff-mode {linear,nn}] [--ff-min-rows FF_MIN_ROWS]
                          [--ff-warmup-windows FF_WARMUP_WINDOWS]
                          [--per-window-build] [--warmup-days WARMUP_DAYS]
                          [--account ACCOUNT] [--leverage LEVERAGE]

Walk-Forward Optimization driver for gp_optimize

options:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR
  --start START         Global start timestamp (e.g., 2020-01-01)
  --end END             Global end timestamp (e.g., 2024-12-31)
  --train-months TRAIN_MONTHS
  --val-months VAL_MONTHS
  --step-months STEP_MONTHS
  --train-weeks TRAIN_WEEKS
  --val-weeks VAL_WEEKS
  --step-weeks STEP_WEEKS
  --population POPULATION
  --generations GENERATIONS
  --max-depth MAX_DEPTH
  --cost-bps COST_BPS
  --min-hold MIN_HOLD
  --cooldown COOLDOWN
  --subsample SUBSAMPLE
  --seed SEED
  --windows-limit WINDOWS_LIMIT
                        Optional: limit number of windows for quick runs
  --checkpoint-dir CHECKPOINT_DIR
  --no-gen-logs         Do not print per-generation summaries
  --final-top-k FINAL_TOP_K
                        Save only top-K of final population per window (0=all)
  --select-top-k SELECT_TOP_K
                        If >0, build ensemble from top-K candidates per window
  --select-by {ff,fitness}
                        Ranking metric for selection: learned FF or train
                        fitness
  --select-ff-threshold SELECT_FF_THRESHOLD
                        Minimum FF score to include in ensemble (0-1 for FF
                        models)
  --select-train-sharpe-min SELECT_TRAIN_SHARPE_MIN
                        Minimum train Sharpe for selection eligibility
  --select-train-trades-min SELECT_TRAIN_TRADES_MIN
                        Minimum train trades for selection eligibility
  --strategy-type {logic,nn}
                        Strategy representation: logic trees or neural-net
                        genome
  --nn-arch NN_ARCH     Hidden layer sizes for NN, comma-separated (e.g.,
                        256,128,64)
  --input-norm {rolling,affine}
                        Input normalization for NN inputs
  --nn-mutation-prob NN_MUTATION_PROB
  --nn-mutation-sigma NN_MUTATION_SIGMA
  --nn-affine-sigma NN_AFFINE_SIGMA
  --ff-enable           Enable learned fitness function scoring from past
                        windows
  --ff-mode {linear,nn}
                        FF model type: linear logistic or small NN
  --ff-min-rows FF_MIN_ROWS
                        Minimum rows from past windows to train FF
  --ff-warmup-windows FF_WARMUP_WINDOWS
                        Use first N windows to collect data before training FF
  --per-window-build    Build features per window to reduce RAM
  --warmup-days WARMUP_DAYS
                        Warmup days before train start for indicators/z-scores
  --account ACCOUNT     Account base currency size for scaling PnL
  --leverage LEVERAGE   Leverage multiplier for scaled returns

COMMENT
