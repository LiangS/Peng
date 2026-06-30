# Project Specification — Multi-Horizon TCN for A-Share Trend Prediction

Status: draft / living document. See [README.md](README.md) for setup and
[CLAUDE.md](CLAUDE.md) for working conventions. This document defines *what* the
system does and the contracts each component must uphold; the test suite under
`tests/` enforces the load-bearing ones.

## 1. Problem statement

Given the recent daily price history of a Chinese A-share, predict the
**direction of future price movement at multiple horizons simultaneously**.

This is a **multi-model** project: several model families (TCN, XGBoost, ...)
are trained on one shared dataset and compared with identical metrics. New
models plug into the same `prepare_windowed` → metrics flow.

- Horizons (trading days): `1, 5, 20` (configurable).
- For each horizon, output a probability distribution (softmax) over
  `num_classes` **symmetric return buckets**. The bucket edges come from
  `Config.class_thresholds` (default `[0.5%, 2%, 5%]` → 7 classes); `num_classes`
  is derived as `2*len(class_thresholds)+1`. Class 0 = biggest drop, the middle
  class = flat, the last class = biggest gain.
- This is a **multi-task classification** problem: one shared representation,
  one independent prediction per horizon.

Non-goals: price-level regression, intraday/tick data, order execution,
portfolio construction, and live trading. The output is a directional
probability, not a trading signal.

## 2. Data contract

**Source.** AKShare `stock_zh_a_hist` (daily, forward-adjusted `qfq`).

**Canonical frame** (`data.load_prices`): a `DataFrame` indexed by ascending
`DatetimeIndex` with float columns `open, high, low, close, volume` (plus
`amount, turnover` when available). Cached as parquet under `data/cache/`.

**Known data hazard.** `qfq` adjustment can produce **non-positive `close`** in
the oldest rows (cumulative dividend subtraction). Such rows are invalid and
must be removed before feature computation (see §3).

## 3. Features and labels

**Features** (`features.build_features`), all derived from OHLCV and designed to
be roughly stationary:

| Feature       | Definition                                  |
|---------------|---------------------------------------------|
| `log_ret`     | `log(close_t / close_{t-1})`, guarded > 0   |
| `ma5_ratio`   | `close / SMA(close, 5) - 1`                 |
| `ma10_ratio`  | `close / SMA(close, 10) - 1`                |
| `ma20_ratio`  | `close / SMA(close, 20) - 1`                |
| `vol_10`      | rolling std of `log_ret` over 10 days       |
| `rsi_14`      | RSI(14), scaled to ~[0, 1]                  |
| `vol_change`  | `log((volume+1) / (volume_{t-1}+1))`        |
| `hl_range`    | `(high - low) / close`                      |

**Labels** (`features.build_labels`). For horizon `h`, forward return
`r = close_{t+h}/close_t - 1`. The sorted positive `class_thresholds` are each
scaled by `sqrt(h)` (so longer horizons aren't dominated by drift) and mirrored
around zero into ascending boundaries; the bucket is `np.digitize(r, edges)`,
giving classes `0 .. 2*len(thresholds)`. Example for `[0.005, 0.02, 0.05]`:
`0 down>5%, 1 down2-5%, 2 down0.5-2%, 3 flat(±0.5%), 4 up0.5-2%, 5 up2-5%,
6 up>5%` (edges shown at `h=1`). `features.class_names(thresholds)` returns the
human-readable label per class. Rows whose future price is unknown (last `h`
rows) are `NaN` and dropped from training windows.

**Pipeline contract** (`features.build_dataset_frame`):
1. Drop the leading block of non-positive-`close` rows (`_drop_nonpositive_prices`).
2. Compute features and labels on the cleaned frame.
3. Drop warm-up rows where any feature is `NaN`.
4. Returns `(features_df, labels_df)` sharing one index; features contain no
   `NaN`/`inf`; the `np.log` calls never receive a non-positive value.

## 4. Windowing, split, scaling (`dataset.build_windowed_data`)

- **Window**: each sample is the last `window` (default 60) timesteps, shaped
  `(num_features, window)` for `Conv1d`. Label is the multi-horizon class
  vector at the window's final timestep.
- **Sample validity**: a window is kept only if every horizon label at its end
  index is known (no `NaN`).
- **Split**: strictly time-ordered. `cutoff = int(n * (1 - val_ratio))`; a window
  belongs to train iff its end index `< cutoff`, else validation.
- **No leakage** (invariants the tests enforce):
  - `StandardScaler` is fit on training rows **only**, then applied to all.
  - Every validation window's end index `>= cutoff`; every training window's
    end index `< cutoff`. No future label informs training.
- **Class weights**: inverse-frequency per horizon, computed from the training
  split, for use in the loss.

## 4a. Shared prep & model interface

`prepare.prepare_windowed(cfg)` runs §3–§4 once and returns numpy arrays
(`X_train/X_val` as `(N, F, window)`, `y_*` as `(N, H)`, class weights, scaler
stats). Every model trains on this identical output:

- **TCN** consumes the windows directly (`train.train_from_arrays`).
- **XGBoost** consumes a tabular projection of the same windows
  (`tabular.window_to_tabular`: each feature at lags `{0,1,5,20}` from the window
  end, plus window mean/std → `F*6` columns), with one classifier per horizon
  (`train_xgb.run_xgb`).

`compare.py` trains all models on one `prepare_windowed` result and prints a
side-by-side table. Adding a model = add a trainer that returns a
`horizon_report` and register it in `compare`.

## 5. TCN model (`model.MultiHorizonTCN`)

- TCN backbone: stacked **dilated causal** 1-D conv residual blocks
  (Bai et al. 2018), dilation `2**i`, weight-normalised via
  `torch.nn.utils.parametrizations.weight_norm`.
- The last timestep of the final feature map summarises the window.
- One `Linear` **head per horizon** → class logits.
- `forward(x)` returns a list of `(batch, num_classes)` logit tensors, one per
  horizon. `predict_proba(x)` returns the per-horizon softmax (each row sums
  to 1).
- Receptive field must cover `window`:
  `1 + 2*(kernel_size-1)*(2**num_layers - 1)`. Defaults give 61 ≥ 60.

## 6. Training & evaluation (`train.py`)

- Loss: mean over horizons of class-weighted cross-entropy.
- Optimiser AdamW + cosine LR; gradient clipping; early stopping on **mean
  validation accuracy**; best checkpoint saved with scaler stats, config, and the
  per-epoch training `history`.
- Reported metrics (`metrics.py`, shared by all models): per-horizon
  **accuracy, macro-F1, and majority-class baseline** (+ lift over baseline).
  Class-weighted training makes raw accuracy misleading on its own, so the
  baseline and macro-F1 are always shown alongside it.
- **Training history.** `train_from_arrays(..., history=[])` records one dict per
  epoch (`train_loss`, per-horizon `val_acc`, `mean_acc`); `train()` stores it in
  the checkpoint under `"history"`. XGBoost's per-round validation curve comes from
  each fitted model's `evals_result()` — `train_xgb` returns `(report, models)` so
  callers can read it.

## 6a. Visualization (`plotting.py`)

Plotting reads the artifacts above; it never re-runs training.
`plot_tcn_history(history, horizons)` charts training loss and per-horizon
validation accuracy vs epoch; `plot_xgb_evals(models, horizons)` charts each
horizon's validation log-loss vs boosting round. Each returns a matplotlib
`Figure`. The per-model notebooks (`colab_train_tcn.ipynb`,
`colab_train_xgb.ipynb`) call these between training and inference.

## 7. Success criteria

- **Correctness (must hold, tested):** no look-ahead leakage; labels match their
  definition; non-positive prices removed; no `NaN`/`inf` in features; model
  output shapes and softmax normalisation correct.
- **Modelling (aspirational):** beat the per-horizon majority-class baseline
  out-of-sample. Daily direction is near-random; sustained **53–55%** directional
  accuracy across many symbols is a strong result. A single symbol (~1,850 days)
  is expected to sit near chance — multi-symbol training is the intended path to
  signal. Note: finer buckets lower per-class chance (≈1/`num_classes`) and split
  the data thinner, so judge them by **macro-F1 and lift over the baseline**, not
  raw accuracy, and collapse buckets if the tails stay empty.

## 8. Open items / roadmap

1. ~~Majority-baseline + macro-F1 reporting~~ — done (`metrics.py`).
2. ~~Second model family (XGBoost) for comparison~~ — done (`train_xgb.py`,
   `compare.py`).
3. Multi-symbol training: pool many tickers, window each **independently** (never
   across symbols), to expand the dataset by orders of magnitude. (Biggest
   expected lift; applies to all models.)
4. Optional: more model families (LightGBM, logistic-regression baseline);
   tune `class_thresholds` (fewer/coarser buckets, e.g. `[0.005]` for down/flat/up);
   richer features.
5. Optional: push checkpoints to Hugging Face so they survive Colab resets.
