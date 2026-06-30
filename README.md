# Peng — Multi-horizon TCN for A-share trend prediction

A PyTorch Temporal Convolutional Network (TCN) that predicts **directional trend
at multiple horizons** (1, 5, and 20 trading days). Each horizon has its own
softmax head, so the model outputs a probability distribution over
`{down, flat, up}` for every horizon independently.

Data comes from **AKShare** (Chinese A-share daily OHLCV). Code lives locally;
training is meant to run on a **Colab GPU runtime** via the VS Code Google Colab
extension.

## Layout

```
src/
  config.py    # one Config dataclass — all hyperparameters
  data.py      # AKShare download + parquet caching
  features.py  # technical features + 3-class forward-direction labels
  dataset.py   # sliding windows, time-ordered split, feature scaling, class weights
  model.py     # TCN backbone + one softmax head per horizon
  train.py     # multi-task training loop, early stopping, checkpointing
  train_xgb.py # per-horizon XGBoost classifiers on tabular features
  compare.py   # train every model on one shared split, print side-by-side
  plotting.py  # matplotlib training-curve helpers for the notebooks
notebooks/
  colab_train_tcn.ipynb   # train TCN + plot curves + inference on Colab
  colab_train_xgb.ipynb   # same flow for XGBoost
```

## Quick start (local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only, for XGBoost's OpenMP runtime:
brew install libomp

python -m src.train     --symbol 600519 --epochs 60   # TCN
python -m src.train_xgb --symbol 600519               # XGBoost
python -m src.compare   --symbol 600519 --epochs 60   # both, side-by-side
```

The TCN's best checkpoint (by mean validation accuracy) is written to
`checkpoints/tcn_<symbol>.pt` with scaler stats and config; XGBoost models go to
`checkpoints/xgb_<symbol>/h{horizon}.json`.

## Comparing models

This repo trains multiple model families on the **same** windowed train/val
split and reports identical metrics, so results are directly comparable.
`python -m src.compare` prints per-horizon **accuracy / macro-F1** for each
model next to the **majority-class baseline** — the number that actually matters
(beating it is the bar). Add a model by writing a trainer that returns a
`metrics.horizon_report` and registering it in [src/compare.py](src/compare.py).

## Running on a Colab GPU (recommended)

There is one notebook per model — `notebooks/colab_train_tcn.ipynb` and
`notebooks/colab_train_xgb.ipynb` — sharing the same setup cell. The TCN
notebook benefits from a GPU; XGBoost is CPU-bound.

1. Push this repo to GitHub.
2. Open the notebook you want in VS Code.
3. In the kernel picker (top-right), connect to a **Colab** runtime (choose a
   **GPU** runtime for the TCN).
4. Set `REPO_URL` in the first cell to your repo URL (the Colab VM is a fresh
   cloud machine and can't see your local files), then run all cells.

> The notebooks auto-detect Colab vs. a local kernel. On a local kernel they
> import `src` directly; on Colab they clone `REPO_URL`.

Each notebook trains, then **plots its training curves** — the TCN's loss and
per-horizon validation accuracy per epoch, XGBoost's per-round validation
log-loss — and finally prints per-horizon direction probabilities for the most
recent window.

## Tests

The contracts in [SPEC.md](SPEC.md) (no look-ahead leakage, label correctness,
feature sanity, model output shapes) are enforced by a pytest suite:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Modelling notes

- **Labels.** For horizon `h`, the forward return `close[t+h]/close[t] - 1` is
  bucketed into down / flat / up. The flat band widens with `sqrt(h)` so longer
  horizons aren't swamped by drift. Tune `flat_threshold` in `config.py`.
- **No leakage.** The train/val split is strictly time-ordered; the scaler is fit
  on training rows only, and a window is dropped if its future label is unknown.
- **Class imbalance** is handled with inverse-frequency class weights per horizon.
- **Receptive field** must cover `window`. With kernel 3 and 4 layers it's 61
  steps (printed at startup) — increase `channels`/`kernel_size` if you enlarge
  `window`.

## Caveats

Stock direction prediction is genuinely hard and markets are close to efficient;
treat accuracy modestly above the majority-class baseline as a real result, and
**do not** use this for live trading without rigorous walk-forward backtesting,
transaction-cost modelling, and out-of-sample validation across many symbols.
