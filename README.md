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
notebooks/
  colab_train.ipynb   # run training + inference on a Colab runtime
```

## Quick start (local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --symbol 600519 --epochs 60
```

The best checkpoint (by mean validation accuracy) is written to
`checkpoints/tcn_<symbol>.pt` along with the scaler stats and config.

## Running on a Colab GPU (recommended)

1. Push this repo to GitHub.
2. Open `notebooks/colab_train.ipynb` in VS Code.
3. In the kernel picker (top-right), connect to a **Colab** runtime and choose a
   **GPU** runtime.
4. Set `REPO_URL` in the first cell to your repo URL (the Colab VM is a fresh
   cloud machine and can't see your local files), then run all cells.

> The notebook auto-detects Colab vs. a local kernel. On a local kernel it
> imports `src` directly; on Colab it clones `REPO_URL`.

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
