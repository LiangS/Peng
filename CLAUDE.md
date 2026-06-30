# CLAUDE.md

Guidance for working in this repo. See [README.md](README.md) for the full project overview; this file records conventions and the non-obvious gotchas.

## What this is

A PyTorch **Temporal Convolutional Network (TCN)** that predicts **multi-horizon directional trend** for Chinese A-shares. One shared TCN backbone feeds **one softmax head per horizon** (`[1, 5, 20]` trading days), each emitting a distribution over `{down, flat, up}`. Data comes from **AKShare**; training is meant to run on a **Colab GPU runtime** via the VS Code Google Colab extension, with code version-controlled here.

## Layout

- `src/config.py` — single `Config` dataclass; all hyperparameters live here.
- `src/data.py` — AKShare OHLCV download + parquet caching (Chinese→English columns).
- `src/features.py` — technical features + 3-class forward-direction labels.
- `src/dataset.py` — sliding windows, time-ordered split, train-only scaler, class weights.
- `src/model.py` — `MultiHorizonTCN` (dilated causal convs + per-horizon heads).
- `src/train.py` — multi-task weighted-CE loop, early stopping, checkpointing.
- `notebooks/colab_train.ipynb` — clone/pull repo on Colab, train, run inference.

## Commands

```bash
# Local (CPU is fine; the model is small)
pip install -r requirements.txt
python -m src.train --symbol 600519 --epochs 60   # writes checkpoints/tcn_<symbol>.pt

# Syntax/JSON sanity checks
python -m py_compile src/*.py
python -c "import json; json.load(open('notebooks/colab_train.ipynb'))"
```

On Colab: set `REPO_URL` in the notebook's first cell, connect a GPU runtime, Run all.

## Gotchas (learned the hard way)

- **AKShare `qfq` prices can be ≤ 0** in the oldest history (dividend subtraction). They break `log` returns and contaminate rolling features. `features._drop_nonpositive_prices` drops the leading bad block; the log-return ratio is also guarded. Don't remove these.
- **No look-ahead leakage** is load-bearing: train/val split is strictly time-ordered, the scaler is fit on train rows only, and a window is dropped if its future label is unknown. Preserve this whenever touching `dataset.py`.
- **Colab clone must stay idempotent.** The setup cell pins an absolute `REPO_DIR=/content/Peng` and removes any nested `/content/Peng/Peng` before clone/pull — re-running it from inside the checkout previously created a nested copy running stale code.
- **Validation accuracy is misleading.** The loss uses inverse-frequency class weights, so raw accuracy can sit near chance (~0.33 for 3 classes) while predictions are actually balanced. Always compare against the **majority-class baseline** and prefer **macro-F1 / balanced accuracy**.
- **Edit the notebook via the JSON**, not the Edit tool (it rejects `.ipynb`). When the user runs on Colab, executed outputs live in the editor session, not the file — they must save (⌘S) for outputs to reach disk.

## Conventions

- Keep all tunables in `Config`; thread them through, don't hardcode.
- Use `torch.nn.utils.parametrizations.weight_norm` (not the deprecated `torch.nn.utils.weight_norm`); init conv weights **before** wrapping.
- Don't commit notebooks with executed outputs baked in; strip them first.

## Modelling status / next steps

First single-symbol baseline sits at chance (~0.33). Highest-leverage improvements, in order: (1) report majority-class baseline + macro-F1, (2) **train across many symbols** (one stock ≈ 1,850 days is too little data; window each symbol independently — never across symbols), (3) cheap knobs: try binary up/down, widen `flat_threshold`, lower `lr`. Daily direction is near-random; ~53–55% is a genuinely good result.
