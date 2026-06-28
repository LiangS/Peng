"""AKShare data download with on-disk caching.

AKShare returns Chinese column names; we normalise them to lowercase English
OHLCV so the rest of the pipeline is source-agnostic.
"""

from __future__ import annotations

import os

import pandas as pd

# Map AKShare's Chinese columns -> canonical English names.
_RENAME = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_change",
    "换手率": "turnover",
}

_KEEP = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]


def load_prices(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    cache_dir: str = "data/cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a daily OHLCV DataFrame indexed by date (ascending).

    Results are cached as parquet so repeated runs (and offline Colab cells)
    don't re-hit the network.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"{symbol}_{start_date}_{end_date}_{adjust}.parquet"
    )

    if os.path.exists(cache_path) and not refresh:
        df = pd.read_parquet(cache_path)
    else:
        import akshare as ak  # imported lazily so the module loads without akshare

        raw = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        if raw is None or raw.empty:
            raise ValueError(
                f"AKShare returned no data for symbol={symbol}. "
                "Check the code (6-digit A-share) and date range."
            )
        df = raw.rename(columns=_RENAME)
        df = df[[c for c in _KEEP if c in df.columns]].copy()
        df.to_parquet(cache_path, index=False)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


if __name__ == "__main__":  # quick smoke test
    out = load_prices("600519", "20230101", "20231231")
    print(out.tail())
    print(f"{len(out)} rows, columns: {list(out.columns)}")
