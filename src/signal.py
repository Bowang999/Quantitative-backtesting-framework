# src/signal.py
import pandas as pd
import numpy as np

def long_short_half_weight(signal: pd.Series) -> pd.Series:
    """
    前50%做多，后50%做空，等权；多头权重和=+1，空头权重和=-1
    """
    s = signal.dropna()
    w = pd.Series(0.0, index=signal.index)

    n = len(s)
    if n < 2:
        return w

    s = s.sort_values(ascending=False)

    n_long = n // 2
    n_short = n - n_long  # 保证覆盖所有（奇数时空头多1个也可以）

    long_idx = s.index[:n_long]
    short_idx = s.index[n_long:]

    if len(long_idx) > 0:
        w.loc[long_idx] = 1.0 / len(long_idx)
    if len(short_idx) > 0:
        w.loc[short_idx] = -1.0 / len(short_idx)

    return w


def zscore_cs(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd is None or np.isnan(sd) or sd < 1e-12:
        return x * 0.0
    return (x - mu) / sd

def topk_equal_weight(signal: pd.Series, k: int = 200) -> pd.Series:
    s = signal.dropna().sort_values(ascending=False)
    picks = s.index[:k]
    w = pd.Series(0.0, index=signal.index)
    if len(picks) > 0:
        w.loc[picks] = 1.0 / len(picks)
    return w
