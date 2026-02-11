# src/metrics.py
import pandas as pd
import numpy as np

def annual_return(nav: pd.Series, ann: int = 252) -> float:
    """
    年化收益率（CAGR）：(终值/初值)^(ann/交易日数) - 1
    """
    x = nav.dropna()
    if len(x) < 2:
        return np.nan
    total = x.iloc[-1] / x.iloc[0]
    n = len(x) - 1
    if n <= 0:
        return np.nan
    return total ** (ann / n) - 1

def sharpe_ratio(daily_ret: pd.Series, rf: float = 0.0, ann: int = 252) -> float:
    """
    年化夏普：sqrt(ann) * mean(excess daily ret) / std(daily ret)
    rf 为年化无风险利率（默认0）
    """
    r = daily_ret.dropna() - rf / ann
    sd = r.std(ddof=0)
    if sd < 1e-12:
        return np.nan
    return np.sqrt(ann) * r.mean() / sd

def max_drawdown(nav: pd.Series) -> float:
    """
    最大回撤（负数），例如 -0.25 表示最大回撤25%
    """
    x = nav.dropna()
    if x.empty:
        return np.nan
    peak = x.cummax()
    dd = x / peak - 1.0
    return dd.min()
