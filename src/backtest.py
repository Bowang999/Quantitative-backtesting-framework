from __future__ import annotations
import pandas as pd
import numpy as np

def run_backtest_weights(
    ret: pd.DataFrame,
    signal: pd.DataFrame,
    universe: pd.DataFrame,
    weight_fn,
    rebalance_freq: int = 5,
    cost_bps: float = 10.0,
):
    dates = ret.index
    stocks = ret.columns

    nav = pd.Series(1.0, index=dates)
    daily_ret = pd.Series(0.0, index=dates)

    weights = pd.DataFrame(0.0, index=dates, columns=stocks)
    turnover = pd.Series(0.0, index=dates)

    w_prev = pd.Series(0.0, index=stocks)

    for i in range(len(dates) - 1):
        t = dates[i]

        if (i % rebalance_freq) == 0:
            sig_t = signal.loc[t].where(universe.loc[t])
            w_t = weight_fn(sig_t).fillna(0.0)
        else:
            w_t = w_prev

        to = (w_t - w_prev).abs().sum()
        cost = to * (cost_bps / 10000.0)
        turnover.iloc[i+1] = to

        r_next = ret.iloc[i+1].fillna(0.0)
        pr = (w_t * r_next).sum() - cost

        daily_ret.iloc[i+1] = pr
        nav.iloc[i+1] = nav.iloc[i] * (1.0 + pr)

        weights.iloc[i+1] = w_t
        w_prev = w_t

    return nav, daily_ret, weights, turnover
