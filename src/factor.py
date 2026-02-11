# src/factor.py
from __future__ import annotations
import numpy as np
import pandas as pd

def winsorize_mad_cs(x: pd.Series, k: float = 5.0) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if mad is None or mad < 1e-12 or np.isnan(mad):
        return x
    lo = med - k * 1.4826 * mad
    hi = med + k * 1.4826 * mad
    return x.clip(lo, hi)

def zscore_cs(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd is None or sd < 1e-12 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd

def neutralize_markettype_cs(
    f: pd.Series,
    mtype: pd.Series,
    mktcap: pd.Series | None = None,
) -> pd.Series:
    df = pd.concat([f.rename("f"), mtype.rename("mtype")], axis=1)

    if mktcap is not None:
        df = pd.concat([df, mktcap.rename("mktcap")], axis=1)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["f", "mtype"])
    if df.empty:
        return f * np.nan

    # 稳定行业类型
    df["mtype"] = df["mtype"].astype("category")

    y = df["f"].astype(float).to_numpy()

    dummies = pd.get_dummies(df["mtype"], drop_first=True, dtype=float)
    X_parts = [np.ones((len(df), 1), dtype=float)]
    if dummies.shape[1] > 0:
        X_parts.append(dummies.to_numpy())

    if mktcap is not None:
        cap = df["mktcap"].astype(float)
        cap_mask = np.isfinite(cap) & (cap > 0)
        df2 = df.loc[cap_mask].copy()
        if df2.empty:
            return f * np.nan

        df2["mtype"] = df2["mtype"].astype("category")
        y2 = df2["f"].astype(float).to_numpy()
        d2 = pd.get_dummies(df2["mtype"], drop_first=True, dtype=float)

        # const + dummies + logcap
        p = 1 + d2.shape[1] + 1
        if len(df2) < p + 5:
            return f * np.nan  # 固定规则：样本不足就不出值（避免时变口径）

        X2_parts = [np.ones((len(df2), 1), dtype=float)]
        if d2.shape[1] > 0:
            X2_parts.append(d2.to_numpy(dtype=float))

        logcap = np.log(df2["mktcap"].astype(float).clip(lower=1.0).to_numpy()).reshape(-1, 1)
        X2_parts.append(logcap)

        X2 = np.concatenate(X2_parts, axis=1)
        beta = np.linalg.lstsq(X2, y2, rcond=None)[0]
        resid = y2 - X2 @ beta

        out = pd.Series(index=f.index, dtype=float)
        out.loc[df2.index] = resid
        return out

    # 仅行业回归
    X = np.concatenate(X_parts, axis=1)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta

    out = pd.Series(index=f.index, dtype=float)
    out.loc[df.index] = resid
    return out

def factor_turnover_liquidity(
    amount: pd.DataFrame,
    float_mktcap: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    因子 = 20日均值( log(amount / float_mktcap) )
    """
    x = amount / float_mktcap
    x = np.log(x.replace([np.inf, -np.inf], np.nan))
    f = x.rolling(window).mean()
    return f

def factor_std20_ret(
    ret: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    std20 波动率因子（收益率标准差）：
    = 20日滚动标准差( daily return )

    参数：
    - ret: 日收益率面板 (date × stock)
    - window: 滚动窗口，默认20日
    """
    f = ret.rolling(window).std(ddof=0)
    return f

def factor_amihud_illiq(
    ret: pd.DataFrame,
    amount: pd.DataFrame,
    window: int = 20,
    eps: float = 1e-12
) -> pd.DataFrame:
    """
    Amihud Illiquidity 因子（非流动性）：
    illiq_t = |ret_t| / amount_t
    factor = rolling mean over window

    ret: 日收益率面板
    amount: 日成交额面板（元）
    """
    amt = amount.clip(lower=eps)  # 防止除0
    illiq = ret.abs() / amt
    f = illiq.rolling(window).mean()
    return f

def factor_liq_risk_mom(
    close: pd.DataFrame,
    ret: pd.DataFrame,
    amount: pd.DataFrame,
    window_mom: int = 20,
    window_rev: int = 5,
    window_vol: int = 20,
    window_illiq: int = 20,
    eps: float = 1e-12,
    w_mom: float = 1.0,
    w_rev: float = 0.5,
    w_vol: float = 0.7,
    w_illiq: float = 0.7,
) -> pd.DataFrame:
    """
    综合因子（越大越做多）：
    score = + w_mom * z(mom20)
            + w_rev * z(rev5)
            - w_vol * z(vol20)
            - w_illiq * z(illiq20)

    其中：
    mom20   = close.pct_change(20)
    rev5    = -close.pct_change(5)  (短期反转，过去涨太多→未来倾向回归)
    vol20   = std20(ret)
    illiq20 = MA20(|ret|/amount)  (Amihud illiquidity)
    """
    # 1) 中期动量
    mom = close.pct_change(window_mom, fill_method=None)

    # 2) 短期反转
    rev = -close.pct_change(window_rev, fill_method=None)

    # 3) 波动率
    vol = ret.rolling(window_vol).std(ddof=0)

    # 4) 非流动性（Amihud）
    amt = amount.clip(lower=eps)
    illiq = (ret.abs() / amt).rolling(window_illiq).mean()

    # 5) 每天截面标准化后组合（更稳健）
    mom_z = mom.apply(zscore_cs, axis=1)
    rev_z = rev.apply(zscore_cs, axis=1)
    vol_z = vol.apply(zscore_cs, axis=1)
    ill_z = illiq.apply(zscore_cs, axis=1)

    f = (w_mom * mom_z) + (w_rev * rev_z) - (w_vol * vol_z) - (w_illiq * ill_z)
    return f

def factor_mom(close, window=60):
    f = close.pct_change(window, fill_method=None)
    return f

def factor_rev(close, window=5):
    f = -close.pct_change(window, fill_method=None)
    return f

def factor_downside_std20(ret, window=20):
    neg = ret.clip(upper=0.0)
    f = neg.rolling(window).std(ddof=0)
    return f

def factor_obv_slope(close, volume, window=20):
    sign = np.sign(close.diff())
    obv = (sign * volume).fillna(0.0).cumsum()
    # 取rolling差分近似 slope
    f = (obv - obv.shift(window)) / window
    return f

def factor_intraday_mom(openp, close):
    f = close / openp - 1
    return f

def factor_range(high, low, close, eps=1e-12):
    
    f = (high - low) / close.clip(lower=eps)
    return f


def factor_abnormal_volume(
    volume: pd.DataFrame,    # 用 Dnshrtrd 面板（date×stock）
    window: int = 252,
    eps: float = 1e-12
) -> pd.DataFrame:
    """
    异常成交量 = 当日成交量 / 过去252日平均成交量
    注意：均值用 shift(1) 防止未来函数（不把当天成交量算进历史均值）
    """
    vol = volume.replace([np.inf, -np.inf], np.nan)
    hist_mean = vol.rolling(window, min_periods=window).mean().shift(1)
    f = vol / (hist_mean + eps)
    return f

def factor_daily_extreme_return(
    ret: pd.DataFrame,
    universe: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    日度极端收益 = (个股当日收益 - 当日横截面均值收益)^2
    如果给 universe，就只在可交易股票池里计算横截面均值
    """
    r = ret.copy()
    if universe is not None:
        r = r.where(universe)

    cs_mean = r.mean(axis=1)                 # 每天的横截面均值（Series）
    extreme = (r.sub(cs_mean, axis=0)) ** 2  # (r_i,t - mean_t)^2
    return extreme

def factor_candle_shadow_daily(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    日频上/下影线：
    upper = High - max(Open, Close)
    lower = min(Open, Close) - Low
    """
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low
    upper = upper.replace([np.inf, -np.inf], np.nan)
    lower = lower.replace([np.inf, -np.inf], np.nan)
    return upper, lower


def factor_candle_shadow_std_daily(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 5,
    eps: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    标准化影线：
    upper_std = upper_t / mean(upper_{t-1..t-window})
    lower_std = lower_t / mean(lower_{t-1..t-window})
    注意：用 shift(1) 避免把当天算进“过去均值”（更严谨）
    """
    upper, lower = factor_candle_shadow_daily(open_, high, low, close)
    upper_mean = upper.rolling(window, min_periods=window).mean().shift(1)
    lower_mean = lower.rolling(window, min_periods=window).mean().shift(1)
    upper_std = upper / (upper_mean + eps)
    lower_std = lower / (lower_mean + eps)
    return upper_std, lower_std


def factor_candle_shadow_monthly(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    std_window: int = 5,
    lookback: int = 20,
    which: str = "upper_mean",
) -> pd.DataFrame:
    """
    月频因子（在月末计算）：
    先计算 daily 标准化影线 upper_std/lower_std，
    再在每个月末，回看过去 lookback 日：
      upper_mean: mean(upper_std)
      upper_std : std(upper_std)
      lower_mean: mean(lower_std)
      lower_std : std(lower_std)

    返回：月末那天有值，其它日为 NaN（后续再 forward fill 给回测用）
    """
    upper_std, lower_std = factor_candle_shadow_std_daily(
        open_, high, low, close, window=std_window
    )

    if which == "upper_mean":
        daily = upper_std
        agg = daily.rolling(lookback, min_periods=lookback).mean()
    elif which == "upper_std":
        daily = upper_std
        agg = daily.rolling(lookback, min_periods=lookback).std(ddof=0)
    elif which == "lower_mean":
        daily = lower_std
        agg = daily.rolling(lookback, min_periods=lookback).mean()
    elif which == "lower_std":
        daily = lower_std
        agg = daily.rolling(lookback, min_periods=lookback).std(ddof=0)
    else:
        raise ValueError("which must be one of upper_mean/upper_std/lower_mean/lower_std")

    # 只保留月末（生成与 agg 同形状的 mask）
    idx = agg.index
    is_month_end = idx.to_series().dt.to_period("M").ne(
        idx.to_series().shift(-1).dt.to_period("M")
    ).values  # shape: (T,)

    mask = pd.DataFrame(
        np.repeat(is_month_end[:, None], agg.shape[1], axis=1),
        index=agg.index,
        columns=agg.columns
    )

    factor_m = agg.where(mask)

    return factor_m


def factor_williams_shadow_monthly(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    std_window: int = 5,
    lookback: int = 20,
    which: str = "upper_mean",  # upper_mean/upper_std/lower_mean/lower_std
) -> pd.DataFrame:
    """
    威廉上下影线月频因子（以收盘价为基准）：
    1) daily: WU = High - Close, WL = Close - Low
    2) daily standardize: divide by past std_window-day mean
    3) month-end: lookback-day mean/std (only month-end has value)
    输出：月末才有值，其余为 NaN
    """

    # 1) daily williams shadows
    wu = (high - close).clip(lower=0)
    wl = (close - low).clip(lower=0)

    # 2) daily standardize by past std_window mean (use shift(1) avoid look-ahead)
    wu_den = wu.rolling(std_window).mean().shift(1)
    wl_den = wl.rolling(std_window).mean().shift(1)

    wu_std = wu / wu_den.replace(0, np.nan)
    wl_std = wl / wl_den.replace(0, np.nan)

    # 3) month-end aggregation over past lookback days (again shift(1) so month-end uses info up to that day)
    upper_mean = wu_std.rolling(lookback).mean()
    upper_std  = wu_std.rolling(lookback).std(ddof=0)

    lower_mean = wl_std.rolling(lookback).mean()
    lower_std  = wl_std.rolling(lookback).std(ddof=0)

    if which == "upper_mean":
        agg = upper_mean
    elif which == "upper_std":
        agg = upper_std
    elif which == "lower_mean":
        agg = lower_mean
    elif which == "lower_std":
        agg = lower_std
    else:
        raise ValueError("which must be one of: upper_mean/upper_std/lower_mean/lower_std")

    # 4) keep month-end only
    idx = agg.index
    # month end = this date's month != next date's month
    is_month_end = idx.to_series().dt.to_period("M").ne(
        idx.to_series().shift(-1).dt.to_period("M")
    )
    factor_m = agg.where(is_month_end, np.nan)

    return factor_m

def monthly_to_daily_hold(factor_m: pd.DataFrame) -> pd.DataFrame:
    """
    把“月末因子”扩展成“月初~月末持有”的日频信号：
    月末计算出来的因子 -> 下一天开始生效（避免未来函数）
    实现：先 shift(1) 再 forward fill
    """
    return factor_m.shift(1).ffill()

def monthly_to_daily_next_month(factor_m: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    严格月频持有：
    - factor_m：只有月末有值的DataFrame（index=交易日，月末那天有值，其它NaN）
    - 输出：日频DataFrame，在“下个月第一个交易日”开始持有该月末因子值，并一直持有到下个月第一个交易日前一天
    """
    # 1) 把月末值取出来：每月最后一个交易日的截面
    fm = factor_m.reindex(daily_index)
    month_end_vals = fm.dropna(how="all")

    # 2) 找到每个月的“第一个交易日”
    daily = pd.DataFrame(index=daily_index)
    month = daily_index.to_period("M")
    first_trade = pd.Series(daily_index, index=daily_index).groupby(month).min()

    # 3) 构造：把“上个月月末值”放到“本月第一个交易日”上
    #    即：month_end_vals 的月份 m -> 生效日是月份 m+1 的 first_trade
    eff = pd.DataFrame(index=daily_index, columns=fm.columns, dtype=float)

    # month_end_vals 的 key 是具体日期，我们转成月份
    me_month = month_end_vals.index.to_period("M")
    for d, m in zip(month_end_vals.index, me_month):
        next_m = (m + 1)
        if next_m in first_trade.index:
            eff_day = first_trade.loc[next_m]
            eff.loc[eff_day] = month_end_vals.loc[d]

    # 4) 月内持有：从生效日开始向后填充
    eff = eff.ffill()

    return eff

def neutralize_size_cs(f: pd.Series, mktcap: pd.Series) -> pd.Series:
    df = pd.concat([f, mktcap], axis=1)
    df.columns = ["f", "mktcap"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.shape[0] < 20:
        return f - f.mean()

    x = np.log(df["mktcap"].clip(lower=1.0)).values
    y = df["f"].values
    X = np.column_stack([np.ones_like(x), x])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta

    out = pd.Series(index=f.index, dtype=float)
    out.loc[df.index] = resid
    return out

def factor_ubl_monthly(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    mktcap: pd.DataFrame,
    std_window: int = 5,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    UBL（月末）= zscore(candle_upper_std_desize) + zscore(williams_lower_mean_desize)
    输出：月末才有值
    """
    # 1) 月末：蜡烛上_std
    candle_upper_std = factor_candle_shadow_monthly(
        open_=open_, high=high, low=low, close=close,
        std_window=std_window, lookback=lookback,
        which="upper_std"
    )

    # 2) 月末：威廉下_mean
    will_lower_mean = factor_williams_shadow_monthly(
        open_=open_, high=high, low=low, close=close,
        std_window=std_window, lookback=lookback,
        which="lower_mean"
    )

    # 3) 月末：市值中性化 desize（逐日横截面）
    candle_desize = pd.DataFrame(
        [neutralize_size_cs(candle_upper_std.loc[t], mktcap.loc[t]) for t in candle_upper_std.index],
        index=candle_upper_std.index, columns=candle_upper_std.columns
    )

    will_desize = pd.DataFrame(
        [neutralize_size_cs(will_lower_mean.loc[t], mktcap.loc[t]) for t in will_lower_mean.index],
        index=will_lower_mean.index, columns=will_lower_mean.columns
    )

    # 4) 月末：横截面标准化 + 等权相加
    candle_z = candle_desize.apply(zscore_cs, axis=1)
    will_z   = will_desize.apply(zscore_cs, axis=1)

    ubl_m = candle_z + will_z
    return ubl_m

def factor_high0_daily(
    high: pd.DataFrame,
    close: pd.DataFrame,
    eps: float = 1e-12,
) -> pd.DataFrame:
    
    f = high / (close + eps)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f

def KLEN(high: pd.DataFrame,
         low: pd.DataFrame,
         open1:pd.DataFrame,
         eps:float = 1e-12) -> pd.DataFrame:
    
    f = (high - low) / (open1)
    f = f.replace([np.inf, -np.inf], np.nan)
    
    return f

def VWAP0(vwap: pd.DataFrame,
          close: pd.DataFrame) -> pd.DataFrame:
    
    f = vwap / close
    f = f.replace([np.inf, -np.inf], np.nan)
    
    return f

def factor_corr(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    
    if min_periods is None:
        min_periods = d

    x = close
    y = np.log(volume.clip(lower=0) + 1.0)

    f = x.rolling(window=d, min_periods=min_periods).corr(y)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f

def factor_beta_20(
    close: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    beta20 = Slope(close, d) / close
    where Slope is the OLS slope of close over time index [0..d-1] in a rolling window.
    """
    if min_periods is None:
        min_periods = d

    # time index within window
    x = np.arange(d, dtype=float)
    x_mean = x.mean()
    x_demean = x - x_mean
    denom = (x_demean ** 2).sum()  # Var(x) * d

    # rolling mean of close
    y = close.astype(float)
    y_mean = y.rolling(d, min_periods=min_periods).mean()

    # sum((x-xbar)*y)
    # rolling dot product: Σ (x_demean[i] * y_{t-d+1+i})
    # use rolling apply on numpy array for each column (fast enough in pandas)
    def _rolling_sxy(a: np.ndarray) -> float:
        return np.dot(x_demean, a)

    sxy = y.rolling(d, min_periods=min_periods).apply(_rolling_sxy, raw=True)

    slope = sxy / denom

    beta = slope / y  # divide by today's close
    beta = beta.replace([np.inf, -np.inf], np.nan)
    return beta


def factor_cntd20(close: pd.DataFrame, d: int = 20, min_periods: int | None = None) -> pd.DataFrame:
    """
    cntd_d = Mean(close > Ref(close,1), d) - Mean(close < Ref(close,1), d)
    """
    if min_periods is None:
        min_periods = d

    prev = close.shift(1)
    up = (close > prev).astype(float)     # True->1, False->0, NaN->NaN
    down = (close < prev).astype(float)

    up_mean = up.rolling(d, min_periods=min_periods).mean()
    down_mean = down.rolling(d, min_periods=min_periods).mean()

    fac = up_mean - down_mean
    return fac

def factor_cntn20(close: pd.DataFrame, d: int = 20, min_periods: int | None = None) -> pd.DataFrame:
    """
    cntn_d = Mean(close < Ref(close, 1), d)
    过去d天“下跌天数占比”
    """
    if min_periods is None:
        min_periods = d

    prev = close.shift(1)
    down = (close < prev).astype(float)  # True->1, False->0

    fac = down.rolling(d, min_periods=min_periods).mean()
    return fac

def factor_cntp20(close: pd.DataFrame, d: int = 20, min_periods: int | None = None) -> pd.DataFrame:
    """
    cntp_d = Mean(close > Ref(close, 1), d)
    过去d天“上涨天数占比”
    """
    if min_periods is None:
        min_periods = d

    prev = close.shift(1)
    up = (close > prev).astype(float)

    fac = up.rolling(d, min_periods=min_periods).mean()
    return fac

def factor_imax20(high: pd.DataFrame, d: int = 20, min_periods: int | None = None) -> pd.DataFrame:
    if min_periods is None:
        min_periods = d

    def _idxmax(a: np.ndarray) -> float:
        return float(np.nanargmax(a) + 1)  # 1..d

    idx = high.rolling(d, min_periods=min_periods).apply(_idxmax, raw=True)
    fac = idx / float(d)
    return fac

def factor_imin20(low: pd.DataFrame, d: int = 20, min_periods: int | None = None) -> pd.DataFrame:
    if min_periods is None:
        min_periods = d

    def _idxmin(a: np.ndarray) -> float:
        return float(np.nanargmin(a) + 1)  # 1..d

    idx = low.rolling(d, min_periods=min_periods).apply(_idxmin, raw=True)
    fac = idx / float(d)
    return fac

def factor_max20(
    high: pd.DataFrame,
    close: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    max_d = Max(high, d) / close
    """
    if min_periods is None:
        min_periods = d

    max_high = high.rolling(d, min_periods=min_periods).max()
    fac = max_high.div(close)

    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_min20(
    low: pd.DataFrame,
    close: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    min_d = Min(low, d) / close
    """
    if min_periods is None:
        min_periods = d

    min_low = low.rolling(d, min_periods=min_periods).min()
    fac = min_low.div(close)

    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_qtld20(close: pd.DataFrame, d: int = 20, q: float = 0.2,
                min_periods: int | None = None) -> pd.DataFrame:
    """
    qtld_d = Quantile(close, d, q) / close
    """
    if min_periods is None:
        min_periods = d

    q_close = close.rolling(d, min_periods=min_periods).quantile(q)
    fac = q_close.div(close)

    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_qtlu20(close: pd.DataFrame, d: int = 20, q: float = 0.8,
                min_periods: int | None = None) -> pd.DataFrame:
    """
    qtlu_d = Quantile(close, d, q) / close
    """
    if min_periods is None:
        min_periods = d

    q_close = close.rolling(d, min_periods=min_periods).quantile(q)
    fac = q_close.div(close)

    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_rank20(close: pd.DataFrame, d: int = 20, min_periods: int | None = None) -> pd.DataFrame:
    """
    rank_d = Rank(close, d)
    Time-series rank: position of close[t] within last d closes, scaled to [0,1].
    """
    if min_periods is None:
        min_periods = d

    def _ts_rank(a: np.ndarray) -> float:
        # a is length d, oldest -> latest
        x = a[-1]
        if np.isnan(x):
            return np.nan
        # percentile rank: proportion of values <= current
        return np.nanmean(a <= x)

    fac = close.rolling(d, min_periods=min_periods).apply(_ts_rank, raw=True)
    return fac

def factor_resi20(
    close: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    resi_d = Resi(close, d) / close
    Resi is the last-day residual of OLS fit close ~ a + b*time over last d days.
    """
    if min_periods is None:
        min_periods = d

    y = close.astype(float)

    # time index 0..d-1
    x = np.arange(d, dtype=float)
    x_mean = x.mean()
    x_demean = x - x_mean
    denom = (x_demean ** 2).sum()  # Σ (x-xbar)^2

    # rolling means
    y_mean = y.rolling(d, min_periods=min_periods).mean()

    # rolling sum of (x-xbar)*y
    def _rolling_sxy(a: np.ndarray) -> float:
        return float(np.dot(x_demean, a))

    sxy = y.rolling(d, min_periods=min_periods).apply(_rolling_sxy, raw=True)

    # slope b and intercept a in rolling window
    b = sxy / denom
    # a = ybar - b*xbar
    a = y_mean - b * x_mean

    # predicted value at last point x_last = d-1
    x_last = float(d - 1)
    y_hat_last = a + b * x_last

    # residual at time t (last point in the window)
    resi = y - y_hat_last

    fac = resi.div(y)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_roc20(close: pd.DataFrame, d: int = 20) -> pd.DataFrame:
    """
    roc_d = Ref(close, d) / close = close[t-d] / close[t]
    """
    ref = close.shift(d)
    fac = ref.div(close)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_rsqr20(close: pd.DataFrame, d: int = 20, min_periods: int | None = None) -> pd.DataFrame:
    """
    rsqr_d = Rsquare(close, d)
    Rolling R^2 of close regressed on time index [0..d-1].
    """
    if min_periods is None:
        min_periods = d

    x = np.arange(d, dtype=float)

    def _r2(a: np.ndarray) -> float:
        # a: length d array (oldest -> latest)
        if np.isnan(a).any():
            return np.nan
        y = a
        # linear fit y = p1*x + p0
        p1, p0 = np.polyfit(x, y, 1)
        y_hat = p1 * x + p0
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot == 0:
            return np.nan
        return 1.0 - ss_res / ss_tot

    fac = close.rolling(d, min_periods=min_periods).apply(_r2, raw=True)
    return fac

def factor_rsv20(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    rsv_d = (close - Min(low,d)) / (Max(high,d) - Min(low,d) + eps)
    """
    if min_periods is None:
        min_periods = d

    low_min = low.rolling(d, min_periods=min_periods).min()
    high_max = high.rolling(d, min_periods=min_periods).max()

    denom = (high_max - low_min) + eps
    fac = (close - low_min).div(denom)

    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_sumd20(
    close: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    sumd_d = (Sum(pos_delta,d) - Sum(pos_-delta,d)) / (Sum(abs_delta,d) + eps)
    where delta = close - close.shift(1)
    """
    if min_periods is None:
        min_periods = d

    delta = close - close.shift(1)

    up = delta.clip(lower=0.0)          # max(delta,0)
    down = (-delta).clip(lower=0.0)     # max(-delta,0)
    absd = delta.abs()

    up_sum = up.rolling(d, min_periods=min_periods).sum()
    down_sum = down.rolling(d, min_periods=min_periods).sum()
    abs_sum = absd.rolling(d, min_periods=min_periods).sum()

    fac = (up_sum - down_sum).div(abs_sum + eps)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_sumn20(
    close: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    sumn_d = Sum(max(Ref(close,1)-close, 0), d) / (Sum(abs(close-Ref(close,1)), d) + eps)
           = Sum(max(-delta, 0), d) / (Sum(abs(delta), d) + eps)
    """
    if min_periods is None:
        min_periods = d

    delta = close - close.shift(1)

    down = (-delta).clip(lower=0.0)   # max(-delta,0)
    absd = delta.abs()

    down_sum = down.rolling(d, min_periods=min_periods).sum()
    abs_sum = absd.rolling(d, min_periods=min_periods).sum()

    fac = down_sum.div(abs_sum + eps)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_sump20(
    close: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    sump_d = Sum(max(close-Ref(close,1),0), d) / (Sum(abs(close-Ref(close,1)), d) + eps)
          = Sum(max(delta,0), d) / (Sum(abs(delta), d) + eps)
    """
    if min_periods is None:
        min_periods = d

    delta = close - close.shift(1)

    up = delta.clip(lower=0.0)      # max(delta,0)
    absd = delta.abs()

    up_sum = up.rolling(d, min_periods=min_periods).sum()
    abs_sum = absd.rolling(d, min_periods=min_periods).sum()

    fac = up_sum.div(abs_sum + eps)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_vma20(
    volume: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    vma_d = Mean(volume, d) / (volume + eps)
    """
    if min_periods is None:
        min_periods = d

    vmean = volume.rolling(d, min_periods=min_periods).mean()
    fac = vmean.div(volume + eps)

    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_vstd20(
    volume: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    vstd_d = Std(volume, d) / (volume + eps)
    """
    if min_periods is None:
        min_periods = d

    vstd = volume.rolling(d, min_periods=min_periods).std(ddof=0)  # ddof=0 更贴近量化库
    fac = vstd.div(volume + eps)

    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_vsumd20(
    volume: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    vsumd_d = (Sum(max(dv,0), d) - Sum(max(-dv,0), d)) / (Sum(abs(dv), d) + eps)
    where dv = volume - volume.shift(1)
    """
    if min_periods is None:
        min_periods = d

    dv = volume - volume.shift(1)

    up = dv.clip(lower=0.0)        # max(dv,0)
    down = (-dv).clip(lower=0.0)   # max(-dv,0)
    absd = dv.abs()

    up_sum = up.rolling(d, min_periods=min_periods).sum()
    down_sum = down.rolling(d, min_periods=min_periods).sum()
    abs_sum = absd.rolling(d, min_periods=min_periods).sum()

    fac = (up_sum - down_sum).div(abs_sum + eps)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_vsumn20(
    volume: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    vsumn_d = Sum(max(Ref(volume,1)-volume, 0), d) / (Sum(abs(volume-Ref(volume,1)), d) + eps)
           = Sum(max(-dv,0), d) / (Sum(abs(dv), d) + eps)
    """
    if min_periods is None:
        min_periods = d

    dv = volume - volume.shift(1)

    down = (-dv).clip(lower=0.0)    # max(-dv,0)
    absd = dv.abs()

    down_sum = down.rolling(d, min_periods=min_periods).sum()
    abs_sum = absd.rolling(d, min_periods=min_periods).sum()

    fac = down_sum.div(abs_sum + eps)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_vsump20(
    volume: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    vsump_d = Sum(max(volume-Ref(volume,1), 0), d) / (Sum(abs(volume-Ref(volume,1)), d) + eps)
           = Sum(max(dv,0), d) / (Sum(abs(dv), d) + eps)
    """
    if min_periods is None:
        min_periods = d

    dv = volume - volume.shift(1)

    up = dv.clip(lower=0.0)         # max(dv,0)
    absd = dv.abs()

    up_sum = up.rolling(d, min_periods=min_periods).sum()
    abs_sum = absd.rolling(d, min_periods=min_periods).sum()

    fac = up_sum.div(abs_sum + eps)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_wvma20(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
    ddof: int = 0,   # 因子库常用 0；想用样本标准差可改 1
) -> pd.DataFrame:
    """
    wvma_d = Std( abs(close/Ref(close,1)-1) * volume , d ) /
             (Mean( abs(close/Ref(close,1)-1) * volume , d ) + eps)
    """
    if min_periods is None:
        min_periods = d

    ret_abs = (close.div(close.shift(1)) - 1.0).abs()
    x = ret_abs * volume

    x_mean = x.rolling(d, min_periods=min_periods).mean()
    x_std  = x.rolling(d, min_periods=min_periods).std(ddof=ddof)

    fac = x_std.div(x_mean + eps)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_cord20(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    cord_d = Corr( close/Ref(close,1), Log(volume/Ref(volume,1) + 1), d )
    Rolling (time-series) correlation for each stock separately.
    """
    if min_periods is None:
        min_periods = d

    # x: close/prev_close (gross return, not minus 1)
    prev_close = close.shift(1)
    x = close.div(prev_close.where(prev_close.abs() > eps))  # avoid /0

    # y: log(volume/prev_volume + 1)
    prev_vol = volume.shift(1)
    vol_ratio = volume.div(prev_vol.where(prev_vol.abs() > eps))  # avoid /0
    y = np.log(vol_ratio.clip(lower=0.0) + 1.0)  # ensure inside log is >= 1

    fac = x.rolling(d, min_periods=min_periods).corr(y)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_corr20(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    corr_d = Corr(close, Log(volume+1), d)
    Rolling (time-series) correlation for each stock separately.
    """
    if min_periods is None:
        min_periods = d

    x = close.astype(float)

    # make sure volume non-negative, avoid log issues
    y = np.log(volume.clip(lower=0.0) + 1.0)

    fac = x.rolling(d, min_periods=min_periods).corr(y)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_std20(
    close: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Std(close, d) / close
    """
    if min_periods is None:
        min_periods = d

    s = close.rolling(d, min_periods=min_periods).std(ddof=ddof)
    fac = s / close
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_std20_improve1(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    d: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Mean(Greater(high-low, Greater(Abs(high-Ref(close,1)), Abs(low-Ref(close,1)))), d) / close
    """
    if min_periods is None:
        min_periods = d

    pc = close.shift(1)  # Ref(close,1)

    a = high - low
    b = (high - pc).abs()
    c = (low - pc).abs()

    inner = np.maximum(b, c)          # Greater(Abs(high-pc), Abs(low-pc))
    tr_like = np.maximum(a, inner)    # Greater(high-low, inner)

    m = tr_like.rolling(d, min_periods=min_periods).mean()
    fac = m / close

    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_std20_improve2(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    adjust: bool = False,   
    min_periods: int | None = None,  
) -> pd.DataFrame:
    """
    EMA(Greater(high-low, Greater(Abs(high-Ref(close,1)), Abs(low-Ref(close,1)))) * volume, d)
    / EMA(volume, d) / close
    """
    if min_periods is None:
        min_periods = d

    pc = close.shift(1)

    a = high - low
    b = (high - pc).abs()
    c = (low - pc).abs()

    # Greater(Abs(high-Ref(close,1)), Abs(low-Ref(close,1)))
    inner = np.maximum(b, c)

    # Greater(high-low, inner)
    tr_like = np.maximum(a, inner)

    num = (tr_like * volume).ewm(span=d, adjust=adjust, min_periods=min_periods).mean()
    den_v = volume.ewm(span=d, adjust=adjust, min_periods=min_periods).mean()

    fac = num / den_v / close
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor_std20_improve3(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    sign_eps: float = 1e-5,
    adjust: bool = False,
) -> pd.DataFrame:
    """
    EMA(Greater(Greater(high-low, Greater(Abs(high-Ref(close,1)), Abs(low-Ref(close,1)))),
                Abs(open-Ref(close,1)))
        * Sign(close-Ref(close,1)+1e-5) * volume, d)
    / EMA(volume, d) / close
    """
    pc = close.shift(1)

    inner = np.maximum((high - low), np.maximum((high - pc).abs(), (low - pc).abs()))

    amp = np.maximum(inner, (open_ - pc).abs())

    sgn = np.sign(close - pc + sign_eps)

    x = amp * sgn * volume

    ema_x = x.ewm(span=d, adjust=adjust, min_periods=d).mean()
    ema_v = volume.ewm(span=d, adjust=adjust, min_periods=d).mean()

    fac = ema_x / ema_v / close
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor1(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    adjust: bool = False,
) -> pd.DataFrame:
    """
    EMA( I(close>pc) * (close/pc-1)^2 * vol, d ) / EMA(vol, d)
  - EMA( I(close<pc) * (pc/close-1)^2 * vol, d ) / EMA(vol, d)
    """
    pc = close.shift(1)

    up = (close - pc) > 0
    dn = (pc - close) > 0

    r2_up = (close / pc - 1.0) ** 2
    r2_dn = (pc / close - 1.0) ** 2

    x_up = r2_up.where(up, 0.0) * volume
    x_dn = r2_dn.where(dn, 0.0) * volume

    ema_v  = volume.ewm(span=d, adjust=adjust, min_periods=d).mean()
    ema_up = x_up.ewm(span=d, adjust=adjust, min_periods=d).mean()
    ema_dn = x_dn.ewm(span=d, adjust=adjust, min_periods=d).mean()

    den = ema_v.where(ema_v != 0)
    fac = ema_up.div(den) - ema_dn.div(den)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    return fac

def factor2(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    adjust: bool = False,
) -> pd.DataFrame:
    """
    EMA( r * Sign(volume - EMA(volume,d)) * Abs(r), d ) / Std(r, d)
    r = close/Ref(close,1) - 1
    """
    pc = close.shift(1)
    r = close.div(pc) - 1.0

    ema_v = volume.ewm(span=d, adjust=adjust, min_periods=d).mean()

    sV = np.sign((volume - ema_v).astype(float))

    x = r * sV * r.abs()

    ema_x = x.ewm(span=d, adjust=adjust, min_periods=d).mean()

    std_r = r.rolling(window=d, min_periods=d).std(ddof=0)

    fac = ema_x.div(std_r).where(std_r > 0)
    return fac


def _rolling_linear_residual_last(y: np.ndarray) -> float:
    """
    给定窗口内 y（长度=window），返回最后一个点的线性回归残差：
    resi_last = y_last - (a + b*x_last), x = 0..n-1
    """
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan

    yy = y[mask]
    x = np.arange(len(y))[mask].astype(float)

    x_mean = x.mean()
    y_mean = yy.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return np.nan

    b = ((x - x_mean) * (yy - y_mean)).sum() / denom
    a = y_mean - b * x_mean

    x_last = float(len(y) - 1)
    y_last = y[-1]
    if not np.isfinite(y_last):
        return np.nan

    return y_last - (a + b * x_last)

def factor3(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    adjust: bool = False,
) -> pd.DataFrame:
    pc = close.shift(1)
    pl = low.shift(1)

    amp = np.maximum(
        (high - pc).astype(float),
        (close - pl).abs().astype(float),
    )

    r = close / pc - 1.0

    ema_v = volume.ewm(span=d, adjust=adjust, min_periods=1).mean()
    vratio = volume / ema_v

    x = amp * r * vratio
    ema_x = x.ewm(span=d, adjust=adjust, min_periods=1).mean()

    resi = close.rolling(window=d, min_periods=1).apply(
        _rolling_linear_residual_last, raw=True
    )

    std_resi = resi.rolling(window=d, min_periods=1).std()

    fac = ema_x / std_resi
    return fac

def factor4(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    adjust: bool = False,
) -> pd.DataFrame:
    pc = close.shift(1)
    pv = volume.shift(1)

    r = close / pc - 1.0
    g = volume / pv - 1.0

    x = r * g
    ema_x = x.ewm(span=d, adjust=adjust, min_periods=1).mean()

    std_r = r.rolling(window=d, min_periods=1).std()

    fac = ema_x / std_r
    return fac

def factor5(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    adjust: bool = False,
) -> pd.DataFrame:
    pc = close.shift(1)
    pv = volume.shift(1)

    r = close / pc - 1.0
    g = volume / pv - 1.0

    s = np.sign(g.astype(float))
    x = r * s

    ema_x = x.ewm(span=d, adjust=adjust, min_periods=1).mean()
    std_r = r.rolling(window=d, min_periods=1).std()

    fac = ema_x / std_r
    return fac

def factor_price_position_volume_zscore(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:

    low_min = low.rolling(d).min()
    high_max = high.rolling(d).max()
    denom = (high_max - low_min).replace(0, np.nan)

    price_pos = (close - low_min) / (high_max - low_min)
    
    vol_mean = volume.rolling(d).mean()
    vol_std = volume.rolling(d).std().replace(0, np.nan)

    vol_z = (volume - vol_mean) / vol_std

    factor = price_pos * vol_z

    return factor

def factor_log_price_pos_vol_ratio(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:

    low_mean = low.rolling(d).mean()
    high_mean = high.rolling(d).mean()

    price_ratio = (close - low_mean) / (high_mean - low_mean)

    price_ratio = np.log(price_ratio)

    vol_mean = volume.rolling(d).mean()
    vol_ratio = volume / vol_mean - 1

    factor = price_ratio * vol_ratio

    return factor

def rolling_slope(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    计算滚动窗口内的线性回归斜率：y ~ a + b*t，返回 b
    df: (T, N) 的DataFrame，T天、N只股票
    """
    if window <= 1:
        return df * np.nan

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_demean = x - x_mean
    denom = np.sum(x_demean ** 2)  # 常数

    def _slope_1d(y: np.ndarray) -> float:
        # y shape: (window,)
        if np.any(~np.isfinite(y)):
            return np.nan
        y_mean = y.mean()
        return np.sum((y - y_mean) * x_demean) / denom

    return df.rolling(window, min_periods=window).apply(_slope_1d, raw=True)

# 1
def factor_price_volume_gate(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d_price: int = 20,
    d_vol: int = 20,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    formula：
    Log(Greater(close, Mean(high,d)) / Less(close, Mean(low,d))) * Sign(Slope(volume,d))
    """
    mean_high = high.rolling(d_price, min_periods=d_price).mean()
    mean_low  = low.rolling(d_price, min_periods=d_price).mean()

    numerator = np.maximum(close, mean_high)
    denominator = np.minimum(close, mean_low)

    ratio = (numerator / (denominator.clip(lower=eps))).clip(lower=eps)
    price_part = np.log(ratio)

    vol_slope = rolling_slope(volume, d_vol)
    vol_part = np.sign(vol_slope)

    factor = price_part * vol_part
    return factor

# 2
def factor_pos_slope_vol_resi(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    eps: float = 1e-12
) -> pd.DataFrame:
    """
    Slope((close-Mean(low,d))/(Mean(high,d)-Mean(low,d)), d)
    * Sign(Resi(volume,d))
    """

    mean_low = low.rolling(d, min_periods=d).mean()
    mean_high = high.rolling(d, min_periods=d).mean()

    pos = (close - mean_low) / (mean_high - mean_low + eps)

    price_slope = rolling_slope(pos, d)

    vol_mean = volume.rolling(d, min_periods=d).mean()
    vol_resi = volume - vol_mean
    vol_sign = np.sign(vol_resi)

    factor = price_slope * vol_sign
    return factor

# 3
def factor_price_trend_x_vol_change(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    close_scale: float = 100.0,   
    d_slope: int = 20,            
    d_mean_vol: int = 20,         
    ref_lag: int = 20,            
    eps: float = 1e-12
) -> pd.DataFrame:
    """
    Sign(Slope(close/close_scale, d_slope)) * (Mean(volume,d_mean_vol)/Ref(volume,ref_lag) - 1)
    """
    price = close / close_scale
    slope_p = rolling_slope(price, d_slope)
    sign_trend = np.sign(slope_p)

    mean_vol = volume.rolling(d_mean_vol, min_periods=d_mean_vol).mean()
    ref_vol = volume.shift(ref_lag)

    vol_change = mean_vol / (ref_vol.clip(lower=eps)) - 1.0

    return sign_trend * vol_change

# 4
def factor_slope_price_x_slope_logvol(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    close_scale: float = 100.0,  
    d_price: int = 20,           
    d_vol: int = 20,            
    eps: float = 1e-12
) -> pd.DataFrame:
    """
    Slope(close/close_scale, d_price) * Slope(log(volume), d_vol)
    """
    price = close / close_scale
    slope_price = rolling_slope(price, d_price)

    log_vol = np.log(volume.clip(lower=eps))
    slope_logvol = rolling_slope(log_vol, d_vol)

    return slope_price * slope_logvol

# 5
def factor_price_slope_x_vol_anom(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    eps: float = 1e-12
) -> pd.DataFrame:
    """
    Slope(close,d) * (Ref(volume,d)/Mean(volume,d) - 1) / Std(volume,d)
    """

    price_slope = rolling_slope(close, d)

    vol_ref = volume.shift(d)
    vol_mean = volume.rolling(d, min_periods=d).mean()
    vol_std  = volume.rolling(d, min_periods=d).std()

    vol_anom = (vol_ref / (vol_mean + eps) - 1.0) / (vol_std + eps)

    return price_slope * vol_anom

# 6
def factor_pos_slope_x_zvol(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    eps: float = 1e-12
) -> pd.DataFrame:

    mean_low = low.rolling(d, min_periods=d).mean()
    mean_high = high.rolling(d, min_periods=d).mean()

    pos = (close - mean_low) / (mean_high - mean_low + eps)
    pos_slope = rolling_slope(pos, d)

    vol_mean = volume.rolling(d, min_periods=d).mean()
    vol_std  = volume.rolling(d, min_periods=d).std()

    zvol = (volume - vol_mean) / (vol_std + eps)

    return pos_slope * zvol

def Ref(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """Ref(X,N): N=0当前；N>0过去N期；N<0未来|N|期。pandas 对应 shift(N)。"""
    if N == 0:
        return df
    return df.shift(N)


def rolling_rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rank 单变量滚动window窗口期排名：返回当前值（窗口最后一个）在窗口内的“排名分位”∈[0,1]。
    这里采用常见定义：rank = mean(x <= last)  （并列按最大名次）
    - 若窗口内有 NaN/inf，则该窗口返回 NaN（严格语义）
    """
    def _rank_last(x: np.ndarray) -> float:
        if np.any(~np.isfinite(x)):
            return np.nan
        last = x[-1]
        # 分位：小于等于last的比例
        return float(np.mean(x <= last))

    return df.rolling(window, min_periods=window).apply(_rank_last, raw=True)

# 7
def factor_rel_strength_slope_x_vol_rank_change(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:

    mean_close = close.rolling(d, min_periods=d).mean()
    rel = close / mean_close  # mean_close==0 -> inf/NaN，按严格语义保留

    part1 = rolling_slope(rel, d)

    rk = rolling_rank(volume, d)
    part2 = rk - Ref(rk, d)

    return part1 * part2

# 8
def factor_log_pos_x_vol_ratio(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
) -> pd.DataFrame:

    mean_low = low.rolling(d, min_periods=d).mean()
    mean_high = high.rolling(d, min_periods=d).mean()

    pos = (close - mean_low) / (mean_high - mean_low)
    log_pos = np.log(pos)

    mean_vol = volume.rolling(d, min_periods=d).mean()
    vol_ratio = volume / (mean_vol) - 1.0

    return log_pos * vol_ratio

def safe_div0(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    out = a / b
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0)

# 9
def factor_pos_x_sign_abn_rvol(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d_pos: int = 20,
    k_ref: int = 1,
    d_mean: int = 20
) -> pd.DataFrame:

    # --- pos = (close - Mean(low)) / (Mean(high)-Mean(low)) ---
    mean_low = low.rolling(d_pos, min_periods=d_pos).mean()
    mean_high = high.rolling(d_pos, min_periods=d_pos).mean()
    denom = mean_high - mean_low

    pos = safe_div0(close - mean_low, denom)

    # --- rvol = volume/Ref(volume,k)-1 ---
    vol_ref = Ref(volume, k_ref)
    rvol = safe_div0(volume, vol_ref) - 1.0

    # --- abn = rvol - Mean(rvol,d_mean) ---
    rvol_mean = rvol.rolling(d_mean, min_periods=d_mean).mean()
    abn = rvol - rvol_mean

    gate = np.sign(abn)  # 一般引擎也是：>0 1；<0 -1；=0 0

    return pos * gate

# 10
def factor_price_bias_z_x_vol_z(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:

    # ---- price part ----
    mean_close = close.rolling(d, min_periods=d).mean()
    std_close  = close.rolling(d, min_periods=d).std()

    std_close = std_close.where(std_close != 0)
    mean_close = mean_close.where(mean_close != 0)

    price_part = (close / mean_close - 1.0) / std_close

    # ---- volume part ----
    mean_vol = volume.rolling(d, min_periods=d).mean()
    std_vol  = volume.rolling(d, min_periods=d).std()

    std_vol = std_vol.where(std_vol != 0)

    vol_part = (volume - mean_vol) / std_vol

    return price_part * vol_part

# 11
def factor_log_pos_x_vol_slope_std(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:

    # ---------- 价格区间位置 ----------
    mean_low = low.rolling(d, min_periods=d).mean()
    mean_high = high.rolling(d, min_periods=d).mean()

    denom = mean_high - mean_low
    denom = denom.where(denom != 0)

    pos = (close - mean_low) / denom
    pos = pos.where(pos > 0)      
    log_pos = np.log(pos)

    # ---------- 成交量趋势强度 ----------
    vol_slope = rolling_slope(volume, d)
    vol_std = volume.rolling(d, min_periods=d).std()
    vol_std = vol_std.where(vol_std != 0)

    vol_part = vol_slope / vol_std

    return log_pos * vol_part

# 12
def factor_log_close_over_meanhigh_x_refvol_over_meanvol(
    close: pd.DataFrame,
    high: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20,
    k: int = 1
) -> pd.DataFrame:

    mean_high = high.rolling(d, min_periods=d).mean()
    ratio_p = close / mean_high
    ratio_p = ratio_p.where(ratio_p > 0)         
    part_p = np.log(ratio_p)

    mean_vol = volume.rolling(d, min_periods=d).mean()
    ref_vol = Ref(volume, k)
    ratio_v = ref_vol / mean_vol
    ratio_v = ratio_v.replace([np.inf, -np.inf], np.nan)
    part_v = ratio_v - 1.0

    return part_p * part_v

#13 
def factor_log_close_over_meanhigh_x_vol_z(
    close: pd.DataFrame,
    high: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:

    # ----- price part -----
    mean_high = high.rolling(d, min_periods=d).mean()
    ratio_p = close / mean_high
    ratio_p = ratio_p.where(ratio_p > 0)     # log 只对正数
    part_p = np.log(ratio_p)

    # ----- volume part -----
    mean_vol = volume.rolling(d, min_periods=d).mean()
    std_vol  = volume.rolling(d, min_periods=d).std()
    std_vol = std_vol.where(std_vol != 0)

    part_v = (volume - mean_vol) / std_vol

    return part_p * part_v

#14
def factor_log_rel_close_x_vol_slope(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:
    """
    Log(close/Mean(close,d)) * Slope(volume,d)
    """

    # ---- price part ----
    mean_close = close.rolling(d, min_periods=d).mean()
    ratio_p = close / mean_close
    ratio_p = ratio_p.where(ratio_p > 0)   
    part_p = np.log(ratio_p)

    # ---- volume part ----
    vol_slope = rolling_slope(volume, d)

    return part_p * vol_slope

#15
def factor_log_rel_close_x_sign_vol_slope(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:
    """
    Log(close/Mean(close,d)) * Sign(Slope(volume,d))
    """

    mean_close = close.rolling(d, min_periods=d).mean()
    ratio_p = close / mean_close
    ratio_p = ratio_p.where(ratio_p > 0)
    part_p = np.log(ratio_p)

    vol_slope = rolling_slope(volume, d)
    gate = np.sign(vol_slope)

    return part_p * gate

#16
def factor_log_pos_x_rvol(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:

    mean_low = low.rolling(d, min_periods=d).mean()
    mean_high = high.rolling(d, min_periods=d).mean()

    denom = mean_high - mean_low
    denom = denom.where(denom != 0)

    pos = (close - mean_low) / denom
    pos = pos.where(pos > 0)          # log 只对正数
    log_pos = np.log(pos)

    mean_vol = volume.rolling(d, min_periods=d).mean()
    mean_vol = mean_vol.where(mean_vol != 0)

    rvol = volume / mean_vol - 1.0

    return log_pos * rvol

# 18
def factor_log_pos_minmax_x_vol_slope_std(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:

    # ---- price position using Min/Max ----
    min_low = low.rolling(d, min_periods=d).min()
    max_high = high.rolling(d, min_periods=d).max()

    denom = (max_high - min_low).where((max_high - min_low) != 0)
    pos = (close - min_low) / denom
    pos = pos.where(pos > 0)  # log domain
    log_pos = np.log(pos)

    # ---- volume slope / std ----
    vol_slope = rolling_slope(volume, d)
    vol_std = volume.rolling(d, min_periods=d).std()
    vol_std = vol_std.where(vol_std != 0)

    vol_part = vol_slope / vol_std

    return log_pos * vol_part

#19
def factor_log_pricepos_minmax_x_volpos_minmax(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:

    # ---- price position (Min/Max) ----
    min_low = low.rolling(d, min_periods=d).min()
    max_high = high.rolling(d, min_periods=d).max()

    # denom
    denom_p = (max_high - min_low).where((max_high - min_low) != 0)

    price_pos = (close - min_low) / denom_p
    price_pos = price_pos.where(price_pos > 0)   # log domain
    log_price_pos = np.log(price_pos)

    # ---- volume position (Min/Max) ----
    min_vol = volume.rolling(d, min_periods=d).min()
    max_vol = volume.rolling(d, min_periods=d).max()
    denom_v = (max_vol - min_vol).where((max_vol - min_vol) != 0)

    vol_pos = (volume - min_vol) / denom_v

    return log_price_pos * vol_pos

#20
def factor_log_pos_meanhl_x_sign_vol_slope(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    d: int = 20
) -> pd.DataFrame:

    mean_low = low.rolling(d, min_periods=d).mean()
    mean_high = high.rolling(d, min_periods=d).mean()

    denom = (mean_high - mean_low).where((mean_high - mean_low) != 0)
    pos = (close - mean_low) / denom
    pos = pos.where(pos > 0)      
    log_pos = np.log(pos)

    vol_slp = rolling_slope(volume, d)
    gate = np.sign(vol_slp)

    return log_pos * gate


def clean_factor_with_markettype(
    factor: pd.DataFrame,
    universe: pd.DataFrame,
    markettype: pd.DataFrame,
    #mad_k: float = 5.0,
    mktcap: pd.DataFrame | None = None,   
) -> pd.DataFrame:
    # 1) universe过滤
    f = factor.where(universe)

    # 2) 先做 MAD winsor
    #f = f.apply(lambda row: winsorize_mad_cs(row, k=mad_k), axis=1)

    # 3) 行业/市值中性化
    f = pd.DataFrame(
        [
            neutralize_markettype_cs(
                f.loc[t],
                markettype.loc[t],
                mktcap=None if mktcap is None else mktcap.loc[t],
            )
            for t in f.index
        ],
        index=f.index,
        columns=f.columns,
    )

    # 4) zscore
    #f = f.apply(zscore_cs, axis=1)

    return f

def make_signal(factor_clean: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    return factor_clean.shift(lag)
