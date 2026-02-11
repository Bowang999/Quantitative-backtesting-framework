import sys
import os
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_panel
from src.signal import long_short_half_weight
from src.backtest import run_backtest_weights
from src.metrics import annual_return, sharpe_ratio, max_drawdown
universe = load_panel("../data/cache/universe_v3.parquet").astype(bool)
ret = load_panel("../data/cache/ret.parquet")
idx = load_panel("../data/cache/index_ret.parquet")

def run_factor_backtest_report(
    signal_path: str,
    title: str = "Factor Backtest",
    negate_signal: bool = False,
    n_groups: int = 10,
    rebalance: str = "daily",       # "daily" | "weekly" | "monthly"
    cost_bps: float = 0.0,
    cache_dir: str = "../data/cache",
    annual_days: int | None = None,  
    start_date: str = "2009-01-01",
    end_date: str = "2023-07-31",
    ascending: bool = True,
    long_group: int = 1,
    short_group: int = 10,
    min_n_per_day: int = 200,
    signal_shift_1: bool = True,
    debug_log: bool = True,
    use_vwap_price: bool = False,
    vwap_price_path: str | None = None,
    monthly_use_prev_month_end_signal: bool = False,
    weekly_use_nonoverlap_vwap: bool = False,

    # 基准相关
    benchmark: str = "mkt_ew",  # "mkt_ew" | "index" | "none"
    benchmark_index_code: str = "000001", 
    benchmark_ret_path: str = "../data/cache/index_ret.parquet",
    excess_vs_benchmark: bool = True,
):
    # ===================== 指标函数 =====================
    def ann_return_linear(r: pd.Series, days: int) -> float:
        r = r.dropna()
        if len(r) == 0:
            return np.nan
        return r.mean() * days

    def ann_vol_from_daily(r: pd.Series, days: int) -> float:
        r = r.dropna()
        if len(r) == 0:
            return np.nan
        return r.std(ddof=1) * np.sqrt(days)

    def sharpe_from_daily(r: pd.Series, days: int, rf: float = 0.0) -> float:
        r = r.dropna()
        if len(r) == 0:
            return np.nan
        rf_period = (1 + rf) ** (1 / days) - 1
        ex = r - rf_period
        vol = ex.std(ddof=1) * np.sqrt(days)
        if np.isnan(vol) or vol == 0:
            return np.nan
        ann_ex = ex.mean() * days
        return ann_ex / vol

    def max_drawdown_from_nav(nav: pd.Series) -> float:
        nav = nav.dropna()
        if len(nav) == 0:
            return np.nan
        peak = nav.cummax()
        dd = nav / peak - 1
        return dd.min()

    def apply_weight_fixed_gross_no_leverage(
        w: pd.Series, r: pd.Series, target_gross: float | None
    ) -> pd.Series:
        w0 = w.copy()
        if target_gross is not None:
            gross0 = float(w0.abs().sum())
            if gross0 > 0:
                w0 = w0 / gross0 * target_gross
        return w0.where(r.notna(), 0.0)

    def grouped_nav_with_calendar(
        ret: pd.DataFrame,
        signal: pd.DataFrame,
        universe: pd.DataFrame,
        n_groups: int,
        rebalance: str,
        cost_bps: float,
        min_n_per_day: int,
        ascending: bool,
        long_group: int,
        short_group: int,
        debug_log: bool,
        monthly_use_prev_month_end_signal: bool,
        vwap_panel: pd.DataFrame | None,
        weekly_use_nonoverlap_vwap: bool,
    ):
        dates = ret.index.intersection(signal.index).intersection(universe.index)
        dates = pd.DatetimeIndex(dates).sort_values()

        ret = ret.loc[dates]
        signal = signal.loc[dates]
        universe = universe.loc[dates].astype(bool)

        common_cols = ret.columns.intersection(signal.columns).intersection(universe.columns)
        ret = ret[common_cols]
        signal = signal[common_cols]
        universe = universe[common_cols]

        grp_ret = pd.DataFrame(index=dates, columns=[f"G{i}" for i in range(1, n_groups + 1)], dtype=float)
        ls_ret = pd.Series(np.nan, index=dates)
        turnover = pd.Series(0.0, index=dates)

        ls50_ret = pd.Series(np.nan, index=dates)
        turnover50 = pd.Series(0.0, index=dates)

        mkt_ew_ret = pd.Series(np.nan, index=dates)
        trade_log: list[dict] = []

        if rebalance not in ("daily", "weekly", "monthly"):
            raise ValueError("rebalance must be 'daily' | 'weekly' | 'monthly'")

        # ---- rebalance days ----
        if rebalance == "daily":
            rebalance_days = dates[:-1]
        elif rebalance == "weekly":
            per = pd.Series(dates, index=dates).dt.to_period("W-FRI")
            is_week_begin = per.ne(per.shift(1)).fillna(True).values
            rebalance_days = dates[is_week_begin]
        else:
            per = pd.Series(dates, index=dates).dt.to_period("M")
            is_month_begin = per.ne(per.shift(1)).fillna(True).values
            rebalance_days = dates[is_month_begin]

        rebalance_days = pd.DatetimeIndex(rebalance_days)
        rebalance_set = set(rebalance_days)

        # ---- monthly prev-month-end mapping ----
        month_end_map: dict[pd.Timestamp, pd.Timestamp] = {}
        if rebalance == "monthly" and monthly_use_prev_month_end_signal:
            per_m = pd.Series(dates, index=dates).dt.to_period("M")
            is_month_end = per_m.ne(per_m.shift(-1)).fillna(True).values
            month_end_dates = dates[is_month_end]
            for d0 in rebalance_days:
                prev_ends = month_end_dates[month_end_dates < d0]
                if len(prev_ends) > 0:
                    month_end_map[d0] = prev_ends[-1]

        def make_groups(sig_t: pd.Series):
            ranks = sig_t.rank(method="first", ascending=ascending)
            g = pd.qcut(ranks, n_groups, labels=False) + 1
            return {k: g.index[g == k].tolist() for k in range(1, n_groups + 1)}

        def equal_weight(names: list[str]) -> pd.Series:
            w = pd.Series(0.0, index=common_cols)
            if len(names) > 0:
                w.loc[names] = 1.0 / len(names)
            return w

        def make_half_by_stock(sig_raw: pd.Series, tradable_t: pd.Series, ascending: bool) -> pd.Series:
            s = sig_raw.where(tradable_t).dropna()
            if len(s) < 2:
                return pd.Series(0.0, index=common_cols)
            s = s.sort_values(ascending=ascending)
            n = len(s)
            half = n // 2
            if half == 0:
                return pd.Series(0.0, index=common_cols)
            long_names = s.index[:half]
            short_names = s.index[-half:]
            w_long = equal_weight(list(long_names))
            w_short = equal_weight(list(short_names))
            return w_long - w_short

        # ============ 持仓/权重状态（固定到下一次调仓）============
        current_groups = None
        w_grp_hold = {k: pd.Series(0.0, index=common_cols) for k in range(1, n_groups + 1)}
        w_ls_hold = pd.Series(0.0, index=common_cols)
        w_ls50_hold = pd.Series(0.0, index=common_cols)

        w_ls_prev_hold = pd.Series(0.0, index=common_cols)
        w_ls50_prev_hold = pd.Series(0.0, index=common_cols)

        # -------------- weekly + 非重叠VWAP --------------
        if rebalance == "weekly" and weekly_use_nonoverlap_vwap:
            if vwap_panel is None:
                raise ValueError("weekly_use_nonoverlap_vwap=True requires vwap_panel (use_vwap_price=True).")

            rb = pd.DatetimeIndex(sorted(rebalance_days))
            exit_map = {rb[i]: rb[i + 1] for i in range(len(rb) - 1)}

            pending_cost_ls = 0.0
            pending_cost_ls50 = 0.0

            for t_entry, t_exit in exit_map.items():
                if rebalance == "monthly" and monthly_use_prev_month_end_signal:
                    t_sig = month_end_map.get(t_entry, t_entry)
                else:
                    t_sig = t_entry

                sig = signal.loc[t_sig].where(universe.loc[t_sig]).dropna()
                if len(sig) < min_n_per_day:
                    if debug_log:
                        trade_log.append({
                            "rebalance_day": t_entry,
                            "signal_day_used": t_sig,
                            "rebalance": "weekly_nonoverlap_vwap",
                            "did_rebalance": False,
                            "ok": False,
                            "reason": "insufficient cross-section",
                            "n_sig": int(len(sig)),
                        })
                    continue

                current_groups = make_groups(sig)
                tradable_entry = universe.loc[t_entry]

                for k in range(1, n_groups + 1):
                    names_k = [x for x in current_groups[k] if tradable_entry.get(x, False)]
                    w_grp_hold[k] = equal_weight(names_k)

                long_names = [x for x in current_groups[long_group] if tradable_entry.get(x, False)]
                short_names = [x for x in current_groups[short_group] if tradable_entry.get(x, False)]
                w_long = equal_weight(long_names)
                w_short = equal_weight(short_names)
                w_ls_hold = w_long - w_short

                w_ls50_hold = make_half_by_stock(
                    sig_raw=signal.loc[t_sig],
                    tradable_t=tradable_entry,
                    ascending=ascending,
                )

                to = float((w_ls_hold - w_ls_prev_hold).abs().sum())
                turnover.loc[t_exit] = to
                pending_cost_ls = to * (cost_bps / 10000.0)
                w_ls_prev_hold = w_ls_hold.copy()

                to50 = float((w_ls50_hold - w_ls50_prev_hold).abs().sum())
                turnover50.loc[t_exit] = to50
                pending_cost_ls50 = to50 * (cost_bps / 10000.0)
                w_ls50_prev_hold = w_ls50_hold.copy()

                if debug_log:
                    trade_log.append({
                        "rebalance_day": t_entry,
                        "signal_day_used": t_sig,
                        "rebalance": "weekly_nonoverlap_vwap",
                        "did_rebalance": True,
                        "ok": True,
                        "n_sig": int(len(sig)),
                        "g1_size": int((w_grp_hold[1] != 0).sum()),
                        f"g{n_groups}_size": int((w_grp_hold[n_groups] != 0).sum()),
                        "ls50_long_size": int((w_ls50_hold > 0).sum()),
                        "ls50_short_size": int((w_ls50_hold < 0).sum()),
                    })

                px_entry = vwap_panel.loc[t_entry, common_cols]
                px_exit = vwap_panel.loc[t_exit, common_cols]
                r_period = (px_exit / px_entry - 1.0)

                mkt_names = tradable_entry[tradable_entry].index.tolist()
                mkt_ew_ret.loc[t_exit] = r_period[mkt_names].mean(skipna=True) if len(mkt_names) else np.nan

                for k in range(1, n_groups + 1):
                    w_g_eff = apply_weight_fixed_gross_no_leverage(w_grp_hold[k], r_period, target_gross=1.0)
                    grp_ret.loc[t_exit, f"G{k}"] = float((w_g_eff * r_period.fillna(0.0)).sum())

                w_ls_eff = apply_weight_fixed_gross_no_leverage(w_ls_hold, r_period, target_gross=2.0)
                ls_ret.loc[t_exit] = float((w_ls_eff * r_period.fillna(0.0)).sum()) - pending_cost_ls
                pending_cost_ls = 0.0

                w_ls50_eff = apply_weight_fixed_gross_no_leverage(w_ls50_hold, r_period, target_gross=2.0)
                ls50_ret.loc[t_exit] = float((w_ls50_eff * r_period.fillna(0.0)).sum()) - pending_cost_ls50
                pending_cost_ls50 = 0.0

            return (
                grp_ret,
                ls_ret, turnover,
                ls50_ret, turnover50,
                mkt_ew_ret,
                pd.DataFrame(trade_log),
            )

        # -------------- “逐日滚动收益”分支 --------------
        pending_cost_ls = 0.0
        pending_cost_ls50 = 0.0

        for i in range(len(dates) - 1):
            t = dates[i]
            t_next = dates[i + 1]

            if (t in rebalance_set) or (current_groups is None):
                if rebalance == "monthly" and monthly_use_prev_month_end_signal:
                    t_sig = month_end_map.get(t, t)
                else:
                    t_sig = t

                sig = signal.loc[t_sig].where(universe.loc[t_sig]).dropna()

                if len(sig) >= min_n_per_day:
                    current_groups = make_groups(sig)
                    did_rebalance = True
                else:
                    did_rebalance = False
                    if current_groups is None:
                        if debug_log:
                            trade_log.append({
                                "rebalance_day": t,
                                "signal_day_used": t_sig,
                                "rebalance": rebalance,
                                "did_rebalance": False,
                                "ok": False,
                                "reason": "insufficient cross-section at first rebalance",
                                "n_sig": int(len(sig)),
                            })
                        continue

                if did_rebalance:
                    tradable_t = universe.loc[t]

                    for k in range(1, n_groups + 1):
                        names_k = [x for x in current_groups[k] if tradable_t.get(x, False)]
                        w_grp_hold[k] = equal_weight(names_k)

                    long_names = [x for x in current_groups[long_group] if tradable_t.get(x, False)]
                    short_names = [x for x in current_groups[short_group] if tradable_t.get(x, False)]
                    w_long = equal_weight(long_names)
                    w_short = equal_weight(short_names)
                    w_ls_hold = w_long - w_short

                    w_ls50_hold = make_half_by_stock(
                        sig_raw=signal.loc[t_sig],
                        tradable_t=tradable_t,
                        ascending=ascending,
                    )

                    to = float((w_ls_hold - w_ls_prev_hold).abs().sum())
                    turnover.loc[t_next] = to
                    pending_cost_ls = to * (cost_bps / 10000.0)
                    w_ls_prev_hold = w_ls_hold.copy()

                    to50 = float((w_ls50_hold - w_ls50_prev_hold).abs().sum())
                    turnover50.loc[t_next] = to50
                    pending_cost_ls50 = to50 * (cost_bps / 10000.0)
                    w_ls50_prev_hold = w_ls50_hold.copy()

                    if debug_log:
                        trade_log.append({
                            "rebalance_day": t,
                            "signal_day_used": t_sig,
                            "rebalance": rebalance,
                            "did_rebalance": True,
                            "ok": True,
                            "n_sig": int(len(sig)),
                            "g1_size": int((w_grp_hold[1] != 0).sum()),
                            f"g{n_groups}_size": int((w_grp_hold[n_groups] != 0).sum()),
                            "ls50_long_size": int((w_ls50_hold > 0).sum()),
                            "ls50_short_size": int((w_ls50_hold < 0).sum()),
                        })

            if current_groups is None:
                continue

            r_next = ret.loc[t_next]

            tradable_next = universe.loc[t_next]
            mkt_names = tradable_next[tradable_next].index.tolist()
            mkt_ew_ret.loc[t_next] = r_next[mkt_names].mean(skipna=True) if len(mkt_names) else np.nan

            for k in range(1, n_groups + 1):
                w_g_eff = apply_weight_fixed_gross_no_leverage(w_grp_hold[k], r_next, target_gross=1.0)
                grp_ret.loc[t_next, f"G{k}"] = float((w_g_eff * r_next.fillna(0.0)).sum())

            w_ls_eff = apply_weight_fixed_gross_no_leverage(w_ls_hold, r_next, target_gross=2.0)
            ls_ret.loc[t_next] = float((w_ls_eff * r_next.fillna(0.0)).sum()) - pending_cost_ls
            pending_cost_ls = 0.0

            w_ls50_eff = apply_weight_fixed_gross_no_leverage(w_ls50_hold, r_next, target_gross=2.0)
            ls50_ret.loc[t_next] = float((w_ls50_eff * r_next.fillna(0.0)).sum()) - pending_cost_ls50
            pending_cost_ls50 = 0.0

        return (
            grp_ret,
            ls_ret, turnover,
            ls50_ret, turnover50,
            mkt_ew_ret,
            pd.DataFrame(trade_log),
        )

    # ===================== 读取数据 =====================
    universe = load_panel(f"{cache_dir}/universe_v3.parquet").astype(bool)
    signal = load_panel(signal_path)

    vwap_panel = None
    if use_vwap_price:
        if vwap_price_path is None:
            vwap_price_path = f"{cache_dir}/vwap.parquet"
        vwap_panel = load_panel(vwap_price_path).loc[start_date:end_date]
        ret = vwap_panel.pct_change(fill_method=None)
    else:
        ret = load_panel(f"{cache_dir}/ret.parquet").loc[start_date:end_date]

    ret = ret.loc[start_date:end_date]
    universe = universe.loc[ret.index]
    signal = signal.loc[ret.index]

    if negate_signal:
        signal = -signal
    if signal_shift_1:
        signal = signal.shift(1)

    if annual_days is None:
        if rebalance == "weekly" and weekly_use_nonoverlap_vwap:
            annual_days = 52
        elif rebalance == "monthly":
            annual_days = 252
        else:
            annual_days = 252

    grp_ret, ls_ret, turnover, ls50_ret, turnover50, mkt_ew_ret, trade_log = grouped_nav_with_calendar(
        ret=ret,
        signal=signal,
        universe=universe,
        n_groups=n_groups,
        rebalance=rebalance,
        cost_bps=cost_bps,
        min_n_per_day=min_n_per_day,
        ascending=ascending,
        long_group=long_group,
        short_group=short_group,
        debug_log=debug_log,
        monthly_use_prev_month_end_signal=monthly_use_prev_month_end_signal,
        vwap_panel=vwap_panel,
        weekly_use_nonoverlap_vwap=weekly_use_nonoverlap_vwap,
    )

    long_ret = grp_ret[f"G{long_group}"]

    # ===================== 构造 benchmark 收益 =====================
    bench_ret = None

    if benchmark == "mkt_ew":
        bench_ret = mkt_ew_ret.copy()

    elif benchmark == "index":
        idx = pd.read_parquet(benchmark_ret_path)

        # long-form: index_code,date,ret
        if {"index_code", "date", "ret"}.issubset(idx.columns):
            idx["date"] = pd.to_datetime(idx["date"])
            idx = idx.set_index("date").sort_index()
            
            mask = idx["index_code"].astype(str).str.contains(str(benchmark_index_code))
            s = idx.loc[mask, "ret"].sort_index()
        else:
            # wide-form
            idx.index = pd.to_datetime(idx.index)
            s = idx[str(benchmark_index_code)].sort_index()

        # 若是 weekly_nonoverlap_vwap：日收益 => entry->exit 复利周收益（记在 exit 日）
        if rebalance == "weekly" and weekly_use_nonoverlap_vwap:
            rb = pd.to_datetime(
                trade_log.loc[trade_log["did_rebalance"], "rebalance_day"]
            ).sort_values().unique()
            rb = pd.DatetimeIndex(rb)

            exit_map = {rb[i]: rb[i + 1] for i in range(len(rb) - 1)}
            bench_weekly = pd.Series(index=list(exit_map.values()), dtype=float)

            for t_entry, t_exit in exit_map.items():
                seg = s.loc[t_entry:t_exit]
                if len(seg) == 0:
                    continue
                bench_weekly.loc[t_exit] = (1 + seg).prod() - 1

            bench_ret = bench_weekly.reindex(grp_ret.index)
        else:
            bench_ret = s.reindex(grp_ret.index)

    elif benchmark == "none":
        bench_ret = None
    else:
        raise ValueError("benchmark must be 'mkt_ew' | 'index' | 'none'")

    # ===================== 超额收益定义 =====================
    if excess_vs_benchmark and (bench_ret is not None):
        long_excess = long_ret - bench_ret
    else:
        long_excess = long_ret.copy()

    ls_excess = ls_ret.copy()
    ls50_excess = ls50_ret.copy()

    # ===================== NAV & 统计 =====================
    grp_nav = (1 + grp_ret.fillna(0.0)).cumprod()
    long_excess_nav = (1 + long_excess.fillna(0.0)).cumprod()
    ls_excess_nav = (1 + ls_excess.fillna(0.0)).cumprod()
    ls50_excess_nav = (1 + ls50_excess.fillna(0.0)).cumprod()

    bench_nav = (1 + bench_ret.fillna(0.0)).cumprod() if bench_ret is not None else None
    mkt_nav = (1 + mkt_ew_ret.fillna(0.0)).cumprod()

    monthly_long = (1 + long_excess.dropna()).resample("ME").prod() - 1 if long_excess.notna().any() else pd.Series(dtype=float)
    monthly_ls = (1 + ls_excess.dropna()).resample("ME").prod() - 1 if ls_excess.notna().any() else pd.Series(dtype=float)
    monthly_ls50 = (1 + ls50_excess.dropna()).resample("ME").prod() - 1 if ls50_excess.notna().any() else pd.Series(dtype=float)

    ann_long = ann_return_linear(long_excess, annual_days)
    vol_long = ann_vol_from_daily(long_excess, annual_days)
    sr_long = sharpe_from_daily(long_excess, annual_days, rf=0.0)
    mwr_long = (monthly_long > 0).mean() if len(monthly_long) else np.nan
    mdd_long = max_drawdown_from_nav(long_excess_nav)

    ann = ann_return_linear(ls_excess, annual_days)
    vol = ann_vol_from_daily(ls_excess, annual_days)
    sr = sharpe_from_daily(ls_excess, annual_days, rf=0.0)
    mwr = (monthly_ls > 0).mean() if len(monthly_ls) else np.nan
    mdd = max_drawdown_from_nav(ls_excess_nav)

    ann50 = ann_return_linear(ls50_excess, annual_days)
    vol50 = ann_vol_from_daily(ls50_excess, annual_days)
    sr50 = sharpe_from_daily(ls50_excess, annual_days, rf=0.0)
    mwr50 = (monthly_ls50 > 0).mean() if len(monthly_ls50) else np.nan
    mdd50 = max_drawdown_from_nav(ls50_excess_nav)

    # ===================== 打印 =====================
    price_tag = "VWAP" if use_vwap_price else "RET(parquet)"
    extra_tag = "weekly_nonoverlap_vwap" if (rebalance == "weekly" and weekly_use_nonoverlap_vwap) else "mark_to_market"
    bench_tag = f"{benchmark}" + (f":{benchmark_index_code}" if benchmark == "index" else "")

    print(f"[LONG] EXCESS vs {bench_tag if excess_vs_benchmark else 'RAW'} | price={price_tag} | rebalance={rebalance}({extra_tag}) | Long=G{long_group}")
    print(f"Annual Return:     {ann_long:.2%}")
    print(f"Annual Volatility: {vol_long:.2%}")
    print(f"Sharpe Ratio:      {sr_long:.2f}")
    print(f"Monthly Win Rate:  {mwr_long:.2%}")
    print(f"Max Drawdown:      {mdd_long:.2%}")

    print(f"[LS] price={price_tag} | rebalance={rebalance}({extra_tag}) | LS=G{long_group}-G{short_group}")
    print(f"Annual Return:     {ann:.2%}")
    print(f"Annual Volatility: {vol:.2%}")
    print(f"Sharpe Ratio:      {sr:.2f}")
    print(f"Monthly Win Rate:  {mwr:.2%}")
    print(f"Max Drawdown:      {mdd:.2%}")

    print(f"[LS-50] price={price_tag} | rebalance={rebalance}({extra_tag}) | LS=Top50%-Bottom50% (by STOCKS)")
    print(f"Annual Return:     {ann50:.2%}")
    print(f"Annual Volatility: {vol50:.2%}")
    print(f"Sharpe Ratio:      {sr50:.2f}")
    print(f"Monthly Win Rate:  {mwr50:.2%}")
    print(f"Max Drawdown:      {mdd50:.2%}")

    if debug_log:
        if len(trade_log) > 0 and "did_rebalance" in trade_log.columns:
            n_reb = int(trade_log["did_rebalance"].sum())
            print(f"[DEBUG] actual rebalances logged: {n_reb}")
            print("[DEBUG] first 10 rebalance days:")
            print(trade_log.loc[trade_log["did_rebalance"], ["rebalance_day", "signal_day_used"]].head(10))
        else:
            print("[DEBUG] trade_log empty or missing did_rebalance.")

    # ===================== 画图 =====================
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=120)

    long_excess_nav.plot(ax=axes[0], lw=2.0, label=f"LONG(G{long_group}) excess")
    ls_excess_nav.plot(ax=axes[0], lw=2.2, label=f"LS(G{long_group}-G{short_group})")
    ls50_excess_nav.plot(ax=axes[0], lw=2.2, label=f"LS(Top50%-Bottom50%)")

    if bench_nav is not None:
        bench_nav.plot(ax=axes[0], lw=1.2, alpha=0.7, label=f"BENCH NAV ({bench_tag})")
    else:
        mkt_nav.plot(ax=axes[0], lw=1.0, alpha=0.5, label="MKT_EW NAV (ref)")

    axes[0].set_title(f"{title}", fontsize=12)
    axes[0].set_ylabel("NAV")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    grp_nav.plot(ax=axes[1], lw=1.2, alpha=0.85)
    axes[1].set_title(f"Grouped NAV (rebalance={rebalance}({extra_tag}))", fontsize=12)
    axes[1].set_ylabel("NAV")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()

    # ===================== 返回 =====================
    return {
        "ret_used": ret,
        "vwap_used": vwap_panel,
        "universe": universe,
        "signal": signal,
        "mkt_ew_ret": mkt_ew_ret,
        "bench_ret": bench_ret,
        "group_daily_ret": grp_ret,
        "group_nav": grp_nav,
        "long_ret_raw": long_ret,
        "long_ret_excess": long_excess,
        "long_nav_excess": long_excess_nav,
        "group_ls_ret_raw": ls_ret,
        "group_ls50_ret_raw": ls50_ret,
        "turnover_ls": turnover,
        "turnover_ls50": turnover50,
        "group_ls_ret_excess": ls_excess,
        "group_ls_nav_excess": ls_excess_nav,
        "group_ls50_ret_excess": ls50_excess,
        "group_ls50_nav_excess": ls50_excess_nav,
        "trade_log": trade_log,
        "metrics_long_excess": {
            "benchmark": bench_tag if excess_vs_benchmark else "RAW",
            "price": "VWAP" if use_vwap_price else "RET(parquet)",
            "rebalance": rebalance,
            "extra": extra_tag,
            "annual_days": annual_days,
            "long_mode": f"G{long_group}",
            "annual_return": ann_long,
            "annual_vol": vol_long,
            "sharpe": sr_long,
            "monthly_win_rate": mwr_long,
            "max_drawdown": mdd_long,
        },
        "metrics_ls_excess": {
            "mode": "LS (no extra benchmark)",
            "annual_return": ann,
            "annual_vol": vol,
            "sharpe": sr,
            "monthly_win_rate": mwr,
            "max_drawdown": mdd,
        },
        "metrics_ls50_excess": {
            "mode": "LS-50 (no extra benchmark)",
            "annual_return": ann50,
            "annual_vol": vol50,
            "sharpe": sr50,
            "monthly_win_rate": mwr50,
            "max_drawdown": mdd50,
        }
    }