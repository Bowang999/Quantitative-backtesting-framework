# src/data.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def load_raw_excels(
    raw_dir: str | Path,
    pattern: str = "TRD_Dalyr*.xlsx",
) -> pd.DataFrame:
    
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {raw_dir}/{pattern}")

    dfs = []
    for fp in files:
        df = pd.read_excel(
            fp,
            header=0,          
            skiprows=[1, 2],   
            engine="openpyxl"
        )
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    return out


def preprocess_long_table(
    df: pd.DataFrame,
    code_col: str,
    date_col: str,
    numeric_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()

    # 1) 日期解析
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[code_col, date_col])

    # 2) 数值列转 float（避免 $、字符串等导致后续报错）
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) 去重：确保 (code, date) 唯一
    df = df.sort_values([code_col, date_col])
    df = df.drop_duplicates(subset=[code_col, date_col], keep="last")

    return df

def to_panel(
    df: pd.DataFrame,
    value_col: str,
    code_col: str,
    date_col: str,
) -> pd.DataFrame:
    """长表 -> 面板（date × stock）"""
    panel = df.pivot(index=date_col, columns=code_col, values=value_col).sort_index()
    return panel

def calc_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)

def make_universe(close: pd.DataFrame, min_price: float = 0.01) -> pd.DataFrame:
    """最简股票池：有价格且价格>min_price"""
    return close.notna() & (close > min_price)

def save_panel(df: pd.DataFrame, path: str) -> None:
    """保存面板（推荐 parquet）"""
    df.to_parquet(path)

def load_panel(path: str) -> pd.DataFrame:
    """读取面板"""
    return pd.read_parquet(path)

def build_suspend_panel_from_excel(
    path: str,
    close: pd.DataFrame,
    code_col: str = "Stkcd",
    date_col: str = "Suspdate",
    flag_col: str = "ST",
    y_value: str = "Y",
) -> pd.DataFrame:
    """
    读取停牌表（long table: code, date, Y/N），转成 panel（date x code），并对齐到 close 的 index/columns。
    返回：suspend_panel(bool)，True 表示停牌
    """
    df = pd.read_excel(path)

    # 统一列名
    df = df.rename(columns={code_col: "Stkcd", date_col: "Trddt", flag_col: "flag"})

    # dtype 统一
    df["Stkcd"] = df["Stkcd"].astype(str).str.zfill(6)
    df["Trddt"] = pd.to_datetime(df["Trddt"], errors="coerce")
    df = df.dropna(subset=["Trddt"])

    df["flag"] = df["flag"].astype(str).str.upper().map({y_value: True, "N": False})
    df = df.dropna(subset=["flag"])

    # long -> panel
    panel = df.pivot_table(index="Trddt", columns="Stkcd", values="flag", aggfunc="last")

    # 对齐到主面板
    panel = panel.reindex(index=close.index, columns=close.columns)

    # 没记录默认 False（不停牌）
    panel = panel.fillna(False).astype(bool)

    return panel


def upgrade_universe_with_suspend(
    universe: pd.DataFrame,
    suspend_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    universe 升级：剔除停牌日
    suspend_panel=True 表示停牌 -> 不可交易
    """
    suspend_panel = suspend_panel.reindex_like(universe).fillna(False).astype(bool)
    return universe.astype(bool) & (~suspend_panel)
