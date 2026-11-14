"""Stress rules. Every flag at date t uses only data up to the close of t."""
import numpy as np
import pandas as pd


def primary_flags(vix, lookback=252, percentile=80):
    """VIX[t] > percentile of the previous `lookback` closes, origin excluded.
    False until a full lookback of history exists."""
    thr = vix.rolling(lookback).quantile(percentile / 100).shift(1)
    return (vix > thr).fillna(False)


def count_episodes(flags, max_gap=5):
    """Number of stress episodes; gaps of <= max_gap calm days do not split one."""
    idx = np.flatnonzero(np.asarray(flags, dtype=bool))
    if idx.size == 0:
        return 0
    return 1 + int((np.diff(idx) - 1 > max_gap).sum())


def all_flags(vix, spy_close, spy_logret, cfg):
    s = cfg["stress"]
    out = pd.DataFrame(index=vix.index)
    out["vix"] = vix
    out["primary"] = primary_flags(vix, s["primary"]["lookback"], s["primary"]["percentile"])
    out["r1"] = vix > s["r1_vix_above"]
    out["r2"] = vix > s["r2_vix_above"]
    high = spy_close.rolling(s["r3"]["lookback"]).max()  # trailing high of closes, origin included
    out["r3"] = (spy_close < (1 - s["r3"]["drawdown"]) * high).fillna(False)
    rv = spy_logret.rolling(s["r4"]["rv_days"]).std()
    thr = rv.rolling(s["r4"]["lookback"]).quantile(s["r4"]["percentile"] / 100).shift(1)
    out["r4"] = (rv > thr).fillna(False)
    return out
