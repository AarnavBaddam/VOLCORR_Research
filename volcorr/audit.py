"""Lookahead audit. Prints one line per check and exits non-zero if any fails.

Structural checks read the results table. Perturbation checks corrupt every observation after a cut
date and confirm that forecasts and stress flags at or before the cut do not move."""
import sys

import numpy as np
import pandas as pd

from .classical import run_asset
from .common import CLASSICAL, MODELS, ROOT, load_cfg, load_panel, origin_range
from .foundation import CHUNK, chunk_inputs
from .stress import all_flags

FAILED = []


def check(name, ok, detail=""):
    print(("PASS" if ok else "FAIL"), "-", name, detail)
    if not ok:
        FAILED.append(name)


def main():
    cfg = load_cfg()
    df = pd.read_parquet(ROOT / "results" / "forecasts.parquet")
    gk, ret = load_panel()
    cal = gk.index
    first, last = origin_range(cfg, cal)
    w = cfg["windows"]

    # 1. structure of the results table
    pos_o, pos_t = cal.get_indexer(df.origin_date), cal.get_indexer(df.target_date)
    check("every target date is exactly h trading days after its origin", bool((pos_t - pos_o == df.horizon).all()),
          f"({len(df)} rows)")
    c = df[df.recal.notna()]
    check("every calibration pair's target date <= the origin of its fit <= forecast origin",
          bool(((c.calib_last_target_date <= c.fit_origin_date) & (c.fit_origin_date <= c.origin_date)).all()),
          f"({len(c)} recalibrated rows)")
    check("no recalibrated forecast without a full calibration fit",
          bool(c.fit_origin_date.notna().all() and df[df.recal.isna()].fit_origin_date.isna().all()))
    prim = c[(c.origin_date >= w["primary"]["start"]) & (c.origin_date <= w["primary"]["end"])]
    check("every primary-window target date is after the 16 Apr 2025 cutoff bound",
          bool((prim.target_date > "2025-04-16").all()), f"(earliest {prim.target_date.min().date()})")
    keys = {m: set(zip(g.asset, g.horizon, g.origin_date)) for m, g in c.groupby("model")}
    check("all models are scored on identical asset-days", len(keys) == len(MODELS) and
          all(k == keys[MODELS[0]] for k in keys.values()), f"({len(keys[MODELS[0]])} asset-day-horizons x {len(keys)} models)")

    # 2. perturbation: corrupt the future, recompute, compare with stored forecasts
    rng = np.random.default_rng(0)
    cut = first + CHUNK * ((cal.get_loc(w["primary"]["start"]) - first) // CHUNK + 3)  # a refit origin inside the primary window
    start = cut - 2 * cfg["classical"]["refit_every"]
    gk2, ret2 = gk.copy(), ret.copy()
    gk2.iloc[cut + 1:] *= np.exp(rng.standard_normal(gk2.iloc[cut + 1:].shape))
    ret2.iloc[cut + 1:] = rng.standard_normal(ret2.iloc[cut + 1:].shape) * 0.05
    dates = cal[start:cut + 1]
    worst = 0.0
    for a in ("SPY", "TLT"):
        for m in CLASSICAL:
            new = run_asset(m, np.log(gk2[a].values), ret2[a].values, start, cut, cfg["classical"])[:, 0]
            old = df[(df.model == m) & (df.asset == a) & (df.horizon == 1)].set_index("origin_date").raw.loc[dates].values
            worst = max(worst, float(np.abs(new / old - 1).max()))
    check("classical forecasts unchanged when all data after the origin is corrupted", worst < 1e-9,
          f"(max relative change {worst:.1e}, {len(dates)} origins x 2 assets x 4 models)")

    same = True
    for length in (cfg["foundation"]["context"], cfg["foundation"]["context"] + 5):
        for k in ((start - first) // CHUNK, (cut - first) // CHUNK - 1):
            i1, x1 = chunk_inputs(gk, first, last, k, length)
            i2, x2 = chunk_inputs(gk2, first, last, k, length)
            same &= bool(i1.max() <= cut and np.array_equal(x1, x2))
    check("foundation-model inputs unchanged when all data after the origin is corrupted", same)
    k = (cut - first) // CHUNK - 1
    idx, x = chunk_inputs(gk, first, last, k, cfg["foundation"]["context"])
    y = np.log(gk.values).astype(np.float32)
    check("each foundation context is exactly the 512 log-variance values ending at its origin",
          all(np.array_equal(x[a * len(idx) + j], y[i - 511:i + 1, a]) for a in range(y.shape[1]) for j, i in enumerate(idx)))

    st = pd.read_parquet(ROOT / "data" / "clean" / "stress.parquet").set_index("date")
    spy = pd.read_parquet(ROOT / "data" / "clean" / "panel.parquet").query("asset == 'SPY'").set_index("date")
    vix2, close2, lr2 = st["vix"].copy(), spy["close"].copy(), spy["logret"].copy()
    vix2.iloc[cut + 1:] *= 3
    close2.iloc[cut + 1:] *= 0.5
    lr2.iloc[cut + 1:] *= 10
    rules = ["primary", "r1", "r2", "r3", "r4"]
    f2 = all_flags(vix2, close2, lr2, cfg)
    check("stress flags (primary, R1-R4) unchanged when all data after the origin is corrupted",
          bool((f2[rules].iloc[:cut + 1].values == st[rules].iloc[:cut + 1].values).all()))

    print("LOOKAHEAD AUDIT:", "FAILED " + str(FAILED) if FAILED else "PASSED")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
