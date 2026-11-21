"""Recalibration: ln(target) = a + b ln(raw) on the most recent 1,000 forecast-target pairs whose
target date is on or before the origin, refit every 20 trading days; recal = exp(fit + 0.5 * resid var).
Writes the single results table results/forecasts.parquet."""
import numpy as np
import pandas as pd

from .common import CLASSICAL, MODELS, RAW_DIR, RESULTS_DIR, load_cfg, load_panel, log_run

OUT = RESULTS_DIR / "forecasts.parquet"


def recalibrate(raw, target, h, window, every):
    """raw[k], target[k]: forecast made at consecutive origin k and its realized target (known at k+h).
    Returns recal[k], fit_k[k] (origin of the fit used) and last_pair[k] (newest pair in that fit);
    -1 / NaN until `window` usable pairs exist."""
    n = len(raw)
    lx, ly = np.log(raw), np.log(target)
    ok = np.isfinite(lx) & np.isfinite(ly)
    recal, fit_k, last_pair = np.full(n, np.nan), np.full(n, -1), np.full(n, -1)
    coef, since = None, 0
    for k in range(n):
        if coef is None or since >= every:
            j = np.flatnonzero(ok[:max(k - h + 1, 0)])[-window:]  # pairs with target date <= origin k
            if len(j) < window:
                continue
            b, a = np.polyfit(lx[j], ly[j], 1)
            s2 = np.sum((ly[j] - a - b * lx[j]) ** 2) / (window - 2)
            coef, since, fk, lp = (a, b, s2), 0, k, j[-1]
        since += 1
        recal[k] = np.exp(coef[0] + coef[1] * lx[k] + 0.5 * coef[2])
        fit_k[k], last_pair[k] = fk, lp
    return recal, fit_k, last_pair


def load_raw():
    parts = []
    for m in MODELS:
        files = [RAW_DIR / f"{m}.parquet"] if m in CLASSICAL else sorted((RAW_DIR / m).glob("chunk_*.parquet"))
        parts += [pd.read_parquet(f) for f in files if f.exists()]
    return pd.concat(parts, ignore_index=True)


def targets(gk, ret, h, floor):
    """Realized h-day targets aligned to the origin row: mean of days t+1..t+h."""
    def fwd(x):
        return x.rolling(h).mean().shift(-h)
    return fwd(gk), fwd((ret ** 2).clip(lower=floor))


def main():
    cfg = load_cfg()
    log_run("recalibrate")
    rc = cfg["recalibration"]
    gk, ret = load_panel()
    cal = gk.index
    raw = load_raw()
    models = [m for m in MODELS if m in set(raw.model)]
    sizes = raw.groupby("model").size()
    if sizes.nunique() != 1:  # a missing row is an unfinished run, not a failed forecast
        raise SystemExit(f"forecast stage incomplete, rerun it first:\n{sizes}")
    wide = raw.pivot(index=["asset", "horizon", "origin_date"], columns="model", values="raw")[models]
    failed = wide.isna().any(axis=1)
    dropped = wide[failed].reset_index()[["asset", "horizon", "origin_date"]]
    wide[failed] = np.nan  # a failed asset-day is dropped for every model
    out = []
    for h in cfg["target"]["horizons"]:
        t_gk, t_sq = targets(gk, ret, h, cfg["target"]["gk_floor"])
        for a in gk.columns:
            w = wide.loc[(a, h)]
            pos = cal.get_indexer(w.index)
            assert (np.diff(pos) == 1).all(), "origins must be consecutive trading days"
            tg, ts = t_gk[a].values[pos], t_sq[a].values[pos]
            for m in models:
                r = w[m].values
                recal, fit_k, last_pair = recalibrate(r, tg, h, rc["window"], rc["refit_every"])
                recal_sq, _, _ = recalibrate(r, ts, h, rc["window"], rc["refit_every"])
                has = fit_k >= 0
                out.append(pd.DataFrame({
                    "model": m, "asset": a, "horizon": h, "origin_date": w.index, "target_date": cal[pos + h],
                    "raw": r, "recal": recal, "target": tg,
                    "fit_origin_date": pd.Series(cal[pos[np.where(has, fit_k, 0)]]).where(has).values,
                    "calib_last_target_date": pd.Series(cal[pos[np.where(has, last_pair, 0)] + h]).where(has).values,
                    "target_sqret": ts, "recal_sqret": recal_sq}))
    df = pd.concat(out, ignore_index=True)
    df = df[df.raw.notna()].sort_values(["model", "asset", "horizon", "origin_date"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    dropped.to_csv(RESULTS_DIR / "dropped_days.csv", index=False, lineterminator="\n")
    print(f"forecasts.parquet: {len(df)} rows, models={models}, dropped asset-days={len(dropped)}")
    print("first recalibrated origin per horizon:",
          df[df.recal.notna()].groupby("horizon").origin_date.min().dt.date.to_dict())


if __name__ == "__main__":
    main()
