"""Spot check: recompute three random (model, asset, date) forecasts from the raw CSV files with
standalone code and compare with the results table. Foundation rows re-run their inference batch."""
import math
import sys

from . import foundation  # noqa: F401  first, so torch/JAX load before pandas
import numpy as np
import pandas as pd
from arch import arch_model

from .common import CLASSICAL, DATA_DIR, FOUNDATION, RESULTS_DIR, load_cfg

SEED = 20261110


def raw_series(asset):
    """GK variance and log returns straight from the raw CSV, independent of the data stage."""
    r = pd.read_csv(DATA_DIR / "raw" / f"{asset}.csv", index_col=0, parse_dates=True)
    gk = [max(0.5 * math.log(h / l) ** 2 - (2 * math.log(2) - 1) * math.log(c / o) ** 2, 1e-8)
          for o, h, l, c in zip(r["Open"], r["High"], r["Low"], r["Close"])]
    return r.index, np.array(gk), np.log(r["Adj Close"]).diff().values


def classical(model, gk, ret, i, first):
    fit_i = i - (i - first) % 20  # last refit origin
    if model == "ewma":
        w = ret[i - 999:i + 1]
        s2 = float(np.mean(w[:30] ** 2))
        for x in w[30:]:
            s2 = 0.94 * s2 + 0.06 * x * x
        return np.full(5, s2)
    if model in ("garch", "gjr"):
        kw = dict(mean="Constant", vol="GARCH", p=1, o=int(model == "gjr"), q=1, dist="normal", rescale=False)
        params = arch_model(100 * ret[fit_i - 999:fit_i + 1], **kw).fit(disp="off", show_warning=False).params
        return arch_model(100 * ret[i - 999:i + 1], **kw).fix(params).forecast(horizon=5, reindex=False).variance.values[-1] / 1e4
    y = pd.Series(np.log(gk))
    X = pd.DataFrame({"c": 1.0, "d": y, "w": y.rolling(5).mean(), "m": y.rolling(22).mean()})
    rows = slice(fit_i - 1000, fit_i)  # regressors dated s, targets dated s+1 <= fit origin
    beta = np.linalg.lstsq(X.iloc[rows].values, y.shift(-1).iloc[rows].values, rcond=None)[0]
    hist, out = list(y.iloc[:i + 1]), []
    for _ in range(5):
        out.append(beta[0] + beta[1] * hist[-1] + beta[2] * np.mean(hist[-5:]) + beta[3] * np.mean(hist[-22:]))
        hist.append(out[-1])
    return np.exp(out)


def foundation_chunk(model, asset, i, first, last, cfg, assets):
    """Rebuild the whole inference batch from raw CSVs (Moirai sampling depends on the batch) and re-run it."""
    import torch
    fc = cfg["foundation"]
    predict, length = foundation.LOADERS[model](fc["models"][model], fc, "cuda" if torch.cuda.is_available() else "cpu")
    k = (i - first) // foundation.CHUNK
    idx = range(first + k * foundation.CHUNK, min(first + (k + 1) * foundation.CHUNK, last + 1))
    x = np.array([np.log(raw_series(a)[1])[j - length + 1:j + 1] for a in assets for j in idx], dtype=np.float32)
    torch.manual_seed(fc["seed"] + k)
    out = np.exp(np.asarray(predict(x), dtype=np.float64))
    return out[assets.index(asset) * len(idx) + (i - idx[0])]


def main():
    cfg = load_cfg()
    df = pd.read_parquet(RESULTS_DIR / "forecasts.parquet")
    assets = sorted(df.asset.unique())
    pool = df[df.recal.notna()].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    picks = [pool[pool.model.isin(fam)].iloc[int(rng.integers(len(pool[pool.model.isin(fam)])))]
             for fam in (FOUNDATION, CLASSICAL)] + [pool.iloc[int(rng.integers(len(pool)))]]
    ok = True
    for p in picks:
        cal, gk, ret = raw_series(p.asset)
        i, h = cal.get_loc(p.origin_date), int(p.horizon)
        first, last = 1021, cal.get_loc(cfg["windows"]["primary"]["end"])
        steps = (classical(p.model, gk, ret, i, first) if p.model in CLASSICAL
                 else foundation_chunk(p.model, p.asset, i, first, last, cfg, assets))
        raw = float(np.mean(steps[:h]))
        # recalibration: refit origin, its 1,000 pairs, OLS on logs, exp(fit + half residual variance)
        s = df[(df.model == p.model) & (df.asset == p.asset) & (df.horizon == h)].sort_values("origin_date")
        k = int(np.searchsorted(s.origin_date.values, p.origin_date.to_datetime64()))
        k0 = 999 + h
        fit_k = k0 + 20 * ((k - k0) // 20)
        j = np.arange(fit_k - h - 999, fit_k - h + 1)
        pos = cal.get_indexer(s.origin_date)
        tgt = np.array([gk[q + 1:q + h + 1].mean() for q in pos[j]])
        lx, ly = np.log(s.raw.values[j]), np.log(tgt)
        A = np.column_stack([np.ones(1000), lx])
        coef, res = np.linalg.lstsq(A, ly, rcond=None)[:2]
        recal = math.exp(coef[0] + coef[1] * math.log(raw) + 0.5 * res[0] / 998)
        tol = 1e-8 if p.model in CLASSICAL else 1e-4
        good = abs(raw / p.raw - 1) < tol and abs(recal / p.recal - 1) < tol and abs(gk[i + 1:i + h + 1].mean() / p.target - 1) < 1e-12
        ok &= good
        print(f"{'MATCH' if good else 'MISMATCH'}  {p.model:18s} {p.asset} {p.origin_date.date()} h={h}  "
              f"raw {p.raw:.10e} vs {raw:.10e}  recal {p.recal:.10e} vs {recal:.10e}")
    print("SPOT CHECK:", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
