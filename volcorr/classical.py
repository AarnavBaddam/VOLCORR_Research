"""Classical forecasts: GARCH(1,1), GJR-GARCH(1,1,1), EWMA(0.94), log-HAR-RV.
Every function takes data that ends at the origin and returns per-day variance forecasts for t+1..t+5."""
import numpy as np
from arch import arch_model

from .common import CLASSICAL, MAX_H, RAW_DIR, attempt, load_cfg, load_panel, log_run, origin_range, raw_rows

SCALE = 100.0  # returns in percent for the optimizer; variances are scaled back


def _garch(r, o):
    return arch_model(SCALE * r, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="normal", rescale=False)


def garch_fit(r, o):
    return _garch(r, o).fit(disp="off", show_warning=False).params.values


def garch_forecast(r, o, params):
    """Analytic multi-step variance from fixed parameters filtered over the window ending at the origin."""
    return _garch(r, o).fix(params).forecast(horizon=MAX_H, reindex=False).variance.values[-1] / SCALE ** 2


def ewma_forecast(r, lam, init_obs):
    """RiskMetrics recursion over the window, started at the mean of the first init_obs squared returns."""
    r2 = r ** 2
    n = len(r2) - init_obs
    s2 = lam ** n * r2[:init_obs].mean() + (1 - lam) * (lam ** np.arange(n - 1, -1, -1)) @ r2[init_obs:]
    return np.full(MAX_H, s2)  # EWMA forecasts are flat across horizons


def har_features(y, s):
    return np.array([1.0, y[s], y[s - 4:s + 1].mean(), y[s - 21:s + 1].mean()])


def har_fit(y, window):
    """OLS of y[s+1] on daily, 5-day and 22-day means at s, last `window` rows, all targets inside y."""
    n = len(y)
    rows = range(n - 1 - window, n - 1)
    X = np.array([har_features(y, s) for s in rows])
    return np.linalg.lstsq(X, y[[s + 1 for s in rows]], rcond=None)[0]


def har_forecast(y, beta):
    """Iterated log forecasts for t+1..t+5, exponentiated per day."""
    y = list(y[-22:])
    out = []
    for _ in range(MAX_H):
        nxt = float(har_features(np.array(y), len(y) - 1) @ beta)
        out.append(nxt)
        y.append(nxt)
    return np.exp(out)


def run_asset(model, y, r, first, last, c):
    """y: log GK variance, r: log returns, both full series; origins first..last inclusive."""
    W, every = c["window"], c["refit_every"]
    steps, params = [], None
    for k, i in enumerate(range(first, last + 1)):
        rw, refit = r[i - W + 1:i + 1], k % every == 0
        if model in ("garch", "gjr"):
            o = int(model == "gjr")
            if refit:
                params = garch_fit(rw, o)
            steps.append(attempt(garch_forecast, rw, o, params))
        elif model == "ewma":
            steps.append(attempt(ewma_forecast, rw, c["ewma"]["lambda"], c["ewma"]["init_obs"]))
        else:
            if refit:
                params = har_fit(y[:i + 1], W)
            steps.append(attempt(har_forecast, y[:i + 1], params))
    return np.array(steps)


def main():
    import pandas as pd
    cfg = load_cfg()
    log_run("forecast-classical")
    gk, ret = load_panel()
    first, last = origin_range(cfg, gk.index)
    dates = gk.index[first:last + 1]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for model in CLASSICAL:
        out = RAW_DIR / f"{model}.parquet"
        if out.exists():  # resumable: finished models are skipped
            continue
        parts = [raw_rows(model, a, dates, run_asset(model, np.log(gk[a].values), ret[a].values, first, last, cfg["classical"]))
                 for a in gk.columns]
        tmp = out.with_suffix(".tmp")
        pd.concat(parts, ignore_index=True).to_parquet(tmp, index=False)
        tmp.replace(out)
        print(model, "done", flush=True)


if __name__ == "__main__":
    main()
