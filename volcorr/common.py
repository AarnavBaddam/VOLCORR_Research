"""Shared helpers: config, target and loss formulas, run logging."""
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from .window import resolve_window

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["numpy", "pandas", "pyarrow", "scipy", "yfinance", "arch", "torch",
            "chronos-forecasting", "timesfm", "uni2ts", "transformers"]

# VOLCORR_CONFIG selects the study (default v1.0 config.yaml). VOLCORR_VARIANT=e3 is the v2.0 input-confound run.
VARIANT = os.environ.get("VOLCORR_VARIANT", "")
DEFAULT_PATHS = {"data": "data", "results": "results", "dashboard": "dashboard"}


def config_path():
    return ROOT / os.environ.get("VOLCORR_CONFIG", "config.yaml")


def _read_cfg():
    return yaml.safe_load(config_path().read_text(encoding="utf-8"))


def _paths(cfg):
    p = {**DEFAULT_PATHS, **(cfg.get("paths") or {})}
    if VARIANT == "e3":
        p["results"] = p["results"] + "/e3"
    return p


PATHS = _paths(_read_cfg())
DATA_DIR = ROOT / PATHS["data"]
RESULTS_DIR = ROOT / PATHS["results"]
DASH_DIR = ROOT / PATHS["dashboard"]


def load_cfg():
    cfg = _read_cfg()
    cfg["paths"] = dict(PATHS)
    cfg["variant"] = VARIANT or None
    cfg["forecast_input"] = "sqret" if VARIANT == "e3" else cfg.get("forecast_input", "gk")
    if cfg.get("study") == "v2.0":
        resolve_window(cfg, DATA_DIR, ROOT)
    return cfg


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gk_variance(o, h, l, c, floor=1e-8):
    """Garman-Klass variance, floored. Returns (variance, was_floored)."""
    gk = 0.5 * np.log(h / l) ** 2 - (2 * math.log(2) - 1) * np.log(c / o) ** 2
    floored = gk < floor
    return np.where(floored, floor, gk), floored


def qlike(r, f):
    x = np.asarray(r) / np.asarray(f)
    return x - np.log(x) - 1


def package_versions():
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return versions


def log_run(stage, **extra):
    """Append package versions and the config hash to logs/runs.jsonl."""
    rec = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stage": stage,
           "python": sys.version.split()[0], "platform": platform.platform(),
           "config": config_path().name, "config_sha256": sha256_file(config_path()),
           "packages": package_versions(), **extra}
    (ROOT / "logs").mkdir(exist_ok=True)
    with open(ROOT / "logs" / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


FOUNDATION = ["chronos_bolt_base", "timesfm_2_0_500m", "moirai_1_1_r_base"]
CLASSICAL = ["garch", "gjr", "ewma", "har"]
MODELS = FOUNDATION + CLASSICAL
RAW_DIR = RESULTS_DIR / "raw"
MAX_H = 5


def input_series(gk, ret, cfg):
    """HAR input: log GK variance (v1, v2 default) or log squared returns floored at gk_floor (v2.0 E3 only).
    Squared returns have no value on the first calendar day, so that day takes the next day's value."""
    if cfg["forecast_input"] == "gk":
        return gk
    return (ret ** 2).bfill().clip(lower=cfg["target"]["gk_floor"])


def load_panel():
    """Wide (date x asset) frames of GK variance and log returns."""
    import pandas as pd
    p = pd.read_parquet(DATA_DIR / "clean" / "panel.parquet")
    return p.pivot(index="date", columns="asset", values="gk"), p.pivot(index="date", columns="asset", values="logret")


def origin_range(cfg, calendar):
    """Positions of the first and last forecast origin in the calendar.
    First: earliest day on which every model can be estimated (HAR needs window + 22 lags).
    Last: end of the primary window, so every 5-day target is realized."""
    first = cfg["classical"]["window"] + max(cfg["classical"]["har"]["lags"]) - 1
    last = calendar.get_loc(cfg["windows"]["primary"]["end"])
    return first, last


def attempt(fn, *args):
    """Pre-registered failure rule: retry once, then give up (NaN drops the asset-day for all models)."""
    for _ in range(2):
        try:
            out = np.asarray(fn(*args), dtype=float)
            if np.isfinite(out).all() and (out > 0).all():
                return out
        except Exception:
            pass
    return np.full(MAX_H, np.nan)


def raw_rows(model, asset, dates, steps):
    """steps: (n_origins, 5) per-day variance forecasts -> long rows for horizons 1 and 5."""
    import pandas as pd
    steps = np.asarray(steps)
    return pd.concat([pd.DataFrame({"model": model, "asset": asset, "horizon": h, "origin_date": dates,
                                    "raw": steps[:, :h].mean(axis=1)}) for h in (1, 5)], ignore_index=True)
