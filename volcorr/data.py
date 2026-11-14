"""Data stage: download once to data/raw (never overwritten), verify the SHA-256 manifest,
build the clean panel and stress table, print the pre-model-run report."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .common import ROOT, gk_variance, load_cfg, log_run, sha256_file
from .stress import all_flags, count_episodes

RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
MANIFEST = RAW / "MANIFEST.json"


def raw_path(ticker):
    return RAW / f"{ticker.lstrip('^')}.csv"


def download(cfg):
    """Fetch each missing ticker once. Existing raw files are never touched."""
    import yfinance as yf
    d = cfg["data"]
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": {}}
    end = (pd.Timestamp(d["sample_end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive
    for t in d["assets"] + [d["vix"]]:
        p = raw_path(t)
        if not p.exists():
            for attempt in (1, 2):  # retry once
                df = yf.download(t, start=d["sample_start"], end=end, auto_adjust=d["auto_adjust"],
                                 actions=False, progress=False, multi_level_index=False)
                if len(df):
                    break
            if not len(df):
                raise RuntimeError(f"download failed twice for {t}")
            df.to_csv(p, lineterminator="\n")
            manifest["files"][p.name] = {
                "ticker": t, "sha256": sha256_file(p), "rows": len(df),
                "first": str(df.index[0].date()), "last": str(df.index[-1].date()),
                "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "yfinance": yf.__version__}
            MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for name, m in manifest["files"].items():  # raw files must match the snapshot forever
        if sha256_file(RAW / name) != m["sha256"]:
            raise RuntimeError(f"raw file {name} does not match manifest hash")
    return manifest


def read_raw(ticker, cfg):
    df = pd.read_csv(raw_path(ticker), index_col=0, parse_dates=True)
    df.index.name = "date"
    return df.loc[cfg["data"]["sample_start"]:cfg["data"]["sample_end"]]


def clean(cfg):
    d = cfg["data"]
    raws = {t: read_raw(t, cfg) for t in d["assets"]}
    calendar = sorted(set().union(*(r.index for r in raws.values())))
    rows, drops = [], []
    for t, r in raws.items():
        r = r.reindex(calendar)
        o, h, l, c, a = (r[k] for k in ["Open", "High", "Low", "Close", "Adj Close"])
        cols = pd.concat([o, h, l, c, a], axis=1)
        missing = cols.isna().any(axis=1)
        invalid = ~missing & ((cols <= 0).any(axis=1) | (h < l))
        for dt in r.index[missing]:
            drops.append((t, dt.date(), "missing"))
        for dt in r.index[invalid]:
            drops.append((t, dt.date(), "invalid"))
        r = r[~(missing | invalid)]
        gk, floored = gk_variance(r["Open"].values, r["High"].values, r["Low"].values,
                                  r["Close"].values, cfg["target"]["gk_floor"])
        rows.append(pd.DataFrame({
            "date": r.index, "asset": t, "open": r["Open"].values, "high": r["High"].values,
            "low": r["Low"].values, "close": r["Close"].values, "adj_close": r["Adj Close"].values,
            "gk": gk, "gk_floored": floored,
            "logret": np.log(r["Adj Close"]).diff().values}))
    panel = pd.concat(rows, ignore_index=True)
    drops = pd.DataFrame(drops, columns=["asset", "date", "reason"])

    cal = pd.DatetimeIndex(calendar, name="date")
    vix_raw = read_raw(d["vix"], cfg)["Close"]
    vix = vix_raw.reindex(cal).ffill()  # missing VIX on an origin: most recent earlier close
    spy = panel[panel.asset == "SPY"].set_index("date")
    stress = all_flags(vix, spy["close"].reindex(cal), spy["logret"].reindex(cal), cfg)
    stress["vix_filled"] = vix_raw.reindex(cal).isna().values

    CLEAN.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CLEAN / "panel.parquet", index=False)
    stress.reset_index().to_parquet(CLEAN / "stress.parquet", index=False)
    drops.to_csv(CLEAN / "dropped_bars.csv", index=False, lineterminator="\n")
    return panel, stress, drops


def report(cfg, manifest, panel, stress, drops):
    w, s = cfg["windows"]["primary"], cfg["stress"]
    prim = stress.loc[w["start"]:w["end"]]
    rules = {}
    for rule in ["primary", "r1", "r2", "r3", "r4"]:
        rules[rule] = {"stress_days": int(prim[rule].sum()),
                       "episodes": count_episodes(prim[rule].values, s["episode_max_gap"])}
    p = rules["primary"]
    ohlc_odd = panel[(panel[["open", "close"]].max(axis=1) > panel.high) |
                     (panel[["open", "close"]].min(axis=1) < panel.low)]
    rep = {
        "raw_files": {k: {"rows": v["rows"], "first": v["first"], "last": v["last"], "sha256": v["sha256"]}
                      for k, v in manifest["files"].items()},
        "calendar_days": len(stress),
        "panel_rows_per_asset": panel.groupby("asset").size().to_dict(),
        "floored_per_asset": panel.groupby("asset")["gk_floored"].sum().astype(int).to_dict(),
        "dropped_bars": drops.groupby(["asset", "reason"]).size().rename("n").reset_index().to_dict("records"),
        "open_or_close_outside_high_low_kept": ohlc_odd.groupby("asset").size().to_dict(),
        "vix_days_filled_forward": int(stress["vix_filled"].sum()),
        "primary_window_origins": len(prim),
        "primary_window_stress": rules,
        "stress_minimum_met": bool(p["stress_days"] >= s["min_stress_days"] and p["episodes"] >= s["min_episodes"]),
    }
    (CLEAN / "data_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    return rep


def main():
    cfg = load_cfg()
    log_run("data")
    manifest = download(cfg)
    rep = report(cfg, manifest, *clean(cfg))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
