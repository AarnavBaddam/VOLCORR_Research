"""v2.0 driver. `status` is the only stage allowed while the window is open: it downloads ^VIX and nothing else.
`run` refuses until the stopping rule has fired (or the hard stop has passed) and T + 5 trading days have elapsed.

    python -m volcorr.v2 status
    python -m volcorr.v2 run
Both must be started through ./run.sh v2-status / ./run.sh v2-all (they set VOLCORR_CONFIG=config_v2.yaml)."""
import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta

import pandas as pd

from .common import DATA_DIR, RESULTS_DIR, ROOT, load_cfg
from .nyse import add_trading_days, last_trading_day_of_year, trading_days
from .stress import count_episodes, primary_flags
from .window import ET

DEVIATIONS = ROOT / "DEVIATIONS_v2.0.md"
STAGES = ["data", "classical", "foundation", "recalibrate", "stats", "robustness", "audit"]


# ---------- stopping rule ----------

def scan_stop(flags, min_days, min_eps, max_gap, hard_stop):
    """Walk the trading days of the window in order. Fires on the first day T where the window so far holds
    at least min_days stress days and min_eps episodes (gaps of <= max_gap calm days do not split an episode).
    If the hard-stop day is reached first, T is that day with fired=False."""
    n = eps = 0
    last_pos = None
    days = list(flags.index)
    for pos, (d, f) in enumerate(zip(days, flags.values)):
        if f:
            if last_pos is None or pos - last_pos - 1 > max_gap:
                eps += 1
            last_pos, n = pos, n + 1
            if n >= min_days and eps >= min_eps:
                return {"fired": True, "hard_stop": False, "T": d, "n_stress": n, "n_episodes": eps}
        if d.date() == hard_stop:
            return {"fired": False, "hard_stop": True, "T": d, "n_stress": n, "n_episodes": eps}
    return {"fired": False, "hard_stop": False, "T": None, "n_stress": n, "n_episodes": eps,
            "through": days[-1].date() if days else None}


# ---------- monitor (VIX only) ----------

def fetch_vix(cfg, today):
    """Download the monitor ticker only, into DATA_DIR/monitor. Nothing else is requested."""
    import yfinance as yf
    d = cfg["data"]
    tickers = [cfg["stopping"]["monitor_ticker"]]
    df = yf.download(tickers[0], start=d["sample_start"], end=(today + timedelta(days=1)).isoformat(),
                     auto_adjust=False, actions=False, progress=False, multi_level_index=False)
    if not len(df):
        raise SystemExit("monitor download returned no VIX rows; nothing was written")
    out = DATA_DIR / "monitor"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "VIX.csv", lineterminator="\n")
    return df["Close"]


def monitor(cfg, today, vix=None):
    """Stress count and stopping state on the NYSE calendar, from window start to the last VIX close."""
    if vix is None:
        vix = fetch_vix(cfg, today)
    s = cfg["stress"]
    last = vix.index.max().date()
    start_all = date.fromisoformat(cfg["data"]["sample_start"])
    cal = pd.DatetimeIndex([pd.Timestamp(x) for x in trading_days(start_all, last)], name="date")
    vc = vix.reindex(cal).ffill()
    flags = primary_flags(vc, s["primary"]["lookback"], s["primary"]["percentile"])
    start = pd.Timestamp(cfg["v2"]["window_start"])
    fl = flags[flags.index >= start]
    hard = last_trading_day_of_year(int(cfg["stopping"]["hard_stop"][:4]))
    st = scan_stop(fl, s["min_stress_days"], s["min_episodes"], s["episode_max_gap"], hard)
    st.update({"today": today, "last_vix": last, "window_start": start.date(),
               "earliest_download": add_trading_days(st["T"].date(), cfg["stopping"]["download_lag_trading_days"])
               if st["T"] is not None else None})
    return st


def print_status(st):
    print(f"window start (from tag): {st['window_start']}")
    print(f"last VIX close: {st['last_vix']}   today: {st['today']}")
    print(f"stress days: {st['n_stress']}   episodes: {st['n_episodes']}   (minimum 90 and 6)")
    if st["T"] is not None:
        kind = "fired" if st["fired"] else "hard stop reached"
        print(f"stopping rule: {kind}, T = {st['T'].date()}")
        print(f"earliest allowed download (must be after this date): {st['earliest_download']}")
    else:
        print("stopping rule: not fired, window open")


# ---------- gates ----------

def assert_download_allowed(st, today):
    if st["T"] is None:
        raise SystemExit(f"refused: stopping rule not fired ({st['n_stress']} of 90 stress days, "
                         f"{st['n_episodes']} of 6 episodes, through {st.get('through')})")
    if today <= st["earliest_download"]:
        raise SystemExit(f"refused: download not allowed on or before {st['earliest_download']} "
                         f"(T + 5 trading days); today is {today}")


def freeze_window(cfg, st):
    """Write data_v2/window.json once. A later call must give the same record."""
    from .window import window_record
    rec = window_record(date.fromisoformat(cfg["v2"]["window_start"]), st["T"].date(),
                        "hard_stop" if st["hard_stop"] else "rule", cfg["v2"]["tag"], cfg["v2"]["tag_commit_iso"],
                        {"monitor_n_stress": st["n_stress"], "monitor_n_episodes": st["n_episodes"],
                         "monitor_last_vix": str(st["last_vix"]), "config": "config_v2.yaml"})
    f = DATA_DIR / "window.json"
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old != rec:
            raise SystemExit(f"{f} exists with different content; refusing to overwrite the frozen window")
        return rec
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


# ---------- checks after the data stage ----------

def overlap_check(cfg):
    """Largest relative difference between the new download and the v1.0 raw files over their common dates."""
    ref = ROOT / cfg["overlap_reference"]
    cols = ["Open", "High", "Low", "Close", "Adj Close"]
    report = {}
    for t in cfg["data"]["assets"] + [cfg["data"]["vix"]]:
        name = f"{t.lstrip('^')}.csv"
        a = pd.read_csv(ref / name, index_col=0, parse_dates=True)
        b = pd.read_csv(DATA_DIR / "raw" / name, index_col=0, parse_dates=True)
        common = a.index.intersection(b.index)
        a, b = a.loc[common, cols], b.loc[common, cols]
        rel = (b - a).abs() / a.abs()
        worst = rel.max()
        report[t] = {"common_dates": int(len(common)), "v1_only_dates": int(len(a.index.difference(b.index))),
                     "max_rel_diff": {c: float(worst[c]) for c in cols},
                     "largest": {"value": float(worst.max()), "column": str(worst.idxmax()),
                                 "date": str(rel[worst.idxmax()].idxmax().date()) if len(common) else None}}
    (DATA_DIR / "overlap_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for t, r in report.items():
        print(f"overlap {t}: {r['common_dates']} common dates, largest relative difference "
              f"{r['largest']['value']:.3e} ({r['largest']['column']}, {r['largest']['date']})")
    return report


def etf_recount(cfg, st):
    """Stress count on the ETF calendar (from the downloaded data) governs. Any disagreement with the monitor
    is written to DEVIATIONS_v2.0.md."""
    gap = cfg["stress"]["episode_max_gap"]
    tab = pd.read_parquet(DATA_DIR / "clean" / "stress.parquet").set_index("date")
    w = cfg["windows"]["primary"]
    flags = tab.loc[w["start"]:w["end"], "primary"].astype(bool).values
    n, eps = int(flags.sum()), count_episodes(flags, gap)
    minimum = n >= cfg["stress"]["min_stress_days"] and eps >= cfg["stress"]["min_episodes"]
    print(f"ETF-calendar count in window {w['start']} to {w['end']}: {n} stress days, {eps} episodes, "
          f"minimum met: {minimum}")
    if (n, eps) != (st["n_stress"], st["n_episodes"]):
        line = (f"- {date.today().isoformat()}: stress count, ETF calendar vs monitor (NYSE calendar). "
                f"Monitor counted {st['n_stress']} stress days and {st['n_episodes']} episodes through T={w['end']}; "
                f"ETF calendar counts {n} and {eps}. The ETF count governs (PREREGISTRATION_v2.0.md section 4). "
                f"Minimum met on the ETF count: {minimum}. Effect: inconclusive if not met, no other change.")
        text = DEVIATIONS.read_text(encoding="utf-8")
        marker = "## A. Pre-registered contingencies triggered\n"
        text = text.replace(marker, marker + "\n" + line + "\n", 1)
        DEVIATIONS.write_text(text, encoding="utf-8")
        print("logged the disagreement in", DEVIATIONS.name)
    return {"n_stress": n, "n_episodes": eps, "minimum_met": minimum}


def copy_classical_for_e3(main_results, e3_results):
    """E3 changes only HAR-RV and the foundation models; the other classical forecasts are copied unchanged."""
    src, dst = main_results / "raw", e3_results / "raw"
    dst.mkdir(parents=True, exist_ok=True)
    for m in ("garch", "gjr", "ewma"):
        shutil.copy2(src / f"{m}.parquet", dst / f"{m}.parquet")


# ---------- stage runner ----------

def run_stage(module, variant=None):
    env = {**os.environ, "VOLCORR_CONFIG": "config_v2.yaml"}
    env.pop("VOLCORR_VARIANT", None)
    if variant:
        env["VOLCORR_VARIANT"] = variant
    print(f"--- {module}" + (f" [{variant}]" if variant else ""), flush=True)
    subprocess.run([sys.executable, "-m", f"volcorr.{module}"], cwd=ROOT, env=env, check=True)


def run_all(cfg, st, today):
    assert_download_allowed(st, today)
    freeze_window(cfg, st)
    cfg = load_cfg()                                   # now resolved with the frozen end and sample_end
    for stage in ("data", "classical", "foundation", "recalibrate"):
        run_stage(stage)
    overlap_check(cfg)
    etf_recount(cfg, st)
    for stage in ("stats", "robustness", "audit"):
        run_stage(stage)
    # E3: input confound. HAR-RV and the three foundation models on log squared returns, under results_v2/e3.
    run_stage("classical", variant="e3")
    copy_classical_for_e3(RESULTS_DIR, RESULTS_DIR / "e3")
    run_stage("foundation", variant="e3")
    run_stage("recalibrate", variant="e3")
    run_stage("stats", variant="e3")
    print("v2-all finished: outputs under", DATA_DIR.name, "and", RESULTS_DIR.name, "(dashboard not built for v2.0)")


def main(argv):
    if os.environ.get("VOLCORR_CONFIG") != "config_v2.yaml":
        raise SystemExit("run this through ./run.sh v2-status or ./run.sh v2-all")
    cmd = argv[1] if len(argv) > 1 else ""
    cfg = load_cfg()                                   # refuses here if the prereg-v2.0 tag does not exist
    today = datetime.now(ET).date()
    st = monitor(cfg, today)
    if cmd == "status":
        print_status(st)
        return
    if cmd != "run":
        raise SystemExit("usage: python -m volcorr.v2 status|run")
    run_all(cfg, st, today)


if __name__ == "__main__":
    main(sys.argv)
