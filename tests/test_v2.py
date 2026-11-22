"""v2.0 tests: calendar, window start, stopping rule, verdicts, monitor scope, refusals, config diff, v2 stats path."""
import os
import subprocess
from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

from volcorr import nyse, v2, window
from volcorr.common import ROOT
from volcorr.stats import holm, load_inputs, compute
from volcorr.stats import verdicts_v2, verdict_v2


def base_cfg():
    return yaml.safe_load((ROOT / "config_v2.yaml").read_text(encoding="utf-8"))


# ---------- NYSE calendar ----------

def test_easter_and_good_friday():
    assert nyse.easter(2026) == date(2026, 4, 5)
    assert date(2026, 4, 3) in nyse.holidays(2026)


def test_fixed_holidays_observed_on_weekend_rules():
    h26 = nyse.holidays(2026)
    assert date(2026, 7, 3) in h26        # 4 July 2026 is a Saturday: closed Friday 3 July
    assert date(2026, 11, 26) in h26      # Thanksgiving
    assert date(2026, 12, 25) in h26
    assert date(2026, 6, 19) in h26       # Juneteenth, Friday
    assert nyse.is_trading_day(date(2027, 12, 31))  # 1 Jan 2028 is a Saturday: NYSE open on 31 Dec 2027


def test_add_trading_days_skips_weekend_and_thanksgiving():
    # Fri 20 Nov 2026 -> Mon 23, Tue 24, Wed 25, (Thu 26 closed), Fri 27, Mon 30
    assert nyse.add_trading_days(date(2026, 11, 20), 5) == date(2026, 11, 30)


# ---------- window start ----------

@pytest.mark.parametrize("commit, expected", [
    ("2026-11-21T10:00:00-05:00", date(2026, 11, 23)),  # Saturday: next trading day is Monday
    ("2026-11-21T02:00:00+00:00", date(2026, 11, 23)),  # UTC Saturday 02:00 is still Friday 20 Nov in US Eastern
    ("2026-11-25T12:00:00-05:00", date(2026, 11, 27)),  # Wednesday before Thanksgiving: Friday, not the holiday
    ("2026-11-26T12:00:00-05:00", date(2026, 11, 27)),  # commit on the holiday itself
    ("2026-11-20T09:00:00-05:00", date(2026, 11, 23)),  # Friday: strictly after the calendar date
])
def test_window_start_from_commit_date(commit, expected):
    assert window.window_start(commit) == expected


def test_tag_commit_date_from_real_git_repo(tmp_path):
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-11-21T10:00:00-05:00",
           "GIT_COMMITTER_DATE": "2026-11-21T10:00:00-05:00",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, env=env, check=True, capture_output=True)
    run("init", "-q")
    (tmp_path / "f.txt").write_text("x")
    run("add", "f.txt")
    run("commit", "-q", "-m", "freeze")
    run("tag", "prereg-v2.0")
    iso = window.tag_commit_iso("prereg-v2.0", tmp_path)
    assert iso == "2026-11-21T10:00:00-05:00"
    assert window.window_start(iso) == date(2026, 11, 23)


def test_missing_tag_refuses(tmp_path):
    with pytest.raises(SystemExit, match="prereg-v2.0"):
        window.tag_commit_iso("prereg-v2.0", tmp_path)  # not a git repo: tag cannot exist


def test_resolve_window_refuses_when_window_json_disagrees(tmp_path, monkeypatch):
    monkeypatch.setattr(window, "tag_commit_iso", lambda tag, repo: "2026-11-21T10:00:00-05:00")
    (tmp_path / "window.json").write_text('{"start": "2030-01-02", "end": "2030-02-01", "sample_end": "2030-02-08", "stop": "rule"}')
    with pytest.raises(SystemExit, match="refusing"):
        window.resolve_window(base_cfg(), tmp_path, ROOT)


# ---------- stopping rule ----------

def days_flags(pattern, start=date(2027, 1, 4)):
    """Trading days from start, flagged by a pattern of (value, count) runs."""
    seq = []
    for val, n in pattern:
        seq += [val] * n
    days = nyse.trading_days(start, date(2032, 12, 31))[:len(seq)]
    return pd.Series(seq[:len(days)], index=pd.DatetimeIndex(days), dtype=bool)


def test_stop_fires_on_exact_first_day_with_six_episodes_and_gaps_of_six():
    pattern = []
    for _ in range(6):
        pattern += [(True, 15), (False, 6)]
    pattern = pattern[:-1]                       # no calm tail after the sixth episode
    flags = days_flags(pattern)
    out = v2.scan_stop(flags, 90, 6, 5, date(2031, 12, 31))
    assert out["fired"] and out["n_stress"] == 90 and out["n_episodes"] == 6
    assert out["T"] == flags[flags].index[-1]    # the 90th stress day, the last day of episode six


def test_gap_of_six_splits_episode_gap_of_five_does_not():
    gap5 = days_flags([(True, 50), (False, 5), (True, 50)])
    gap6 = days_flags([(True, 50), (False, 6), (True, 50)])
    r5 = v2.scan_stop(gap5, 90, 2, 5, date(2031, 12, 31))
    r6 = v2.scan_stop(gap6, 90, 2, 5, date(2031, 12, 31))
    assert not r5["fired"] and r5["n_episodes"] == 1          # one episode: no fire
    assert r6["fired"] and r6["n_episodes"] == 2
    assert r6["T"] == gap6[gap6].index[89]                    # the 90th stress day, not earlier


def test_hard_stop_reached_without_firing():
    flags = pd.Series(False, index=pd.DatetimeIndex(nyse.trading_days(date(2031, 1, 2), date(2031, 12, 31))))
    out = v2.scan_stop(flags, 90, 6, 5, nyse.last_trading_day_of_year(2031))
    assert not out["fired"] and out["hard_stop"]
    assert out["T"].date() == nyse.last_trading_day_of_year(2031)


# ---------- Holm and verdicts ----------

def test_holm_two_p_values():
    assert np.allclose(holm([0.01, 0.04]), [0.02, 0.04])
    assert np.allclose(holm([0.03, 0.03]), [0.06, 0.06])


@pytest.mark.parametrize("est, p, minimum, expected", [
    (0.06, 0.01, True, "supported"),
    (0.03, 0.01, True, "detectable, below practical threshold"),
    (-0.04, 0.01, True, "opposite direction"),
    (0.06, 0.20, True, "not supported"),
    (0.06, 0.01, False, "inconclusive"),
])
def test_verdict_branches(est, p, minimum, expected):
    got, _ = verdict_v2({"est": est}, p, minimum, 0.05, 0.05, 90, 6)
    assert got == expected


def test_verdicts_use_holm_across_two_hypotheses():
    cfg = base_cfg()
    r = lambda est, p: {"est": est, "lo": est - 0.01, "hi": est + 0.01, "p": p, "members": None}
    res = {"H1": r(0.05, 0.0), "H2": r(0.06, 0.03), "H3": r(0.03, 0.03), "H4": r(0.0, 0.5)}
    rows = {x["id"]: x for x in verdicts_v2(res, True, cfg)}
    assert rows["H2"]["p_holm"] == pytest.approx(0.06)        # family of two, not four
    assert rows["H2"]["verdict"] == "not supported"            # 0.06 Holm p is not below 0.05
    assert rows["H1"]["verdict"] is None and rows["H4"]["verdict"] is None  # replications: no verdict


# ---------- monitor scope and refusals ----------

def test_monitor_requests_only_vix(tmp_path, monkeypatch):
    import yfinance as yf
    calls = []

    def fake_download(tickers, **kw):
        calls.append(tickers)
        idx = pd.bdate_range("2015-01-02", "2026-11-20")
        return pd.DataFrame({"Close": np.full(len(idx), 20.0)}, index=idx)

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(v2, "DATA_DIR", tmp_path)
    vix = v2.fetch_vix(base_cfg(), date(2026, 11, 21))
    assert calls == ["^VIX"]
    assert (tmp_path / "monitor" / "VIX.csv").exists()
    assert len(vix) > 0


def test_monitor_constant_vix_has_no_stress(tmp_path):
    cfg = base_cfg()
    cfg["v2"] = {"window_start": "2027-01-04"}
    idx = pd.bdate_range("2015-01-02", "2027-06-30")
    vix = pd.Series(18.0, index=idx)
    st = v2.monitor(cfg, date(2027, 6, 30), vix=vix)
    assert st["n_stress"] == 0 and st["T"] is None and not st["fired"]


def test_download_refused_before_stop_and_before_t_plus_five():
    cfg_st = {"T": None, "n_stress": 10, "n_episodes": 1, "through": date(2027, 2, 1)}
    with pytest.raises(SystemExit, match="stopping rule not fired"):
        v2.assert_download_allowed(cfg_st, date(2030, 1, 1))
    fired = {"T": pd.Timestamp("2028-03-01"), "earliest_download": date(2028, 3, 8)}
    with pytest.raises(SystemExit, match="T \\+ 5"):
        v2.assert_download_allowed(fired, date(2028, 3, 8))
    v2.assert_download_allowed(fired, date(2028, 3, 9))       # allowed the day after


def test_v2_refuses_without_tag_before_any_download(monkeypatch):
    import yfinance as yf
    monkeypatch.setenv("VOLCORR_CONFIG", "config_v2.yaml")

    def boom(*a, **k):
        raise AssertionError("no download may happen without the tag")
    monkeypatch.setattr(yf, "download", boom)

    def no_tag(tag, repo):
        raise SystemExit(f"tag {tag} not found")
    monkeypatch.setattr(window, "tag_commit_iso", no_tag)
    with pytest.raises(SystemExit, match="prereg-v2.0"):
        v2.main(["volcorr.v2", "status"])
    with pytest.raises(SystemExit, match="prereg-v2.0"):
        v2.main(["volcorr.v2", "run"])


# ---------- config diff against v1 ----------

def test_v2_config_differs_from_v1_only_where_preregistered():
    v1 = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    v2c = base_cfg()
    for key in ["target", "foundation", "classical", "recalibration", "inference", "decision"]:
        assert v1[key] == v2c[key], key
    assert v1["data"]["assets"] == v2c["data"]["assets"]
    assert v1["stress"]["primary"] == v2c["stress"]["primary"]
    assert v1["windows"]["secondary"] == v2c["windows"]["secondary"]
    assert v1["windows"]["crisis"] == v2c["windows"]["crisis"]
    assert (v1["stress"]["min_stress_days"], v1["stress"]["min_episodes"]) == (60, 4)
    assert (v2c["stress"]["min_stress_days"], v2c["stress"]["min_episodes"]) == (90, 6)
    assert v2c["paths"] == {"data": "data_v2", "results": "results_v2", "dashboard": "dashboard_v2"}


# ---------- v2 stats path on the existing v1.0 table, in memory only ----------

def test_v2_stats_path_on_v1_table(tmp_path):
    cfg = base_cfg()
    cfg["windows"]["primary"] = {"start": "2025-04-17", "end": "2026-10-30"}
    cfg["inference"]["n_boot"] = 200
    cfg["v2"] = {"tag": "prereg-v2.0", "stop": None}
    df, st = load_inputs()
    def tree():
        skip = {".git", "__pycache__", ".pytest_cache"}
        return {p for p in ROOT.rglob("*") if p.is_file() and not skip & set(p.parts)}
    before = tree()
    out, tab = compute(cfg, df, st)
    after = tree()
    assert before == after, "compute() must not write files"

    by_id = {h["id"]: h for h in out["hypotheses"]}
    assert set(by_id) == {"H1", "H2", "H3", "H4"}
    assert out["meta"]["holm_family"] == ["H2", "H3"]
    assert by_id["H2"]["p_holm"] == pytest.approx(holm([by_id["H2"]["p"], by_id["H3"]["p"]])[0])
    assert by_id["H3"]["p_holm"] == pytest.approx(holm([by_id["H2"]["p"], by_id["H3"]["p"]])[1])
    assert by_id["H2"]["verdict"] == "inconclusive"           # 54 stress days < 90
    assert by_id["H1"]["verdict"] is None and by_id["H4"]["verdict"] is None
    for key in ["meta", "hypotheses", "conclusion", "components", "corr", "qlike", "dm", "mcs", "by_asset",
                "primary_summary"]:
        assert key in out, key
    for key in ["secondary", "crisis", "rolling"]:   # v1.0-only diagnostics, not in v2.0 section 8
        assert key not in out, key
    for key in ["H1", "H2", "H3", "H4"]:
        assert all(f in by_id[key] for f in ["est", "lo", "hi", "p", "p_holm", "verdict", "note"])
    exp = tab["exploratory"]
    assert set(exp) >= {"E1", "E2", "label"}
    assert "foundation" in exp["E1"]["D"] and "classical_4" in exp["E1"]["D"]
    assert len([k for k in exp["E1"]["D"] if k.startswith("classical_")]) == 5   # full family + four 3-model subsets
    assert set(exp["E2"]["rho"]) == {"F", "C"}


# ---------- E3 input and calendar additions ----------

def test_e3_input_is_causal_floored_five_day_mean():
    from volcorr.common import input_series
    idx = pd.bdate_range("2020-01-01", periods=40)
    rng = np.random.default_rng(1)
    ret = pd.DataFrame({"A": rng.normal(0, 0.01, 40)}, index=idx)
    ret.iloc[0] = np.nan
    ret.iloc[10:15] = 0.0                                   # five zero-return days in a row
    gk = ret.abs()
    cfg = {"forecast_input": "sqret", "target": {"gk_floor": 1e-8}}
    x = input_series(gk, ret, cfg)
    assert np.isfinite(x.values).all() and (x.values >= 1e-8).all()
    assert x.iloc[20, 0] == pytest.approx((ret.iloc[16:21, 0] ** 2).mean())
    assert x.iloc[14, 0] == pytest.approx(1e-8)              # all five zero: floored, not log(0)
    ret2 = ret.copy(); ret2.iloc[25:] = 0.5                  # change the future
    assert (input_series(gk, ret2, cfg).iloc[:25] == x.iloc[:25]).all().all()
    assert input_series(gk, ret, {"forecast_input": "gk", "target": {"gk_floor": 1e-8}}) is gk


def test_bush_day_of_mourning_closed():
    assert not nyse.is_trading_day(date(2018, 12, 5))
