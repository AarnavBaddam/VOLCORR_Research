import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from volcorr.classical import run_asset
from volcorr.common import MODELS, ROOT
from volcorr.foundation import contexts
from volcorr.recalibrate import recalibrate
from volcorr.stats import asset_corr, dm_test, evaluate, corr3, fam_rho, holm, losses, n_eff, sb_indices, summ

TABLE = ROOT / "results" / "forecasts.parquet"
CFG = {"window": 200, "refit_every": 20, "ewma": {"lambda": 0.94, "init_obs": 30}}


def series(n=320, seed=1):
    rng = np.random.default_rng(seed)
    ret = rng.standard_normal(n) * 0.01
    y = np.log(ret ** 2 + 1e-6) + rng.standard_normal(n) * 0.1
    return y, ret


# ---------- no lookahead ----------

@pytest.mark.parametrize("model", ["garch", "gjr", "ewma", "har"])
def test_classical_features_end_at_origin(model):
    y, ret = series()
    cut = 260
    base = run_asset(model, y, ret, 221, cut, CFG)
    y2, ret2 = y.copy(), ret.copy()
    y2[cut + 1:] += 5          # corrupt everything after the last origin
    ret2[cut + 1:] *= 40
    assert np.array_equal(base, run_asset(model, y2, ret2, 221, cut, CFG))
    y2[cut] += 5               # ...whereas the origin's own observation must matter
    ret2[cut] *= 40
    assert not np.array_equal(base[-1], run_asset(model, y2, ret2, 221, cut, CFG)[-1])


def test_foundation_context_ends_at_origin():
    y = np.arange(1000.0)
    x = contexts(y, [600, 601], 512)
    assert x.shape == (2, 512)
    assert x[0, -1] == 600 and x[0, 0] == 89 and x[1, -1] == 601   # newest value is the origin, nothing later


@pytest.mark.parametrize("h", [1, 5])
def test_recalibration_uses_only_realized_pairs(h):
    rng = np.random.default_rng(0)
    n, window = 400, 100
    raw = np.exp(rng.standard_normal(n))
    target = np.exp(0.3 + 0.8 * np.log(raw) + 0.2 * rng.standard_normal(n))
    recal, fit_k, last_pair = recalibrate(raw, target, h, window, 20)
    has = fit_k >= 0
    assert np.isnan(recal[:window - 1 + h]).all() and has[window - 1 + h:].all()
    k = np.arange(n)
    assert (last_pair[has] + h <= fit_k[has]).all() and (fit_k[has] <= k[has]).all()   # pair target date <= origin
    assert set(np.diff(np.unique(fit_k[has]))) == {20}                                 # refit every 20 days
    cut = 300                                                                          # targets realized after `cut` are corrupted
    t2 = target.copy()
    t2[cut - h + 1:] *= 100
    assert np.array_equal(recalibrate(raw, t2, h, window, 20)[0][:cut + 1], recal[:cut + 1], equal_nan=True)
    assert not np.allclose(recalibrate(raw, t2, h, window, 20)[0][cut + 21:], recal[cut + 21:])


def test_recalibration_recovers_known_mapping():
    rng = np.random.default_rng(1)
    raw = np.exp(rng.standard_normal(3000))
    target = np.exp(0.5 + 0.7 * np.log(raw) + 0.3 * rng.standard_normal(3000))
    recal, _, _ = recalibrate(raw, target, 1, 1000, 20)
    expect = np.exp(0.5 + 0.7 * np.log(raw) + 0.5 * 0.09)
    assert np.nanmax(np.abs(recal[1000:] / expect[1000:] - 1)) < 0.08


@pytest.mark.skipif(not TABLE.exists(), reason="results table not built")
def test_results_table_no_lookahead_and_identical_days():
    df = pd.read_parquet(TABLE)
    assert (df.target_date > df.origin_date).all()
    c = df[df.recal.notna()]
    assert (c.calib_last_target_date <= c.fit_origin_date).all() and (c.fit_origin_date <= c.origin_date).all()
    keys = [set(zip(g.asset, g.horizon, g.origin_date)) for _, g in c.groupby("model")]
    assert len(keys) == len(MODELS) and all(k == keys[0] for k in keys)   # identical days across models
    assert not df.duplicated(["model", "asset", "horizon", "origin_date"]).any()


# ---------- statistics ----------

def test_stationary_bootstrap_sanity():
    T, n = 387, 4000
    idx = sb_indices(T, n, 10, np.random.default_rng(1))
    assert idx.shape == (n, T) and idx.min() >= 0 and idx.max() < T
    breaks = (np.diff(idx, axis=1) % T != 1).mean()
    assert abs(breaks - 0.1) < 0.005                      # mean block length 10 (new block w.p. 1/10)
    counts = np.bincount(idx.ravel(), minlength=T) / idx.size
    assert np.abs(counts * T - 1).max() < 0.05            # every date equally likely, circular blocks
    x = np.random.default_rng(2).standard_normal(T).cumsum()
    assert abs(x[idx].mean(1).mean() - x.mean()) < 0.05 * x.std()
    assert np.array_equal(idx, sb_indices(T, n, 10, np.random.default_rng(1)))   # seeded
    longer = sb_indices(T, n, 20, np.random.default_rng(1))
    assert abs((np.diff(longer, axis=1) % T != 1).mean() - 0.05) < 0.004


def test_holm():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm([0.0, 0.2, 0.0, 0.01]) == pytest.approx([0.0, 0.2, 0.0, 0.02])


def test_bootstrap_p_value_and_interval():
    d = np.arange(-10, 990) / 100.0           # 11 of 1,000 draws are <= 0
    s = summ(1.0, d)
    assert s["p"] == pytest.approx(2 * 11 / 1000)
    assert s["lo"] == pytest.approx(np.percentile(d, 2.5)) and s["hi"] == pytest.approx(np.percentile(d, 97.5))
    assert summ(0.0, np.array([-1.0, 1.0] * 50))["p"] == 1.0


def test_spearman_matches_scipy_and_family_average():
    rng = np.random.default_rng(3)
    E = rng.standard_normal((7, 2, 80))
    E[1] += E[0]
    C = asset_corr(E)
    for a in range(2):
        assert C[a] == pytest.approx(spearmanr(E[:, a].T).statistic)
    m = C.mean(0)
    assert fam_rho(m, [0, 1, 2]) == pytest.approx((m[0, 1] + m[0, 2] + m[1, 2]) / 3)
    assert fam_rho(m, [3, 4, 5, 6]) == pytest.approx(m[3:, 3:][np.triu_indices(4, 1)].mean())
    assert n_eff(np.eye(7), [0, 1, 2]) == pytest.approx(3) and n_eff(np.ones((7, 7)), [3, 4, 5, 6]) == pytest.approx(1)


def test_statistics_on_constructed_data():
    """Foundation errors share a common factor, more so in stress; the ensemble beats its members."""
    rng = np.random.default_rng(4)
    T, A = 400, 3
    stress = np.zeros(T, bool)
    stress[100:180] = True
    common = rng.standard_normal((A, T))
    E = rng.standard_normal((7, A, T))
    w = np.where(stress, 0.95, 0.5)          # share of error variance that is common; total variance fixed
    E[:3] = np.sqrt(w) * common + np.sqrt(1 - w) * E[:3]
    E *= 0.5
    R = np.exp(rng.standard_normal((A, T)))
    F = R[None] / np.exp(E)
    C0 = corr3(E, stress)
    idx = sb_indices(T, 300, 10, np.random.default_rng(5))
    idx = idx[stress[idx].sum(1) >= 20]              # same discard rule as the pipeline
    Cd = np.stack([corr3(E[:, :, ix], stress[ix]) for ix in idx])
    res = evaluate(C0, Cd, losses(F, R), idx, stress)
    assert res["H1"]["est"] > 0.3 and res["H1"]["p"] < 0.05
    assert res["H2"]["est"] > 0.2
    assert res["H3"]["est"] > 0                              # less diversification when errors move together
    assert all(m["est"] > 0 for m in res["H4"]["members"].values())
    assert res["H4"]["p"] == max(m["p"] for m in res["H4"]["members"].values())
    q = losses(F, R)
    assert res["D_all"]["est"] == pytest.approx(1 - q[7].mean() / q[:3].mean())
    assert q[7] == pytest.approx((R / F[:3].mean(0)) - np.log(R / F[:3].mean(0)) - 1)


def test_dm_matches_hand_formula():
    d = np.random.default_rng(6).standard_normal(200) + 0.2
    u, T = d - d.mean(), 200
    lrv = u @ u / T + 2 * sum((1 - k / 6) * (u[k:] @ u[:-k]) / T for k in range(1, 6))
    assert dm_test(d, 5)["stat"] == pytest.approx(d.mean() / np.sqrt(lrv / T))


# ---------- dashboard ----------

@pytest.mark.skipif(not (ROOT / "dashboard" / "index.html").exists(), reason="dashboard not built")
def test_dashboard_numbers_equal_stats_output():
    html = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
    stats = json.loads((ROOT / "results" / "stats" / "stats.json").read_text(encoding="utf-8"))
    robust = json.loads((ROOT / "results" / "stats" / "robustness.json").read_text(encoding="utf-8"))
    assert data["stats"] == stats and data["robustness"] == robust
    assert [h["id"] for h in data["stats"]["hypotheses"]] == ["H1", "H2", "H3", "H4"]
