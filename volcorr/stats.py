"""Confirmatory tests H1-H4 and exploratory statistics on the 1-day horizon.
All inference uses one stationary bootstrap of dates (jointly across assets and models)."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .common import CLASSICAL, FOUNDATION, MODELS, ROOT, load_cfg, log_run
from .stress import count_episodes

OUT = ROOT / "results" / "stats"
FI, CI = [0, 1, 2], [3, 4, 5, 6]            # positions of each family in MODELS
ENS = {"ens_foundation": FI, "ens_classical": CI, "ens_all": FI + CI}
SERIES = MODELS + list(ENS)                  # rows of every loss array
ENS_F = SERIES.index("ens_foundation")
REGIMES = ["all", "stress", "calm"]
ASSET_CLASS = {"SPY": "US equity", "QQQ": "US equity", "IWM": "US equity", "XLF": "US equity sector",
               "XLE": "US equity sector", "XLK": "US equity sector", "EFA": "International equity",
               "EEM": "International equity", "TLT": "Treasury bonds", "GLD": "Gold"}


# ---------- data ----------

def prepare(df, h, start, end, target="target", recal="recal", models=MODELS):
    """Balanced arrays for one horizon and window: forecasts F[model, asset, day], targets R[asset, day]."""
    d = df[(df.horizon == h) & (df.origin_date >= start) & (df.origin_date <= end) & df.model.isin(models)]
    f = d.pivot(index="origin_date", columns=["model", "asset"], values=recal)
    # shortcut: a date missing for any model/asset is dropped for all assets, not only that asset;
    # identical while no asset-day fails (results/dropped_days.csv is empty). Upgrade if one ever does.
    f = f.dropna()
    assets = sorted(d.asset.unique())
    F = np.stack([f[m][assets].values.T for m in models])
    R = d[d.model == models[0]].pivot(index="origin_date", columns="asset", values=target).loc[f.index, assets].values.T
    return {"dates": f.index, "assets": assets, "F": F, "R": R, "E": np.log(R)[None] - np.log(F)}


def losses(F, R, kind="qlike", ens_weights=None):
    """Loss per series, asset and day: 7 models then the 3 equal-weight ensembles."""
    ens = [F[idx].mean(0) if ens_weights is None or name != "ens_foundation"
           else np.tensordot(ens_weights, F[idx], 1) for name, idx in ENS.items()]
    f = np.concatenate([F, np.stack(ens)])
    if kind == "mse":
        return (R[None] - f) ** 2
    x = R[None] / f
    return x - np.log(x) - 1


# ---------- correlation ----------

def asset_corr(E):
    """Spearman correlation between models, per asset. E[model, asset, day] -> [asset, model, model]."""
    if E.shape[-1] < 3:
        return np.full((E.shape[1], E.shape[0], E.shape[0]), np.nan)
    r = rankdata(E, axis=-1)
    r -= r.mean(-1, keepdims=True)
    r /= np.sqrt((r ** 2).sum(-1, keepdims=True))
    return np.einsum("man,kan->amk", r, r)


def corr3(E, stress):
    """Per-asset correlation matrices on all, stress and calm days: [3, asset, model, model]."""
    return np.stack([asset_corr(E), asset_corr(E[:, :, stress]), asset_corr(E[:, :, ~stress])])


def fam_rho(C, fam):
    """Mean pairwise correlation within a family, from matrices [..., model, model]."""
    fam = np.asarray(fam)
    i, j = np.triu_indices(len(fam), 1)
    return C[..., fam[i], fam[j]].mean(-1)


def n_eff(C, fam):
    """Effective number of independent models: (sum eig)^2 / sum(eig^2)."""
    ev = np.linalg.eigvalsh(C[..., np.ix_(fam, fam)[0], np.ix_(fam, fam)[1]])
    return ev.sum(-1) ** 2 / (ev ** 2).sum(-1)


# ---------- inference ----------

def sb_indices(T, n, mean_block, rng):
    """Stationary bootstrap (Politis-Romano): circular blocks with geometric lengths, mean `mean_block`."""
    new = rng.random((n, T)) < 1.0 / mean_block
    new[:, 0] = True
    starts = rng.integers(0, T, (n, T))
    t = np.arange(T)
    last = np.maximum.accumulate(np.where(new, t, 0), axis=1)
    return (np.take_along_axis(starts, last, 1) + t - last) % T


def bootstrap(E, stress, inf, block=None, discard=True):
    """Resampled date indices and per-draw correlation matrices [draw, 3, asset, model, model].
    Draws with fewer than the minimum number of stress days are discarded and counted."""
    rng = np.random.default_rng(inf["seed"])
    idx = sb_indices(E.shape[-1], inf["n_boot"], block or inf["mean_block"], rng)
    keep = stress[idx].sum(1) >= inf["min_stress_days_per_draw"] if discard else np.ones(len(idx), bool)
    idx = idx[keep]
    Cd = np.empty((len(idx), 3) + (E.shape[1], E.shape[0], E.shape[0]), np.float32)
    for b, ix in enumerate(idx):
        Cd[b] = corr3(E[:, :, ix], stress[ix])
    return idx, Cd, int((~keep).sum())


def summ(est, draws):
    """Point estimate, 95% percentile interval, two-sided p (twice the smaller tail share around zero)."""
    draws = np.asarray(draws, dtype=float)
    if not np.isfinite(est) or len(draws) == 0 or not np.isfinite(draws).all():
        return {"est": None, "lo": None, "hi": None, "p": None}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "p": float(p)}


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def dm_test(d, bandwidth):
    """Diebold-Mariano on a loss differential with a Newey-West (Bartlett) long-run variance."""
    d = np.asarray(d, dtype=float)
    T, u = len(d), d - d.mean()
    lrv = u @ u / T + 2 * sum((1 - k / (bandwidth + 1)) * (u[k:] @ u[:-k]) / T for k in range(1, bandwidth + 1))
    stat = d.mean() / np.sqrt(lrv / T)
    return {"mean_diff": float(d.mean()), "stat": float(stat), "p": float(2 * norm.sf(abs(stat)))}


def regime_means(L, s):
    """Mean loss on all, stress and calm days. L[..., day], s[..., day] -> [3, ...]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([L.mean(-1), (L * s).sum(-1) / s.sum(-1), (L * ~s).sum(-1) / (~s).sum(-1)])


def evaluate(C0, Cd, La, idx, stress, fam_f=FI, fam_c=CI, assets=None, members=FI, ens=ENS_F):
    """Every pre-registered statistic with its bootstrap summary.
    C0: point correlations [3, asset, m, m]; Cd: draws of the same; La: losses [series, asset, day]."""
    A = slice(None) if assets is None else assets
    c0, cd = C0[:, A].mean(1), Cd[:, :, A].mean(2)
    out = {}
    rho0 = {"F": fam_rho(c0, fam_f), "C": fam_rho(c0, fam_c)}
    rhod = {"F": fam_rho(cd, fam_f), "C": fam_rho(cd, fam_c)}
    for fam in "FC":
        for r, name in enumerate(REGIMES):
            out[f"rho_{fam}_{name}"] = summ(rho0[fam][r], rhod[fam][:, r])
    out["H1"] = summ(rho0["F"][0] - rho0["C"][0], rhod["F"][:, 0] - rhod["C"][:, 0])
    out["H2"] = summ((rho0["F"][1] - rho0["F"][2]) - (rho0["C"][1] - rho0["C"][2]),
                     (rhod["F"][:, 1] - rhod["F"][:, 2]) - (rhod["C"][:, 1] - rhod["C"][:, 2]))
    for fam, pos in (("F", fam_f), ("C", fam_c)):
        e0, ed = n_eff(c0, pos), n_eff(cd.astype(float), pos)
        for r, name in enumerate(REGIMES):
            out[f"neff_{fam}_{name}"] = summ(e0[r], ed[:, r])

    L = La[list(members) + [ens]][:, A].mean(1)          # daily loss = mean across assets
    m0 = regime_means(L, stress[None])                    # [3, 4]
    md = regime_means(L[:, idx], stress[idx][None])       # [3, 4, draw]

    def D(m):
        return 1 - m[-1] / m[:-1].mean(0)
    for r, name in enumerate(REGIMES):
        out[f"D_{name}"] = summ(D(m0[r]), D(md[r]))
    out["H3"] = summ(D(m0[2]) - D(m0[1]), D(md[2]) - D(md[1]))
    mem = {SERIES[m]: summ(m0[0][j] - m0[0][-1], md[0][j] - md[0][-1]) for j, m in enumerate(members)}
    ps = [v["p"] for v in mem.values()]
    worst = min(mem.values(), key=lambda v: v["est"])
    out["H4"] = {"est": worst["est"], "lo": worst["lo"], "hi": worst["hi"],
                 "p": None if None in ps else max(ps), "members": mem}
    return out


# ---------- reporting ----------

def clean(o):
    """JSON-safe: NaN -> null, numpy scalars -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(clean(obj), indent=1) + "\n", encoding="utf-8")


def verdicts(res, minimum_met, cfg):
    """Holm adjustment across H1-H4 and the pre-registered decision rules."""
    alpha, dec = cfg["inference"]["alpha"], cfg["decision"]
    adj = holm([res[h]["p"] for h in ("H1", "H2", "H3", "H4")])
    small = {"H1": dec["min_corr_effect"], "H2": dec["min_corr_effect"], "H3": dec["min_d_effect"]}
    text = {"H1": "Foundation-model errors are more correlated with each other than classical-model errors.",
            "H2": "The correlation gap is larger in stress than in calm periods.",
            "H3": "The ensemble's diversification benefit is smaller in stress than in calm periods.",
            "H4": "The equal-weight foundation ensemble beats each of its three members."}
    rows = []
    for h, pa in zip(("H1", "H2", "H3", "H4"), adj):
        r = res[h]
        positive = all(m["est"] > 0 for m in r["members"].values()) if h == "H4" else r["est"] > 0
        supported = bool(positive and pa < alpha)
        if h in ("H2", "H3") and not minimum_met:
            verdict = "inconclusive"
            note = "Insufficient stress: the pre-registered minimum of stress days and episodes is not met."
        elif supported and h in small and abs(r["est"]) < small[h]:
            verdict, note = "supported", "Detectable but not practically meaningful (below the pre-registered effect threshold)."
        else:
            verdict, note = ("supported" if supported else "not supported"), ""
        rows.append({"id": h, "hypothesis": text[h], "est": r["est"], "lo": r["lo"], "hi": r["hi"], "p": r["p"],
                     "p_holm": float(pa), "verdict": verdict, "note": note,
                     "members": r.get("members")})
    return rows


def conclusion(rows, minimum_met):
    v = {r["id"]: r["verdict"] for r in rows}
    out = []
    if v["H1"] != "supported":
        out.append("No evidence that foundation-model errors are more correlated than classical-model errors.")
    elif v["H2"] == "supported":
        out.append("Foundation-model errors are more correlated than classical-model errors, and the gap widens in stress.")
    elif v["H2"] == "inconclusive":
        out.append("Foundation-model errors are more correlated than classical-model errors. "
                   "Whether stress changes this cannot be concluded.")
    else:
        out.append("Foundation-model errors are more correlated at all times, without evidence that stress changes this.")
    if v["H4"] != "supported":
        out.append("Ensembling the three foundation models does not reliably beat its members.")
    elif v["H3"] == "supported":
        out.append("Ensembling helps overall, but the benefit is smaller in stress.")
    elif v["H3"] == "inconclusive":
        out.append("Ensembling foundation models helps. Whether the benefit shrinks in stress cannot be concluded.")
    else:
        out.append("Ensembling foundation models helps, and the benefit does not shrink detectably in stress.")
    if not minimum_met:
        out.append("Stress minimum not met: H2 and H3 are inconclusive because of insufficient stress. H1 and H4 are reported.")
    return out


def window_summary(d, stress_flags=None):
    """Point estimates for one window: correlations, losses and the foundation / HAR-RV QLIKE ratio."""
    s = np.zeros(len(d["dates"]), bool) if stress_flags is None else stress_flags
    c = corr3(d["E"], s).mean(1)
    q = losses(d["F"], d["R"]).mean(1)
    har = q[MODELS.index("har")].mean()
    return {"n_days": len(d["dates"]), "first": str(d["dates"][0].date()), "last": str(d["dates"][-1].date()),
            "rho_F": fam_rho(c[0], FI), "rho_C": fam_rho(c[0], CI), "corr": c[0],
            "qlike": {n: q[i].mean() for i, n in enumerate(SERIES)},
            "qlike_ratio_vs_har": {n: q[SERIES.index(n)].mean() / har for n in FOUNDATION + ["ens_foundation"]}}


def rolling_rho(E, window):
    """Trailing-window family correlations, averaged across assets."""
    out = np.full((2, E.shape[-1]), np.nan)
    for t in range(window - 1, E.shape[-1]):
        c = asset_corr(E[:, :, t - window + 1:t + 1]).mean(0)
        out[:, t] = fam_rho(c, FI), fam_rho(c, CI)
    return out


def primary_setup(cfg, df, stress_tab, h=1, rule="primary", **kw):
    w = cfg["windows"]["primary"]
    d = prepare(df, h, w["start"], w["end"], **kw)
    d["stress"] = stress_tab.loc[d["dates"], rule].values.astype(bool)
    return d


def load_inputs():
    df = pd.read_parquet(ROOT / "results" / "forecasts.parquet")
    st = pd.read_parquet(ROOT / "data" / "clean" / "stress.parquet").set_index("date")
    return df, st


def smoke(cfg, df):
    """Classical-only check before any foundation model exists."""
    w = cfg["windows"]["primary"]
    d = prepare(df, 1, w["start"], w["end"], models=CLASSICAL)
    c = asset_corr(d["E"]).mean(0)
    x = d["R"][None] / d["F"]
    print("days", len(d["dates"]), "| rho_C", round(float(fam_rho(c, range(4))), 4))
    print("QLIKE", dict(zip(CLASSICAL, np.round((x - np.log(x) - 1).mean((1, 2)), 4))))
    print(pd.DataFrame(c, CLASSICAL, CLASSICAL).round(3))


def main():
    cfg = load_cfg()
    df, st = load_inputs()
    if "--smoke" in sys.argv:
        return smoke(cfg, df)
    log_run("stats")
    inf, sc = cfg["inference"], cfg["stress"]
    d = primary_setup(cfg, df, st)
    s, E = d["stress"], d["E"]
    n_stress, n_ep = int(s.sum()), count_episodes(s, sc["episode_max_gap"])
    minimum_met = n_stress >= sc["min_stress_days"] and n_ep >= sc["min_episodes"]

    C0 = corr3(E, s)
    idx, Cd, discarded = bootstrap(E, s, inf)
    Lq, Lm = losses(d["F"], d["R"]), losses(d["F"], d["R"], "mse")
    res = evaluate(C0, Cd, Lq, idx, s)
    hyp = verdicts(res, minimum_met, cfg)

    daily = Lq.mean(1)                                   # [series, day]
    rm, rm_mse = regime_means(daily, s[None]), regime_means(Lm.mean(1), s[None])
    qlike = [{"series": n, "family": "foundation" if i in FI else "classical" if i in CI else "ensemble",
              "qlike_all": rm[0][i], "qlike_stress": rm[1][i], "qlike_calm": rm[2][i], "mse_all": rm_mse[0][i]}
             for i, n in enumerate(SERIES)]
    dm = {"h4": {SERIES[m]: dm_test(daily[m] - daily[ENS_F], inf["dm_bandwidth"]) for m in FI},
          "vs_har": {e: dm_test(daily[SERIES.index(e)] - daily[MODELS.index("har")], inf["dm_bandwidth"]) for e in ENS}}

    from arch.bootstrap import MCS
    mcs = MCS(pd.DataFrame(daily.T, columns=SERIES), size=inf["mcs_level"], reps=inf["n_boot"],
              block_size=inf["mean_block"], method="R", bootstrap="stationary", seed=inf["seed"])
    mcs.compute()
    mcs_out = {"level": inf["mcs_level"], "included": list(mcs.included),
               "pvalues": {k: float(v) for k, v in mcs.pvalues["Pvalue"].items()}}

    by_asset = []
    for a, name in enumerate(d["assets"]):
        rf, rc = fam_rho(C0[0, a], FI), fam_rho(C0[0, a], CI)
        dr = fam_rho(Cd[:, 0, a], FI) - fam_rho(Cd[:, 0, a], CI)
        by_asset.append({"asset": name, "asset_class": ASSET_CLASS[name], "rho_F": rf, "rho_C": rc,
                         "diff": summ(rf - rc, dr), "qlike": {n: Lq[i, a].mean() for i, n in enumerate(SERIES)}})

    w = cfg["windows"]
    sec = prepare(df, 1, w["secondary"]["start"], w["secondary"]["end"])
    secondary = window_summary(sec)
    secondary["requested"] = w["secondary"]
    crisis = []
    for cw in w["crisis"]:
        m = (sec["dates"] >= cw["start"]) & (sec["dates"] <= cw["end"])
        row = {"start": cw["start"], "end": cw["end"], "n_days": int(m.sum()), "rho_F": None, "rho_C": None}
        if m.sum() >= 5:
            c = asset_corr(sec["E"][:, :, m]).mean(0)
            row.update(rho_F=fam_rho(c, FI), rho_C=fam_rho(c, CI))
        crisis.append(row)

    full = prepare(df, 1, w["secondary"]["start"], w["primary"]["end"])
    roll_window = 63
    roll = rolling_rho(full["E"], roll_window)
    stf = st.loc[full["dates"]]

    out = {
        "meta": {"horizon": 1, "window": w["primary"], "n_days": len(s), "n_assets": len(d["assets"]),
                 "assets": d["assets"], "models": MODELS, "series": SERIES,
                 "n_stress": n_stress, "n_calm": int((~s).sum()), "n_episodes": n_ep,
                 "min_stress_days": sc["min_stress_days"], "min_episodes": sc["min_episodes"],
                 "stress_minimum_met": bool(minimum_met), "n_boot": inf["n_boot"], "mean_block": inf["mean_block"],
                 "seed": inf["seed"], "discarded_draws": discarded, "valid_draws": len(idx)},
        "hypotheses": hyp,
        "conclusion": conclusion(hyp, minimum_met),
        "components": {k: v for k, v in res.items() if not k.startswith("H")},
        "corr": {name: C0[r].mean(0) for r, name in enumerate(REGIMES)},
        "qlike": qlike,
        "dm": dm,
        "mcs": mcs_out,
        "by_asset": by_asset,
        "primary_summary": window_summary(d, s),
        "secondary": secondary,
        "crisis": crisis,
        "rolling": {"window": roll_window, "dates": [str(x.date()) for x in full["dates"]],
                    "rho_F": roll[0], "rho_C": roll[1], "vix": stf["vix"].values,
                    "stress": stf["primary"].values.astype(int), "primary_start": w["primary"]["start"]},
    }
    write_json("stats.json", out)
    cols = ["id", "est", "lo", "hi", "p", "p_holm", "verdict", "note"]
    pd.DataFrame(hyp)[cols].to_csv(OUT / "hypotheses.csv", index=False, lineterminator="\n")
    pd.DataFrame(qlike).to_csv(OUT / "qlike.csv", index=False, lineterminator="\n")
    pd.DataFrame([{"stat": k, **v} for k, v in out["components"].items()]).to_csv(
        OUT / "components.csv", index=False, lineterminator="\n")
    for name in REGIMES:
        pd.DataFrame(out["corr"][name], MODELS, MODELS).to_csv(OUT / f"corr_{name}.csv", lineterminator="\n")
    pd.DataFrame([{"asset": b["asset"], "asset_class": b["asset_class"], "rho_F": b["rho_F"], "rho_C": b["rho_C"],
                   **{f"diff_{k}": v for k, v in b["diff"].items()}, **{f"qlike_{k}": v for k, v in b["qlike"].items()}}
                  for b in by_asset]).to_csv(OUT / "by_asset.csv", index=False, lineterminator="\n")

    print(f"primary window: {len(s)} days, {n_stress} stress days, {n_ep} episodes, minimum met: {minimum_met}")
    print(f"bootstrap: {len(idx)} valid draws, {discarded} discarded")
    print(pd.DataFrame(hyp)[cols[:-1]].to_string(index=False))
    for line in out["conclusion"]:
        print("-", line)


if __name__ == "__main__":
    main()
