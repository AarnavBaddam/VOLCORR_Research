"""Section 8 exploratory analyses for v2.0. Estimates, 95% bootstrap intervals and unadjusted p-values only.
Never given a verdict.

E1: diversification benefit D = 1 - QLIKE(equal-weight ensemble) / mean QLIKE(members), foundation vs classical,
    on all, stress and calm days. Classical is shown as the full family of 4 and as each matched 3-model subset.
E2: error-style correlation (Spearman across models, per asset, averaged) of the LOG RECALIBRATED forecasts."""
from itertools import combinations

import numpy as np

from .common import MODELS, qlike
from .stats import CI, FI, bootstrap, corr3, fam_rho, regime_means, summ

REGIMES = ["all", "stress", "calm"]


def _div_daily(F, R, members):
    """Daily (asset-averaged) QLIKE of the equal-weight ensemble, and the mean daily QLIKE of its members."""
    qe = qlike(R, F[list(members)].mean(0)).mean(0)
    qm = np.mean([qlike(R, F[m]).mean(0) for m in members], axis=0)
    return qe, qm


def _D(qe, qm, s, idx):
    """D by regime: point estimate on all days, and one draw per resample."""
    point = 1 - regime_means(qe, s) / regime_means(qm, s)
    draws = 1 - regime_means(qe[idx], s[idx]) / regime_means(qm[idx], s[idx])
    return point, draws


def e1(d, s, idx):
    families = {"foundation": list(FI), "classical_4": list(CI)}
    for sub in combinations(CI, 3):
        families["classical_" + "+".join(MODELS[i] for i in sub)] = list(sub)
    D = {name: _D(*_div_daily(d["F"], d["R"], m), s, idx) for name, m in families.items()}
    p_f, d_f = D["foundation"]
    out = {"D": {}, "diff_classical_minus_foundation": {}}
    for name, (p, dr) in D.items():
        out["D"][name] = {REGIMES[r]: summ(p[r], dr[r]) for r in range(3)}
        if name != "foundation":
            out["diff_classical_minus_foundation"][name] = {
                REGIMES[r]: summ(p[r] - p_f[r], dr[r] - d_f[r]) for r in range(3)}
    return out


def e2(d, s, inf, idx):
    Efc = np.log(d["F"])                       # log recalibrated forecasts, [model, asset, day]
    C0 = corr3(Efc, s)
    idx2, Cd, _ = bootstrap(Efc, s, inf)
    assert np.array_equal(idx2, idx), "bootstrap draws must match the primary run"
    c0, cd = C0.mean(1), Cd.mean(2)            # [3, m, m], [draw, 3, m, m]
    rho0 = {"F": fam_rho(c0, FI), "C": fam_rho(c0, CI)}
    rhod = {"F": fam_rho(cd, FI), "C": fam_rho(cd, CI)}
    out = {"rho": {}, "gap_rho_F_minus_rho_C": {}}
    for fam in "FC":
        out["rho"][fam] = {REGIMES[r]: summ(rho0[fam][r], rhod[fam][:, r]) for r in range(3)}
    out["gap_rho_F_minus_rho_C"] = {REGIMES[r]: summ(rho0["F"][r] - rho0["C"][r], rhod["F"][:, r] - rhod["C"][:, r])
                                    for r in range(3)}
    return out


def run(cfg, d, idx, s, inf):
    return {"label": "exploratory: no verdict, unadjusted p-values",
            "E1": e1(d, s, idx), "E2": e2(d, s, inf, idx)}
