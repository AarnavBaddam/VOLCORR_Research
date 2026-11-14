"""Pre-registered robustness checks. All exploratory: none can overturn a confirmatory result."""
from itertools import combinations

import numpy as np
import pandas as pd

from .common import MODELS, load_cfg, log_run
from .stats import (CI, FI, OUT, SERIES, bootstrap, corr3, evaluate, load_inputs, losses, prepare, primary_setup,
                    write_json)
from .stress import count_episodes

HS = ("H1", "H2", "H3", "H4")


def row(check, variant, res, keep=HS, **extra):
    out = {"check": check, "variant": variant, **extra}
    for h in HS:
        r = res.get(h) if res and h in keep else None
        out[h] = None if r is None else {k: r[k] for k in ("est", "lo", "hi", "p")}
    return out


def full_run(d, inf, block=None, kind="qlike"):
    C0 = corr3(d["E"], d["stress"])
    idx, Cd, discarded = bootstrap(d["E"], d["stress"], inf, block)
    La = losses(d["F"], d["R"], kind)
    return {"C0": C0, "Cd": Cd, "idx": idx, "La": La, "discarded": discarded}


def ev(run, d, **kw):
    return evaluate(run["C0"], run["Cd"], kw.pop("La", run["La"]), run["idx"], d["stress"], **kw)


def main():
    cfg = load_cfg()
    log_run("robustness")
    inf, sc = cfg["inference"], cfg["stress"]
    df, st = load_inputs()
    d = primary_setup(cfg, df, st)
    base = full_run(d, inf)
    rows = [row("Primary result", "1-day, GK target, primary stress rule, QLIKE, block 10", ev(base, d),
                discarded=base["discarded"])]

    d5 = primary_setup(cfg, df, st, h=5)
    r5 = full_run(d5, inf)
    rows.append(row("Horizon", "5-day", ev(r5, d5), discarded=r5["discarded"], n_days=len(d5["dates"])))

    for rule in ("r1", "r2", "r3", "r4"):
        dr = primary_setup(cfg, df, st, rule=rule)
        n = int(dr["stress"].sum())
        extra = {"n_stress": n, "n_episodes": count_episodes(dr["stress"], sc["episode_max_gap"])}
        if n < inf["min_stress_days_per_draw"]:
            rows.append(row("Stress rule", rule.upper(), None, note="Not estimable: fewer than 20 stress days.", **extra))
            continue
        rr = full_run(dr, inf)
        rows.append(row("Stress rule", rule.upper(), ev(rr, dr), keep=("H2", "H3"), discarded=rr["discarded"], **extra))

    dq = primary_setup(cfg, df, st, target="target_sqret", recal="recal_sqret")
    rq = full_run(dq, inf)
    rows.append(row("Target", "Squared daily returns", ev(rq, dq), discarded=rq["discarded"]))

    w = cfg["windows"]["secondary"]
    cal = prepare(df, 1, w["start"], w["end"])           # calibration data: before the primary window
    q = losses(cal["F"], cal["R"]).mean((1, 2))[FI]
    wts = (1 / q) / (1 / q).sum()
    rows.append(row("Ensemble weights", "Inverse-QLIKE", ev(base, d, La=losses(d["F"], d["R"], ens_weights=wts)),
                    keep=("H3", "H4"), note="weights " + ", ".join(f"{MODELS[i]}={x:.3f}" for i, x in zip(FI, wts))))

    for a, name in enumerate(d["assets"]):
        rows.append(row("Leave one asset out", f"without {name}",
                        ev(base, d, assets=[i for i in range(len(d["assets"])) if i != a])))

    for sub in combinations(CI, 3):
        names = "+".join(MODELS[i] for i in sub)
        note = "least redundant subset" if names == "garch+ewma+har" else ""
        rows.append(row("Matched family size", names, ev(base, d, fam_c=list(sub)), keep=("H1", "H2"), note=note))

    for block in inf["robustness_blocks"]:
        rb = full_run(d, inf, block)
        rows.append(row("Block length", f"mean block {block}", ev(rb, d), discarded=rb["discarded"]))

    rows.append(row("Loss", "MSE", ev(base, d, La=losses(d["F"], d["R"], "mse")), keep=("H3", "H4")))

    write_json("robustness.json", {"rows": rows})
    flat = [{"check": r["check"], "variant": r["variant"], "note": r.get("note", ""),
             **{f"{h}_{k}": (r[h] or {}).get(k) for h in HS for k in ("est", "lo", "hi", "p")}} for r in rows]
    pd.DataFrame(flat).to_csv(OUT / "robustness.csv", index=False, lineterminator="\n")
    show = pd.DataFrame([{"check": r["check"], "variant": r["variant"][:28],
                          **{h: (None if r[h] is None or r[h]["est"] is None else round(r[h]["est"], 4)) for h in HS}} for r in rows])
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
