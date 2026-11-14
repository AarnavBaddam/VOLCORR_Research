# Results

Pre-registration v1.0. Primary window: forecast origins 2025-04-17 to 2026-10-30, 387 trading days,
10 ETFs, 1-day Garman-Klass variance. Stationary bootstrap, 10,000 resamples (164 discarded for
fewer than 20 stress days), mean block 10, seed 20261110. Holm across H1 to H4.
Full numbers: `results/stats/`, dashboard: `dashboard/index.html`.

## Stress minimum

54 stress days in 9 episodes. The minimum is 60 days and 4 episodes, so **the stress minimum is not
met** and H2 and H3 are inconclusive by the pre-registered rule.

## Confirmatory hypotheses

| | Statistic | Estimate | 95% CI | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| H1 | rho_F - rho_C, all days | 0.052 | [0.041, 0.062] | <0.0001 | <0.0001 | **Supported** |
| H2 | stress-minus-calm gap, foundation minus classical | 0.025 | [-0.024, 0.056] | 0.356 | 0.712 | Inconclusive (insufficient stress) |
| H3 | D(calm) - D(stress) | -0.0145 | [-0.0262, -0.0028] | 0.012 | 0.036 | Inconclusive (insufficient stress) |
| H4 | smallest member-minus-ensemble QLIKE | -0.0011 | [-0.0061, 0.0028] | 0.660 | 0.712 | **Not supported** |

- **H1.** rho_F = 0.978, rho_C = 0.926. The difference, 0.052, is just above the pre-registered
  0.05 threshold for practical meaning.
- **H4.** The ensemble beats Moirai (0.0142, p < 0.0001) but not TimesFM (0.0017, p = 0.48) and is
  slightly worse than Chronos-Bolt (-0.0011, p = 0.66). The Diebold-Mariano checks agree
  (statistics -0.52, 0.69, 3.26).
- **H3.** The point estimate has the opposite sign to the hypothesis: D is 2.8% on stress days and
  1.3% on calm days. No conclusion is drawn.

Conclusion from the pre-registered decision table:

- Foundation-model errors are more correlated than classical-model errors. Whether stress changes
  this cannot be concluded.
- Ensembling the three foundation models does not reliably beat its members.
- H2 and H3 are inconclusive because of insufficient stress.

## Exploratory

- **Where the H1 gap comes from.** log-HAR-RV errors correlate 0.97 to 0.98 with each foundation
  model and only 0.88 to 0.90 with the three return-based models. With the classical family reduced
  to GARCH, GJR and EWMA the gap shrinks to 0.017 [0.008, 0.025]; with the pre-specified least
  redundant subset (GARCH, EWMA, HAR-RV) it is 0.064 [0.049, 0.078].
- **Effective number of independent models.** Foundation 1.03 of 3, classical 1.12 of 4.
- **Accuracy (QLIKE, lower is better).** HAR-RV 0.323, Chronos-Bolt 0.325, foundation ensemble
  0.326, TimesFM 0.327, Moirai 0.340, GJR 0.348, GARCH 0.366, EWMA 0.396. No ensemble beats HAR-RV
  (foundation ensemble minus HAR-RV: 0.0031, DM p = 0.62).
- **Model Confidence Set (10%).** Kept: HAR-RV, Chronos-Bolt, TimesFM, GJR, foundation ensemble,
  all-model ensemble. Dropped: Moirai, GARCH, EWMA, classical ensemble.
- **By asset.** rho_F - rho_C is positive for all 10 assets, from 0.035 (TLT) to 0.086 (XLF).

## Robustness (exploratory)

| Check | H1 | H4 (smallest) | Note |
|---|---|---|---|
| 5-day horizon | 0.075 [0.059, 0.093] | 0.0005, p = 0.83 | |
| Squared-return target | 0.011 [0.009, 0.013] | -0.0088, p = 0.85 | H1 gap much smaller with the noisier target |
| Leave one asset out | 0.048 to 0.053 | -0.0016 to -0.0001 | No single asset drives either result |
| Block length 5 / 20 | p < 0.0001 both | p = 0.63 / 0.71 | |
| Inverse-QLIKE weights | unchanged | -0.0011, p = 0.68 | Weights 0.340, 0.334, 0.326 |
| MSE loss | unchanged | 3.4e-12, p = 0.96 | |
| Stress rules R1, R4 | | | H2 0.012 for both; R2 and R3 not estimable (4 and 7 stress days) |

## Secondary window (possibly contaminated, diagnostic only)

Available origins: 2023-01-12 to 2025-04-15, 566 days (see DEVIATIONS.md, section B).

- Foundation / HAR-RV QLIKE ratio, secondary vs primary: Chronos-Bolt 0.975 vs 1.006, TimesFM 0.995
  vs 1.015, Moirai 1.018 vs 1.054. Each foundation model looks 2 to 4% better relative to HAR-RV
  where memorization is possible. This is a single comparison of two periods with no interval.
- rho_F is 0.978 in both windows; rho_C is 0.937 (secondary) and 0.926 (primary).
- Crisis windows available: March 2023, August 2024, early April 2025 (10 to 18 days each). The 2020
  and 2022 windows are not available.

## Integrity

- Data: 11 raw files, 2,980 bars per ETF, 0 dropped bars, 0 floored GK values, 0 VIX fills.
- 0 asset-days dropped for model failure.
- `pytest`: 22 passed. Lookahead audit: 9 checks passed. Spot check: 3 of 3 forecasts matched.
- Rerun from a clean copy with cached raw data: all forecast chunks, the results table and every
  stats file byte-identical.

## What did not match the pre-registration

1. The stress minimum failed (a pre-registered outcome).
2. The secondary window starts in January 2023 instead of January 2019, because the 2015 sample
   start leaves no room for both 1,000-day burn-ins.
3. Five ambiguous points were resolved with stated defaults, and the freeze steps (PDF hash, tag,
   commits) were not performed by the pipeline. Details in DEVIATIONS.md.
