# Pre-registration v2.0: confirmatory test of H2 and H3

**Status: DRAFT. Not frozen.** This plan takes effect only when the commit containing it is tagged
`prereg-v2.0` (see "Freeze procedure"). The evaluation window starts after that tag, so no v2.0 data
exists while this document can still change.

Authors: Aarnav Baddam, Arjun Allu. Advisor copy: Prof. López de Prado.

## 1. Relationship to v1.0

v1.0 (pre-registration frozen 10 November 2026; code, config and results at git tag `v1.0-complete`)
is the first study. Its record is final and is not edited by this plan:

- H1 supported: rho_F - rho_C = 0.052 [0.041, 0.062].
- H4 not supported.
- H2 and H3 inconclusive: 54 stress days in 9 episodes against a minimum of 60 days and 4 episodes.

v2.0 is a new, separate study. It re-tests H2 and H3 on data collected only after this plan is frozen.
It does not reuse any v1.0 evaluation day, and it cannot change a v1.0 verdict. `DEVIATIONS.md` and
`RESULTS.md` belong to v1.0 and stay as they are; v2.0 writes `DEVIATIONS_v2.0.md` and
`RESULTS_v2.0.md`.

A v1.1 amendment (extend the sample back to 2005 and score 2019 to 2025 as exploratory) was drafted
on 19 November 2026 and not adopted. No v1.1 analysis was run.

## 2. Hypotheses

The test family is H2 and H3, worded as in v1.0:

- **H2.** The gap between foundation-model and classical-model error correlation is larger on stress
  days than on calm days: (rho_F,stress - rho_F,calm) - (rho_C,stress - rho_C,calm) > 0.
- **H3.** The equal-weight foundation ensemble's diversification benefit is smaller on stress days than
  on calm days: D_calm - D_stress > 0, where D = 1 - QLIKE(ensemble) / mean QLIKE(members).

Statistics, error definition (log realized minus log recalibrated forecast), per-asset Spearman
correlation averaged across assets, family averages and D are computed exactly as in v1.0
(`volcorr/stats.py: evaluate()`).

H1 and H4 are re-estimated on the v2.0 window as a replication. They are reported with intervals but
are not part of the v2.0 test family and receive no verdict.

## 3. Data

- Assets: SPY, QQQ, IWM, EFA, EEM, TLT, GLD, XLF, XLE, XLK; ^VIX for the stress rule. Source yfinance,
  `auto_adjust: false`.
- History: sample start 2015-01-02, as in v1.0, so the warm-up and recalibration schedule match v1.0.
- **Evaluation window start:** the first US trading day after the calendar date (US Eastern time) of
  the commit tagged `prereg-v2.0`.
- **Evaluation window end:** set by the stopping rule (section 5).
- Raw data are downloaded once, after the window has ended, into a new folder with its own SHA-256
  manifest. v1.0 raw files are not touched.
- Prices from the new download that overlap the v1.0 sample are compared with the v1.0 files, and the
  largest difference is reported.

## 4. Stress rule

v2.0 states its own stress rule. It is the same definition as the v1.0 primary rule, kept so that the
two studies measure the same thing:

- A forecast origin t is a stress day if the VIX close on t is above the 80th percentile (linear
  interpolation) of the previous 252 VIX closes, excluding t, on the ETF trading calendar. Missing VIX
  closes are filled forward.
- Stress days separated by 5 or fewer calm days belong to one episode.

## 5. Minimum evidence and stopping rule

- **Minimum:** 90 stress days and 6 episodes inside the evaluation window.
- **Stopping rule:** the window ends on the first trading day T on which the window, counted from its
  start through T, contains at least 90 stress days and at least 6 episodes. The data download happens
  after T + 5 trading days so every 1-day and 5-day target is realized.
- **Hard stop:** if the minimum is not reached by 2031-12-31, the window ends on the last trading day
  of 2031. H2 and H3 are then estimated and reported as "inconclusive, insufficient stress".
- **Monitoring:** while the window is open, the only data anyone may download or look at is the VIX
  close, to count stress days and episodes. No ETF prices, forecasts, errors or losses for the window
  are computed until the stopping rule has fired. The stress count depends only on VIX, so checking it
  does not reveal any model result.

**Why 90 and 6.** From v1.0, the bootstrap standard error of the H2 statistic with 54 stress days was
about 0.020 (95% interval [-0.024, 0.056]). For 80% power to detect the practical threshold of 0.05 at
a two-sided level of 0.025 (Holm's first step with two tests), the standard error must be at most
0.05 / (2.24 + 0.84) = 0.016. Assuming the standard error scales with 1 / sqrt(stress days), that needs
about 54 x (0.020 / 0.016)^2 = 84 stress days, rounded up to 90 because the scaling is approximate. H3
is already adequately powered at this size (v1.0 standard error about 0.006 against a threshold of
0.02). Six episodes keeps the result from resting on one or two events. v1.0 recorded about 14% of days
as stress days; at 14 to 20%, 90 stress days take roughly 450 to 640 trading days (about 1.8 to 2.5
years).

## 6. Models, target and recalibration

Identical to v1.0 `config.yaml`: the three pinned foundation checkpoints (same Hugging Face revisions,
context 512, seed 20261110); GARCH(1,1), GJR-GARCH(1,1,1), EWMA (lambda 0.94) and log-HAR-RV on a rolling
1,000-day window refit every 20 origins; next-day Garman-Klass variance floored at 1e-8; log-log
recalibration on the latest 1,000 realized pairs, refit every 20 origins, anchored as in v1.0.

## 7. Inference and decision rules

- Stationary bootstrap of dates, jointly across assets and models: 10,000 draws, mean block 10, draws
  with fewer than 20 stress days discarded, seed 20261110.
- p-values are two-sided: twice the smaller share of draws on either side of zero.
- Holm adjustment across the two-test family {H2, H3}, alpha 0.05.
- Practical thresholds: 0.05 for H2, 0.02 for H3 (unchanged from v1.0).
- Verdicts, applied mechanically:
  - **Inconclusive** if the minimum in section 5 is not met.
  - **Supported** if the estimate is positive, the Holm p is below 0.05, and the estimate is at least
    the practical threshold.
  - **Detectable, below practical threshold** if positive and Holm p below 0.05, but under the threshold.
  - **Opposite direction** if the estimate is negative and the Holm p is below 0.05.
  - **Not supported** otherwise.

## 8. Exploratory analyses

Reported with estimates, 95% intervals and unadjusted p-values, labelled exploratory, never given a
verdict:

- Robustness: stress rules R1 (VIX > 25) and R4 (SPY 21-day volatility above its 90th percentile),
  5-day horizon, block lengths 5 and 20, leave one asset out, squared-return target.
- E1: foundation versus classical diversification benefit, D_C - D_F, on all, stress and calm days, also
  with matched 3-model classical families.
- E2: correlation of log recalibrated forecasts, computed like the error correlation.
- E3: input confound. The three foundation models and HAR-RV rerun with a return-based input instead of
  log Garman-Klass variance: the log of the trailing 5-day mean of squared daily log returns (days t-4 to t),
  floored at 1e-8. A 5-day mean is used instead of single-day squared returns because near-zero-return days
  would otherwise put spikes of log(1e-8) into the input series. GARCH, GJR-GARCH, EWMA, the target,
  recalibration and the bootstrap are unchanged.
- Descriptive tables on the v2.0 window, carried over from the v1.0 code: mean QLIKE by model and regime,
  per-asset correlations, Diebold-Mariano tests and the 10% model confidence set.

v1.0's secondary-window (2019 to 2025), crisis-window and rolling-correlation diagnostics are not computed
in v2.0. They concern the possibly contaminated period and are already reported in v1.0.

## 9. What changed from v1.0, and why

| Item | v1.0 | v2.0 | Reason |
|---|---|---|---|
| Test family | H1 to H4 | H2, H3 | H1 and H4 already have confirmatory verdicts |
| Holm family size | 4 | 2 | Follows from the test family |
| Direction of test | Verdict only for the hypothesized direction | Two-sided, with an "opposite direction" verdict | v1.0's inconclusive H2 and H3 estimates leaned against the hypotheses; a confirmatory test should be able to say so |
| Minimum stress days | 60 | 90 | Power calculation in section 5 |
| Minimum episodes | 4 | 6 | Longer window; guards against a single event |
| Window | Fixed dates | Starts after the freeze tag; ends by stopping rule or 2031-12-31 | Data must not exist when the plan is frozen |
| Stress rule | VIX above 80th percentile of previous 252 closes | Same definition | Keeps the two studies comparable |
| Practical thresholds | 0.05 (H2), 0.02 (H3) | Same | Not changed after seeing v1.0 estimates |

## 10. Freeze procedure

1. Both authors review this document and the v2.0 analysis code.
2. Change "Status" above to "FROZEN", commit this file together with the analysis code, and tag the
   commit `prereg-v2.0`.
3. Push the commit and tag to GitHub, and email a PDF of this file with its SHA-256 to the advisor the
   same day.
4. Collection starts on the next US trading day. Any later change to this plan is a deviation and is
   logged in `DEVIATIONS_v2.0.md` with its date and reason.
