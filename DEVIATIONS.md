# Deviations and interpretation log

Pre-registration v1.0, frozen 10 November 2026. This file records (A) pre-registered contingencies
that were triggered, (B) one design point that could not be followed as written, (C) defaults taken
on ambiguous points, and (D) implementation readings where the pre-registration is silent.
No hypothesis, model, window, threshold, loss or test was changed.

## A. Pre-registered contingencies triggered

| Date | Section | What happened | Consequence fixed by the pre-reg |
|---|---|---|---|
| 2026-11-10 | Stress definition, minimum evidence | Primary rule gives **54 stress days** in 9 episodes over the 387 primary-window origins. The minimum is 60 days and 4 episodes. The day minimum is not met. | H2 and H3 are reported "inconclusive, insufficient stress". H1 and H4 are reported. The rule is not changed. |

The count was recomputed with an independent loop. It is 53 or 54 under every percentile
interpolation convention (linear, lower, higher, nearest, midpoint) and under both the ETF and the
VIX trading calendar, so no reading of the rule reaches 60. The H2 and H3 estimates are still
computed and shown, labelled inconclusive.

## B. Could not be followed as written

| Date | Section | Problem | What was done | Effect on results |
|---|---|---|---|---|
| 2026-11-10 | Recalibration; evaluation windows (secondary) | The sample starts 2 Jan 2015. Classical models need 1,000 days before their first raw forecast, and recalibration needs 1,000 forecast-target pairs after that. The first origin every model can forecast is 2019-01-24 and the first recalibrated origin is **2023-01-12** (1-day) and 2023-01-19 (5-day). The secondary window is specified as 2019-01-02 to 2025-04-15. | The secondary diagnostics use only origins with a full, as-specified recalibration: 2023-01-12 to 2025-04-15, 566 days. No rule was shortened or replaced. | Confirmatory results: none (the primary window has its full 1,000-pair burn-in). Diagnostics: the 2020 and 2022 crisis windows are not available; the March 2023, August 2024 and April 2025 windows are. The alternative (an expanding recalibration window before 2023) was not run; it would need author approval as a deviation. |

## C. Defaults taken on ambiguous points

These were raised with the authors on 2026-11-10 with a recommended reading. The instruction
received was to finish the work, so the recommended reading was used. Each can be changed and
rerun; none affects which hypothesis is tested.

| # | Section | Ambiguity | Default used | Effect |
|---|---|---|---|---|
| C2 | Models, Moirai | "Patch size is the library default for daily data." | The `uni2ts` constructor default, `patch_size="auto"`: for each series the library scores its patch sizes (8 to 128) on the last 5 observed days and uses the best. This needs 517 values: the forecast context is the 512 days ending at the origin, and the 5 days before that window are used only for patch selection. Nothing after the origin is used. | Moirai forecasts only. A fixed patch size of 16 or 32 was not run. |
| C3 | Robustness, squared-return target | Whether the check changes only the target or also the model inputs. | Target only: raw forecasts are unchanged, and are recalibrated and scored against squared daily log returns (5-day: mean of daily squared returns), each day floored at 1e-8. | Robustness row "Squared daily returns". |
| C4 | Models, log-HAR-RV | "Rolling 1,000-day window." | 1,000 regression rows (target days), the newest being the origin. | First forecast origin is 2019-01-24. |
| C5 | Models, recalibration | Anchor of "refit every 20 trading days". | Classical models refit on the first forecast origin and every 20th origin after it. Recalibration refits on the first origin with 1,000 realized pairs and every 20th after it. | None on design. |
| C6 | Reproducibility | GPU inference is not guaranteed bit-identical. | Tested instead of assumed: a full rerun from a clean copy with cached raw data reproduced every forecast chunk, the results table and all stats files byte for byte on the same machine (RTX 4060, same lock file). Other hardware may differ in the last floating-point digits of foundation forecasts. | None observed. |

## D. Implementation readings where the pre-reg is silent

| Date | Section | Reading used | Effect |
|---|---|---|---|
| 2026-11-10 | Data, point-in-time rules | A bar is "invalid" if any of open, high, low, close, adjusted close is missing or non-positive, or high < low. | None: 0 missing, 0 invalid bars; 0 GK values floored. |
| 2026-11-10 | Data, sample | yfinance called with end = 2026-11-07 (its end date is exclusive). Last bar is 2026-11-06. Yahoo returns ^VIX rows on 2026-05-25 and 2026-09-07 (US holidays); they are ignored because origins exist only on ETF trading days. | None. |
| 2026-11-10 | Stress definition | 80th percentile by linear interpolation over the previous 252 closes on the ETF calendar; a full 252 prior closes required. Stress days and episodes are counted inside the primary window. | 54 days under this reading, 53 or 54 under the others. |
| 2026-11-10 | Stress rules R3, R4 | R3: SPY unadjusted close more than 10% below the maximum close of the 252 days ending at the origin. R4: 21-day standard deviation of SPY adjusted-close log returns above the 90th percentile of its previous 252 values. | Robustness only. |
| 2026-11-10 | Models, checkpoints | Revisions pinned to the repository heads on 10 Nov 2026 (hashes in `config.yaml`). Weight files last changed on 2024-11-13 (Chronos-Bolt), 2025-04-16 (TimesFM), 2024-06-14 (Moirai). Chronos-Bolt has README-only commits after the 16 Apr 2025 cutoff bound; its weights are unchanged, so the bound holds. | None. |
| 2026-11-10 | Models, TimesFM | Loaded with the `timesfm` 1.3.0 package from `torch_model.ckpt`, with the model-card settings for this checkpoint (50 layers, no positional embedding), context 512, frequency 0 (daily). "Point forecast" is the library's default point output, which in this version is the median head. | TimesFM forecasts only. |
| 2026-11-10 | Models, multi-step | Each foundation model forecasts 5 steps in one pass; the 1-day forecast is step 1 and the 5-day forecast is the mean of the five exponentiated steps. HAR-RV iterates its one-step forecast; GARCH models use analytic multi-step variance; EWMA is flat. | 5-day robustness only. |
| 2026-11-10 | Models, GARCH | Fitted with the `arch` package on returns in percent (scaled back afterwards). Between refits the fixed parameters are filtered over the rolling 1,000-day window ending at each origin. | None on design. |
| 2026-11-10 | Models, EWMA | Variance starts at the mean of the first 30 squared returns of the rolling 1,000-day window, then the 0.94 recursion runs over the remaining 970. | None on design. |
| 2026-11-10 | Models, Moirai sampling | 100 samples per series, seeded per batch of 20 origins (seed 20261110 + batch number), so results do not depend on where a run resumes. | None on design. |
| 2026-11-10 | Recalibration | Residual variance uses n - 2 degrees of freedom. For the 5-day horizon the newest usable pair is the forecast made 5 days before the origin. | None on design. |
| 2026-11-10 | Failures | A missing forecast is an unfinished run and stops the pipeline; only a forecast that fails twice is dropped for all models. | 0 asset-days dropped. |
| 2026-11-10 | Inference | The same 10,000 draws serve all four hypotheses; draws with fewer than 20 stress days are discarded for all of them (164 discarded, 9,836 kept). The two-sided p is exactly twice the smaller tail share, so it can be 0; it is then displayed as below 1/9,836. | H1 p displayed as <0.0001. |
| 2026-11-10 | Inference, Holm | H2 and H3 stay in the family of four even though they are inconclusive. | Conservative for H1 and H4. |
| 2026-11-10 | H4 reporting | The headline H4 estimate and interval are those of the member with the smallest difference; the p-value is the largest of the three. | Presentation only. |
| 2026-11-10 | Exploratory, MCS | `arch` implementation, range statistic, stationary bootstrap with block 10, 10,000 replications, seed 20261110. | Exploratory. |
| 2026-11-10 | Exploratory, rolling chart | The dashboard's rolling correlation uses a 63-day window, which is not pre-registered. It is descriptive only. | Exploratory. |
| 2026-11-10 | Robustness, inverse-QLIKE weights | "Calibration window" read as all recalibrated origins before the primary window (2023-01-12 to 2025-04-15). Weights are proportional to 1 / mean daily QLIKE, one set for all assets, fixed over the primary window. | Robustness row. |
| 2026-11-10 | Robustness, stress rules | A rule with fewer than 20 stress days in the window cannot pass the per-draw minimum and is reported as not estimable (R2: 4 days, R3: 7 days). | Robustness rows. |
| 2026-11-10 | Results by asset class | Reported per asset with its class label; no separate class-level aggregates. | Exploratory. |
| 2026-11-10 | Pinned configuration | Python 3.11; `timesfm` 1.3.0; `uni2ts` 1.1.1; `pyarrow` < 18, `transformers` < 5 and `jaxtyping` < 0.2.35 for compatibility with `uni2ts` (NumPy 1.26, torch 2.5.1). All in `uv.lock`. | None on design. |

## Not done by the pipeline

The pre-registration's freeze steps (PDF export and hash, repository tag `prereg-v1.0`, committing
the manifest, config and lock file before the first evaluation run) are the authors' actions. The
project directory was not a git repository when the forecasts were produced, so the order "commit,
then run" is not evidenced by commit history; the manifest timestamps and `logs/runs.jsonl` record
when the data were downloaded and when each stage ran.
