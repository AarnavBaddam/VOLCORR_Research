# Error correlation of time series foundation models vs classical volatility models

Pipeline and dashboard for the pre-registration v1.0 (frozen 10 November 2026). The pre-registration
is the source of truth; every point where it was silent or could not be followed is in
[DEVIATIONS.md](DEVIATIONS.md). Results are summarised in [RESULTS.md](RESULTS.md).

There are no LLM calls anywhere. The only learned models are the three pinned foundation-model
checkpoints, run zero-shot.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a bash shell (Git Bash on Windows). uv installs
Python 3.11 and every locked dependency on first use.

```bash
./run.sh test        # creates the environment from uv.lock and runs the tests
```

The environment is created at `~/.venvs/volcorr` (outside the project, so cloud-sync clients do not
touch it). Set `UV_PROJECT_ENVIRONMENT` to move it.

## One command

```bash
./run.sh all
```

runs `data`, `forecast-classical`, `forecast-foundation`, `recalibrate`, `stats`, `robustness`,
`dashboard` in order and reproduces every table and figure from the raw files. Each stage can be run
alone (`./run.sh stats`). Verification stages: `./run.sh test`, `./run.sh audit` (lookahead audit),
`./run.sh spotcheck` (recomputes three random forecasts from the raw CSVs).

Open `dashboard/index.html` in a browser. It is one self-contained file with the stats output
embedded: no server, no network. A single HTML file was chosen over Streamlit because it needs
nothing installed to view, can be archived with the results, and its embedded numbers can be
asserted equal to the stats output in a test.

| Stage | Reads | Writes |
|---|---|---|
| `data` | Yahoo Finance (first run only) | `data/raw/*.csv`, `data/raw/MANIFEST.json`, `data/clean/` |
| `forecast-classical` | `data/clean/` | `results/raw/{garch,gjr,ewma,har}.parquet` |
| `forecast-foundation` | `data/clean/`, Hugging Face (first run only) | `results/raw/<model>/chunk_*.parquet` |
| `recalibrate` | `results/raw/` | `results/forecasts.parquet` (the one results table), `results/dropped_days.csv` |
| `stats` | results table | `results/stats/stats.json` and CSVs |
| `robustness` | results table | `results/stats/robustness.json`, `robustness.csv` |
| `dashboard` | stats output, manifest, config, lock, DEVIATIONS.md | `dashboard/index.html` |
| `archive` | every tracked file under `data/`, `results/`, `logs/`, plus config, lock and docs | `archive/runs/<run_id>.json` (committed), `archive/objects/` (gitignored blob store) |

History: `./run.sh all` ends with an `archive` snapshot. Each run gets an immutable manifest (`archive/runs/<run_id>.json`) that records the git commit, dirty files, config hash, package versions and the SHA-256 of every tracked file. Identical files are stored once. Run `bash run.sh archive snapshot <label>` after any stage you want kept, `bash run.sh archive list` to see runs, and `bash run.sh archive restore <run_id> <dir>` to recover one. Results are overwritten in place, so the archive is the only record of earlier runs; back up `archive/objects/` because it is gitignored.

Caching and resuming: raw files are downloaded once and never overwritten; every run re-verifies
their SHA-256 hashes against the manifest and stops on a mismatch. Forecast stages skip work already
on disk (foundation inference is checkpointed every 20 origins), so a crash loses at most one batch.
To recompute a stage from scratch, delete its outputs. Package versions, the config hash and model
revisions are appended to `logs/runs.jsonl` on every run.

## Configuration

All design numbers are in `config.yaml`, dependencies in `uv.lock`, and the Hugging Face revision
hash of each checkpoint in `config.yaml` under `foundation.models`.

## Hardware notes

Developed on Windows 11, RTX 4060 (8 GB), 28 logical cores. Measured stage times: data under 10 s,
classical forecasts 1 min, foundation forecasts about 10 min on the GPU (Moirai dominates),
recalibration 10 s, stats 1 min, robustness about 10 min. Foundation inference falls back to CPU
when CUDA is unavailable (`./run.sh forecast-foundation --device cpu`); expect it to be much slower.
Moirai with 100 samples peaks near 5.4 GB of GPU memory at 200 series per batch.

On Windows the torch and JAX DLLs must load before pyarrow; `volcorr/foundation.py` handles the
import order.

## Freeze steps left to the authors

The pre-registration's freeze procedure (PDF hash in this README, `prereg-v1.0` tag, committing the
manifest, config and lock file) is the authors' to perform. This directory is not yet a git
repository.
