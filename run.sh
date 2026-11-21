#!/usr/bin/env bash
# One entry point for every stage. Usage: ./run.sh <stage>   (./run.sh all reproduces everything)
set -euo pipefail
cd "$(dirname "$0")"
# The virtualenv lives outside the project so OneDrive does not sync or lock it.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/volcorr}"
export PYTHONHASHSEED=0

py() { uv run --locked python -m "volcorr.$1" "${@:2}"; }

stage() {
  case "$1" in
    data)                py data ;;
    forecast-classical)  py classical ;;
    forecast-foundation) py foundation "${@:2}" ;;
    recalibrate)         py recalibrate ;;
    stats)               py stats "${@:2}" ;;
    robustness)          py robustness ;;
    dashboard)           py dashboard ;;
    audit)               py audit ;;
    spotcheck)           py spotcheck ;;
    archive)             py archive "${@:2}" ;;
    test)                uv run --locked pytest -q ;;
    v2-status)           VOLCORR_CONFIG=config_v2.yaml uv run --locked python -m volcorr.v2 status ;;
    v2-all)              VOLCORR_CONFIG=config_v2.yaml uv run --locked python -m volcorr.v2 run ;;
    all) for s in data forecast-classical forecast-foundation recalibrate stats robustness dashboard archive; do stage "$s"; done ;;
    *) echo "stages: data forecast-classical forecast-foundation recalibrate stats robustness dashboard audit spotcheck archive test all | v2-status v2-all"; exit 2 ;;
  esac
}
stage "${@:-all}"
