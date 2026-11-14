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
    test)                uv run --locked pytest -q ;;
    all) for s in data forecast-classical forecast-foundation recalibrate stats robustness dashboard; do stage "$s"; done ;;
    *) echo "stages: data forecast-classical forecast-foundation recalibrate stats robustness dashboard audit spotcheck test all"; exit 2 ;;
  esac
}
stage "${@:-all}"
