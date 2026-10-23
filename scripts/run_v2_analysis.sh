#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT="results/_spectral/v2_experiments"
OUTPUT="results/_spectral/v2_experiments/analysis"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            INPUT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

if command -v uv >/dev/null 2>&1; then
    uv run python scripts/analyze_v2_experiments.py --input "$INPUT" --output "$OUTPUT"
else
    python3 scripts/analyze_v2_experiments.py --input "$INPUT" --output "$OUTPUT"
fi

echo ""
echo "V2 analysis complete: $OUTPUT"
echo "Report: $OUTPUT/assessment_report.md"
echo "Plots:  $OUTPUT/plots"
echo "Tables: $OUTPUT/tables"
