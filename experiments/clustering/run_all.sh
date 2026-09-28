#!/usr/bin/env bash
# Every selected dataset, sequentially, on one machine.
#
#   ./run_all.sh                    # [extra args go to every dataset]
#   ./run_all.sh --device cpu --methods euclidean dtw
set -euo pipefail
cd "$(dirname "$0")"

TABLE="results/ucr_nn_datasets.csv"
if [ ! -f "$TABLE" ]; then
    python3 run.py --list --device cpu
fi

DATASETS=$(awk -F, 'NR>1 {sub(/\r$/,"",$7); if ($7=="keep") print $1}' "$TABLE")
for d in $DATASETS; do
    echo "=== $d"
    python3 run.py "$d" "$@"
done

python3 summarize.py --csv
