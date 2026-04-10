#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_DATASET=$(find "${PROJECT_ROOT}/design3toolbenchdataset/async_agent_bench" -maxdepth 1 -type f -name '*.jsonl' ! -name '_tmp_*' | sort | tail -n 1)
DATASET="${1:-$DEFAULT_DATASET}"
NUM_SAMPLES=10
ATTN_TYPE="blocked"
CONDA_ENV="${CONDA_ENV:-lsa2}"

if [ -z "$DATASET" ] || [ ! -f "$DATASET" ]; then
    echo "No valid dataset found. Generate one under design3toolbenchdataset/async_agent_bench first."
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
OUTPUT_DIR="result/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "Running ALL TESTS"
echo "Dataset: ${DATASET}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "========================================================================"
echo ""

echo "========================================================================"
echo "Running TEST 1: STANDARD SYNC"
echo "========================================================================"
conda run -n $CONDA_ENV python test_1_standard_sync.py --num_samples $NUM_SAMPLES --attn_type $ATTN_TYPE --dataset_path $DATASET --output_dir $OUTPUT_DIR

echo -e "\n\n========================================================================"
echo "Running TEST 3: CHUNK ASYNC"
echo "========================================================================"
conda run -n $CONDA_ENV python test_3_simple_async.py --num_samples $NUM_SAMPLES --attn_type $ATTN_TYPE --dataset_path $DATASET --output_dir $OUTPUT_DIR

echo -e "\n\n========================================================================"
echo "Running TEST 4: BLOCK THEN LINKS"
echo "========================================================================"
conda run -n $CONDA_ENV python test_4_lap_async.py --num_samples $NUM_SAMPLES --attn_type $ATTN_TYPE --dataset_path $DATASET --output_dir $OUTPUT_DIR

echo -e "\n\n========================================================================"
echo "Running TEST 5: BLOCK LINK CHAIN"
echo "========================================================================"
conda run -n $CONDA_ENV python test_5_advanced_lap.py --num_samples $NUM_SAMPLES --attn_type $ATTN_TYPE --dataset_path $DATASET --output_dir $OUTPUT_DIR

echo -e "\n\n========================================================================"
echo "ALL TESTS COMPLETED!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================================================"

echo ""
echo "=== RESULT SUMMARY ==="
for stats_file in ${OUTPUT_DIR}/*stats.json; do
    if [ -f "$stats_file" ]; then
        echo ""
        echo "File: $(basename $stats_file)"
        python -c "import json; data = json.load(open('$stats_file')); print(f'  Accuracy: {data.get(\"accuracy\", 0):.4f}'); print(f'  Avg TTFT (compute only): {data.get(\"avg_ttft\", 0):.4f}s'); print(f'  Avg REAL TTFT (from start): {data.get(\"avg_real_ttft\", 0):.4f}s'); print(f'  === KEY METRIC ==='); print(f'  Avg TTFT AFTER LAST TOOL ARRIVED: {data.get(\"avg_ttft_after_last_tool\", 0):.4f}s'); print(f'  Total compute: {data.get(\"total_time_compute\", 0):.4f}s'); print(f'  Total wait: {data.get(\"total_wait_time\", 0):.4f}s'); print(f'  Total: {data.get(\"total_time_total\", 0):.4f}s')"
    fi
done