#!/bin/bash
set -e

# Confusable Pairs Analysis Pipeline
# Usage: ./run_analysis.sh <model_path> [output_dir] [--use_cache]

MODEL_PATH="${1:-outputs/audio_text_mix}"
OUTPUT_DIR="${2:-Experiment_2/output}"
TEST_FILE="${3:-slurp/dataset/slurp/test.jsonl}"
AUDIO_DIR="${4:-slurp/audio/slurp_real}"
USE_CACHE="${5:-}"  # Pass "--use_cache" to skip inference

# Find latest checkpoint if directory given
if [ -d "$MODEL_PATH" ]; then
    CHECKPOINT=$(ls -d "$MODEL_PATH"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        MODEL_PATH="$CHECKPOINT"
    fi
fi

echo "============================================================"
echo " Confusable Pairs Analysis Pipeline"
echo "============================================================"
echo " Model:      $MODEL_PATH"
echo " Output Dir: $OUTPUT_DIR"
echo " Test File:  $TEST_FILE"
echo " Audio Dir:  $AUDIO_DIR"
echo " Use Cache:  ${USE_CACHE:-No}"
echo "============================================================"

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Step 1: Run main analysis (feature extraction + confusion matrix)
echo ""
echo "[Step 1/4] Running main analysis..."

CACHE_ARG=""
if [ "$USE_CACHE" = "--use_cache" ]; then
    CACHE_ARG="--use_cache"
fi

python Experiment_2/run_analysis.py \
    --model_path "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --audio_dir "$AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_beams 3 \
    --top_k_confusable 10 \
    $CACHE_ARG
    # Add --save_attention for attention analysis (requires more memory)
    # Add --max_samples 100 for quick testing

echo ""
echo "[Step 2/4] Generating t-SNE/UMAP visualizations..."
python Experiment_2/visualize_features.py \
    --input_dir "$OUTPUT_DIR" \
    --method both

echo ""
echo "[Step 3/4] Running entropy analysis..."
python Experiment_2/entropy_analysis.py \
    --input_dir "$OUTPUT_DIR"

# Step 4: Attention analysis (optional, requires --save_attention in step 1)
if [ -f "$OUTPUT_DIR/attention_weights.pt" ]; then
    echo ""
    echo "[Step 4/4] Running attention analysis..."
    python Experiment_2/attention_analysis.py \
        --input_dir "$OUTPUT_DIR"
else
    echo ""
    echo "[Step 4/4] Skipping attention analysis (no attention weights saved)"
    echo "         To enable, add --save_attention to step 1"
fi

echo ""
echo "============================================================"
echo " Analysis Complete!"
echo "============================================================"
echo " Results saved to: $OUTPUT_DIR"
echo ""
echo " Key outputs:"
echo "   - cached_inference_data.pt : Full cached data (for re-analysis)"
echo "   - analysis_summary.json    : Overall summary"
echo "   - sample_results.json      : Per-sample results"
echo "   - confusion_matrix.npy     : Confusion matrix"
echo "   - hidden_states.pt         : Feature embeddings"
echo "   - figures/                 : Visualizations"
echo "   - entropy_figures/         : Entropy analysis"
echo ""
echo " To re-run analysis without inference:"
echo "   ./run_analysis.sh $MODEL_PATH $OUTPUT_DIR --use_cache"
echo "============================================================"
