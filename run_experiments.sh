#!/bin/bash
# run_experiments.sh
# Run Sequential LoRA and OrthoHist-LoRA on the five TRACE tasks.
# Usage:
#   bash run_experiments.sh [gpu_id] [k ...]
# Examples:
#   bash run_experiments.sh 0          # run k=4,8,16 if configs exist
#   bash run_experiments.sh 0 8        # run only k=8
#   bash run_experiments.sh 0 4 8 16

set -euo pipefail

GPU=${1:-0}
shift || true

KS=("$@")
if [ ${#KS[@]} -eq 0 ]; then
    KS=(4 8 16)
fi

export CUDA_VISIBLE_DEVICES="$GPU"
CONFIG_DIR=${CONFIG_DIR:-configs}

RUN_SEQ=${RUN_SEQ:-1}
RUN_ORTHO=${RUN_ORTHO:-1}

run_config() {
    local label="$1"
    local cfg="$2"

    if [ ! -f "$cfg" ]; then
        echo "[Skip] $label: config not found: $cfg"
        return 0
    fi

    echo ""
    echo ">>> $label"
    echo "    config: $cfg"
    python main.py --config "$cfg"
}


echo "===== Continual LoRA Experiments ====="
echo "GPU: $GPU"
echo "CONFIG_DIR: $CONFIG_DIR"
echo "OrthoHist k values: ${KS[*]}"
echo ""

if [ "$RUN_SEQ" = "1" ]; then
    run_config "Sequential LoRA" "$CONFIG_DIR/sequential_lora.yaml"
else
    echo "[Skip] Sequential LoRA disabled by RUN_SEQ=0"
fi

if [ "$RUN_ORTHO" = "1" ]; then
    for K in "${KS[@]}"; do
        run_config "OrthoHist-LoRA mt_rank=$K" "$CONFIG_DIR/ortho_hist_k${K}.yaml"
    done
else
    echo "[Skip] OrthoHist-LoRA disabled by RUN_ORTHO=0"
fi


echo ""
echo "===== Finished ====="
echo "To generate the table:"
echo "  python summarise_results.py"
