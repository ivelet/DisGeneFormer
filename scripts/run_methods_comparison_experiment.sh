#!/usr/bin/env bash
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --account=bi-dlc2
#SBATCH --gres=gpu:1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DisGeneFormer_env
nvidia-smi
set -euo pipefail

ROOT="results/model_comparison"         

find "$ROOT" -type d -name ranked_genes -print0 |
while IFS= read -r -d '' RG_DIR; do
    EXP=$(dirname "$RG_DIR")          # go one level up: the method folder
    echo -e "\n=== Processing $EXP ==="
    python evaluate.py "$EXP"
done

