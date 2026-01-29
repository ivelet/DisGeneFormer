#!/usr/bin/env bash
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --account=bi-dlc2
#SBATCH --gres=gpu:1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DisGeneFormer_env
nvidia-smi
set -euo pipefail

ROOT="results/model_comparison" 

python evaluate_fold.py results/model_comparison/DisGeneFormer_XC_V3

python evaluate_fold.py results/model_comparison/DisGeneFormer

find "$ROOT" -type d -name ranked_genes -print0 |
while IFS= read -r -d '' RG_DIR; do
    EXP=$(dirname "$RG_DIR")          
    echo -e "\n=== Processing $EXP ==="
    python evaluate.py "$EXP"
done

echo -e "\n=== Evaluation complete ==="
echo -e "\n=== Generating plots ==="

python plots/scripts/plot_tp_curves.py results/model_comparison --output-dir plots/results/method_comparison_tp_curves --exclude-methods graph_baseline

echo -e "\n=== Combining results ==="

python plots/scripts/combine_results.py results/model_comparison

echo -e "\n=== Experiment complete ==="
