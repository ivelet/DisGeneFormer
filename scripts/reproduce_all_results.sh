#!/bin/bash
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --account=bi-dlc2
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
nvidia-smi
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DisGeneFormer_env
set -euo pipefail

# This script assumes the trained model files are already present in the results directories.
# By default, runs all experiments found under results
ROOT="results"          

find "$ROOT" -type f -name config.yml -print0 |
  xargs -0 -n1 dirname |
  sort -u |
  while read -r EXP; do
      # Only process if directory has exactly 1 file (config.yml only)
      FILE_COUNT=$(find "$EXP" -maxdepth 1 -type f | wc -l)
      if [ "$FILE_COUNT" -eq 1 ]; then
          echo -e "\n=== Processing $EXP ==="
          (
            set -x               
            python predict_genes.py "$EXP"
            python evaluate.py "$EXP"
          )
      else
          echo -e "\n⊘ Skipping $EXP (already processed)"
      fi
  done