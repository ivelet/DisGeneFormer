#!/bin/bash
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --account=bi-dlc2
#SBATCH --gres=gpu:1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DisGeneFormer_env
nvidia-smi
#!/usr/bin/env bash
set -euo pipefail

ROOT="experiments/hard_negatives_ablation"         

find "$ROOT" -type f -name config.yml -print0 |
  xargs -0 -n1 dirname      |  
  sort -u                   |   
  while read -r EXP; do
      echo -e "\n=== Processing $EXP ==="
      (
        set -x               
        python train.py  "$EXP"
        python predict_genes_fold.py  "$EXP"
        python evaluate.py       "$EXP"
      )
  done
