#!/bin/bash
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --account=bi-dlc2
#SBATCH --gres=gpu:1
# export CONDA_ROOT=/home/fr/fr_fr/fr_rk1111/miniconda3
# eval "$($CONDA_ROOT/bin/conda shell.bash hook)"
# conda activate DisGeneFormer_env2
# ml devel/cuda
# ml compiler/gnu
# nvidia-smi
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DisGeneFormer_env
nvidia-smi
#!/usr/bin/env bash
set -euo pipefail

ROOT="results/humannet_comparison"         

find "$ROOT" -type f -name config.yml -print0 |
  xargs -0 -n1 dirname      |  
  sort -u                   |   
  while read -r EXP; do
      echo -e "\n=== Processing $EXP ==="
      (
        set -x               
        python train.py  "$EXP"
        python predict_genes.py  "$EXP"
        python evaluate.py       "$EXP"
      )
  done
