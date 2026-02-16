#!/bin/bash
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --account=bi-dlc2
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
nvidia-smi
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DisGeneFormer_env
set -euo pipefail

echo "Running GeneNet Feature Removal Experiment"
ROOT="results/gene_net_feature_removal"         

find "$ROOT" -type f -name config.yml -print0 |
  xargs -0 -n1 dirname      |  
  sort -u                   |   
  while read -r EXP; do
      echo -e "\n=== Processing $EXP ==="
      (
        set -x               
        python train.py  "$EXP"
        python predict_genes_fold.py  "$EXP"
        python evaluate_fold.py       "$EXP"
      )
  done

echo "Running GeneNet XC V3 Feature Removal Experiment"
ROOT="results/gene_net_xc_v3_feature_removal"         

find "$ROOT" -type f -name config.yml -print0 |
  xargs -0 -n1 dirname      |  
  sort -u                   |   
  while read -r EXP; do
      echo -e "\n=== Processing $EXP ==="
      (
        set -x               
        python train.py  "$EXP"
        python predict_genes_fold.py  "$EXP"
        python evaluate_fold.py       "$EXP"
      )
  done

echo "Running DiseaseNet Feature Removal Experiment"
ROOT="results/disease_net_feature_removal"         

find "$ROOT" -type f -name config.yml -print0 |
  xargs -0 -n1 dirname      |  
  sort -u                   |   
  while read -r EXP; do
      echo -e "\n=== Processing $EXP ==="
      (
        set -x               
        python train.py  "$EXP"
        python predict_genes_fold.py  "$EXP"
        python evaluate_fold.py       "$EXP"
      )
  done

  echo "Combine results and generate tables for GeneNet and DiseaseNet feature removal"
  python plots/scripts/combine_results.py results/disease_net_feature_removal
  python plots/scripts/combine_results.py results/gene_net_feature_removal
