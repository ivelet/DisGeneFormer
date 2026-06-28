#!/bin/bash
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --account=bi-dlc2
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
nvidia-smi
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DisGeneFormer_env

echo "Running GeneNet Feature Removal Experiment"
ROOT="results/gene_net_feature_removal"         

find "$ROOT" -type f -name config.yml -print0 |
  xargs -0 -n1 dirname |
  sort -u |
  while read -r EXP; do
      # Only process if directory has exactly 1 file (config.yml only)
      FILE_COUNT=$(find "$EXP" -maxdepth 1 -type f | wc -l)
      if [ "$FILE_COUNT" -eq 1 ]; then
          echo -e "\\n=== Processing $EXP ==="
          (
            set -euo pipefail
            set -x               
            python train.py "$EXP"
            python predict_genes_fold.py "$EXP"
            python evaluate_fold.py "$EXP"
          )
      else
          echo -e "\\n⊘ Skipping $EXP (already processed - has $FILE_COUNT files)"
      fi
  done

echo -e "\\nRunning DiseaseNet Feature Removal Experiment"
ROOT="results/disease_net_feature_removal"         

find "$ROOT" -type f -name config.yml -print0 |
  xargs -0 -n1 dirname |
  sort -u |
  while read -r EXP; do
      # Only process if directory has exactly 1 file (config.yml only)
      FILE_COUNT=$(find "$EXP" -maxdepth 1 -type f | wc -l)
      if [ "$FILE_COUNT" -eq 1 ]; then
          echo -e "\\n=== Processing $EXP ==="
          (
            set -euo pipefail
            set -x               
            python train.py "$EXP"
            python predict_genes_fold.py "$EXP"
            python evaluate_fold.py "$EXP"
          )
      else
          echo -e "\\n⊘ Skipping $EXP (already processed - has $FILE_COUNT files)"
      fi
  done

echo -e "\\nCombine results and generate tables for GeneNet and DiseaseNet feature removal"
python plots/scripts/combine_results.py results/disease_net_feature_removal
python plots/scripts/combine_results.py results/gene_net_feature_removal

echo -e "\\n=== All experiments completed ==="