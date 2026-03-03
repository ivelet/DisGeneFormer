#!/usr/bin/env bash

ROOT="results"

echo "======================================================================="
echo "Evaluating DisGeneFormer methods (with folds)"
echo "======================================================================="

# Find all experiments with ranked_genes_* directories (NOT in model_comparison)
find "$ROOT" -type d \( -name "ranked_genes_best" -o -name "ranked_genes_fold_*" \) -print0 |
  xargs -0 -n1 dirname |
  sort -u |
  grep -v "model_comparison" |
  while read -r EXP; do
      echo ""
      echo ">>> python evaluate_fold.py $EXP"
      python evaluate_fold.py "$EXP"
  done

echo ""
echo "======================================================================="
echo "Evaluating existing methods (model_comparison)"
echo "======================================================================="

# Find all experiments in model_comparison with ranked_genes directory
find "$ROOT/model_comparison" -type d -name "ranked_genes" -print0 2>/dev/null |
  xargs -0 -n1 dirname |
  sort -u |
  while read -r EXP; do
      echo ""
      echo ">>> python evaluate.py $EXP"
      python evaluate.py "$EXP"
  done

echo ""
echo "======================================================================="
echo "All evaluations complete!"
echo "======================================================================="