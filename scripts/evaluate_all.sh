#!/usr/bin/env bash

ROOT="results"

# Evaluate all experiments that have ranked_genes directory
find "$ROOT" -type d -name ranked_genes -print0 |
  xargs -0 -n1 dirname |                
  sort -u |                             
  while read -r EXP; do
      echo ">>> python evaluate.py $EXP"
      python evaluate.py "$EXP"
  done
