#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=64G
nvidia-smi
conda activate DisGeneFormer_env


# Gradual HPO phenotype annotation removal experiment

# Step 1 — pick the gene population (larger n than the 14). 
# Build the Phen2Gene-conditioned set at top-50 (the recommended headline population — large and fair) and the strict intersection as confirmation:
python build_shared_gene_list.py results/omim_track/longer_training \
    --mode phen2gene_conditioned --topk-threshold 50 --out shared_p2g_top50.tsv

python build_shared_gene_list.py results/omim_track/longer_training \
    --mode intersection --topk-threshold 100 --out shared_intersection_top100.tsv

# Step 2 — build the removal schedule (defines, identically for both methods, the order terms are stripped). 
# IC-order is the clinically realistic primary; random is the robustness check:

python build_removal_schedule.py results/omim_track/longer_training \
    --gene-list shared_p2g_top50.tsv --order ic --out schedule_ic.tsv

python build_removal_schedule.py results/omim_track/longer_training \
    --gene-list shared_p2g_top50.tsv --order random --seed 0 --out schedule_rand0.tsv

# Step 3 — run the graded sweep. One run per level per method. 
# Save each summary to a per-level filename so the plotter can find them:
for F in 0.00 0.25 0.50 0.75 1.00; do
  # DisGeneFormer (needs working GPU)
  python predict_ranking_blinded_same_genes.py results/omim_track/longer_training \
      --checkpoint best_model.ptm --ks 1,5,10,20,50 \
      --schedule results/omim_track/longer_training/schedule_ic.tsv \
      --remove-fraction $F
  cp results/omim_track/longer_training/blinded_eval/summary_blinded.json \
     results/omim_track/longer_training/blinded_eval/summary_f${F}.json

  # Phen2Gene (CPU, slower)
  python predict_phen2gene_blinded_same_genes.py results/omim_track/longer_training \
      --phen2gene-dir /home/fr/fr_fr/fr_rk1111/Phen2Gene \
      --ks 1,5,10,20,50 \
      --schedule results/omim_track/longer_training/schedule_ic.tsv \
      --remove-fraction $F
  cp results/omim_track/longer_training/phen2gene_blinded_eval/summary_blinded.json \
     results/omim_track/longer_training/phen2gene_blinded_eval/summary_f${F}.json
done

# Step 4 — plot the curves and emit the paper table:
python plot_degradation_curves.py results/omim_track/longer_training \
    --metric hit_rate@50 --out degradation_hit50.png

python plot_degradation_curves.py results/omim_track/longer_training \
    --metric MRR --out degradation_mrr.png