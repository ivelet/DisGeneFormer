#!/bin/bash
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --account=bi-dlc2
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
nvidia-smi
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DisGeneFormer_env

# Your model — blinded (frozen weights, zero each causal gene's HPO features)
# python predict_ranking_blinded.py results/omim_track/longer_training \
#     --checkpoint best_model.ptm \
#     --ks 1,5,10,20,50 \
#     --blind-set topk --topk-threshold 50

# # Phen2Gene — strict-blinded (gene removed from entire KB per query)
# # python predict_phen2gene_blinded.py results/omim_track/longer_training \
# #     --phen2gene-dir /work/dlclarge1/koeksalr-disgeneformer/DisGeneFormer_test/Phen2Gene \
# #     --ks 1,5,10,20,50 \
# #     --blind-set topk --topk-threshold 50

# python predict_phen2gene_blinded.py results/omim_track/longer_training \
#     --phen2gene-dir /work/dlclarge1/koeksalr-disgeneformer/DisGeneFormer_test/Phen2Gene \
#     --ks 1,5,10,20,50 \
#     --blind-set topk --topk-threshold 50


# # Same gene set experiment
# # Step 1 — build the shared set from the per-gene files your two runs already produced (no re-running needed for this step):
# python build_shared_gene_list.py results/omim_track/longer_training --topk-threshold 50

# # Step 2 — blind exactly that shared set in both methods:
# python predict_ranking_blinded_same_genes.py results/omim_track/longer_training \
#     --checkpoint best_model.ptm --ks 1,5,10,20,50 \
#     --gene-list results/omim_track/longer_training/shared_blind_set.tsv

# python predict_phen2gene_blinded_same_genes.py results/omim_track/longer_training \
#     --phen2gene-dir /home/fr/fr_fr/fr_rk1111/Phen2Gene \
#     --ks 1,5,10,20,50 \
#     --gene-list results/omim_track/longer_training/shared_blind_set.tsv



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


# Experiment: complementary project 
python analyze_complementarity.py \
  --dgf  your_dgf_predictions.csv \
  --p2g  your_phen2gene_predictions.csv \
  --truth your_causal_genes.csv