"""
predict_genes.py - Generate ranked gene predictions for diseases

This script loads a trained DisGeneFormer model and generates ranked gene lists
for each disease in the evaluation set.

Usage:
    python predict_genes.py experiment_dir                  # Generate predictions and evaluate
    python predict_genes.py experiment_dir --predict-only   # Only generate predictions
"""

import os
import os.path as osp
import ast
import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from box import Box

from source.DiseaseNet import DiseaseNet
from source.GeneNet import GeneNet
from source.DisGeneFormer import DisGeneFormer

# Suppress matplotlib warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ranked gene predictions for diseases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Directory containing config.yml and trained model"
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Only generate predictions without evaluation"
    )
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_disease_omim_positive_list(all_omim_associations, omim_ids_list, disease_id, disease_name, test_path):
    """Get OMIM positive genes for a disease from multiple OMIM IDs."""
    disease_omim_positive_list_path = osp.join(
        test_path, 'omim_positive_lists', f'{disease_id}_omim_positives.tsv'
    )
    
    disease_omim_positive_list = []
    all_omim_associations = all_omim_associations.copy()
    all_omim_associations.columns = ['gene_id', 'omim_id']
    
    for omim_id in omim_ids_list:
        omim_positive_list_path = osp.join(
            test_path, 'omim_positive_lists', f'{omim_id}_positives.tsv'
        )
        
        if os.path.exists(omim_positive_list_path):
            omim_positive_list = pd.read_csv(omim_positive_list_path, sep='\t')
        else:
            mask = all_omim_associations['omim_id'] == omim_id
            omim_positive_list = all_omim_associations[mask].copy()  # FIX: Added .copy()
            
            if len(omim_positive_list) > 0:  # FIX: Only add 'y' if DataFrame is not empty
                omim_positive_list['y'] = 1.0  # FIX: Changed from .loc[:, 'y']
                
                os.makedirs(osp.join(test_path, 'omim_positive_lists'), exist_ok=True)
                omim_positive_list.to_csv(omim_positive_list_path, sep='\t', index=False)
        
        disease_omim_positive_list.append(omim_positive_list)
    
    disease_omim_positives = pd.concat(disease_omim_positive_list, ignore_index=True)
    disease_omim_positives.to_csv(disease_omim_positive_list_path, sep='\t', index=False)
    
    return disease_omim_positives


def get_omim_test_set_pairs(all_genes, omim_id, omim_positive_list):
    """Create test pairs (positives + negatives) for evaluation."""
    omim_positive_list = omim_positive_list.copy()
    omim_positive_list.columns = ['gene_id', 'omim_id', 'y']
    
    negatives = pd.DataFrame({
        'gene_id': sorted(set(all_genes) - set(omim_positive_list['gene_id'])),
        'omim_id': omim_id,
        'y': 0.0
    })
    
    test_pairs = pd.concat([omim_positive_list, negatives], ignore_index=True)
    return test_pairs


def format_test_set(test_set_pairs, gene_id_idx_mapping, omim_id_idx_mapping):
    """Format test pairs with index mappings."""
    test_set_pairs = test_set_pairs.copy()
    test_set_pairs['gene_idx'] = test_set_pairs['gene_id'].map(gene_id_idx_mapping).astype(float)
    test_set_pairs['disease_idx'] = test_set_pairs['omim_id'].map(omim_id_idx_mapping).astype(float)
    
    # Remove unmapped entries
    test_set_pairs = test_set_pairs.dropna(subset=["gene_idx", "disease_idx"]).copy()
    test_set_pairs["gene_idx"] = test_set_pairs["gene_idx"].astype(float)
    test_set_pairs["disease_idx"] = test_set_pairs["disease_idx"].astype(float)
    
    X_pairs = []
    y_labels = []
    
    for _, row in test_set_pairs.iterrows():
        X_pairs.append((float(row['gene_idx']), float(row['disease_idx'])))
        y_labels.append(float(row['y']))
    
    test_data_df = pd.DataFrame(
        [[X_pairs], [y_labels]], 
        index=["X", "y"]
    )
    
    return test_data_df


def get_id_mappings(experiment_dir):
    """Load all ID mappings."""
    # Gene mappings
    gene_id_mapping_path = osp.join(experiment_dir, 'gene_net', 'processed', 'gene_id_index_mapping.tsv')
    df_gene = pd.read_csv(gene_id_mapping_path, sep='\t', header=None, skiprows=1)
    gene_id_to_idx = dict(zip(df_gene[0].astype(int), df_gene[1].astype(int)))
    gene_idx_to_id = dict(zip(df_gene[1].astype(int), df_gene[0].astype(int)))
    
    # Disease mappings
    disease_id_mapping_path = osp.join(experiment_dir, 'disease_net', 'processed', 'disease_id_feature_index_mapping.tsv')
    df_disease = pd.read_csv(disease_id_mapping_path, sep='\t', header=None, skiprows=1)
    omim_id_to_idx = dict(zip(df_disease[0].astype(str), df_disease[1].astype(int)))
    omim_idx_to_id = dict(zip(df_disease[1].astype(int), df_disease[0].astype(str)))
    
    return gene_id_to_idx, gene_idx_to_id, omim_id_to_idx, omim_idx_to_id


def get_disease_name(disease_id):
    """Get disease name from ID."""
    disease_id_name_mapping = {
        'C0006142': 'Malignant_Neoplasm_Of_breast',
        'C0009402': 'Colorectal_Carcinoma',
        'C0023893': 'Liver_Cirrhosis_Experimental',
        'C0036341': 'Schizophrenia',
        'C0376358': 'Malignant_Neoplasm_Of_Prostate',
        'C0001973': 'Alcoholic_Intoxication_Chronic',
        'C0011581': 'Depressive_Disorder',
        'C0860207': 'Drug_Induced_Liver_Disease',
        'C3714756': 'Intellectual_Disability',
        'C0005586': 'Bipolar_Disorder'
    }
    return disease_id_name_mapping.get(disease_id, 'Unknown_Disease')


def evaluate_gnet(cfg, experiment_dir, predict_only):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load networks
    print("Loading GeneNet...")
    gene_dataset = GeneNet(cfg=cfg, experiment_dir=experiment_dir)
    gene_net_data = gene_dataset[0].to(device)
    
    print("Loading DiseaseNet...")
    disease_dataset = DiseaseNet(cfg=cfg, experiment_dir=experiment_dir)
    disease_net_data = disease_dataset[0].to(device)
    
    # Initialize model
    print("Initializing model...")
    ablate_cfg = cfg.get('model', {}).get('ablate', {})
    ablate_dict = {k: bool(v) for k, v in ablate_cfg.items() if v}
    
    model = DisGeneFormer(
        gene_feature_dim=gene_net_data.x.shape[1],
        disease_feature_dim=disease_net_data.x.shape[1],
        fc_hidden_dim=cfg.model.fc_hidden_dim,
        gene_net_hidden_dim=cfg.model.gene_net_hidden_dim,
        disease_net_hidden_dim=cfg.model.disease_net_hidden_dim,
        mode='DGP',
        ablate=ablate_dict
    ).to(device)
    
    # Load model weights
    best_model_path = osp.join(experiment_dir, 'best_model.ptm')
    if not os.path.exists(best_model_path):
        best_model_path = osp.join(experiment_dir, 'best_model_fold_5.ptm')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Model not found in {experiment_dir}")
    
    print(f"Loading model weights from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    # Pre-compute embeddings for efficiency
    print("Pre-computing node embeddings...")
    with torch.no_grad():
        gene_emb_cache, dis_emb_cache = model.encode_nodes(gene_net_data, disease_net_data)
        gene_emb_cache = gene_emb_cache.cpu().numpy()
        dis_emb_cache = dis_emb_cache.cpu().numpy()
        mlp = model.final_fc.cpu()
        mlp.eval()
    
    def fast_predict(pair_idx_np, batch=2_000_000):
        """Vectorized prediction using cached embeddings."""
        out = np.empty((pair_idx_np.shape[0],), dtype=np.float32)
        
        for i in range(0, pair_idx_np.shape[0], batch):
            blk = pair_idx_np[i:i+batch]
            g = torch.from_numpy(gene_emb_cache[blk[:, 0]])
            d = torch.from_numpy(dis_emb_cache[blk[:, 1]])
            feats = torch.cat([g, d], dim=1)
            
            with torch.no_grad():
                logits = mlp(feats).float()
            out[i:i+batch] = logits[:, 1].numpy()
        
        return out
    
    # Load evaluation diseases
    eval_path = "data/eval_diseases.tsv"
    test_sets = (
        pd.read_csv(eval_path, sep="\t", header=None)
        .iloc[:, 0]
        .dropna()
        .astype(str)
        .tolist()
    )
    
    # Get mappings
    gene_id_to_idx, gene_idx_to_id, omim_id_to_idx, omim_idx_to_id = get_id_mappings(experiment_dir)
    all_genes = list(gene_id_to_idx.keys())
    
    # Load associations
    test_path = cfg.evaluation.test_path
    all_omim_associations = pd.read_csv(cfg.evaluation.all_omim_associations_path, sep='\t')
    umls_omim_mapping = pd.read_csv(cfg.evaluation.umls_omim_mapping_path, sep='\t', header=None)
    
    # Process each disease
    k_rows = []
    
    for test_name in test_sets:
        disgenet_disease_id = str(test_name).strip()
        omim_ids_list = umls_omim_mapping[
            umls_omim_mapping.iloc[:, 0] == disgenet_disease_id
        ].iloc[:, 1].tolist()
        
        if not omim_ids_list:
            print(f"No OMIM mapping for {disgenet_disease_id}, skipping...")
            continue
        
        omim_ids_list = ast.literal_eval(omim_ids_list[0])
        model_disease_pred_rows = []
        
        # Get positive genes
        disease_omim_positive_list = get_disease_omim_positive_list(
            all_omim_associations, omim_ids_list, disgenet_disease_id, test_name, test_path
        )
        
        if not predict_only:
            positive_genes_list_path = osp.join(
                test_path, 'omim_positive_lists', f'{disgenet_disease_id}_omim_positives.tsv'
            )
            positive_genes = pd.read_csv(positive_genes_list_path, sep='\t')
            positive_genes = positive_genes.iloc[:, 0].astype(int).tolist()
        
        # Predict for each OMIM ID
        for omim_id in tqdm(omim_ids_list, desc=test_name):
            test_set_pairs = get_omim_test_set_pairs(all_genes, omim_id, disease_omim_positive_list)
            test_set = format_test_set(test_set_pairs, gene_id_to_idx, omim_id_to_idx)
            
            if test_set.shape[1] == 1:
                X = test_set.at["X", test_set.columns[0]]
            else:
                X = ast.literal_eval(test_set.loc['X'])
            
            x_tensor = np.array(X, dtype=np.int64)
            prob_scores = fast_predict(x_tensor)
            
            for (gene_idx, omim_idx), p in zip(X, prob_scores):
                model_disease_pred_rows.append({
                    'gene_idx': int(gene_idx),
                    'omim_idx': int(omim_idx),
                    'model_prob': float(p),
                })
        
        # Rank genes
        model_disease_preds = pd.DataFrame(model_disease_pred_rows)
        model_disease_preds['gene_id'] = model_disease_preds['gene_idx'].map(gene_idx_to_id)
        model_disease_preds['omim_id'] = model_disease_preds['omim_idx'].map(omim_idx_to_id)
        
        ranked_genes = model_disease_preds.sort_values(by='model_prob', ascending=False)
        ranked_genes_unique = ranked_genes.groupby('gene_id').agg({'model_prob': 'mean'}).reset_index()
        ranked_genes_unique = ranked_genes_unique.sort_values(by='model_prob', ascending=False)
        
        # Save ranked genes
        ranked_genes_output_path = osp.join(
            experiment_dir, 'ranked_genes', f'{disgenet_disease_id}_ranked_genes.tsv'
        )
        os.makedirs(osp.join(experiment_dir, 'ranked_genes'), exist_ok=True)
        ranked_genes_unique.to_csv(ranked_genes_output_path, sep='\t', index=False)
        print(f"Saved ranked genes: {ranked_genes_output_path}")
        
        # Evaluate if not predict-only
        if not predict_only:
            ranked_genes_list = ranked_genes_unique['gene_id'].astype(int).tolist()
            disease_name = get_disease_name(disgenet_disease_id)
            
            print(f"\n{'='*60}")
            print(f"{disease_name} ({disgenet_disease_id})")
            print('='*60)
            
            top_k_list = ast.literal_eval(cfg.evaluation.top_k)
            
            for top_k in top_k_list:
                top_k_genes = ranked_genes_list[:top_k]
                intersection = set(top_k_genes).intersection(set(positive_genes))
                
                tp = len(intersection)
                precision = tp / top_k if top_k > 0 else 0
                recall = tp / len(positive_genes) if positive_genes else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"K={top_k}: TP={tp}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                
                k_rows.append({
                    "Disease": test_name,
                    "K": top_k,
                    "Precision": round(precision, 4)
                })
    
    # Save results
    if not predict_only and k_rows:
        table = pd.DataFrame(k_rows)
        out_csv = osp.join(experiment_dir, "precision_by_k.csv")
        table.to_csv(out_csv, index=False)
        print(f"\nSaved results to {out_csv}")
    
    print("\nDone!")


def main():
    args = parse_args()
    
    # Load config
    cfg_path = args.experiment_dir / "config.yml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{cfg_path} not found")
    
    with cfg_path.open() as f:
        cfg = Box(yaml.safe_load(f))
    
    # Set seed
    seed = cfg.get('train', {}).get('seed', 42)
    set_seed(seed)
    
    # Run evaluation
    evaluate_gnet(cfg, args.experiment_dir, args.predict_only)


if __name__ == '__main__':
    main()