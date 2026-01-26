import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Disease ID to name mapping
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ranked gene lists")
    parser.add_argument("experiment_dir", type=Path,
                       help="Directory containing 'ranked_genes/' subfolder")
    parser.add_argument("--max-k", type=int, default=250,
                       help="Maximum K value for evaluation (default: 250)")
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                       help="Root directory for data files (default: data)")
    parser.add_argument("--include-disgenet", action="store_true",
                       help="Include DisGeNET metrics in evaluation")
    parser.add_argument("--include-tp-ranks", action="store_true", default=True,
                       help="Generate tp_ranks.csv output file")
    return parser.parse_args()


def clean_id(x) -> str:
    """Make every gene identifier comparable (strip whitespace, cut version)."""
    s = str(x).strip()
    return s.split(".", 1)[0] if "." in s else s


def _split_omim(col: pd.Series) -> list:
    """Return list of 'OMIM:123456' strings from a Series of messy cells."""
    ids = (col.dropna()
             .astype(str)
             .apply(lambda x: re.findall(r"OMIM:?\d+|\d+", x))
             .explode()
             .dropna()
             .unique())
    return ["OMIM:" + i.lstrip("OMIM:") for i in ids]


def create_omim_positive_list(disease_id: str, umls_omim_mapping_path: Path,
                               all_omim_associations_path: Path, omim_pos_dir: Path,
                               overwrite=False) -> Path:
    """Return path to <disease>_omim_positives.tsv (create if missing)."""
    omim_pos_dir.mkdir(parents=True, exist_ok=True)

    if disease_id.startswith("OMIM:"):
        omim_ids = [disease_id]
        stem = disease_id.replace(":", "_")
    else:
        mapping = pd.read_csv(umls_omim_mapping_path, sep="\t", header=None,
                             names=["UMLS", "OMIM"])
        if disease_id not in mapping["UMLS"].values:
            raise ValueError(f"{disease_id} not in UMLS–OMIM map")
        omim_ids = _split_omim(mapping.loc[mapping["UMLS"] == disease_id, "OMIM"])
        stem = disease_id

    out_path = omim_pos_dir / f"{stem}_omim_positives.tsv"
    if out_path.exists() and not overwrite:
        return out_path

    assoc = pd.read_csv(all_omim_associations_path, sep="\t", header=None,
                       names=["gene_id", "omim_id"], usecols=[0, 1])
    genes = (assoc.loc[assoc["omim_id"].isin(omim_ids), "gene_id"]
                  .map(clean_id)
                  .unique())
    if genes.size == 0:
        raise RuntimeError(f"No OMIM associations for {disease_id}")

    pd.DataFrame(genes).to_csv(out_path, sep="\t", index=False, header=False)
    print(f"  • wrote OMIM positives → {out_path.name}  ({genes.size} genes)")
    return out_path


def evaluate_ranked_dir(args):
    ranked_dir = args.experiment_dir / "ranked_genes"
    if not ranked_dir.is_dir():
        raise SystemExit(f"Expected 'ranked_genes/' subfolder in {args.experiment_dir}")

    # Setup paths
    data_root = args.data_root
    umls_omim_mapping_path = data_root / "test/UMLS_OMIM_map.tsv"
    all_omim_associations_path = data_root / "test/raw/all_omim_associations.tsv"
    omim_pos_dir = data_root / "test/disease_specific/omim_positive_lists"
    eval_diseases_path = data_root / "eval_diseases.tsv"
    
    # Load evaluation diseases
    eval_diseases = pd.read_csv(eval_diseases_path, sep="\t", header=None)[0].astype(str).tolist()
    
    # Load DisGeNET if requested
    disgenet_all = None
    if args.include_disgenet:
        all_disgenet_path = data_root / "test/raw/all_disgenet_associations.tsv"
        disgenet_all = pd.read_csv(all_disgenet_path, sep="\t")

    # Output containers
    metrics_rows = []
    tp_rank_rows = []

    k_start, k_stop, k_step = 5, args.max_k, 1

    for tsv in sorted(ranked_dir.glob("*.tsv")):
        m = re.match(r"([^_]+).*ranked_genes\.tsv$", tsv.name)
        if not m:
            print(f"  ✘︎ skip weird file name {tsv.name}")
            continue
        dis_id = m.group(1)
        if dis_id not in eval_diseases:
            print(f"  ✘︎ {dis_id} not in eval set; skipping")
            continue
        dis_name = disease_id_name_mapping.get(dis_id, "Unknown_Disease")

        ranked_genes = (pd.read_csv(tsv, sep="\t", usecols=[0], dtype=str)
                         .iloc[:, 0].map(clean_id).tolist())

        omim_pos_path = create_omim_positive_list(
            dis_id, umls_omim_mapping_path, all_omim_associations_path, omim_pos_dir
        )
        omim_pos = pd.read_csv(omim_pos_path, sep="\t", header=None)[0].map(clean_id).tolist()

        disgenet_pos = None
        if args.include_disgenet:
            disgenet_pos = (disgenet_all[disgenet_all.iloc[:, 4] == dis_id]
                           .iloc[:, 0].map(clean_id).tolist())

        max_k = min(len(ranked_genes), args.max_k)
        for k in range(k_start, max_k + 1, k_step):
            top_k = ranked_genes[:k]

            # OMIM metrics
            omim_tp = sum(g in omim_pos for g in top_k)
            om_prec = omim_tp / k
            om_rec = omim_tp / len(omim_pos) if omim_pos else 0
            om_f1 = (2 * om_prec * om_rec) / (om_prec + om_rec) if (om_prec + om_rec) else 0

            row = {
                "experiment_path": str(args.experiment_dir),
                "disease_id": dis_id,
                "disease_name": dis_name,
                "K": k,
                "omim_tp": omim_tp,
                "omim_prec": om_prec,
                "omim_recall": om_rec,
                "omim_f1": om_f1,
            }

            # DisGeNET metrics if requested
            if args.include_disgenet and disgenet_pos is not None:
                dis_tp = sum(g in disgenet_pos for g in top_k)
                dis_prec = dis_tp / k
                dis_rec = dis_tp / len(disgenet_pos) if disgenet_pos else 0
                dis_f1 = (2 * dis_prec * dis_rec) / (dis_prec + dis_rec) if (dis_prec + dis_rec) else 0
                
                row.update({
                    "disgenet_tp": dis_tp,
                    "disgenet_prec": dis_prec,
                    "disgenet_recall": dis_rec,
                    "disgenet_f1": dis_f1,
                })

            metrics_rows.append(row)

        # TP ranks (ALL ranks, not limited to MAX_K)
        if args.include_tp_ranks:
            tp_ranks = [i + 1 for i, g in enumerate(ranked_genes) if g in omim_pos]
            tp_rank_rows.append({
                "disease_id": dis_id,
                "disease_name": dis_name,
                "tp_ranks": str(tp_ranks)
            })

    # Save results
    pd.DataFrame(metrics_rows).to_csv(args.experiment_dir / "top_k_eval_metrics.csv", index=False)
    print(f"✔︎  top_k_eval_metrics.csv written to {args.experiment_dir}")
    
    if args.include_tp_ranks:
        pd.DataFrame(tp_rank_rows).to_csv(args.experiment_dir / "true_gene_ranks.csv", index=False)
        print(f"✔︎  true_gene_ranks.csv written to {args.experiment_dir}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_ranked_dir(args)