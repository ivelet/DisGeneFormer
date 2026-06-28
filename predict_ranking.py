#!/usr/bin/env python3
"""
predict_ranking_publication_ready.py

Per-OMIM DGF ranking evaluation against the full mapped gene pool.

This version is designed for a paired DGF-vs-Phen2Gene complementarity analysis.
It makes three evaluation guarantees:

1. Every disease that DGF can score and that has frozen ground truth remains in
   per_omim_results.tsv, even when none of its causal genes is in DGF's candidate
   pool. Such diseases are valid DGF misses, not skipped cases.

2. Invalid/unmapped network indices are removed before ranks are assigned.
   Rank 1 therefore means the highest-scoring valid Entrez gene ID, and ranks are
   contiguous over the actual candidate pool.

3. Every run writes a self-contained, versioned output contract:
      manifest.json
      per_omim_results.tsv
      excluded.tsv
      summary.json
      ranked_genes/<bare_omim_id>.tsv  (with --save-ranked-genes)

Use a fresh --output-dir for each experiment. Existing non-empty output
directories are rejected unless --overwrite is supplied explicitly.

Example:
    python predict_ranking_publication_ready.py /path/to/experiment \
      --output-dir /path/to/experiment/complementarity_runs/dgf_run_001 \
      --checkpoint best_model.ptm \
      --ks 1,5,10,20,50,100 \
      --save-ranked-genes
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from box import Box
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch
from torch.cuda.amp import autocast

from source.DiseaseNet import DiseaseNet
from source.GeneNet import GeneNet
from source.DisGeneFormer import DisGeneFormer


# --------------------------------------------------------------------------- #
# Argument parsing and reproducibility helpers
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("experiment_dir", type=Path)
    p.add_argument(
        "--checkpoint",
        default="best_model.ptm",
        help="Checkpoint filename under experiment_dir/classification_eval/",
    )
    p.add_argument(
        "--holdout",
        default="test_omims.tsv",
        help="Held-out OMIM list filename under experiment_dir/classification_eval/",
    )
    p.add_argument(
        "--ks",
        default="1,5,10,20,50,100",
        help="Comma-separated K values for hit-rate@K",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=2_000_000,
        help="MLP scoring chunk size",
    )
    p.add_argument(
        "--save-ranked-genes",
        action="store_true",
        help="Save the complete normalized ranked gene list for every scoreable OMIM.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Run-specific output directory. Defaults to experiment_dir/ranking_eval, "
            "but a non-empty directory is rejected unless --overwrite is supplied."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before writing this run. Use deliberately.",
    )
    return p.parse_args()


def cget(box, dotted_key, default):
    cur = box
    try:
        for part in dotted_key.split("."):
            cur = cur[part]
        return cur if cur is not None else default
    except Exception:
        return default


def sha256_file(path, block_size=1024 * 1024):
    """Return the SHA-256 digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_mapping(mapping):
    """Hash an ID->index mapping deterministically without depending on dict order."""
    h = hashlib.sha256()
    rows = sorted((str(key), str(value)) for key, value in mapping.items())
    for key, value in rows:
        h.update(key.encode("utf-8"))
        h.update(b"\t")
        h.update(value.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)


def prepare_output_dir(out_dir, experiment_dir, overwrite):
    """Create an empty output directory, preventing accidental stale-file reuse."""
    out_dir = Path(out_dir).expanduser().resolve()
    experiment_dir = Path(experiment_dir).expanduser().resolve()

    if out_dir == experiment_dir:
        raise ValueError("--output-dir must not be the experiment directory itself.")

    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {out_dir}\n"
                "Use a fresh --output-dir for a new evaluation run, or pass --overwrite "
                "only when you intentionally want to delete the existing directory."
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# --------------------------------------------------------------------------- #
# Identifier normalization and input loading
# --------------------------------------------------------------------------- #
def canonical_omim_id(value):
    """
    Convert OMIM:614021 and 614021 to the shared bare-ID form 614021.

    The method deliberately does not coerce arbitrary strings to integers; this
    avoids silently transforming malformed identifiers.
    """
    if pd.isna(value):
        raise ValueError("Encountered a missing OMIM identifier.")

    omim_id = str(value).strip()
    if omim_id.upper().startswith("OMIM:"):
        omim_id = omim_id.split(":", 1)[1].strip()

    if not omim_id:
        raise ValueError("Encountered an empty OMIM identifier.")

    return omim_id


def normalize_disease_index_map(raw_map):
    """Normalize DiseaseNet map keys to bare OMIM IDs and reject ambiguous maps."""
    normalized = {}
    for raw_id, index in raw_map.items():
        omim_id = canonical_omim_id(raw_id)
        index = int(index)
        if omim_id in normalized and normalized[omim_id] != index:
            raise ValueError(
                f"Ambiguous DiseaseNet index mapping for OMIM {omim_id}: "
                f"{normalized[omim_id]} versus {index}."
            )
        normalized[omim_id] = index
    return normalized


def load_holdout_omims(path):
    """Load and canonicalize a unique ordered held-out OMIM list."""
    holdout_df = pd.read_csv(path, sep="\t")
    if "omim_id" not in holdout_df.columns:
        holdout_df = pd.read_csv(path, sep="\t", header=None, names=["omim_id"])

    holdout_omims = [canonical_omim_id(value) for value in holdout_df["omim_id"]]
    if len(holdout_omims) != len(set(holdout_omims)):
        duplicates = pd.Series(holdout_omims).value_counts()
        duplicates = duplicates[duplicates > 1].index.tolist()[:10]
        raise ValueError(
            "The held-out OMIM file contains duplicate disease IDs after "
            f"normalization; examples: {duplicates}"
        )
    return holdout_omims


def load_truth_by_omim(path):
    """
    Load causal OMIM-gene associations, supporting explicit gene_id/omim_id
    columns or the historical first-two-column layout used by this project.
    """
    raw = pd.read_csv(path, sep="\t")
    normalized_cols = {str(column).strip().lower(): column for column in raw.columns}

    if {"gene_id", "omim_id"}.issubset(normalized_cols):
        truth = raw[
            [normalized_cols["gene_id"], normalized_cols["omim_id"]]
        ].copy()
    else:
        if raw.shape[1] < 2:
            raise ValueError(
                f"Ground-truth association file must have at least two columns: {path}"
            )
        truth = raw.iloc[:, :2].copy()

    truth.columns = ["gene_id", "omim_id"]
    truth = truth.dropna().drop_duplicates(["gene_id", "omim_id"])

    try:
        truth["gene_id"] = pd.to_numeric(truth["gene_id"], errors="raise").astype(np.int64)
    except Exception as exc:
        raise ValueError(
            "Ground-truth gene IDs must be numeric Entrez IDs for paired analysis."
        ) from exc

    truth["omim_id"] = truth["omim_id"].map(canonical_omim_id)
    truth = truth.drop_duplicates(["gene_id", "omim_id"])

    return (
        truth.groupby("omim_id")["gene_id"]
        .apply(lambda values: set(int(value) for value in values))
        .to_dict()
    )


# --------------------------------------------------------------------------- #
# Model evaluation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def precompute_embeddings(model, gene_data, disease_data):
    """One-shot graph encoding; embeddings are reused for every held-out OMIM."""
    model.eval()
    g_emb, d_emb = model.encode_nodes(gene_data, disease_data)
    return g_emb, d_emb


@torch.no_grad()
def score_all_genes_for_disease(mlp, g_emb, d_emb, dis_idx, n_genes, chunk, use_amp):
    """
    Score every (gene, disease) pair in chunks.

    Returns log-odds logits[:, 1] - logits[:, 0], which are monotonic with
    the corresponding positive-class softmax probability.
    """
    d_vec = d_emb[dis_idx].unsqueeze(0)
    scores = np.empty((n_genes,), dtype=np.float32)

    for start in range(0, n_genes, chunk):
        end = min(start + chunk, n_genes)
        g_block = g_emb[start:end]
        d_block = d_vec.expand(end - start, -1)

        with autocast(enabled=use_amp):
            features = torch.cat([g_block, d_block], dim=1)
            logits = mlp(features).float()
            chunk_scores = logits[:, 1] - logits[:, 0]

        scores[start:end] = chunk_scores.detach().cpu().numpy()

    return scores


def rank_valid_gene_ids(scores, gene_id_by_idx):
    """
    Return valid Entrez IDs and scores in normalized rank order.

    Invalid/unmapped entries are excluded before ranks are assigned. Ties are
    broken deterministically by ascending Entrez ID.
    """
    if not np.isfinite(scores).all():
        n_nonfinite = int((~np.isfinite(scores)).sum())
        raise RuntimeError(f"Encountered {n_nonfinite} non-finite DGF scores.")

    # lexsort uses the final key as the primary key: score descending, then gene ID.
    order = np.lexsort((gene_id_by_idx, -scores))
    ordered_gene_ids = gene_id_by_idx[order]
    ordered_scores = scores[order]

    valid_mask = ordered_gene_ids > 0
    valid_gene_ids = ordered_gene_ids[valid_mask].astype(np.int64, copy=False)
    valid_scores = ordered_scores[valid_mask].astype(np.float32, copy=False)

    if len(valid_gene_ids) == 0:
        raise RuntimeError("No valid Entrez gene IDs remained after filtering the gene map.")

    if len(np.unique(valid_gene_ids)) != len(valid_gene_ids):
        duplicate_ids = pd.Series(valid_gene_ids).value_counts()
        duplicate_ids = duplicate_ids[duplicate_ids > 1].index.tolist()[:10]
        raise RuntimeError(
            "The GeneNet index mapping contains duplicate Entrez gene IDs; "
            f"examples: {duplicate_ids}"
        )

    return valid_gene_ids, valid_scores


def make_metric_row(
    omim_id,
    true_genes,
    ranked_gene_ids,
    ks,
    rank_file,
):
    """Build one complete DGF evaluation row, retaining out-of-pool truth as a miss."""
    positions = {int(gene_id): rank for rank, gene_id in enumerate(ranked_gene_ids, start=1)}
    true_ranks = sorted(positions[gene_id] for gene_id in true_genes if gene_id in positions)

    n_truth_genes = len(true_genes)
    n_truth_in_pool = len(true_ranks)

    if true_ranks:
        best_rank = int(true_ranks[0])
        reciprocal_rank = float(1.0 / best_rank)
        evaluation_status = "scored_truth_ranked"
        status_reason = ""
        all_true_ranks = ",".join(str(rank) for rank in true_ranks)
    else:
        # A scoreable disease whose truth genes are all outside DGF's candidate
        # pool is an evaluated DGF miss, not a technical exclusion.
        best_rank = None
        reciprocal_rank = 0.0
        evaluation_status = "scored_truth_out_of_pool"
        status_reason = "all_truth_genes_outside_dgf_candidate_pool"
        all_true_ranks = ""

    row = {
        "omim_id": omim_id,
        "evaluation_status": evaluation_status,
        "status_reason": status_reason,
        "n_truth_genes": n_truth_genes,
        "n_truth_genes_in_candidate_pool": n_truth_in_pool,
        "truth_ranked": int(bool(true_ranks)),
        "best_rank": best_rank,
        "reciprocal_rank": reciprocal_rank,
        "all_true_ranks": all_true_ranks,
        "n_candidates": int(len(ranked_gene_ids)),
        "rank_file": rank_file,
    }
    for k in ks:
        row[f"hit@{k}"] = int(best_rank is not None and best_rank <= k)

    return row, true_ranks


def safe_float_or_none(value):
    """Convert a finite numeric value to float; otherwise emit JSON null."""
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc)

    if args.chunk <= 0:
        raise ValueError("--chunk must be a positive integer.")

    cfg_path = args.experiment_dir / "config.yml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{cfg_path} not found")
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = Box(yaml.safe_load(handle))

    cls_dir = args.experiment_dir / "classification_eval"
    out_dir = prepare_output_dir(
        args.output_dir or (args.experiment_dir / "ranking_eval"),
        args.experiment_dir,
        args.overwrite,
    )
    ranked_dir = out_dir / "ranked_genes"
    if args.save_ranked_genes:
        ranked_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_path = cls_dir / args.checkpoint
    holdout_path = cls_dir / args.holdout
    truth_path = Path(cfg.evaluation.all_omim_associations_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{checkpoint_path} not found")
    if not holdout_path.is_file():
        raise FileNotFoundError(f"{holdout_path} not found")
    if not truth_path.is_file():
        raise FileNotFoundError(f"{truth_path} not found")

    ks = sorted(set(int(k.strip()) for k in args.ks.split(",") if k.strip()))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("--ks must contain one or more positive integers.")
    print(f"Evaluating at K = {ks}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cget(cfg, "train.use_autocast", True)) and device.type == "cuda"
    print(f"Device: {device} | autocast: {use_amp}")

    # ----- Load inputs and networks ----- #
    print("Loading GeneNet / DiseaseNet ...")
    gene_dataset = GeneNet(cfg=cfg, experiment_dir=args.experiment_dir)
    disease_dataset = DiseaseNet(cfg=cfg, experiment_dir=args.experiment_dir)
    gene_net_data = gene_dataset[0].to(device)
    disease_net_data = disease_dataset[0].to(device)

    gene_map = gene_dataset.load_node_index_mapping()
    raw_disease_map = disease_dataset.load_disease_index_feature_mapping()
    disease_map = normalize_disease_index_map(raw_disease_map)

    n_network_gene_indices = int(gene_net_data.x.shape[0])
    print(
        f"Gene network indices: {n_network_gene_indices} | "
        f"Disease nodes: {disease_net_data['disease'].x.shape[0]} | "
        f"HPO nodes: {disease_net_data['hpo'].x.shape[0]}"
    )

    # Build gene-index -> Entrez ID lookup. Invalid entries are explicitly filtered
    # before rank assignment inside rank_valid_gene_ids().
    gene_id_by_idx = np.full(n_network_gene_indices, -1, dtype=np.int64)
    for gene_id, index in gene_map.items():
        try:
            index = int(index)
            gene_id = int(gene_id)
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= n_network_gene_indices:
            raise ValueError(
                f"Gene-map index {index} for Entrez ID {gene_id} is outside "
                f"[0, {n_network_gene_indices})."
            )
        if gene_id <= 0:
            raise ValueError(f"Invalid non-positive Entrez gene ID in gene map: {gene_id}")
        gene_id_by_idx[index] = gene_id

    n_unmapped_indices = int((gene_id_by_idx <= 0).sum())
    n_mapped_gene_ids = int((gene_id_by_idx > 0).sum())
    if n_unmapped_indices:
        print(
            f"WARNING: {n_unmapped_indices} network indices have no valid Entrez ID. "
            "They will be removed before ranks are assigned."
        )

    holdout_omims = load_holdout_omims(holdout_path)
    truth_by_omim = load_truth_by_omim(truth_path)
    print(f"Held-out OMIMs: {len(holdout_omims)}")
    print(f"OMIMs with truth labels: {len(truth_by_omim)}")

    # Write the immutable run manifest before scoring. It records the inputs that
    # define the evaluation protocol and lets the fusion analyzer reject mismatches.
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "dgf_ranking_eval_v2",
        "method": "DGF",
        "created_utc": started_at.isoformat(),
        "command": [sys.executable, *sys.argv],
        "arguments": {
            "checkpoint": str(args.checkpoint),
            "holdout": str(args.holdout),
            "ks": ks,
            "chunk": args.chunk,
            "save_ranked_genes": bool(args.save_ranked_genes),
        },
        "paths": {
            "experiment_dir": str(args.experiment_dir.resolve()),
            "output_dir": str(out_dir),
            "checkpoint": str(checkpoint_path.resolve()),
            "config": str(cfg_path.resolve()),
            "holdout": str(holdout_path.resolve()),
            "truth": str(truth_path.resolve()),
            "source_script": str(script_path),
        },
        "sha256": {
            "checkpoint": sha256_file(checkpoint_path),
            "config": sha256_file(cfg_path),
            "holdout": sha256_file(holdout_path),
            "truth": sha256_file(truth_path),
            "source_script": sha256_file(script_path),
            "gene_index_mapping": sha256_mapping(gene_map),
            "disease_index_mapping": sha256_mapping(raw_disease_map),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "autocast": bool(use_amp),
        },
        "identifier_contract": {
            "omim_id": "bare OMIM identifier, e.g. 614021",
            "gene_id": "NCBI Entrez Gene ID (positive integer)",
            "candidate_pool": "valid Entrez IDs mapped from the DGF GeneNet",
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    # ----- Load model ----- #
    print(f"Loading checkpoint: {checkpoint_path}")
    ablate_cfg = cget(cfg, "model.ablate", {}) or {}
    ablate_dict = {key: bool(value) for key, value in dict(ablate_cfg).items() if value}

    model = DisGeneFormer(
        gene_feature_dim=gene_net_data.x.shape[1],
        disease_feature_dim=disease_net_data["disease"].x.shape[1],
        fc_hidden_dim=cfg.model.fc_hidden_dim,
        gene_net_hidden_dim=cfg.model.gene_net_hidden_dim,
        disease_net_hidden_dim=cfg.model.disease_net_hidden_dim,
        mode="DGP",
        ablate=ablate_dict,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("Computing graph embeddings ...")
    gene_embeddings, disease_embeddings = precompute_embeddings(
        model, gene_net_data, disease_net_data
    )
    mlp = model.final_fc
    print(
        f"  gene_emb {tuple(gene_embeddings.shape)} | "
        f"disease_emb {tuple(disease_embeddings.shape)}"
    )

    # ----- Per-OMIM scoring ----- #
    per_omim_rows = []
    excluded_rows = []

    for omim_id in tqdm(holdout_omims, desc="Scoring held-out OMIMs"):
        if omim_id not in disease_map:
            excluded_rows.append(
                {
                    "omim_id": omim_id,
                    "exclusion_reason": "not_in_disease_index_map",
                }
            )
            continue

        true_genes = truth_by_omim.get(omim_id, set())
        if not true_genes:
            excluded_rows.append(
                {
                    "omim_id": omim_id,
                    "exclusion_reason": "no_truth_genes_in_association_file",
                }
            )
            continue

        disease_index = disease_map[omim_id]
        scores = score_all_genes_for_disease(
            mlp=mlp,
            g_emb=gene_embeddings,
            d_emb=disease_embeddings,
            dis_idx=disease_index,
            n_genes=n_network_gene_indices,
            chunk=args.chunk,
            use_amp=use_amp,
        )
        ranked_gene_ids, ranked_scores = rank_valid_gene_ids(scores, gene_id_by_idx)

        relative_rank_file = (
            str(Path("ranked_genes") / f"{omim_id}.tsv") if args.save_ranked_genes else ""
        )
        row, _ = make_metric_row(
            omim_id=omim_id,
            true_genes=true_genes,
            ranked_gene_ids=ranked_gene_ids,
            ks=ks,
            rank_file=relative_rank_file,
        )
        per_omim_rows.append(row)

        if args.save_ranked_genes:
            ranked_df = pd.DataFrame(
                {
                    "omim_id": omim_id,
                    "rank": np.arange(1, len(ranked_gene_ids) + 1, dtype=np.int64),
                    "gene_id": ranked_gene_ids,
                    "score": ranked_scores,
                    "is_truth_gene": np.isin(
                        ranked_gene_ids, np.fromiter(true_genes, dtype=np.int64)
                    ).astype(np.int8),
                }
            )
            ranked_df.to_csv(ranked_dir / f"{omim_id}.tsv", sep="\t", index=False)

    # ----- Aggregate and write outputs ----- #
    base_columns = [
        "omim_id",
        "evaluation_status",
        "status_reason",
        "n_truth_genes",
        "n_truth_genes_in_candidate_pool",
        "truth_ranked",
        "best_rank",
        "reciprocal_rank",
        "all_true_ranks",
        "n_candidates",
        "rank_file",
        *[f"hit@{k}" for k in ks],
    ]
    per_omim_df = pd.DataFrame(per_omim_rows, columns=base_columns)
    excluded_df = pd.DataFrame(
        excluded_rows, columns=["omim_id", "exclusion_reason"]
    )

    per_omim_path = out_dir / "per_omim_results.tsv"
    excluded_path = out_dir / "excluded.tsv"
    per_omim_df.to_csv(per_omim_path, sep="\t", index=False)
    excluded_df.to_csv(excluded_path, sep="\t", index=False)

    ranked_best_ranks = (
        per_omim_df.loc[per_omim_df["best_rank"].notna(), "best_rank"]
        if not per_omim_df.empty
        else pd.Series(dtype=float)
    )
    status_counts = (
        per_omim_df["evaluation_status"].value_counts().sort_index().to_dict()
        if not per_omim_df.empty
        else {}
    )
    exclusion_counts = (
        excluded_df["exclusion_reason"].value_counts().sort_index().to_dict()
        if not excluded_df.empty
        else {}
    )

    summary = {
        "schema_version": "dgf_ranking_eval_v2",
        "method": "DGF",
        "output_dir": str(out_dir),
        "checkpoint": str(checkpoint_path),
        "holdout_file": str(holdout_path),
        "truth_file": str(truth_path),
        "n_held_out_omims": int(len(holdout_omims)),
        "n_evaluated": int(len(per_omim_df)),
        "n_excluded": int(len(excluded_df)),
        "n_truth_ranked": int(per_omim_df["truth_ranked"].sum()) if not per_omim_df.empty else 0,
        "n_truth_out_of_candidate_pool": int(
            (per_omim_df["evaluation_status"] == "scored_truth_out_of_pool").sum()
        ) if not per_omim_df.empty else 0,
        "n_network_gene_indices": int(n_network_gene_indices),
        "n_valid_entrez_candidates": int(n_mapped_gene_ids),
        "n_unmapped_network_indices": int(n_unmapped_indices),
        "MRR": safe_float_or_none(
            per_omim_df["reciprocal_rank"].mean() if not per_omim_df.empty else None
        ),
        "median_best_rank_among_ranked_truth": safe_float_or_none(
            ranked_best_ranks.median() if not ranked_best_ranks.empty else None
        ),
        "mean_best_rank_among_ranked_truth": safe_float_or_none(
            ranked_best_ranks.mean() if not ranked_best_ranks.empty else None
        ),
        "evaluation_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "exclusion_reason_counts": {str(k): int(v) for k, v in exclusion_counts.items()},
    }
    for k in ks:
        summary[f"hit_rate@{k}"] = safe_float_or_none(
            per_omim_df[f"hit@{k}"].mean() if not per_omim_df.empty else None
        )

    summary_path = out_dir / "summary.json"
    write_json(summary_path, summary)

    # ----- Print summary ----- #
    print("\n" + "=" * 72)
    print("DGF RANKING EVALUATION  (paired-analysis output contract)")
    print("=" * 72)
    print(f"  Checkpoint:                       {checkpoint_path.name}")
    print(f"  Held-out OMIMs:                   {summary['n_held_out_omims']}")
    print(f"  Evaluated (scoreable + truth):    {summary['n_evaluated']}")
    print(f"  Excluded (technical/no truth):    {summary['n_excluded']}")
    print(f"  Truth ranked in DGF pool:         {summary['n_truth_ranked']}")
    print(f"  Truth outside DGF candidate pool: {summary['n_truth_out_of_candidate_pool']}")
    print(f"  Valid Entrez candidate genes:     {summary['n_valid_entrez_candidates']}")
    print(f"  MRR (all evaluated OMIMs):        {summary['MRR']}")
    print(
        "  Median best rank (ranked truth only): "
        f"{summary['median_best_rank_among_ranked_truth']}"
    )
    for k in ks:
        print(f"  hit-rate@{k:<3}                     {summary[f'hit_rate@{k}']}")
    print("=" * 72)
    print(f"Manifest:         {out_dir / 'manifest.json'}")
    print(f"Per-OMIM results: {per_omim_path}")
    print(f"Excluded:         {excluded_path}")
    print(f"Summary:          {summary_path}")
    if args.save_ranked_genes:
        print(f"Ranked genes:     {ranked_dir}/")
    print(f"Total time:       {datetime.now(timezone.utc) - started_at}")


if __name__ == "__main__":
    main()