#!/usr/bin/env python3
"""
filter_humannet.py - Filter HumanNet edges by gene set

Filters HumanNet file to keep only edges where at least one gene is in a 
specified gene set derived from OMIM associations.

Usage:
    # Filter for all genes in OMIM associations
    python filter_humannet.py humannet.tsv all_omim_associations.tsv output.tsv
    
    # Filter for genes associated with specific diseases
    python filter_humannet.py humannet.tsv all_omim_associations.tsv output.tsv \
        --disease-map disease_omim_map.tsv \
        --diseases C0006142 C0036341
"""

import argparse
import ast
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter HumanNet edges by gene set from OMIM associations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "humannet_file",
        type=Path,
        help="Input HumanNet file (gene1, gene2, LLS)"
    )
    parser.add_argument(
        "omim_associations_file",
        type=Path,
        help="OMIM associations file (gene_id, omim_id)"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Output filtered HumanNet file"
    )
    parser.add_argument(
        "--disease-map",
        type=Path,
        help="Disease-OMIM mapping file (disease_id, [omim_ids])"
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        help="Disease IDs to filter for (requires --disease-map)"
    )
    parser.add_argument(
        "--both-genes",
        action="store_true",
        help="Require BOTH genes in edge to be in gene set (default: at least one)"
    )
    return parser.parse_args()


def load_gene_set_all(omim_associations_file: Path) -> set:
    """Load all genes from OMIM associations."""
    print(f"Loading all genes from {omim_associations_file}...")
    
    df = pd.read_csv(omim_associations_file, sep="\t", header=None,
                    names=["gene_id", "omim_id"])
    
    genes = set(df["gene_id"].astype(str).str.strip())
    print(f"  ✓ Loaded {len(genes)} unique genes")
    
    return genes


def load_gene_set_diseases(omim_associations_file: Path, 
                          disease_map_file: Path,
                          disease_ids: list) -> set:
    """Load genes associated with specific diseases."""
    
    print(f"Loading disease-OMIM mapping from {disease_map_file}...")
    
    # Read disease-OMIM mapping
    disease_map = {}
    with open(disease_map_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            disease_id = parts[0]
            # Parse the list of OMIM IDs (format: ['OMIM:123', 'OMIM:456'])
            try:
                omim_list = ast.literal_eval(parts[1])
            except:
                # Try splitting by comma if not a proper list
                omim_list = [x.strip() for x in parts[1].split(',')]
            disease_map[disease_id] = omim_list
    
    print(f"  ✓ Loaded mappings for {len(disease_map)} diseases")
    
    # Get OMIM IDs for requested diseases
    target_omim_ids = set()
    for disease_id in disease_ids:
        if disease_id not in disease_map:
            print(f"  ⚠ Warning: Disease {disease_id} not in mapping file")
            continue
        omim_ids = disease_map[disease_id]
        target_omim_ids.update(omim_ids)
        print(f"  • {disease_id}: {len(omim_ids)} OMIM IDs")
    
    print(f"  ✓ Total {len(target_omim_ids)} unique OMIM IDs")
    
    # Load OMIM associations
    print(f"\nLoading OMIM associations from {omim_associations_file}...")
    df = pd.read_csv(omim_associations_file, sep="\t", header=None,
                    names=["gene_id", "omim_id"])
    
    # Filter for target OMIM IDs
    df["omim_id"] = df["omim_id"].astype(str).str.strip()
    filtered = df[df["omim_id"].isin(target_omim_ids)]
    
    genes = set(filtered["gene_id"].astype(str).str.strip())
    print(f"  ✓ Found {len(genes)} genes associated with selected diseases")
    
    return genes


def filter_humannet(humannet_file: Path, gene_set: set, 
                   output_file: Path, both_genes: bool = False):
    """Filter HumanNet file to keep edges with genes in gene_set."""
    
    print(f"\nFiltering HumanNet file: {humannet_file}")
    print(f"  Gene set size: {len(gene_set)}")
    print(f"  Filter mode: {'BOTH genes' if both_genes else 'AT LEAST ONE gene'}")
    
    # Count total lines for progress bar
    with open(humannet_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"  Total edges in input: {total_lines:,}")
    
    kept = 0
    with open(humannet_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in tqdm(fin, total=total_lines, desc="  Filtering"):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            gene1 = parts[0].strip()
            gene2 = parts[1].strip()
            
            # Check filter condition
            if both_genes:
                # Both genes must be in set
                if gene1 in gene_set and gene2 in gene_set:
                    fout.write(line)
                    kept += 1
            else:
                # At least one gene must be in set
                if gene1 in gene_set or gene2 in gene_set:
                    fout.write(line)
                    kept += 1
    
    print(f"  ✓ Kept {kept:,} edges ({100*kept/total_lines:.2f}%)")
    print(f"  ✓ Output written to: {output_file}")


def main():
    args = parse_args()
    
    # Validate arguments
    if args.diseases and not args.disease_map:
        raise ValueError("--diseases requires --disease-map")
    
    # Load gene set
    if args.diseases:
        gene_set = load_gene_set_diseases(
            args.omim_associations_file,
            args.disease_map,
            args.diseases
        )
    else:
        gene_set = load_gene_set_all(args.omim_associations_file)
    
    # Filter HumanNet
    filter_humannet(
        args.humannet_file,
        gene_set,
        args.output_file,
        both_genes=args.both_genes
    )
    
    print("\n" + "="*70)
    print("✓ Filtering complete!")
    print("="*70)


if __name__ == "__main__":
    main()