#!/usr/bin/env python
"""
Create synthetic test data to validate pipeline logic.
Replace with real data downloads for actual analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sclc.config import get_project_root

# SCLC subtype marker genes
SCLC_MARKERS = {
    'SCLC-A': ['ASCL1', 'DLL3', 'INSM1', 'CHGA', 'SYP', 'NCAM1'],
    'SCLC-N': ['NEUROD1', 'HES6', 'ASCL2', 'BEX1', 'NKX2-1'],
    'SCLC-P': ['POU2F3', 'SOX9', 'TRPM5', 'AVIL', 'GFI1B'],
    'SCLC-I': ['HLA-DRA', 'CD74', 'CIITA', 'IRF1', 'STAT1']
}

# Immune marker genes
IMMUNE_MARKERS = {
    'T_cell': ['CD3D', 'CD3E', 'CD4', 'CD8A', 'CD8B'],
    'B_cell': ['CD19', 'CD79A', 'MS4A1', 'CD22'],
    'Myeloid': ['CD14', 'CD68', 'ITGAM', 'CSF1R'],
    'NK': ['NCAM1', 'NKG7', 'GNLY', 'KLRD1'],
    'Exhaustion': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT'],
    'Cytotoxic': ['GZMA', 'GZMB', 'PRF1', 'IFNG']
}

def create_bulk_expression(output_dir: Path, n_samples: int = 100, n_genes: int = 15000):
    """Create synthetic bulk RNA-seq expression matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all marker genes
    all_markers = []
    for markers in SCLC_MARKERS.values():
        all_markers.extend(markers)
    for markers in IMMUNE_MARKERS.values():
        all_markers.extend(markers)
    all_markers = list(set(all_markers))

    # Generate random gene names for non-markers
    other_genes = [f"GENE_{i}" for i in range(n_genes - len(all_markers))]
    all_genes = all_markers + other_genes

    # Create sample IDs with subtype assignments
    subtypes = ['SCLC-A'] * 30 + ['SCLC-N'] * 25 + ['SCLC-P'] * 20 + ['SCLC-I'] * 25
    np.random.shuffle(subtypes)
    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]

    # Generate expression matrix
    np.random.seed(42)

    # Base expression (log2 scale, 0-15)
    expression = np.random.lognormal(mean=2, sigma=1.5, size=(n_genes, n_samples))
    expression = np.log2(expression + 1)

    # Add subtype-specific patterns
    df = pd.DataFrame(expression, index=all_genes, columns=sample_ids)

    for i, (sample, subtype) in enumerate(zip(sample_ids, subtypes)):
        # Upregulate subtype-specific markers
        for marker in SCLC_MARKERS.get(subtype, []):
            if marker in df.index:
                df.loc[marker, sample] += np.random.uniform(3, 5)

        # Add immune variation
        if subtype == 'SCLC-I':
            for marker in IMMUNE_MARKERS['T_cell'] + IMMUNE_MARKERS['Myeloid']:
                if marker in df.index:
                    df.loc[marker, sample] += np.random.uniform(2, 4)

    # Save expression matrix
    expr_path = output_dir / "expression_matrix.tsv"
    df.to_csv(expr_path, sep='\t')
    print(f"[OK] Created bulk expression: {expr_path}")
    print(f"     Samples: {n_samples}, Genes: {n_genes}")

    # Save sample metadata
    meta_df = pd.DataFrame({
        'sample_id': sample_ids,
        'subtype_true': subtypes,
        'treatment': np.random.choice(['naive', 'chemo', 'chemo_io'], n_samples),
        'response': np.random.choice(['responder', 'non_responder'], n_samples)
    })
    meta_path = output_dir / "sample_metadata.tsv"
    meta_df.to_csv(meta_path, sep='\t', index=False)
    print(f"[OK] Created metadata: {meta_path}")

    return expr_path, meta_path


def create_scrna_data(output_dir: Path, n_cells: int = 5000, n_genes: int = 2000):
    """Create synthetic scRNA-seq data in AnnData format."""
    try:
        import scanpy as sc
        import anndata as ad
    except ImportError:
        print("[WARN] scanpy/anndata not available, skipping scRNA test data")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Generate count matrix (sparse-like with many zeros)
    counts = np.random.negative_binomial(n=2, p=0.3, size=(n_cells, n_genes))
    counts = counts.astype(np.float32)

    # Gene names (include markers)
    all_markers = []
    for markers in SCLC_MARKERS.values():
        all_markers.extend(markers)
    for markers in IMMUNE_MARKERS.values():
        all_markers.extend(markers)
    all_markers = list(set(all_markers))[:min(500, n_genes)]

    other_genes = [f"GENE_{i}" for i in range(n_genes - len(all_markers))]
    gene_names = all_markers + other_genes

    # Cell barcodes
    cell_barcodes = [f"CELL_{i:05d}" for i in range(n_cells)]

    # Cell types
    cell_types = np.random.choice(
        ['SCLC-A', 'SCLC-N', 'SCLC-P', 'SCLC-I', 'T_cell', 'Myeloid', 'Fibroblast'],
        n_cells,
        p=[0.25, 0.15, 0.10, 0.10, 0.15, 0.15, 0.10]
    )

    # Create AnnData
    adata = ad.AnnData(X=counts)
    adata.var_names = gene_names
    adata.obs_names = cell_barcodes
    adata.obs['cell_type'] = cell_types
    adata.obs['sample'] = np.random.choice(['sample_1', 'sample_2', 'sample_3'], n_cells)

    # Basic QC metrics
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
    adata.obs['total_counts'] = adata.X.sum(axis=1)

    # Save
    h5ad_path = output_dir / "scrna_processed.h5ad"
    adata.write(h5ad_path)
    print(f"[OK] Created scRNA data: {h5ad_path}")
    print(f"     Cells: {n_cells}, Genes: {n_genes}")

    return h5ad_path


def create_hub_genes(output_dir: Path, n_modules: int = 5, genes_per_module: int = 20):
    """Create synthetic hub genes for testing downstream stages."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    rows = []
    module_names = [f"module_{i+1}" for i in range(n_modules)]

    # Include some real marker genes
    real_genes = list(SCLC_MARKERS['SCLC-A']) + list(IMMUNE_MARKERS['Exhaustion'])

    for i, module in enumerate(module_names):
        # Mix of real and fake genes
        n_real = min(5, len(real_genes))
        module_genes = real_genes[:n_real] + [f"HUB_{module}_{j}" for j in range(genes_per_module - n_real)]
        real_genes = real_genes[n_real:]  # Don't reuse

        for gene in module_genes:
            rows.append({
                'gene': gene,
                'module': module,
                'kME': np.random.uniform(0.7, 0.95),
                'module_membership': np.random.uniform(0.6, 0.9),
                'gene_significance': np.random.uniform(0.3, 0.8)
            })

    df = pd.DataFrame(rows)
    hub_path = output_dir / "hub_genes.tsv"
    df.to_csv(hub_path, sep='\t', index=False)
    print(f"[OK] Created hub genes: {hub_path}")
    print(f"     Modules: {n_modules}, Total genes: {len(df)}")

    return hub_path


def main():
    root = get_project_root()

    print("=" * 60)
    print("Creating synthetic test data for pipeline validation")
    print("=" * 60)

    # Bulk RNA expression
    bulk_dir = root / "data" / "processed" / "bulk"
    create_bulk_expression(bulk_dir)

    # scRNA data
    scrna_dir = root / "data" / "processed" / "scrna"
    create_scrna_data(scrna_dir)

    # Hub genes (for stages 7+)
    modules_dir = root / "results" / "modules"
    create_hub_genes(modules_dir)

    # Create immune scores placeholder
    immune_dir = root / "results" / "immune"
    immune_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples = 100
    immune_df = pd.DataFrame({
        'sample_id': [f"SAMPLE_{i:03d}" for i in range(n_samples)],
        'cytotoxic_score': np.random.uniform(-2, 2, n_samples),
        'exhaustion_score': np.random.uniform(-2, 2, n_samples),
        'immune_infiltration': np.random.uniform(0, 1, n_samples),
        'immune_state': np.random.choice(['hot', 'cold', 'excluded'], n_samples)
    })
    immune_path = immune_dir / "immune_scores.tsv"
    immune_df.to_csv(immune_path, sep='\t', index=False)
    print(f"[OK] Created immune scores: {immune_path}")

    print("=" * 60)
    print("Test data creation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
