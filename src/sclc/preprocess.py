"""
Stages 2-4: Preprocessing Modules

Functions for:
- Bulk RNA-seq preprocessing (QC, normalization, batch correction)
- Single-cell RNA-seq preprocessing (QC, filtering, integration)
- Spatial transcriptomics preprocessing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from datetime import datetime


def normalize_gene_symbols(
    df: pd.DataFrame,
    gene_col: str = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Standardize gene symbols to HGNC format.

    Args:
        df: Expression dataframe (genes as rows or columns)
        gene_col: Column containing gene names (if genes in column)
        logger: Optional logger

    Returns:
        DataFrame with standardized gene symbols
    """
    # Basic symbol cleaning - remove version numbers, standardize case
    if gene_col:
        df[gene_col] = df[gene_col].str.upper()
        df[gene_col] = df[gene_col].str.split('.').str[0]
    else:
        df.index = df.index.str.upper()
        df.index = df.index.str.split('.').str[0]

    if logger:
        logger.info("Gene symbols standardized")

    return df


def log2_transform(
    df: pd.DataFrame,
    pseudocount: float = 1.0
) -> pd.DataFrame:
    """
    Apply log2 transformation with pseudocount.

    Args:
        df: Count matrix
        pseudocount: Value to add before log (default 1)

    Returns:
        Log2-transformed matrix
    """
    return np.log2(df + pseudocount)


def filter_low_expression(
    df: pd.DataFrame,
    min_counts: int = 10,
    min_samples_pct: float = 0.1,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Filter out lowly expressed genes.

    Args:
        df: Expression matrix (genes x samples)
        min_counts: Minimum total counts across samples
        min_samples_pct: Minimum fraction of samples with non-zero expression
        logger: Optional logger

    Returns:
        Filtered expression matrix
    """
    n_genes_before = len(df)

    # Filter by total counts
    total_counts = df.sum(axis=1)
    df = df[total_counts >= min_counts]

    # Filter by detection rate
    detection_rate = (df > 0).sum(axis=1) / df.shape[1]
    df = df[detection_rate >= min_samples_pct]

    n_genes_after = len(df)

    if logger:
        logger.info(f"Filtered genes: {n_genes_before} -> {n_genes_after}")

    return df


def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quantile normalization across samples.

    Args:
        df: Expression matrix (genes x samples)

    Returns:
        Quantile-normalized matrix
    """
    from scipy.stats import rankdata

    ranked = df.apply(rankdata, axis=0)
    sorted_means = np.sort(df.values, axis=0).mean(axis=1)
    normalized = ranked.apply(lambda x: sorted_means[x.astype(int) - 1])

    return normalized


def preprocess_bulk_rna(
    input_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Preprocess bulk RNA-seq data.

    Pipeline:
    1. Load counts/TPM matrix
    2. QC filtering (low expression genes, outlier samples)
    3. Normalization (TPM/CPM, log2)
    4. Batch correction if multiple datasets
    5. Export processed matrix

    Args:
        input_path: Path to input count matrix or directory
        output_dir: Output directory for processed data
        config: Preprocessing configuration
        logger: Optional logger

    Returns:
        Preprocessing results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "stage": "preprocess_bulk_rna",
        "timestamp": datetime.now().isoformat(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "success": False
    }

    try:
        if logger:
            logger.info(f"Preprocessing bulk RNA from {input_path}")

        # Load data - adapt based on input format
        import gzip
        input_path = Path(input_path)

        if input_path.suffix == '.csv':
            df = pd.read_csv(input_path, index_col=0)
        elif input_path.suffix == '.tsv':
            df = pd.read_csv(input_path, sep='\t', index_col=0)
        elif input_path.suffix == '.gz':
            # Compressed file
            df = pd.read_csv(input_path, sep='\t', index_col=0, compression='gzip')
        else:
            # Try to find expression matrix in directory
            # Check supplementary folder first (GEO standard)
            supp_dir = input_path / "supplementary"
            search_dirs = [supp_dir, input_path] if supp_dir.exists() else [input_path]

            possible_files = []
            for search_dir in search_dirs:
                # Check for various expression file patterns
                possible_files.extend(list(search_dir.glob("*normalized*.tsv.gz")))
                possible_files.extend(list(search_dir.glob("*expression*.tsv.gz")))
                possible_files.extend(list(search_dir.glob("*counts*.tsv.gz")))
                possible_files.extend(list(search_dir.glob("*.tsv.gz")))
                possible_files.extend(list(search_dir.glob("*counts*.csv")))
                possible_files.extend(list(search_dir.glob("*expression*.csv")))

            if possible_files:
                expr_file = possible_files[0]
                if logger:
                    logger.info(f"Found expression file: {expr_file}")
                if expr_file.suffix == '.gz':
                    df = pd.read_csv(expr_file, sep='\t', index_col=0, compression='gzip')
                elif expr_file.suffix == '.csv':
                    df = pd.read_csv(expr_file, index_col=0)
                else:
                    df = pd.read_csv(expr_file, sep='\t', index_col=0)
            else:
                raise FileNotFoundError(f"No expression matrix found in {input_path}")

        results["n_genes_raw"] = len(df)
        results["n_samples_raw"] = df.shape[1]

        if logger:
            logger.info(f"Loaded {results['n_genes_raw']} genes x {results['n_samples_raw']} samples")

        # Standardize gene symbols
        df = normalize_gene_symbols(df, logger=logger)

        # Remove duplicates (keep highest mean)
        df = df.groupby(df.index).max()

        # Filter low expression
        df = filter_low_expression(
            df,
            min_counts=config.get("min_counts", 10),
            min_samples_pct=config.get("min_samples_pct", 0.1),
            logger=logger
        )

        results["n_genes_filtered"] = len(df)

        # Log2 transform if counts
        if config.get("log_transform", True):
            df = log2_transform(df, pseudocount=config.get("pseudocount", 1))

        # Save processed matrix
        output_path = output_dir / "bulk_expression_matrix.tsv"
        df.to_csv(output_path, sep='\t')
        results["output_file"] = str(output_path)

        # Generate QC report
        qc_report = {
            "n_genes_raw": results["n_genes_raw"],
            "n_genes_filtered": results["n_genes_filtered"],
            "n_samples": df.shape[1],
            "mean_expression": float(df.mean().mean()),
            "median_expression": float(df.median().median())
        }

        qc_path = output_dir / "bulk_qc_report.json"
        with open(qc_path, 'w') as f:
            json.dump(qc_report, f, indent=2)

        results["qc_report"] = str(qc_path)
        results["success"] = True

        if logger:
            logger.info(f"Bulk RNA preprocessing complete: {output_path}")

    except Exception as e:
        results["error"] = str(e)
        if logger:
            logger.error(f"Bulk RNA preprocessing failed: {e}")

    return results


def preprocess_scrna(
    input_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Preprocess single-cell RNA-seq data.

    Pipeline:
    1. Load h5ad/10X matrix
    2. QC filtering (min genes, max genes, mito %)
    3. Normalization and log transform
    4. Highly variable gene selection
    5. Integration (if multiple samples)
    6. Clustering and UMAP
    7. Export h5ad

    Args:
        input_path: Path to input data
        output_dir: Output directory
        config: Preprocessing configuration
        logger: Optional logger

    Returns:
        Preprocessing results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "stage": "preprocess_scrna",
        "timestamp": datetime.now().isoformat(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "success": False
    }

    try:
        import scanpy as sc

        if logger:
            logger.info(f"Preprocessing scRNA from {input_path}")

        # Load data
        input_path = Path(input_path)

        if input_path.suffix == '.h5ad':
            adata = sc.read_h5ad(input_path)
        elif input_path.is_dir():
            # Try 10X format
            adata = sc.read_10x_mtx(input_path)
        else:
            raise ValueError(f"Unsupported input format: {input_path}")

        results["n_cells_raw"] = adata.n_obs
        results["n_genes_raw"] = adata.n_vars

        if logger:
            logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

        # QC metrics
        sc.pp.calculate_qc_metrics(adata, inplace=True)

        # Calculate mitochondrial %
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

        # Filter cells
        min_genes = config.get("min_genes", 200)
        max_genes = config.get("max_genes", 8000)
        max_mito = config.get("max_mito_pct", 20)

        sc.pp.filter_cells(adata, min_genes=min_genes)
        adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
        adata = adata[adata.obs.pct_counts_mt < max_mito, :]

        # Filter genes
        sc.pp.filter_genes(adata, min_cells=config.get("min_cells", 3))

        results["n_cells_filtered"] = adata.n_obs
        results["n_genes_filtered"] = adata.n_vars

        if logger:
            logger.info(f"After QC: {adata.n_obs} cells x {adata.n_vars} genes")

        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Highly variable genes
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=config.get("n_hvg", 2000),
            flavor='seurat_v3' if 'counts' in adata.layers else 'seurat'
        )

        # PCA
        sc.tl.pca(adata, n_comps=config.get("n_pcs", 50))

        # Neighbors and UMAP
        sc.pp.neighbors(adata, n_neighbors=config.get("n_neighbors", 15))
        sc.tl.umap(adata)

        # Clustering
        sc.tl.leiden(adata, resolution=config.get("resolution", 0.5))

        # Save
        output_path = output_dir / "scrna_processed.h5ad"
        adata.write(output_path)
        results["output_file"] = str(output_path)

        # QC report
        qc_report = {
            "n_cells_raw": results["n_cells_raw"],
            "n_cells_filtered": results["n_cells_filtered"],
            "n_genes_raw": results["n_genes_raw"],
            "n_genes_filtered": results["n_genes_filtered"],
            "n_clusters": len(adata.obs['leiden'].unique())
        }

        qc_path = output_dir / "scrna_qc_report.json"
        with open(qc_path, 'w') as f:
            json.dump(qc_report, f, indent=2)

        results["qc_report"] = str(qc_path)
        results["success"] = True

        if logger:
            logger.info(f"scRNA preprocessing complete: {output_path}")

    except Exception as e:
        results["error"] = str(e)
        if logger:
            logger.error(f"scRNA preprocessing failed: {e}")

    return results


def preprocess_spatial(
    input_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Preprocess spatial transcriptomics data.

    Pipeline:
    1. Load spatial data (Visium/GeoMX)
    2. QC filtering (ROI depth, spot quality)
    3. Normalization
    4. Export processed matrix with coordinates

    Args:
        input_path: Path to input data
        output_dir: Output directory
        config: Preprocessing configuration
        logger: Optional logger

    Returns:
        Preprocessing results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "stage": "preprocess_spatial",
        "timestamp": datetime.now().isoformat(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "success": False
    }

    try:
        import scanpy as sc

        if logger:
            logger.info(f"Preprocessing spatial data from {input_path}")

        input_path = Path(input_path)

        # Load based on format
        if input_path.suffix == '.h5ad':
            adata = sc.read_h5ad(input_path)
        elif (input_path / 'filtered_feature_bc_matrix.h5').exists():
            # 10x Visium format
            adata = sc.read_visium(input_path)
        else:
            # Try as tabular
            df = pd.read_csv(input_path, index_col=0)
            adata = sc.AnnData(df.T)

        results["n_spots_raw"] = adata.n_obs
        results["n_genes_raw"] = adata.n_vars

        if logger:
            logger.info(f"Loaded {adata.n_obs} spots x {adata.n_vars} genes")

        # QC
        sc.pp.calculate_qc_metrics(adata, inplace=True)

        # Filter spots
        min_counts = config.get("min_counts_spot", 500)
        adata = adata[adata.obs.total_counts >= min_counts, :]

        # Filter genes
        sc.pp.filter_genes(adata, min_cells=config.get("min_spots", 10))

        results["n_spots_filtered"] = adata.n_obs
        results["n_genes_filtered"] = adata.n_vars

        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Save
        output_path = output_dir / "spatial_processed.h5ad"
        adata.write(output_path)
        results["output_file"] = str(output_path)

        # Also export as TSV for compatibility
        expr_df = pd.DataFrame(
            adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        )
        expr_df.to_csv(output_dir / "spatial_expression.tsv", sep='\t')

        results["success"] = True

        if logger:
            logger.info(f"Spatial preprocessing complete: {output_path}")

    except Exception as e:
        results["error"] = str(e)
        if logger:
            logger.error(f"Spatial preprocessing failed: {e}")

    return results
