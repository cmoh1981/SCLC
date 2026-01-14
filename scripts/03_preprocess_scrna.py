#!/usr/bin/env python
"""
Stage 3: Preprocess Single-cell RNA-seq

Tasks:
- Load scRNA-seq data
- QC filtering (genes, UMIs, mito%)
- Normalization
- Feature selection (HVG)
- Dimensionality reduction (PCA, UMAP)
- Clustering
- Export h5ad

Usage:
    python scripts/03_preprocess_scrna.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.preprocess import preprocess_scrna


def main():
    """Run Stage 3: Preprocess scRNA-seq."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage3_preprocess_scrna",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 3: Preprocess Single-cell RNA-seq")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage3"):
        logger.warning("Stage 3 is disabled in config. Exiting.")
        return 1

    # Get input path
    input_dir = root / "data" / "raw" / "geo" / "GSE138267"
    output_dir = root / "data" / "processed" / "scrna"

    # Check for h5ad or matrix files
    h5ad_files = list(input_dir.glob("*.h5ad"))
    if h5ad_files:
        input_path = h5ad_files[0]
    else:
        input_path = input_dir

    # Preprocessing config
    preprocess_config = {
        "min_genes": 200,
        "max_genes": 8000,
        "max_mito_pct": 20,
        "min_cells": 3,
        "n_hvg": 2000,
        "n_pcs": 50,
        "n_neighbors": 15,
        "resolution": 0.5
    }

    # Run preprocessing
    results = preprocess_scrna(
        input_path=input_path,
        output_dir=output_dir,
        config=preprocess_config,
        logger=logger
    )

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage3_preprocess_scrna",
        parameters=preprocess_config,
        input_files=[input_path],
        output_files=[
            Path(results.get("output_file", "")),
            Path(results.get("qc_report", ""))
        ],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 3 Summary:")
    logger.info(f"  Cells (raw): {results.get('n_cells_raw', 'N/A')}")
    logger.info(f"  Cells (filtered): {results.get('n_cells_filtered', 'N/A')}")
    logger.info(f"  Genes (filtered): {results.get('n_genes_filtered', 'N/A')}")
    logger.info(f"  Output: {results.get('output_file', 'N/A')}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
