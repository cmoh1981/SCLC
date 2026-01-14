#!/usr/bin/env python
"""
Stage 2: Preprocess Bulk RNA-seq

Tasks:
- Load and merge bulk RNA-seq data
- QC filtering
- Normalization and log2 transform
- Batch correction (if multiple datasets)
- Export processed matrix

Usage:
    python scripts/02_preprocess_bulk.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.preprocess import preprocess_bulk_rna


def main():
    """Run Stage 2: Preprocess Bulk RNA."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage2_preprocess_bulk",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 2: Preprocess Bulk RNA-seq")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage2"):
        logger.warning("Stage 2 is disabled in config. Exiting.")
        return 1

    # Get input path (from downloaded GEO data)
    input_dir = root / "data" / "raw" / "geo" / "GSE60052"
    output_dir = root / "data" / "processed" / "bulk"

    # Preprocessing config
    preprocess_config = {
        "min_counts": 10,
        "min_samples_pct": 0.1,
        "log_transform": True,
        "pseudocount": 1.0
    }

    # Run preprocessing
    results = preprocess_bulk_rna(
        input_path=input_dir,
        output_dir=output_dir,
        config=preprocess_config,
        logger=logger
    )

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage2_preprocess_bulk",
        parameters=preprocess_config,
        input_files=[input_dir],
        output_files=[
            Path(results.get("output_file", "")),
            Path(results.get("qc_report", ""))
        ],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 2 Summary:")
    logger.info(f"  Genes (raw): {results.get('n_genes_raw', 'N/A')}")
    logger.info(f"  Genes (filtered): {results.get('n_genes_filtered', 'N/A')}")
    logger.info(f"  Samples: {results.get('n_samples_raw', 'N/A')}")
    logger.info(f"  Output: {results.get('output_file', 'N/A')}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
