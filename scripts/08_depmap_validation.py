#!/usr/bin/env python
"""
Stage 8: DepMap Validation (Optional)

Tasks:
- Load DepMap CRISPR gene effect data
- Validate hub gene dependencies
- Correlate with drug sensitivity
- Generate Fig4

Usage:
    python scripts/08_depmap_validation.py

Note:
    Requires local DepMap files (not included in repository)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.validation import validate_with_depmap
import pandas as pd


def main():
    """Run Stage 8: DepMap Validation."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage8_depmap",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 8: DepMap Validation")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage8"):
        logger.warning("Stage 8 is disabled in config.")
        logger.info("To enable, set stages.stage8.enabled: true in configs/pipeline.yaml")
        logger.info("And provide DepMap data files in data/depmap/")
        return 0  # Not an error, just disabled

    # Check for DepMap data
    depmap_dir = root / "data" / "depmap"
    depmap_config = config.cohorts.get("external_databases", {}).get("depmap", {})

    if not depmap_dir.exists():
        logger.warning(f"DepMap directory not found: {depmap_dir}")
        logger.info("Download DepMap data from https://depmap.org/portal/download/")
        return 0

    # Load hub genes
    hub_genes_path = root / "results" / "modules" / "hub_genes.tsv"
    if not hub_genes_path.exists():
        logger.error("Hub genes not found. Run Stage 7 first.")
        return 1

    hub_df = pd.read_csv(hub_genes_path, sep='\t')
    hub_genes = hub_df['gene'].unique().tolist()

    # Paths
    immune_scores_path = root / "results" / "immune" / "immune_scores.tsv"
    output_dir = root / "results" / "depmap"

    # Run validation
    results = validate_with_depmap(
        hub_genes=hub_genes,
        immune_scores_path=immune_scores_path,
        depmap_dir=depmap_dir,
        output_dir=output_dir,
        config=depmap_config,
        logger=logger
    )

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage8_depmap",
        parameters=depmap_config,
        input_files=[hub_genes_path, immune_scores_path],
        output_files=[Path(results.get("dependencies_file", ""))],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 8 Summary:")
    logger.info(f"  Status: {results.get('status', 'N/A')}")
    logger.info(f"  Genes with data: {results.get('n_genes_with_data', 'N/A')}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
