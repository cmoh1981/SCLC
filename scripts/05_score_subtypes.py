#!/usr/bin/env python
"""
Stage 5: Subtype Scoring

Tasks:
- Score SCLC subtypes (A/N/P/I) using gene signatures
- Assign samples to subtypes
- Generate Fig1: Subtype landscape

Usage:
    python scripts/05_score_subtypes.py
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.scoring import score_sclc_subtypes


def main():
    """Run Stage 5: Subtype Scoring."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage5_subtype_scoring",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 5: Subtype Scoring (A/N/P/I)")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage5"):
        logger.warning("Stage 5 is disabled in config. Exiting.")
        return 1

    # Load expression data
    expr_path = root / "data" / "processed" / "bulk" / "bulk_expression_matrix.tsv"

    if not expr_path.exists():
        logger.error(f"Expression matrix not found: {expr_path}")
        logger.error("Please run Stage 2 first.")
        return 1

    expression = pd.read_csv(expr_path, sep='\t', index_col=0)
    logger.info(f"Loaded expression: {expression.shape}")

    # Output directory
    output_dir = root / "results" / "subtypes"

    # Run subtype scoring
    results = score_sclc_subtypes(
        expression=expression,
        signatures_config=config.signatures,
        output_dir=output_dir,
        method=config.signatures.get("scoring", {}).get("method", "ssgsea"),
        logger=logger
    )

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage5_subtype_scoring",
        parameters={
            "method": "ssgsea",
            "signatures": list(config.signatures.get("sclc_subtypes", {}).keys())
        },
        input_files=[expr_path],
        output_files=[
            Path(results.get("scores_file", "")),
            Path(results.get("calls_file", ""))
        ],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 5 Summary:")
    logger.info(f"  Samples scored: {results.get('n_samples', 'N/A')}")
    logger.info(f"  Subtype distribution: {results.get('subtype_distribution', {})}")
    logger.info(f"  Output: {results.get('scores_file', 'N/A')}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
