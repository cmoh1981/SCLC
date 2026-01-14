#!/usr/bin/env python
"""
Stage 6: Immune-State Scoring (CORE)

Tasks:
- Score immune axis signatures (T-effector, IFNg, AP, TAM, Treg)
- Cluster samples into immune states
- Generate Fig2: Immune-state map

Usage:
    python scripts/06_score_immune.py
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.scoring import score_immune_states


def main():
    """Run Stage 6: Immune-State Scoring."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage6_immune_scoring",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 6: Immune-State Scoring (CORE)")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage6"):
        logger.warning("Stage 6 is disabled in config. Exiting.")
        return 1

    # Load expression data
    expr_path = root / "data" / "processed" / "bulk" / "bulk_expression_matrix.tsv"

    if not expr_path.exists():
        logger.error(f"Expression matrix not found: {expr_path}")
        return 1

    expression = pd.read_csv(expr_path, sep='\t', index_col=0)
    logger.info(f"Loaded expression: {expression.shape}")

    # Output directory
    output_dir = root / "results" / "immune"

    # Run immune scoring
    results = score_immune_states(
        expression=expression,
        signatures_config=config.signatures,
        output_dir=output_dir,
        method="ssgsea",
        n_clusters=4,
        logger=logger
    )

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage6_immune_scoring",
        parameters={
            "method": "ssgsea",
            "n_clusters": 4,
            "signatures": list(config.signatures.get("immune_signatures", {}).keys())
        },
        input_files=[expr_path],
        output_files=[
            Path(results.get("scores_file", "")),
            Path(results.get("states_file", "")),
            Path(results.get("profiles_file", ""))
        ],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 6 Summary (CORE ANALYSIS):")
    logger.info(f"  Samples scored: {results.get('n_samples', 'N/A')}")
    logger.info(f"  Immune states: {results.get('n_states', 'N/A')}")
    logger.info(f"  State distribution: {results.get('state_distribution', {})}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
