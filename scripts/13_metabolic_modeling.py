#!/usr/bin/env python
"""
Stage 13: Genome-scale Metabolic (GEM) Modeling

Tasks:
- Create SCLC-specific metabolic model
- Integrate transcriptomic data (GIMME-like algorithm)
- Run subtype-specific Flux Balance Analysis (FBA)
- Identify metabolic vulnerabilities
- Map to metabolic drug targets

Usage:
    python scripts/13_metabolic_modeling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.metabolic import run_metabolic_analysis


def main():
    """Run Stage 13: Metabolic Modeling."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage13_metabolic",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 13: Genome-scale Metabolic Modeling")
    logger.info("=" * 60)

    # Input paths
    expression_path = root / "data" / "processed" / "bulk" / "bulk_expression_matrix.tsv"
    subtype_path = root / "results" / "subtypes" / "subtype_scores.tsv"
    output_dir = root / "results" / "metabolic"

    # Check inputs
    if not expression_path.exists():
        logger.error(f"Expression matrix not found: {expression_path}")
        logger.error("Please run Stage 2 first.")
        return 1

    if not subtype_path.exists():
        logger.error(f"Subtype scores not found: {subtype_path}")
        logger.error("Please run Stage 5 first.")
        return 1

    # Run metabolic analysis
    results = run_metabolic_analysis(
        expression_path=expression_path,
        subtype_path=subtype_path,
        output_dir=output_dir,
        logger=logger
    )

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage13_metabolic",
        parameters={},
        input_files=[expression_path, subtype_path],
        output_files=[
            output_dir / "subtype_fluxes.tsv",
            output_dir / "metabolic_vulnerabilities.tsv",
            output_dir / "metabolic_drug_targets.tsv"
        ],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 13 Summary:")
    logger.info(f"  Vulnerabilities identified: {results.get('n_vulnerabilities', 'N/A')}")
    logger.info(f"  Drug targets mapped: {results.get('n_drug_targets', 'N/A')}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
