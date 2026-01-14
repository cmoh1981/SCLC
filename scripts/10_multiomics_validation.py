#!/usr/bin/env python
"""
Stage 10: Multi-Omics Validation

Tasks:
- Proteomics: Validate module directionality at protein level
- Metabolomics: Exploratory metabolic validation
- Microbiome: Exploratory association (separate cohort)
- Generate validation summary

Usage:
    python scripts/10_multiomics_validation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.validation import run_multiomics_validation


def main():
    """Run Stage 10: Multi-Omics Validation."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage10_multiomics_validation",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 10: Multi-Omics Validation")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage10"):
        logger.warning("Stage 10 is disabled in config. Exiting.")
        return 1

    # Paths
    results_dir = root / "results"
    data_dir = root / "data" / "processed"
    output_dir = root / "results" / "validation"

    # Validation config
    validation_config = {
        "proteomics_enabled": True,
        "metabolomics_enabled": True,
        "microbiome_enabled": True,
        "note": "Proteomics and metabolomics are from different cohorts - exploratory validation only"
    }

    # Run validation
    results = run_multiomics_validation(
        results_dir=results_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        config=validation_config,
        logger=logger
    )

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage10_multiomics",
        parameters=validation_config,
        input_files=[results_dir, data_dir],
        output_files=[Path(results.get("summary_file", ""))],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 10 Summary:")
    for val_type, val_result in results.get("validations", {}).items():
        logger.info(f"  {val_type}: {val_result.get('status', 'N/A')}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
