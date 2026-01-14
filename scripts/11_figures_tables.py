#!/usr/bin/env python
"""
Stage 11: Figures and Tables

Tasks:
- Generate Fig1-Fig5 (300 dpi)
- Generate Table1-Table3
- Generate supplementary figures

Usage:
    python scripts/11_figures_tables.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.figures import run_figures_tables


def main():
    """Run Stage 11: Figures and Tables."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage11_figures_tables",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 11: Figures and Tables")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage11"):
        logger.warning("Stage 11 is disabled in config. Exiting.")
        return 1

    # Paths
    results_dir = root / "results"
    config_dir = root / "configs"
    output_dir = root / "results"

    # Run figure/table generation
    results = run_figures_tables(
        results_dir=results_dir,
        config_dir=config_dir,
        output_dir=output_dir,
        logger=logger
    )

    # Write manifest
    output_files = [Path(f) for f in results.get("figures", []) + results.get("tables", [])]

    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage11_figures_tables",
        parameters={
            "dpi": 300,
            "format": "png"
        },
        input_files=[results_dir],
        output_files=output_files,
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 11 Summary:")
    logger.info(f"  Figures generated: {len(results.get('figures', []))}")
    logger.info(f"  Tables generated: {len(results.get('tables', []))}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
