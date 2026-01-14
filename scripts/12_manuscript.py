#!/usr/bin/env python
"""
Stage 12: Manuscript Auto-Draft

Tasks:
- Generate main.md (main text)
- Generate methods.md (methods section)
- Generate supplement.md (supplementary info)

Usage:
    python scripts/12_manuscript.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.manuscript import run_manuscript_generation


def main():
    """Run Stage 12: Manuscript Generation."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage12_manuscript",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 12: Manuscript Auto-Draft")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage12"):
        logger.warning("Stage 12 is disabled in config. Exiting.")
        return 1

    # Paths
    results_dir = root / "results"
    output_dir = root / "manuscript"

    # Run manuscript generation
    results = run_manuscript_generation(
        results_dir=results_dir,
        output_dir=output_dir,
        config=config.pipeline,
        logger=logger
    )

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage12_manuscript",
        parameters={
            "sections": ["main", "methods", "supplement"]
        },
        input_files=[results_dir],
        output_files=[Path(f) for f in results.get("files", [])],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 12 Summary:")
    logger.info(f"  Files generated: {len(results.get('files', []))}")
    for f in results.get("files", []):
        logger.info(f"    - {f}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
