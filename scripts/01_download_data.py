#!/usr/bin/env python
"""
Stage 1: Data Download

Tasks:
- Download open-access datasets (GEO, SRA, PRIDE, MetabolomicsWorkbench)
- Create controlled-access scaffolds
- Write checksums

Usage:
    python scripts/01_download_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.download import run_download


def main():
    """Run Stage 1: Data Download."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage1_download",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 1: Data Download")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage1"):
        logger.warning("Stage 1 is disabled in config. Exiting.")
        return 1

    # Run download
    results = run_download(config=config, logger=logger)

    # Count successful downloads
    successful = [d for d in results.get("downloads", []) if d.get("success")]
    failed = [d for d in results.get("downloads", []) if not d.get("success")]

    # Write manifest
    output_files = []
    for d in successful:
        for f in d.get("files", []):
            output_files.append(Path(f.get("path", "")))

    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage1_download",
        parameters={
            "config_file": "configs/cohorts.yaml",
            "open_access_only": True
        },
        input_files=[root / "configs" / "cohorts.yaml"],
        output_files=output_files,
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 1 Summary:")
    logger.info(f"  Successful downloads: {len(successful)}")
    logger.info(f"  Failed downloads: {len(failed)}")
    logger.info(f"  Controlled access instructions: {results.get('controlled_access_instructions', 'N/A')}")
    logger.info("=" * 60)

    if failed:
        for f in failed:
            logger.warning(f"  Failed: {f.get('accession', 'unknown')} - {f.get('error', 'unknown error')}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
