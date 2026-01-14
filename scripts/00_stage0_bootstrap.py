#!/usr/bin/env python
"""
Stage 0: Asset Bootstrap

Tasks:
- Clone GitHub tools with version pinning
- Extract licenses
- Create asset inventory
- Record version locks

Usage:
    python scripts/00_stage0_bootstrap.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.bootstrap import run_bootstrap


def main():
    """Run Stage 0: Asset Bootstrap."""
    root = get_project_root()
    config = PipelineConfig()

    # Setup logging
    logger = setup_logging(
        stage_name="stage0_bootstrap",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 0: Asset Bootstrap")
    logger.info("=" * 60)

    # Check if stage is enabled
    if not config.is_stage_enabled("stage0"):
        logger.warning("Stage 0 is disabled in config. Exiting.")
        return 1

    # Run bootstrap
    results = run_bootstrap(config=config, logger=logger)

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage0_bootstrap",
        parameters={
            "config_file": "configs/stage0_assets.yaml"
        },
        input_files=[root / "configs" / "stage0_assets.yaml"],
        output_files=[
            Path(results.get("inventory_path", "")),
            Path(results.get("lock_file", ""))
        ],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 0 Summary:")
    logger.info(f"  Tools cloned: {len([t for t in results.get('tools_cloned', []) if t.get('success')])}")
    logger.info(f"  Licenses extracted: {len(results.get('licenses_extracted', []))}")
    logger.info(f"  Inventory: {results.get('inventory_path', 'N/A')}")
    logger.info(f"  Lock file: {results.get('lock_file', 'N/A')}")
    logger.info("=" * 60)

    if results.get("errors"):
        logger.warning(f"Errors encountered: {results['errors']}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
