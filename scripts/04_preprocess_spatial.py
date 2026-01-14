#!/usr/bin/env python
"""
Stage 4: Preprocess Spatial Transcriptomics

Tasks:
- Load spatial data (Visium/GeoMX)
- QC filtering
- Normalization
- Export processed matrix with coordinates

Usage:
    python scripts/04_preprocess_spatial.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.preprocess import preprocess_spatial


def main():
    """Run Stage 4: Preprocess Spatial."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage4_preprocess_spatial",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 4: Preprocess Spatial Transcriptomics")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage4"):
        logger.warning("Stage 4 is disabled in config. Exiting.")
        return 1

    # Get input path
    input_dir = root / "data" / "raw" / "geo" / "GSE267310"
    output_dir = root / "data" / "processed" / "spatial"

    # Preprocessing config
    preprocess_config = {
        "min_counts_spot": 500,
        "min_spots": 10
    }

    # Run preprocessing
    results = preprocess_spatial(
        input_path=input_dir,
        output_dir=output_dir,
        config=preprocess_config,
        logger=logger
    )

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage4_preprocess_spatial",
        parameters=preprocess_config,
        input_files=[input_dir],
        output_files=[Path(results.get("output_file", ""))],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 4 Summary:")
    logger.info(f"  Spots (raw): {results.get('n_spots_raw', 'N/A')}")
    logger.info(f"  Spots (filtered): {results.get('n_spots_filtered', 'N/A')}")
    logger.info(f"  Output: {results.get('output_file', 'N/A')}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
