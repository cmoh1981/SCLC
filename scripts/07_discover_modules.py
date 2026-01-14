#!/usr/bin/env python
"""
Stage 7: Resistance Module Discovery

Tasks:
- WGCNA co-expression network analysis
- Identify resistance-associated modules
- Identify hub genes
- Correlate modules with immune states
- Generate Fig3: Module-trait heatmap

Usage:
    python scripts/07_discover_modules.py
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.modules import run_module_discovery


def main():
    """Run Stage 7: Module Discovery."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage7_module_discovery",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 7: Resistance Module Discovery")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage7"):
        logger.warning("Stage 7 is disabled in config. Exiting.")
        return 1

    # Paths
    expr_path = root / "data" / "processed" / "bulk" / "bulk_expression_matrix.tsv"
    traits_path = root / "results" / "immune" / "immune_scores.tsv"
    output_dir = root / "results" / "modules"

    if not expr_path.exists():
        logger.error(f"Expression matrix not found: {expr_path}")
        return 1

    # Module discovery config
    module_config = {
        "soft_power": 6,
        "min_module_size": 30,
        "merge_threshold": 0.25,
        "top_hub_genes": 10
    }

    # Run module discovery
    results = run_module_discovery(
        expression_path=expr_path,
        traits_path=traits_path if traits_path.exists() else None,
        output_dir=output_dir,
        config=module_config,
        logger=logger
    )

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage7_module_discovery",
        parameters=module_config,
        input_files=[expr_path, traits_path],
        output_files=[
            Path(results.get("modules_file", "")),
            Path(results.get("eigengenes_file", "")),
            Path(results.get("hub_genes_file", ""))
        ],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 7 Summary:")
    logger.info(f"  Modules identified: {results.get('n_modules', 'N/A')}")
    logger.info(f"  Hub genes: {results.get('n_hub_genes', 'N/A')}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
