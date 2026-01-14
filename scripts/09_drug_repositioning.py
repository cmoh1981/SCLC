#!/usr/bin/env python
"""
Stage 9: Drug Repositioning

Tasks:
- Implement 3-leg evidence rule:
  1. Target mapping (DGIdb + STITCH)
  2. Signature reversal (LINCS L1000)
  3. DepMap coherence (from Stage 8)
- Rank candidate drugs
- Generate Fig5: Drug triangle

Usage:
    python scripts/09_drug_repositioning.py

Local Database Support:
    - STITCH: Chemical-protein interactions
    - LINCS: GSE92742 Level 5 signatures
    Configure paths in configs/pipeline.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.drug_repositioning import run_drug_repositioning


def main():
    """Run Stage 9: Drug Repositioning."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage9_drug_repositioning",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 9: Drug Repositioning (3-Leg Evidence)")
    logger.info("=" * 60)

    if not config.is_stage_enabled("stage9"):
        logger.warning("Stage 9 is disabled in config. Exiting.")
        return 1

    # Paths
    hub_genes_path = root / "results" / "modules" / "hub_genes.tsv"
    output_dir = root / "results" / "drugs"

    if not hub_genes_path.exists():
        logger.error(f"Hub genes not found: {hub_genes_path}")
        logger.error("Please run Stage 7 first.")
        return 1

    # Check for DepMap results
    depmap_results = root / "results" / "depmap" / "hub_gene_dependencies.tsv"

    # Local drug database paths from config
    paths = config.pipeline.get("paths", {})
    stitch_dir = paths.get("stitch_dir")
    lincs_dir = paths.get("lincs_dir")

    # Log available databases
    if stitch_dir and Path(stitch_dir).exists():
        logger.info(f"STITCH database found: {stitch_dir}")
    else:
        logger.info("STITCH database not configured or not found")

    if lincs_dir and Path(lincs_dir).exists():
        logger.info(f"LINCS database found: {lincs_dir}")
    else:
        logger.info("LINCS database not configured or not found")

    # Drug repositioning config
    drug_config = {
        "lincs_enabled": lincs_dir is not None and Path(lincs_dir).exists(),
        "stitch_min_score": 400,
        "weights": {
            "target": 1.0,
            "reversal": 1.0,
            "depmap": 1.0
        }
    }

    # Run drug repositioning
    results = run_drug_repositioning(
        hub_genes_path=hub_genes_path,
        output_dir=output_dir,
        config=drug_config,
        depmap_results_path=depmap_results if depmap_results.exists() else None,
        stitch_dir=Path(stitch_dir) if stitch_dir else None,
        lincs_dir=Path(lincs_dir) if lincs_dir else None,
        logger=logger
    )

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage9_drug_repositioning",
        parameters=drug_config,
        input_files=[hub_genes_path, depmap_results],
        output_files=[
            Path(results.get("drug_rank_file", "")),
            Path(results.get("dgidb_file", ""))
        ],
        logger=logger
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 9 Summary:")
    logger.info(f"  Drug interactions found: {results.get('n_drug_interactions', 'N/A')}")
    logger.info(f"  Drugs ranked: {results.get('n_drugs_ranked', 'N/A')}")
    logger.info(f"  Reversal evidence: {results.get('reversal_status', 'N/A')}")
    logger.info(f"  DepMap evidence: {results.get('depmap_status', 'N/A')}")
    logger.info("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
