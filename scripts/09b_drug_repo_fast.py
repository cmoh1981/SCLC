#!/usr/bin/env python
"""
Stage 9b: Fast Drug Repositioning (Top 100 Hub Genes Only)
Optimized version that queries only top hub genes for faster execution.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import PipelineConfig, get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.drug_repositioning import query_dgidb, rank_drugs_3leg


def main():
    """Run fast drug repositioning with top hub genes."""
    root = get_project_root()
    config = PipelineConfig()

    logger = setup_logging(
        stage_name="stage9b_drug_fast",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 9b: Fast Drug Repositioning (Top 100 Hub Genes)")
    logger.info("=" * 60)

    # Load hub genes and filter to top 100 by kME
    hub_genes_path = root / "results" / "modules" / "hub_genes.tsv"
    if not hub_genes_path.exists():
        logger.error(f"Hub genes not found: {hub_genes_path}")
        return 1

    hub_df = pd.read_csv(hub_genes_path, sep='\t')

    # Sort by kME and take top 100
    if 'kME' in hub_df.columns:
        hub_df = hub_df.sort_values('kME', ascending=False)

    top_genes = hub_df['gene'].head(100).unique().tolist()
    logger.info(f"Selected top {len(top_genes)} hub genes for drug query")

    # Output directory
    output_dir = root / "results" / "drugs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Query DGIdb only (skip slow STITCH)
    logger.info("Querying DGIdb for drug-target interactions...")
    target_hits = query_dgidb(top_genes, logger=logger)

    if len(target_hits) > 0:
        target_path = output_dir / "dgidb_interactions.tsv"
        target_hits.to_csv(target_path, sep='\t', index=False)
        logger.info(f"Found {len(target_hits)} drug-gene interactions")

        # Rank drugs
        drug_ranks = rank_drugs_3leg(
            target_hits,
            reversal_scores=None,
            depmap_coherence=None,
            logger=logger
        )

        rank_path = output_dir / "drug_rank.tsv"
        drug_ranks.to_csv(rank_path, sep='\t', index=False)
        logger.info(f"Ranked {len(drug_ranks)} drugs")

        # Top 20 summary
        top_drugs = drug_ranks.head(20)
        top_path = output_dir / "top_drugs_summary.tsv"
        top_drugs.to_csv(top_path, sep='\t', index=False)

        logger.info("=" * 60)
        logger.info("Top 10 Drug Candidates:")
        for _, row in drug_ranks.head(10).iterrows():
            logger.info(f"  {row['rank']}. {row['drug_name']} (targets: {row['n_targets']})")
        logger.info("=" * 60)
    else:
        logger.warning("No drug interactions found")

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage9b_drug_fast",
        parameters={"n_genes_queried": len(top_genes)},
        input_files=[hub_genes_path],
        output_files=[output_dir / "drug_rank.tsv"],
        logger=logger
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
