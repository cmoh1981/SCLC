#!/usr/bin/env python
"""
Stage 9c: Drug Repositioning with Known SCLC Genes
Uses real SCLC-relevant genes for drug target queries.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.config import get_project_root
from sclc.utils import setup_logging, write_manifest
from sclc.drug_repositioning import query_dgidb, rank_drugs_3leg


# Known SCLC-relevant genes for drug repositioning
SCLC_TARGET_GENES = [
    # SCLC subtype markers
    'ASCL1', 'DLL3', 'NEUROD1', 'POU2F3', 'INSM1', 'CHGA', 'SYP',
    # Immune checkpoints
    'PDCD1', 'CD274', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT',
    # Signaling pathways
    'NOTCH1', 'NOTCH2', 'MYC', 'MYCL', 'MYCN', 'BCL2', 'MCL1',
    # DNA damage response
    'ATR', 'ATM', 'PARP1', 'PARP2', 'CHEK1', 'CHEK2', 'WEE1',
    # Cell cycle
    'CDK4', 'CDK6', 'AURKA', 'AURKB', 'PLK1', 'CCNE1',
    # RTKs and signaling
    'EGFR', 'FGFR1', 'IGF1R', 'KIT', 'RET', 'NTRK1',
    # Epigenetic regulators
    'EZH2', 'LSD1', 'HDAC1', 'HDAC2', 'BRD4', 'CREBBP',
    # Apoptosis
    'BCL2L1', 'BIRC5', 'XIAP', 'CFLAR',
    # Metabolism
    'SLC7A11', 'GPX4', 'LDHA', 'HK2',
    # Known SCLC therapeutic targets
    'TROP2', 'B7H3', 'CEACAM5', 'SSTR2'
]


def main():
    """Run drug repositioning with real SCLC genes."""
    root = get_project_root()

    logger = setup_logging(
        stage_name="stage9c_drug_sclc",
        log_dir=root / "results" / "logs"
    )

    logger.info("=" * 60)
    logger.info("Stage 9c: Drug Repositioning (Known SCLC Targets)")
    logger.info("=" * 60)

    logger.info(f"Querying {len(SCLC_TARGET_GENES)} SCLC-relevant genes")

    # Output directory
    output_dir = root / "results" / "drugs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Query DGIdb
    logger.info("Querying DGIdb for drug-target interactions...")
    target_hits = query_dgidb(SCLC_TARGET_GENES, logger=logger)

    if len(target_hits) > 0:
        target_path = output_dir / "dgidb_interactions.tsv"
        target_hits.to_csv(target_path, sep='\t', index=False)
        logger.info(f"Found {len(target_hits)} drug-gene interactions")

        # Count unique drugs
        n_unique_drugs = target_hits['drug_name'].nunique()
        logger.info(f"Unique drugs: {n_unique_drugs}")

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
        logger.info("Top 15 Drug Candidates for SCLC:")
        logger.info("-" * 60)
        for _, row in drug_ranks.head(15).iterrows():
            targets = row['target_genes'][:50] + "..." if len(str(row['target_genes'])) > 50 else row['target_genes']
            logger.info(f"  {row['rank']:2d}. {row['drug_name']}")
            logger.info(f"      Targets ({row['n_targets']}): {targets}")
        logger.info("=" * 60)

        # Gene coverage summary
        covered_genes = set()
        for genes in target_hits['gene']:
            covered_genes.add(genes)
        logger.info(f"Genes with drug interactions: {len(covered_genes)}/{len(SCLC_TARGET_GENES)}")

    else:
        logger.warning("No drug interactions found")
        return 1

    # Write manifest
    write_manifest(
        output_path=root / "results" / "intermediate",
        stage_name="stage9c_drug_sclc",
        parameters={
            "n_genes_queried": len(SCLC_TARGET_GENES),
            "genes": SCLC_TARGET_GENES
        },
        input_files=[],
        output_files=[output_dir / "drug_rank.tsv"],
        logger=logger
    )

    # Regenerate Fig5 with actual data
    logger.info("Regenerating Fig5 with drug data...")
    try:
        from sclc.figures import create_drug_triangle_figure
        fig_path = root / "results" / "figures" / "Fig5_drug_triangle.png"
        create_drug_triangle_figure(drug_ranks.head(20), fig_path, logger)
        logger.info(f"Updated Fig5: {fig_path}")
    except Exception as e:
        logger.warning(f"Could not regenerate Fig5: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
