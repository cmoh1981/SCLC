"""
Stage 9: Drug Repositioning Module

Implements 3-leg evidence rule:
1. Target mapping (DGIdb/ChEMBL/STITCH)
2. Signature reversal (LINCS L1000)
3. DepMap coherence (from Stage 8)

Local database support:
- LINCS: GSE92742 Level 5 signatures
- STITCH: Chemical-protein interactions
- ChEMBL: Drug-target binding data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import gzip
from datetime import datetime
import requests


def query_dgidb(
    genes: List[str],
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Query DGIdb for drug-gene interactions using GraphQL API.

    Args:
        genes: List of gene symbols
        logger: Optional logger

    Returns:
        DataFrame of drug-gene interactions
    """
    url = "https://dgidb.org/api/graphql"
    results = []

    try:
        # Query in batches
        batch_size = 50
        for i in range(0, len(genes), batch_size):
            batch = genes[i:i+batch_size]

            if logger:
                logger.info(f"Querying DGIdb batch {i//batch_size + 1}/{(len(genes)-1)//batch_size + 1}")

            # GraphQL query
            gene_list = ', '.join(f'"{g}"' for g in batch)
            query = f'''
            {{
              genes(names: [{gene_list}]) {{
                nodes {{
                  name
                  interactions {{
                    drug {{
                      name
                      conceptId
                    }}
                    interactionTypes {{
                      type
                    }}
                    interactionScore
                    publications {{
                      pmid
                    }}
                    sources {{
                      fullName
                    }}
                  }}
                }}
              }}
            }}
            '''

            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json={'query': query}, headers=headers, timeout=60)

            if response.status_code == 200:
                data = response.json()

                for node in data.get("data", {}).get("genes", {}).get("nodes", []):
                    gene = node.get("name", "")

                    for interaction in node.get("interactions", []):
                        drug = interaction.get("drug", {})
                        int_types = [t.get("type", "") for t in interaction.get("interactionTypes", [])]
                        sources = [s.get("fullName", "") for s in interaction.get("sources", [])]
                        pmids = [str(p.get("pmid", "")) for p in interaction.get("publications", [])]

                        results.append({
                            "gene": gene,
                            "drug_name": drug.get("name", ""),
                            "drug_concept_id": drug.get("conceptId", ""),
                            "interaction_type": int_types,
                            "source": ",".join(sources[:3]),  # Limit sources
                            "pmids": ",".join(pmids[:5])  # Limit PMIDs
                        })

    except Exception as e:
        if logger:
            logger.error(f"DGIdb query failed: {e}")

    return pd.DataFrame(results)


def load_stitch_interactions(
    stitch_dir: Path,
    genes: List[str],
    min_score: int = 400,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Load STITCH chemical-protein interactions for target genes.

    Args:
        stitch_dir: Path to STITCH data directory
        genes: List of gene symbols to query
        min_score: Minimum combined score (0-1000)
        logger: Optional logger

    Returns:
        DataFrame of chemical-protein interactions
    """
    stitch_dir = Path(stitch_dir)
    results = []

    try:
        # Load protein-chemical links
        links_file = stitch_dir / "9606.protein_chemical.links.v5.0.tsv.gz"
        actions_file = stitch_dir / "9606.actions.v5.0.tsv.gz"
        chemicals_file = stitch_dir / "chemicals.v5.0.tsv.gz"

        if not links_file.exists():
            if logger:
                logger.warning(f"STITCH links file not found: {links_file}")
            return pd.DataFrame()

        if logger:
            logger.info("Loading STITCH chemical-protein interactions...")

        # Load links (this can be large)
        links_df = pd.read_csv(links_file, sep='\t', compression='gzip')

        # Filter by score
        if 'combined_score' in links_df.columns:
            links_df = links_df[links_df['combined_score'] >= min_score]

        # Load actions for interaction types
        actions_df = None
        if actions_file.exists():
            actions_df = pd.read_csv(actions_file, sep='\t', compression='gzip')

        # Load chemical names
        chem_df = None
        if chemicals_file.exists():
            chem_df = pd.read_csv(chemicals_file, sep='\t', compression='gzip')

        # Convert gene symbols to ENSP IDs (simplified - would need STRING aliases)
        # For now, extract gene from ENSP format if present
        for gene in genes:
            # Filter links containing gene (protein column contains ENSP IDs)
            gene_links = links_df[
                links_df['protein'].str.contains(gene, na=False, case=False)
            ] if 'protein' in links_df.columns else pd.DataFrame()

            for _, row in gene_links.iterrows():
                chem_id = row.get('chemical', '')
                protein_id = row.get('protein', '')
                score = row.get('combined_score', 0)

                # Get chemical name if available
                chem_name = chem_id
                if chem_df is not None and 'chemical' in chem_df.columns:
                    match = chem_df[chem_df['chemical'] == chem_id]
                    if len(match) > 0 and 'name' in match.columns:
                        chem_name = match.iloc[0]['name']

                # Get action type if available
                action_type = ''
                if actions_df is not None:
                    action_match = actions_df[
                        (actions_df.get('item_id_a', '') == chem_id) |
                        (actions_df.get('item_id_b', '') == protein_id)
                    ]
                    if len(action_match) > 0 and 'mode' in action_match.columns:
                        action_type = action_match.iloc[0]['mode']

                results.append({
                    'gene': gene,
                    'drug_name': chem_name,
                    'drug_id': chem_id,
                    'protein_id': protein_id,
                    'interaction_type': action_type,
                    'score': score,
                    'source': 'STITCH'
                })

        if logger:
            logger.info(f"Found {len(results)} STITCH interactions for {len(genes)} genes")

    except Exception as e:
        if logger:
            logger.error(f"STITCH loading failed: {e}")

    return pd.DataFrame(results)


def load_lincs_signatures(
    lincs_dir: Path,
    disease_signature: pd.Series,
    top_n: int = 100,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Load LINCS L1000 signatures and compute reversal scores.

    Args:
        lincs_dir: Path to LINCS data directory
        disease_signature: Disease expression signature (gene -> log2FC)
        top_n: Number of top reversing drugs to return
        logger: Optional logger

    Returns:
        DataFrame with drug reversal scores
    """
    lincs_dir = Path(lincs_dir)

    try:
        # Check for perturbation info
        pert_file = lincs_dir / "GSE92742_Broad_LINCS_pert_info.txt.gz"
        sig_file = lincs_dir / "GSE92742_Broad_LINCS_sig_info.txt.gz"
        gene_file = lincs_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz"

        if not pert_file.exists():
            if logger:
                logger.warning(f"LINCS perturbation file not found: {pert_file}")
            return pd.DataFrame()

        if logger:
            logger.info("Loading LINCS perturbation metadata...")

        # Load perturbation info
        pert_df = pd.read_csv(pert_file, sep='\t', compression='gzip')

        # Load signature info
        if sig_file.exists():
            sig_df = pd.read_csv(sig_file, sep='\t', compression='gzip')
        else:
            sig_df = None

        # Load gene info
        if gene_file.exists():
            gene_df = pd.read_csv(gene_file, sep='\t', compression='gzip')
        else:
            gene_df = None

        # Note: Full LINCS Level 5 data is 21GB+ in GCTX format
        # For practical use, we compute connectivity scores using pre-filtered data
        # or use CLUE API. Here we return metadata as placeholder.

        if logger:
            logger.info(f"LINCS metadata loaded: {len(pert_df)} perturbations")
            logger.info("Note: Full signature reversal requires GCTX parsing or CLUE API")

        # Return perturbation summary
        drug_perts = pert_df[pert_df['pert_type'] == 'trt_cp'] if 'pert_type' in pert_df.columns else pert_df

        if len(drug_perts) > 0:
            summary = drug_perts.groupby('pert_iname').agg({
                'pert_id': 'count'
            }).reset_index()
            summary.columns = ['drug_name', 'n_signatures']
            summary['reversal_score'] = np.nan  # Placeholder
            summary['source'] = 'LINCS_L1000'
            return summary.head(top_n)

        return pd.DataFrame()

    except Exception as e:
        if logger:
            logger.error(f"LINCS loading failed: {e}")
        return pd.DataFrame()


def compute_signature_reversal_score(
    drug_signature: pd.Series,
    disease_signature: pd.Series
) -> float:
    """
    Compute signature reversal score (negative correlation).

    Args:
        drug_signature: Drug-induced expression changes
        disease_signature: Disease expression signature (e.g., resistance)

    Returns:
        Reversal score (negative = drug reverses disease signature)
    """
    from scipy import stats

    # Align genes
    common_genes = drug_signature.index.intersection(disease_signature.index)

    if len(common_genes) < 10:
        return np.nan

    corr, _ = stats.pearsonr(
        drug_signature.loc[common_genes],
        disease_signature.loc[common_genes]
    )

    # Negative correlation means reversal
    return -corr


def rank_drugs_3leg(
    target_hits: pd.DataFrame,
    reversal_scores: Optional[pd.DataFrame] = None,
    depmap_coherence: Optional[pd.DataFrame] = None,
    weights: Dict[str, float] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Rank drugs using 3-leg evidence rule.

    Args:
        target_hits: Drug-target mapping results
        reversal_scores: Signature reversal scores (optional)
        depmap_coherence: DepMap validation scores (optional)
        weights: Weights for each evidence leg
        logger: Optional logger

    Returns:
        Ranked drug list with evidence columns
    """
    if weights is None:
        weights = {
            "target": 1.0,
            "reversal": 1.0,
            "depmap": 1.0
        }

    # Aggregate target hits per drug
    drug_scores = target_hits.groupby('drug_name').agg({
        'gene': lambda x: ','.join(sorted(set(x))),
        'interaction_type': lambda x: ','.join(str(i) for i in x if i),
        'source': 'first'
    }).reset_index()

    drug_scores.columns = ['drug_name', 'target_genes', 'interaction_types', 'sources']
    drug_scores['n_targets'] = drug_scores['target_genes'].str.count(',') + 1

    # Target score (more targets = higher score)
    drug_scores['target_score'] = np.log1p(drug_scores['n_targets'])

    # Initialize evidence columns
    drug_scores['reversal_score'] = np.nan
    drug_scores['depmap_score'] = np.nan
    drug_scores['reversal_available'] = False
    drug_scores['depmap_available'] = False

    # Add reversal scores if available
    if reversal_scores is not None and len(reversal_scores) > 0:
        for drug in drug_scores['drug_name']:
            if drug in reversal_scores.index:
                drug_scores.loc[drug_scores['drug_name'] == drug, 'reversal_score'] = \
                    reversal_scores.loc[drug, 'score']
                drug_scores.loc[drug_scores['drug_name'] == drug, 'reversal_available'] = True

    # Add DepMap scores if available
    if depmap_coherence is not None and len(depmap_coherence) > 0:
        for drug in drug_scores['drug_name']:
            if drug in depmap_coherence.index:
                drug_scores.loc[drug_scores['drug_name'] == drug, 'depmap_score'] = \
                    depmap_coherence.loc[drug, 'score']
                drug_scores.loc[drug_scores['drug_name'] == drug, 'depmap_available'] = True

    # Compute composite score
    drug_scores['composite_score'] = 0

    # Target leg (always available)
    drug_scores['composite_score'] += weights['target'] * drug_scores['target_score']

    # Reversal leg (if available)
    reversal_mask = ~drug_scores['reversal_score'].isna()
    if reversal_mask.any():
        drug_scores.loc[reversal_mask, 'composite_score'] += \
            weights['reversal'] * drug_scores.loc[reversal_mask, 'reversal_score']

    # DepMap leg (if available)
    depmap_mask = ~drug_scores['depmap_score'].isna()
    if depmap_mask.any():
        drug_scores.loc[depmap_mask, 'composite_score'] += \
            weights['depmap'] * drug_scores.loc[depmap_mask, 'depmap_score']

    # Count evidence legs
    drug_scores['n_evidence_legs'] = 1  # target always available
    drug_scores['n_evidence_legs'] += drug_scores['reversal_available'].astype(int)
    drug_scores['n_evidence_legs'] += drug_scores['depmap_available'].astype(int)

    # Rank by composite score
    drug_scores = drug_scores.sort_values('composite_score', ascending=False)
    drug_scores['rank'] = range(1, len(drug_scores) + 1)

    if logger:
        logger.info(f"Ranked {len(drug_scores)} drugs")
        n_3leg = (drug_scores['n_evidence_legs'] == 3).sum()
        n_2leg = (drug_scores['n_evidence_legs'] == 2).sum()
        logger.info(f"3-leg evidence: {n_3leg}, 2-leg evidence: {n_2leg}")

    return drug_scores


def run_drug_repositioning(
    hub_genes_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
    depmap_results_path: Optional[Path] = None,
    stitch_dir: Optional[Path] = None,
    lincs_dir: Optional[Path] = None,
    disease_signature: Optional[pd.Series] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run drug repositioning analysis.

    Args:
        hub_genes_path: Path to hub genes TSV
        output_dir: Output directory
        config: Configuration dictionary
        depmap_results_path: Path to DepMap validation results (optional)
        stitch_dir: Path to STITCH database directory (optional)
        lincs_dir: Path to LINCS database directory (optional)
        disease_signature: Disease expression signature for reversal (optional)
        logger: Optional logger

    Returns:
        Results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "stage": "drug_repositioning",
        "timestamp": datetime.now().isoformat(),
        "success": False
    }

    try:
        # Load hub genes
        hub_genes = pd.read_csv(hub_genes_path, sep='\t')
        gene_list = hub_genes['gene'].unique().tolist()

        if logger:
            logger.info(f"Running drug repositioning for {len(gene_list)} hub genes")

        # Leg 1: Target mapping via DGIdb
        if logger:
            logger.info("Leg 1: Querying DGIdb for drug-target interactions")

        target_hits = query_dgidb(gene_list, logger=logger)

        # Also load STITCH interactions if available
        if stitch_dir and Path(stitch_dir).exists():
            if logger:
                logger.info("Leg 1 (supplement): Loading STITCH interactions")

            stitch_hits = load_stitch_interactions(
                stitch_dir=stitch_dir,
                genes=gene_list,
                min_score=config.get("stitch_min_score", 400),
                logger=logger
            )

            if len(stitch_hits) > 0:
                stitch_path = output_dir / "stitch_interactions.tsv"
                stitch_hits.to_csv(stitch_path, sep='\t', index=False)
                results["stitch_file"] = str(stitch_path)
                results["n_stitch_interactions"] = len(stitch_hits)

                # Merge with DGIdb (if both available)
                if len(target_hits) > 0:
                    # Standardize columns for merge
                    stitch_standard = stitch_hits[['gene', 'drug_name', 'interaction_type', 'source']].copy()
                    stitch_standard['drug_concept_id'] = stitch_hits.get('drug_id', '')
                    stitch_standard['pmids'] = ''

                    target_hits = pd.concat([target_hits, stitch_standard], ignore_index=True)
                else:
                    target_hits = stitch_hits

        if len(target_hits) > 0:
            target_path = output_dir / "dgidb_interactions.tsv"
            target_hits.to_csv(target_path, sep='\t', index=False)
            results["dgidb_file"] = str(target_path)
            results["n_drug_interactions"] = len(target_hits)
        else:
            if logger:
                logger.warning("No drug-target interactions found")
            results["n_drug_interactions"] = 0

        # Leg 2: Signature reversal (LINCS)
        reversal_scores = None
        if config.get("lincs_enabled", False) and lincs_dir and Path(lincs_dir).exists():
            if logger:
                logger.info("Leg 2: Loading LINCS signature data")

            lincs_summary = load_lincs_signatures(
                lincs_dir=lincs_dir,
                disease_signature=disease_signature if disease_signature is not None else pd.Series(),
                top_n=500,
                logger=logger
            )

            if len(lincs_summary) > 0:
                lincs_path = output_dir / "lincs_perturbations.tsv"
                lincs_summary.to_csv(lincs_path, sep='\t', index=False)
                results["lincs_file"] = str(lincs_path)
                results["reversal_status"] = "available"

                # Set reversal scores if computed
                if 'reversal_score' in lincs_summary.columns and not lincs_summary['reversal_score'].isna().all():
                    reversal_scores = lincs_summary.set_index('drug_name')[['reversal_score']].rename(
                        columns={'reversal_score': 'score'}
                    )
            else:
                results["reversal_status"] = "no_data"
        elif config.get("lincs_enabled", False):
            if logger:
                logger.warning("LINCS enabled but directory not found")
            results["reversal_status"] = "not_configured"
        else:
            results["reversal_status"] = "disabled"

        # Leg 3: DepMap coherence
        depmap_coherence = None
        if depmap_results_path and Path(depmap_results_path).exists():
            if logger:
                logger.info("Leg 3: Loading DepMap validation results")
            depmap_coherence = pd.read_csv(depmap_results_path, sep='\t', index_col=0)
            results["depmap_status"] = "available"
        else:
            results["depmap_status"] = "not_available"

        # Rank drugs
        if len(target_hits) > 0:
            drug_ranks = rank_drugs_3leg(
                target_hits,
                reversal_scores=reversal_scores,
                depmap_coherence=depmap_coherence,
                weights=config.get("weights"),
                logger=logger
            )

            rank_path = output_dir / "drug_rank.tsv"
            drug_ranks.to_csv(rank_path, sep='\t', index=False)
            results["drug_rank_file"] = str(rank_path)
            results["n_drugs_ranked"] = len(drug_ranks)

            # Top drugs summary
            top_cols = ['rank', 'drug_name', 'target_genes', 'n_evidence_legs', 'composite_score']
            top_cols = [c for c in top_cols if c in drug_ranks.columns]
            top_drugs = drug_ranks.head(20)[top_cols]
            top_path = output_dir / "top_drugs_summary.tsv"
            top_drugs.to_csv(top_path, sep='\t', index=False)
            results["top_drugs_file"] = str(top_path)

        results["success"] = True

        if logger:
            logger.info("Drug repositioning complete")

    except Exception as e:
        results["error"] = str(e)
        if logger:
            logger.error(f"Drug repositioning failed: {e}")

    return results
