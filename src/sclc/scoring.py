"""
Stages 5-6: Subtype and Immune Scoring Modules

Functions for:
- SCLC subtype scoring (A/N/P/I)
- Immune-state scoring (T-effector, IFNg, etc.)
- Sample/cell classification
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from datetime import datetime


def ssgsea_score(
    expression: pd.DataFrame,
    gene_set: List[str],
    normalize: bool = True
) -> pd.Series:
    """
    Compute single-sample GSEA (ssGSEA) score for a gene set.

    Args:
        expression: Expression matrix (genes x samples)
        gene_set: List of genes in the signature
        normalize: Whether to normalize scores

    Returns:
        Series of scores per sample
    """
    # Filter to genes present in data
    genes_present = [g for g in gene_set if g in expression.index]

    if len(genes_present) < 3:
        return pd.Series(np.nan, index=expression.columns)

    # Rank genes per sample
    ranks = expression.rank(axis=0, ascending=True)

    # Calculate enrichment score
    n_genes = len(expression)
    n_set = len(genes_present)

    scores = []
    for sample in expression.columns:
        sample_ranks = ranks[sample]

        # Ranks of genes in set
        set_ranks = sample_ranks.loc[genes_present].values

        # Running sum approach
        hits = np.zeros(n_genes)
        hits[set_ranks.astype(int) - 1] = 1

        # Weight by rank
        weights = np.abs(sample_ranks.values) ** 0.25
        weighted_hits = hits * weights

        # Cumulative sums
        hit_sum = np.cumsum(weighted_hits)
        miss_sum = np.cumsum(1 - hits)

        # Normalize
        if hit_sum[-1] > 0:
            hit_sum = hit_sum / hit_sum[-1]
        if miss_sum[-1] > 0:
            miss_sum = miss_sum / miss_sum[-1]

        # Enrichment score
        es = hit_sum - miss_sum
        score = np.sum(es)

        scores.append(score)

    result = pd.Series(scores, index=expression.columns)

    if normalize:
        result = (result - result.mean()) / result.std()

    return result


def mean_score(
    expression: pd.DataFrame,
    gene_set: List[str],
    normalize: bool = True
) -> pd.Series:
    """
    Compute simple mean expression score for a gene set.

    Args:
        expression: Expression matrix (genes x samples)
        gene_set: List of genes in the signature
        normalize: Whether to z-score normalize

    Returns:
        Series of scores per sample
    """
    genes_present = [g for g in gene_set if g in expression.index]

    if len(genes_present) < 1:
        return pd.Series(np.nan, index=expression.columns)

    scores = expression.loc[genes_present].mean(axis=0)

    if normalize:
        scores = (scores - scores.mean()) / scores.std()

    return scores


def score_signatures(
    expression: pd.DataFrame,
    signatures: Dict[str, Dict],
    method: str = "ssgsea",
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Score multiple gene signatures.

    Args:
        expression: Expression matrix (genes x samples)
        signatures: Dictionary of signature name -> {genes: [...], ...}
        method: Scoring method (ssgsea, mean)
        logger: Optional logger

    Returns:
        DataFrame of scores (samples x signatures)
    """
    scores = {}

    for sig_name, sig_info in signatures.items():
        genes = sig_info.get("genes", sig_info.get("markers", []))

        if method == "ssgsea":
            scores[sig_name] = ssgsea_score(expression, genes)
        else:
            scores[sig_name] = mean_score(expression, genes)

        if logger:
            n_found = len([g for g in genes if g in expression.index])
            logger.info(f"Scored {sig_name}: {n_found}/{len(genes)} genes found")

    return pd.DataFrame(scores)


def assign_subtype(
    subtype_scores: pd.DataFrame,
    subtypes: List[str] = ["SCLC_A", "SCLC_N", "SCLC_P", "SCLC_I"]
) -> pd.Series:
    """
    Assign samples to highest-scoring subtype.

    Args:
        subtype_scores: DataFrame of subtype scores
        subtypes: List of subtype column names

    Returns:
        Series of subtype assignments
    """
    available_subtypes = [s for s in subtypes if s in subtype_scores.columns]
    return subtype_scores[available_subtypes].idxmax(axis=1)


def score_sclc_subtypes(
    expression: pd.DataFrame,
    signatures_config: Dict[str, Any],
    output_dir: Path,
    method: str = "ssgsea",
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Score SCLC subtypes (A/N/P/I) for all samples.

    Args:
        expression: Expression matrix (genes x samples)
        signatures_config: Signatures configuration
        output_dir: Output directory
        method: Scoring method
        logger: Optional logger

    Returns:
        Results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "stage": "subtype_scoring",
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "success": False
    }

    try:
        if logger:
            logger.info("Scoring SCLC subtypes...")

        # Get subtype signatures
        subtype_sigs = signatures_config.get("sclc_subtypes", {})

        # Score each subtype
        subtype_scores = score_signatures(
            expression, subtype_sigs, method=method, logger=logger
        )

        # Assign subtypes
        subtype_calls = assign_subtype(subtype_scores)

        # Save scores
        scores_path = output_dir / "subtype_scores.tsv"
        subtype_scores.to_csv(scores_path, sep='\t')

        # Save calls
        calls_df = pd.DataFrame({
            "sample": subtype_calls.index,
            "subtype": subtype_calls.values
        })
        calls_path = output_dir / "subtype_calls.tsv"
        calls_df.to_csv(calls_path, sep='\t', index=False)

        # Summary statistics
        subtype_counts = subtype_calls.value_counts()
        results["subtype_distribution"] = subtype_counts.to_dict()

        results["scores_file"] = str(scores_path)
        results["calls_file"] = str(calls_path)
        results["n_samples"] = len(subtype_calls)
        results["success"] = True

        if logger:
            logger.info(f"Subtype distribution: {subtype_counts.to_dict()}")

    except Exception as e:
        results["error"] = str(e)
        if logger:
            logger.error(f"Subtype scoring failed: {e}")

    return results


def score_immune_states(
    expression: pd.DataFrame,
    signatures_config: Dict[str, Any],
    output_dir: Path,
    method: str = "ssgsea",
    n_clusters: int = 4,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Score immune axis signatures and cluster into immune states.

    Args:
        expression: Expression matrix (genes x samples)
        signatures_config: Signatures configuration
        output_dir: Output directory
        method: Scoring method
        n_clusters: Number of immune state clusters
        logger: Optional logger

    Returns:
        Results dictionary
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "stage": "immune_scoring",
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "success": False
    }

    try:
        if logger:
            logger.info("Scoring immune signatures...")

        # Get immune signatures
        immune_sigs = signatures_config.get("immune_signatures", {})

        # Score each immune axis
        immune_scores = score_signatures(
            expression, immune_sigs, method=method, logger=logger
        )

        # Save scores
        scores_path = output_dir / "immune_scores.tsv"
        immune_scores.to_csv(scores_path, sep='\t')

        # Cluster into immune states
        if logger:
            logger.info(f"Clustering into {n_clusters} immune states...")

        # Prepare data for clustering
        score_matrix = immune_scores.dropna()
        scaler = StandardScaler()
        scaled_scores = scaler.fit_transform(score_matrix)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_scores)

        # Create immune state labels
        immune_states = pd.Series(
            [f"ImmuneState_{c+1}" for c in clusters],
            index=score_matrix.index
        )

        # Characterize states by mean scores
        state_profiles = pd.DataFrame(
            scaled_scores,
            index=score_matrix.index,
            columns=score_matrix.columns
        )
        state_profiles['immune_state'] = immune_states
        state_means = state_profiles.groupby('immune_state').mean()

        # Name states based on dominant features
        state_names = {}
        for state in state_means.index:
            top_feature = state_means.loc[state].idxmax()
            state_names[state] = f"{state}_{top_feature[:10]}"

        # Save states
        states_df = pd.DataFrame({
            "sample": immune_states.index,
            "immune_state": immune_states.values,
            "immune_state_named": [state_names.get(s, s) for s in immune_states.values]
        })
        states_path = output_dir / "immune_states.tsv"
        states_df.to_csv(states_path, sep='\t', index=False)

        # Save state profiles
        profiles_path = output_dir / "immune_state_profiles.tsv"
        state_means.to_csv(profiles_path, sep='\t')

        results["scores_file"] = str(scores_path)
        results["states_file"] = str(states_path)
        results["profiles_file"] = str(profiles_path)
        results["n_samples"] = len(immune_states)
        results["n_states"] = n_clusters
        results["state_distribution"] = immune_states.value_counts().to_dict()
        results["success"] = True

        if logger:
            logger.info(f"Immune state distribution: {immune_states.value_counts().to_dict()}")

    except Exception as e:
        results["error"] = str(e)
        if logger:
            logger.error(f"Immune scoring failed: {e}")

    return results
