"""
Stage 7: Resistance Module Discovery

Functions for:
- WGCNA-style co-expression network analysis
- Module-trait correlation
- Hub gene identification
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from datetime import datetime
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def compute_correlation_matrix(
    expression: pd.DataFrame,
    method: str = "pearson"
) -> pd.DataFrame:
    """
    Compute gene-gene correlation matrix.

    Args:
        expression: Expression matrix (genes x samples)
        method: Correlation method (pearson, spearman)

    Returns:
        Correlation matrix (genes x genes)
    """
    if method == "spearman":
        corr_matrix = expression.T.corr(method='spearman')
    else:
        corr_matrix = expression.T.corr(method='pearson')

    return corr_matrix


def soft_threshold_power(
    corr_matrix: pd.DataFrame,
    powers: List[int] = None,
    r_squared_cutoff: float = 0.85
) -> int:
    """
    Determine soft-thresholding power for scale-free topology.

    Args:
        corr_matrix: Correlation matrix
        powers: Powers to test
        r_squared_cutoff: R-squared cutoff for scale-free fit

    Returns:
        Optimal soft-threshold power
    """
    if powers is None:
        powers = list(range(1, 21))

    # Take absolute correlation
    adj = np.abs(corr_matrix.values)

    best_power = 6  # default

    for power in powers:
        # Compute adjacency
        adj_powered = adj ** power
        np.fill_diagonal(adj_powered, 0)

        # Compute connectivity
        k = adj_powered.sum(axis=0)

        # Log-log fit for scale-free
        k_log = np.log10(k[k > 0] + 1)

        if len(k_log) > 10:
            # Histogram for scale-free fit
            hist, bins = np.histogram(k_log, bins=10)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            hist_log = np.log10(hist + 1)

            # Linear fit
            mask = hist > 0
            if mask.sum() > 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    bin_centers[mask], hist_log[mask]
                )

                if r_value ** 2 >= r_squared_cutoff:
                    best_power = power
                    break

    return best_power


def identify_modules(
    expression: pd.DataFrame,
    power: int = 6,
    min_module_size: int = 30,
    merge_threshold: float = 0.25,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Identify gene co-expression modules using hierarchical clustering.

    Args:
        expression: Expression matrix (genes x samples)
        power: Soft-threshold power
        min_module_size: Minimum genes per module
        merge_threshold: Distance threshold for merging modules
        logger: Optional logger

    Returns:
        Tuple of (module assignments, module eigengenes)
    """
    if logger:
        logger.info(f"Computing adjacency matrix with power={power}")

    # Correlation matrix
    corr = compute_correlation_matrix(expression)

    # Adjacency matrix (soft-thresholded)
    adj = np.abs(corr.values) ** power
    np.fill_diagonal(adj, 0)

    # Topological overlap matrix (simplified)
    # TOM = (adj @ adj + adj) / (outer_sum - adj + 1)
    connectivity = adj.sum(axis=0)
    outer_sum = np.add.outer(connectivity, connectivity)

    with np.errstate(divide='ignore', invalid='ignore'):
        tom = (adj @ adj + adj) / (outer_sum - adj + 1)
        tom = np.nan_to_num(tom, nan=0, posinf=0, neginf=0)

    np.fill_diagonal(tom, 1)

    # Distance for clustering
    dist = 1 - tom

    # Hierarchical clustering
    if logger:
        logger.info("Performing hierarchical clustering...")

    # Convert to condensed form
    dist_condensed = pdist(dist)
    dist_condensed = np.clip(dist_condensed, 0, None)  # Ensure non-negative

    Z = linkage(dist_condensed, method='average')

    # Cut tree to get modules
    # Use dynamic tree cut approximation
    max_clusters = max(2, len(expression) // min_module_size)
    modules = fcluster(Z, t=max_clusters, criterion='maxclust')

    # Filter small modules
    module_series = pd.Series(modules, index=expression.index)
    module_counts = module_series.value_counts()

    # Reassign small modules to "grey" (module 0)
    small_modules = module_counts[module_counts < min_module_size].index
    module_series[module_series.isin(small_modules)] = 0

    # Renumber modules
    unique_modules = sorted(module_series[module_series > 0].unique())
    module_map = {old: new for new, old in enumerate(unique_modules, 1)}
    module_map[0] = 0
    module_series = module_series.map(module_map)

    if logger:
        n_modules = len(module_series[module_series > 0].unique())
        logger.info(f"Identified {n_modules} modules")

    # Compute module eigengenes (first PC of each module)
    eigengenes = {}
    for mod in module_series.unique():
        if mod == 0:
            continue

        mod_genes = module_series[module_series == mod].index
        mod_expr = expression.loc[mod_genes]

        if len(mod_genes) > 1:
            # PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            eigengene = pca.fit_transform(mod_expr.T.values)
            eigengenes[f"ME{mod}"] = eigengene.flatten()

    eigengene_df = pd.DataFrame(eigengenes, index=expression.columns)

    return module_series, eigengene_df


def identify_hub_genes(
    expression: pd.DataFrame,
    modules: pd.Series,
    eigengenes: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Identify hub genes for each module.

    Hub genes have highest correlation with module eigengene.

    Args:
        expression: Expression matrix
        modules: Module assignments
        eigengenes: Module eigengenes
        top_n: Number of hub genes per module

    Returns:
        DataFrame of hub genes
    """
    hub_genes = []

    for mod in modules.unique():
        if mod == 0:
            continue

        me_col = f"ME{mod}"
        if me_col not in eigengenes.columns:
            continue

        mod_genes = modules[modules == mod].index
        me = eigengenes[me_col]

        # Correlation with eigengene
        correlations = {}
        for gene in mod_genes:
            if gene in expression.index:
                corr, _ = stats.pearsonr(expression.loc[gene], me)
                correlations[gene] = abs(corr)

        # Top hub genes
        sorted_genes = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        for rank, (gene, corr) in enumerate(sorted_genes[:top_n], 1):
            hub_genes.append({
                "gene": gene,
                "module": mod,
                "module_eigengene_corr": corr,
                "hub_rank": rank
            })

    return pd.DataFrame(hub_genes)


def correlate_modules_traits(
    eigengenes: pd.DataFrame,
    traits: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Correlate module eigengenes with sample traits.

    Args:
        eigengenes: Module eigengenes (samples x modules)
        traits: Trait matrix (samples x traits)

    Returns:
        Tuple of (correlation matrix, p-value matrix)
    """
    # Align samples
    common_samples = eigengenes.index.intersection(traits.index)
    eigengenes = eigengenes.loc[common_samples]
    traits = traits.loc[common_samples]

    corr_matrix = pd.DataFrame(
        index=eigengenes.columns,
        columns=traits.columns,
        dtype=float
    )
    pval_matrix = pd.DataFrame(
        index=eigengenes.columns,
        columns=traits.columns,
        dtype=float
    )

    for me in eigengenes.columns:
        for trait in traits.columns:
            try:
                corr, pval = stats.pearsonr(
                    eigengenes[me].astype(float),
                    pd.to_numeric(traits[trait], errors='coerce')
                )
                corr_matrix.loc[me, trait] = corr
                pval_matrix.loc[me, trait] = pval
            except:
                corr_matrix.loc[me, trait] = np.nan
                pval_matrix.loc[me, trait] = np.nan

    return corr_matrix, pval_matrix


def run_module_discovery(
    expression_path: Path,
    traits_path: Optional[Path],
    output_dir: Path,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run complete module discovery pipeline.

    Args:
        expression_path: Path to expression matrix
        traits_path: Path to traits matrix (optional)
        output_dir: Output directory
        config: Configuration dictionary
        logger: Optional logger

    Returns:
        Results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "stage": "module_discovery",
        "timestamp": datetime.now().isoformat(),
        "success": False
    }

    try:
        # Load expression
        expression = pd.read_csv(expression_path, sep='\t', index_col=0)

        if logger:
            logger.info(f"Loaded expression: {expression.shape}")

        # Identify modules
        power = config.get("soft_power", 6)
        min_size = config.get("min_module_size", 30)

        modules, eigengenes = identify_modules(
            expression,
            power=power,
            min_module_size=min_size,
            logger=logger
        )

        # Save module assignments
        modules_df = pd.DataFrame({
            "gene": modules.index,
            "module": modules.values
        })
        modules_path = output_dir / "module_genes.tsv"
        modules_df.to_csv(modules_path, sep='\t', index=False)

        # Save eigengenes
        eigengenes_path = output_dir / "module_eigengenes.tsv"
        eigengenes.to_csv(eigengenes_path, sep='\t')

        # Identify hub genes
        hub_genes = identify_hub_genes(
            expression, modules, eigengenes,
            top_n=config.get("top_hub_genes", 10)
        )
        hub_path = output_dir / "hub_genes.tsv"
        hub_genes.to_csv(hub_path, sep='\t', index=False)

        results["modules_file"] = str(modules_path)
        results["eigengenes_file"] = str(eigengenes_path)
        results["hub_genes_file"] = str(hub_path)
        results["n_modules"] = len(modules[modules > 0].unique())
        results["n_hub_genes"] = len(hub_genes)

        # Correlate with traits if provided
        if traits_path and Path(traits_path).exists():
            traits = pd.read_csv(traits_path, sep='\t', index_col=0)

            corr, pval = correlate_modules_traits(eigengenes, traits)

            corr_path = output_dir / "module_trait_correlation.tsv"
            corr.to_csv(corr_path, sep='\t')

            pval_path = output_dir / "module_trait_pvalue.tsv"
            pval.to_csv(pval_path, sep='\t')

            results["trait_correlation_file"] = str(corr_path)

            if logger:
                logger.info("Module-trait correlations computed")

        results["success"] = True

        if logger:
            logger.info(f"Module discovery complete: {results['n_modules']} modules")

    except Exception as e:
        results["error"] = str(e)
        if logger:
            logger.error(f"Module discovery failed: {e}")

    return results
