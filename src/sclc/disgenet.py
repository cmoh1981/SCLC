"""
Stage 7b: DisGeNET Evidence Module

Functions for:
- Querying DisGeNET API for gene-disease associations
- Caching responses
- Filtering to SCLC/lung cancer/immune relevant diseases
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import requests


def get_disgenet_api_key() -> Optional[str]:
    """
    Get DisGeNET API key from environment.
    Never log or store the actual key value.

    Returns:
        API key or None
    """
    return os.environ.get("DISGENET_API_KEY")


def get_cache_path(gene: str, cache_dir: Path) -> Path:
    """Get cache file path for a gene query."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Hash the gene name for filename
    gene_hash = hashlib.md5(gene.encode()).hexdigest()[:8]
    return cache_dir / f"disgenet_{gene}_{gene_hash}.json"


def load_from_cache(cache_path: Path, max_age_days: int = 30) -> Optional[Dict]:
    """
    Load cached DisGeNET response if valid.

    Args:
        cache_path: Path to cache file
        max_age_days: Maximum age of cache in days

    Returns:
        Cached data or None if expired/missing
    """
    if not cache_path.exists():
        return None

    # Check age
    file_age = time.time() - cache_path.stat().st_mtime
    if file_age > max_age_days * 24 * 3600:
        return None

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None


def save_to_cache(data: Dict, cache_path: Path) -> None:
    """Save DisGeNET response to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def query_disgenet_gene(
    gene_symbol: str,
    api_key: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Query DisGeNET for gene-disease associations.

    Args:
        gene_symbol: HGNC gene symbol
        api_key: DisGeNET API key (from env if not provided)
        cache_dir: Directory for caching responses
        logger: Optional logger

    Returns:
        Dictionary with gene-disease associations
    """
    result = {
        "gene": gene_symbol,
        "success": False,
        "associations": [],
        "error": None
    }

    # Check cache first
    if cache_dir:
        cache_path = get_cache_path(gene_symbol, cache_dir)
        cached = load_from_cache(cache_path)
        if cached:
            if logger:
                logger.debug(f"Cache hit for {gene_symbol}")
            return cached

    # Get API key
    if api_key is None:
        api_key = get_disgenet_api_key()

    if api_key is None:
        result["error"] = "DISGENET_API_KEY not set in environment"
        if logger:
            logger.warning(f"DisGeNET API key not available for {gene_symbol}")
        return result

    # Query API
    base_url = "https://www.disgenet.org/api"
    endpoint = f"/gda/gene/{gene_symbol}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    try:
        if logger:
            logger.info(f"Querying DisGeNET for {gene_symbol}")

        response = requests.get(
            f"{base_url}{endpoint}",
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            associations = []
            for item in data:
                associations.append({
                    "disease_id": item.get("diseaseid", ""),
                    "disease_name": item.get("disease_name", ""),
                    "disease_class": item.get("disease_class_name", ""),
                    "score": item.get("score", 0),
                    "ei": item.get("ei", 0),
                    "el": item.get("el", ""),
                    "n_pmids": item.get("NofPmids", 0),
                    "source": item.get("source", "")
                })

            result["associations"] = associations
            result["n_associations"] = len(associations)
            result["success"] = True

        elif response.status_code == 404:
            result["success"] = True
            result["associations"] = []
            result["n_associations"] = 0

        else:
            result["error"] = f"HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        result["error"] = str(e)
        if logger:
            logger.error(f"DisGeNET query failed for {gene_symbol}: {e}")

    # Cache successful response
    if result["success"] and cache_dir:
        save_to_cache(result, cache_path)

    return result


def filter_sclc_relevant(
    associations: List[Dict],
    keywords: List[str] = None
) -> List[Dict]:
    """
    Filter associations to SCLC/lung cancer/immune relevant diseases.

    Args:
        associations: List of gene-disease associations
        keywords: Keywords to filter by (case-insensitive)

    Returns:
        Filtered associations
    """
    if keywords is None:
        keywords = [
            "lung", "sclc", "small cell",
            "carcinoma", "neoplasm", "cancer", "tumor",
            "immune", "autoimmune", "inflammation",
            "checkpoint", "pd-1", "pd-l1",
            "chemotherapy", "resistance", "platinum"
        ]

    keywords_lower = [k.lower() for k in keywords]

    filtered = []
    for assoc in associations:
        disease_name = assoc.get("disease_name", "").lower()
        disease_class = assoc.get("disease_class", "").lower()

        for kw in keywords_lower:
            if kw in disease_name or kw in disease_class:
                filtered.append(assoc)
                break

    return filtered


def run_disgenet_evidence(
    hub_genes_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run DisGeNET evidence gathering for hub genes.

    Args:
        hub_genes_path: Path to hub genes TSV
        output_dir: Output directory
        config: Configuration dictionary
        logger: Optional logger

    Returns:
        Results dictionary
    """
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / "cache"

    results = {
        "stage": "disgenet_evidence",
        "timestamp": datetime.now().isoformat(),
        "success": False
    }

    try:
        # Load hub genes
        hub_genes = pd.read_csv(hub_genes_path, sep='\t')
        gene_list = hub_genes['gene'].unique().tolist()

        if logger:
            logger.info(f"Querying DisGeNET for {len(gene_list)} hub genes")

        # Query each gene
        all_associations = []
        for gene in gene_list:
            gene_result = query_disgenet_gene(
                gene,
                cache_dir=cache_dir,
                logger=logger
            )

            if gene_result["success"]:
                for assoc in gene_result.get("associations", []):
                    assoc["gene"] = gene
                    all_associations.append(assoc)

            # Rate limiting
            time.sleep(0.5)

        # Save all associations
        if all_associations:
            all_df = pd.DataFrame(all_associations)
            all_path = output_dir / "hubgene_disease_evidence.tsv"
            all_df.to_csv(all_path, sep='\t', index=False)
            results["all_evidence_file"] = str(all_path)
            results["total_associations"] = len(all_associations)

            # Filter to SCLC-relevant
            filtered = filter_sclc_relevant(all_associations)
            if filtered:
                filtered_df = pd.DataFrame(filtered)
                filtered_path = output_dir / "sclc_filtered_evidence.tsv"
                filtered_df.to_csv(filtered_path, sep='\t', index=False)
                results["filtered_evidence_file"] = str(filtered_path)
                results["sclc_relevant_associations"] = len(filtered)

                if logger:
                    logger.info(f"Found {len(filtered)} SCLC-relevant associations")

        results["n_genes_queried"] = len(gene_list)
        results["success"] = True

        if logger:
            logger.info("DisGeNET evidence gathering complete")

    except Exception as e:
        results["error"] = str(e)
        if logger:
            logger.error(f"DisGeNET evidence failed: {e}")

    return results
