"""
Stage 1: Data Download Module

Functions for:
- Downloading open-access datasets (GEO, SRA, PRIDE, etc.)
- Downloading GEO supplementary files (expression matrices)
- Creating controlled-access scaffolds
- Checksum verification
"""

import os
import hashlib
import json
import gzip
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .config import PipelineConfig, get_project_root


def download_file_with_retry(url: str, dest_path: Path, max_retries: int = 3, logger: Optional[logging.Logger] = None) -> bool:
    """Download a file with retry logic."""
    for attempt in range(max_retries):
        try:
            if logger:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
            urllib.request.urlretrieve(url, str(dest_path))
            if dest_path.exists() and dest_path.stat().st_size > 0:
                return True
        except Exception as e:
            if logger:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    return False


def download_geo_supplementary(
    accession: str,
    dest_dir: Path,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """
    Download GEO supplementary files (expression matrices).

    Args:
        accession: GEO accession (e.g., GSE60052)
        dest_dir: Destination directory
        logger: Optional logger

    Returns:
        List of downloaded file paths
    """
    import GEOparse
    import tempfile

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    try:
        # Get GEO metadata to find supplementary files
        with tempfile.TemporaryDirectory() as tmpdir:
            gse = GEOparse.get_GEO(geo=accession, destdir=tmpdir, silent=True)
            suppl_files = gse.metadata.get('supplementary_file', [])

        if not suppl_files:
            if logger:
                logger.warning(f"No supplementary files found for {accession}")
            return downloaded

        if logger:
            logger.info(f"Found {len(suppl_files)} supplementary files for {accession}")

        for url in suppl_files:
            if not url.startswith('ftp://') and not url.startswith('http'):
                continue

            filename = url.split('/')[-1]
            dest_path = dest_dir / filename

            try:
                download_file_with_retry(url, dest_path, logger=logger)

                if dest_path.exists():
                    downloaded.append(dest_path)
                    if logger:
                        logger.info(f"Downloaded: {filename} ({dest_path.stat().st_size / 1024 / 1024:.2f} MB)")

                    # Extract tar files
                    if filename.endswith('.tar'):
                        extract_dir = dest_dir / filename.replace('.tar', '')
                        extract_dir.mkdir(exist_ok=True)
                        with tarfile.open(dest_path, 'r') as tar:
                            tar.extractall(extract_dir)
                        if logger:
                            logger.info(f"Extracted {filename} to {extract_dir}")

            except Exception as e:
                if logger:
                    logger.error(f"Failed to download {filename}: {e}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to get supplementary files for {accession}: {e}")

    return downloaded


def download_geo_dataset(
    accession: str,
    dest_dir: Path,
    include_supplementary: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Download a GEO dataset using GEOparse, including supplementary files.

    Args:
        accession: GEO accession (e.g., GSE60052)
        dest_dir: Destination directory
        include_supplementary: Whether to download supplementary files (expression data)
        logger: Optional logger

    Returns:
        Download result dictionary
    """
    import GEOparse

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "accession": accession,
        "portal": "GEO",
        "dest_dir": str(dest_dir),
        "success": False,
        "files": [],
        "supplementary_files": [],
        "error": None
    }

    try:
        if logger:
            logger.info(f"Downloading {accession} from GEO...")

        gse = GEOparse.get_GEO(
            geo=accession,
            destdir=str(dest_dir),
            silent=False
        )

        # Record SOFT files
        for f in dest_dir.glob(f"{accession}*"):
            result["files"].append({
                "path": str(f),
                "size": f.stat().st_size if f.is_file() else 0
            })

        # Download supplementary files (expression data)
        if include_supplementary:
            if logger:
                logger.info(f"Downloading supplementary files for {accession}...")
            suppl_dir = dest_dir / "supplementary"
            suppl_files = download_geo_supplementary(accession, suppl_dir, logger)
            for sf in suppl_files:
                result["supplementary_files"].append({
                    "path": str(sf),
                    "size": sf.stat().st_size if sf.is_file() else 0
                })

        # Save metadata summary
        metadata = {
            "accession": accession,
            "title": gse.metadata.get("title", [""])[0],
            "summary": gse.metadata.get("summary", [""])[0],
            "type": gse.metadata.get("type", [""])[0],
            "platform": list(gse.gpls.keys()),
            "n_samples": len(gse.gsms),
            "supplementary_urls": gse.metadata.get("supplementary_file", [])
        }

        meta_path = dest_dir / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        result["metadata"] = metadata
        result["success"] = True
        result["has_expression_data"] = len(result["supplementary_files"]) > 0

        if logger:
            logger.info(f"Successfully downloaded {accession}: {len(gse.gsms)} samples, {len(result['supplementary_files'])} supplementary files")

    except Exception as e:
        result["error"] = str(e)
        if logger:
            logger.error(f"Failed to download {accession}: {e}")

    return result


def download_sra_metadata(
    accession: str,
    dest_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Download SRA project metadata using pysradb.

    Args:
        accession: SRA BioProject accession (e.g., PRJNA575243)
        dest_dir: Destination directory
        logger: Optional logger

    Returns:
        Download result dictionary
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "accession": accession,
        "portal": "SRA",
        "dest_dir": str(dest_dir),
        "success": False,
        "files": [],
        "error": None
    }

    try:
        from pysradb.sraweb import SRAweb

        if logger:
            logger.info(f"Fetching SRA metadata for {accession}...")

        db = SRAweb()
        df = db.sra_metadata(accession)

        if df is not None and len(df) > 0:
            csv_path = dest_dir / f"{accession}_metadata.csv"
            df.to_csv(csv_path, index=False)

            result["files"].append({
                "path": str(csv_path),
                "size": csv_path.stat().st_size
            })

            result["n_runs"] = len(df)
            result["success"] = True

            if logger:
                logger.info(f"Downloaded metadata for {len(df)} SRA runs")

    except Exception as e:
        result["error"] = str(e)
        if logger:
            logger.error(f"Failed to download SRA metadata for {accession}: {e}")

    return result


def download_metabolomics_workbench(
    study_id: str,
    dest_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Download data from Metabolomics Workbench REST API.

    Args:
        study_id: Study ID (e.g., ST000220)
        dest_dir: Destination directory
        logger: Optional logger

    Returns:
        Download result dictionary
    """
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "accession": study_id,
        "portal": "MetabolomicsWorkbench",
        "dest_dir": str(dest_dir),
        "success": False,
        "files": [],
        "error": None
    }

    base_url = "https://www.metabolomicsworkbench.org/rest/study"
    endpoints = ["summary", "analysis", "metabolites", "factors", "mwtab"]

    try:
        if logger:
            logger.info(f"Downloading {study_id} from Metabolomics Workbench...")

        for endpoint in endpoints:
            url = f"{base_url}/study_id/{study_id}/{endpoint}"
            resp = requests.get(url, timeout=60)

            if resp.status_code == 200:
                if endpoint == "mwtab":
                    file_path = dest_dir / f"{study_id}.mwtab"
                else:
                    file_path = dest_dir / f"{endpoint}.json"

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(resp.text)

                result["files"].append({
                    "path": str(file_path),
                    "size": file_path.stat().st_size
                })

        result["success"] = len(result["files"]) > 0

        if logger:
            logger.info(f"Downloaded {len(result['files'])} files for {study_id}")

    except Exception as e:
        result["error"] = str(e)
        if logger:
            logger.error(f"Failed to download {study_id}: {e}")

    return result


def download_pride_metadata(
    accession: str,
    dest_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Download PRIDE project metadata via REST API.

    Args:
        accession: PRIDE accession (e.g., PXD052033)
        dest_dir: Destination directory
        logger: Optional logger

    Returns:
        Download result dictionary
    """
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "accession": accession,
        "portal": "PRIDE",
        "dest_dir": str(dest_dir),
        "success": False,
        "files": [],
        "error": None
    }

    api_base = "https://www.ebi.ac.uk/pride/ws/archive/v2"

    try:
        if logger:
            logger.info(f"Downloading PRIDE metadata for {accession}...")

        # Get project metadata
        project_url = f"{api_base}/projects/{accession}"
        resp = requests.get(project_url, timeout=60)

        if resp.status_code == 200:
            meta_path = dest_dir / "project_metadata.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(resp.json(), f, indent=2)
            result["files"].append({"path": str(meta_path)})

        # Get file list
        files_url = f"{api_base}/projects/{accession}/files"
        resp = requests.get(files_url, timeout=60)

        if resp.status_code == 200:
            files_data = resp.json()
            files_path = dest_dir / "files_list.json"
            with open(files_path, 'w', encoding='utf-8') as f:
                json.dump(files_data, f, indent=2)
            result["files"].append({"path": str(files_path)})

            # Write download links
            links_path = dest_dir / "download_links.txt"
            with open(links_path, 'w', encoding='utf-8') as f:
                f.write(f"# PRIDE Project: {accession}\n")
                f.write("# Download using Aspera or FTP\n\n")
                for file_info in files_data:
                    for loc in file_info.get('publicFileLocations', []):
                        f.write(f"{loc.get('value', '')}\n")
            result["files"].append({"path": str(links_path)})

            result["n_files"] = len(files_data)

        result["success"] = len(result["files"]) > 0

        if logger:
            logger.info(f"Downloaded metadata for {accession}")

    except Exception as e:
        result["error"] = str(e)
        if logger:
            logger.error(f"Failed to download PRIDE metadata for {accession}: {e}")

    return result


def write_controlled_access_instructions(
    controlled_datasets: List[Dict],
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Write scaffold instructions for controlled-access datasets.

    Args:
        controlled_datasets: List of controlled-access dataset configs
        output_path: Path for instructions file
        logger: Optional logger

    Returns:
        Path to instructions file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Controlled Access Dataset Instructions",
        "# =====================================",
        "",
        "The following datasets require controlled access applications.",
        "Do NOT attempt to download without proper authorization.",
        "",
    ]

    for ds in controlled_datasets:
        lines.extend([
            f"## {ds.get('name', 'Unknown')}",
            f"- Portal: {ds.get('portal', 'Unknown')}",
            f"- Accession: {ds.get('accession', 'Unknown')}",
            f"- Access: {ds.get('access', 'controlled')}",
            f"- Description: {ds.get('description', '')}",
            f"- Reference: {ds.get('reference', '')}",
            "",
            "### Application Process:",
        ])

        portal = ds.get('portal', '').upper()
        if portal == "EGA":
            lines.extend([
                "1. Register at https://ega-archive.org/register",
                "2. Submit Data Access Request to the DAC",
                "3. Use pyega3 for download after approval",
                f"   pyega3 fetch {ds.get('accession', '')}",
            ])
        elif portal == "NGDC":
            lines.extend([
                "1. Register at https://ngdc.cncb.ac.cn/",
                "2. Submit collaboration request",
                "3. Follow data sharing agreement",
            ])
        elif portal == "PDC":
            lines.extend([
                "1. Access via Proteomic Data Commons portal",
                "2. May require dbGaP application for raw data",
            ])

        lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    if logger:
        logger.info(f"Controlled access instructions written to {output_path}")

    return output_path


def compute_checksum(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def write_checksums(
    files: List[Path],
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Write checksums for downloaded files.

    Args:
        files: List of file paths
        output_path: Path for checksum file
        logger: Optional logger

    Returns:
        Path to checksum file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checksums = {}

    for f in files:
        f = Path(f)
        if f.exists() and f.is_file():
            checksums[str(f.name)] = {
                "path": str(f),
                "sha256": compute_checksum(f),
                "size": f.stat().st_size
            }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(checksums, f, indent=2)

    if logger:
        logger.info(f"Checksums written to {output_path}")

    return output_path


def run_download(
    config: Optional[PipelineConfig] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run the complete Stage 1 download process.

    Args:
        config: Pipeline configuration
        logger: Optional logger

    Returns:
        Download results dictionary
    """
    if config is None:
        config = PipelineConfig()

    root = get_project_root()
    raw_dir = root / "data" / "raw"
    logs_dir = root / "results" / "logs"

    results = {
        "stage": "stage1_download",
        "timestamp": datetime.now().isoformat(),
        "downloads": [],
        "errors": []
    }

    # Get open-access cohorts
    open_cohorts = config.get_open_access_cohorts()

    # Download GEO datasets
    for ds in open_cohorts.get("bulk_rna", []) + open_cohorts.get("scrna", []) + open_cohorts.get("spatial", []):
        if ds.get("portal") == "GEO":
            dest = raw_dir / "geo" / ds["accession"]
            result = download_geo_dataset(ds["accession"], dest, logger)
            results["downloads"].append(result)

    # Download SRA metadata
    for ds in open_cohorts.get("scrna", []) + open_cohorts.get("spatial", []):
        bioproject = ds.get("sra_bioproject")
        if bioproject:
            dest = raw_dir / "sra" / bioproject
            result = download_sra_metadata(bioproject, dest, logger)
            results["downloads"].append(result)

    # Download Metabolomics Workbench
    for ds in open_cohorts.get("metabolomics", []):
        if ds.get("portal") == "MetabolomicsWorkbench":
            dest = raw_dir / "metabolomics" / ds["accession"]
            result = download_metabolomics_workbench(ds["accession"], dest, logger)
            results["downloads"].append(result)

    # Download PRIDE metadata
    for ds in open_cohorts.get("proteomics", []):
        if ds.get("portal") == "PRIDE":
            dest = raw_dir / "pride" / ds["accession"]
            result = download_pride_metadata(ds["accession"], dest, logger)
            results["downloads"].append(result)

    # Write controlled access instructions
    controlled = config.get_controlled_cohorts()
    instructions_path = logs_dir / "controlled_access_instructions.txt"
    write_controlled_access_instructions(controlled, instructions_path, logger)
    results["controlled_access_instructions"] = str(instructions_path)

    return results
