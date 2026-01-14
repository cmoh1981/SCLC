"""
Stage 0: Asset Bootstrap Module

Functions for:
- Cloning and pinning GitHub tools
- Recording version locks
- Creating asset inventory
- License extraction
"""

import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime

from .config import PipelineConfig, get_project_root
from .utils import write_manifest


def clone_github_repo(
    url: str,
    dest_dir: Path,
    pin: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Clone a GitHub repository and optionally checkout a specific version.

    Args:
        url: GitHub repository URL
        dest_dir: Destination directory
        pin: Tag, branch, or commit to checkout (None = latest)
        logger: Optional logger

    Returns:
        Dictionary with clone info (commit hash, etc.)
    """
    dest_dir = Path(dest_dir)
    repo_name = url.rstrip('/').split('/')[-1].replace('.git', '')

    result = {
        "url": url,
        "repo_name": repo_name,
        "dest_path": str(dest_dir),
        "pin": pin,
        "success": False,
        "commit_hash": None,
        "error": None
    }

    try:
        # Clone if not exists
        if not dest_dir.exists():
            if logger:
                logger.info(f"Cloning {url} to {dest_dir}")

            clone_cmd = ["git", "clone", "--depth", "1"]

            if pin and pin not in ["pip", "latest"]:
                clone_cmd.extend(["--branch", pin])

            clone_cmd.extend([url, str(dest_dir)])

            subprocess.run(clone_cmd, check=True, capture_output=True, text=True)
        else:
            if logger:
                logger.info(f"Repository already exists at {dest_dir}")

        # Get current commit hash
        commit_result = subprocess.run(
            ["git", "-C", str(dest_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True
        )

        if commit_result.returncode == 0:
            result["commit_hash"] = commit_result.stdout.strip()

        result["success"] = True

    except subprocess.CalledProcessError as e:
        result["error"] = f"Git error: {e.stderr}"
        if logger:
            logger.error(f"Failed to clone {url}: {e.stderr}")

    except Exception as e:
        result["error"] = str(e)
        if logger:
            logger.error(f"Failed to clone {url}: {e}")

    return result


def extract_license(
    repo_dir: Path,
    licenses_dir: Path,
    repo_name: str,
    logger: Optional[logging.Logger] = None
) -> Optional[Path]:
    """
    Extract LICENSE file from repository to licenses directory.

    Args:
        repo_dir: Repository directory
        licenses_dir: Destination for licenses
        repo_name: Name for the license file
        logger: Optional logger

    Returns:
        Path to extracted license or None
    """
    repo_dir = Path(repo_dir)
    licenses_dir = Path(licenses_dir)
    licenses_dir.mkdir(parents=True, exist_ok=True)

    license_names = ["LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING"]

    for name in license_names:
        license_path = repo_dir / name
        if license_path.exists():
            dest_path = licenses_dir / f"{repo_name}_LICENSE.txt"
            shutil.copy(license_path, dest_path)

            if logger:
                logger.info(f"Extracted license for {repo_name}")

            return dest_path

    if logger:
        logger.warning(f"No license file found for {repo_name}")

    return None


def write_version_lock(
    tools: List[Dict[str, Any]],
    lock_file: Path,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Write tool version lock file.

    Args:
        tools: List of tool info dicts with commit hashes
        lock_file: Path to lock file
        logger: Optional logger
    """
    lock_file = Path(lock_file)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    lock_data = {
        "generated": datetime.now().isoformat(),
        "tools": {}
    }

    for tool in tools:
        if tool.get("success"):
            lock_data["tools"][tool["repo_name"]] = {
                "url": tool["url"],
                "pin": tool["pin"],
                "commit_hash": tool["commit_hash"],
                "local_path": tool["dest_path"]
            }

    with open(lock_file, 'w', encoding='utf-8') as f:
        json.dump(lock_data, f, indent=2)

    if logger:
        logger.info(f"Version lock written to {lock_file}")


def create_asset_inventory(
    config: PipelineConfig,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Create asset inventory TSV from configuration.

    Args:
        config: Pipeline configuration
        output_path: Path for inventory TSV
        logger: Optional logger

    Returns:
        Path to inventory file
    """
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # Process datasets from stage0_assets
    assets = config.assets
    datasets = assets.get("datasets", [])

    for ds in datasets:
        rows.append({
            "name": ds.get("name", "Unknown"),
            "kind": ds.get("kind", "unknown"),
            "portal": ds.get("portal", "unknown"),
            "accession": ds.get("accession", ""),
            "access": ds.get("access", "unknown"),
            "samples": ds.get("samples", ""),
            "status": ds.get("status", "pending"),
            "local_path": ds.get("local_path", ""),
            "reference": ds.get("reference", ""),
            "notes": ds.get("notes", "")
        })

    # Process GitHub tools
    tools = assets.get("tools_github", [])

    for tool in tools:
        rows.append({
            "name": tool.get("name", "Unknown"),
            "kind": "github_tool",
            "portal": "GitHub",
            "accession": tool.get("url", ""),
            "access": "open",
            "samples": "",
            "status": tool.get("status", "pending"),
            "local_path": tool.get("local_path", ""),
            "reference": "",
            "notes": tool.get("purpose", "")
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)

    if logger:
        logger.info(f"Asset inventory written to {output_path}")
        logger.info(f"Total assets: {len(rows)}")

    return output_path


def run_bootstrap(
    config: Optional[PipelineConfig] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run the complete Stage 0 bootstrap process.

    Args:
        config: Pipeline configuration (loaded if not provided)
        logger: Optional logger

    Returns:
        Dictionary with bootstrap results
    """
    if config is None:
        config = PipelineConfig()

    root = get_project_root()
    assets = config.assets

    results = {
        "stage": "stage0_bootstrap",
        "timestamp": datetime.now().isoformat(),
        "tools_cloned": [],
        "licenses_extracted": [],
        "errors": []
    }

    # Paths
    tools_dir = root / "external" / "tools"
    licenses_dir = root / "external" / "licenses"
    locks_dir = root / "configs" / "locks"
    tables_dir = root / "results" / "tables"

    # Clone GitHub tools
    github_tools = assets.get("tools_github", [])

    for tool in github_tools:
        repo_name = tool.get("name", "").split('/')[-1]
        dest_dir = tools_dir / repo_name

        clone_result = clone_github_repo(
            url=tool.get("url", ""),
            dest_dir=dest_dir,
            pin=tool.get("pin"),
            logger=logger
        )

        results["tools_cloned"].append(clone_result)

        # Extract license
        if clone_result["success"]:
            license_path = extract_license(
                dest_dir, licenses_dir, repo_name, logger
            )
            if license_path:
                results["licenses_extracted"].append(str(license_path))

    # Write version lock
    lock_file = locks_dir / "tool_versions.lock"
    write_version_lock(results["tools_cloned"], lock_file, logger)

    # Create asset inventory
    inventory_path = tables_dir / "stage0_asset_inventory.tsv"
    create_asset_inventory(config, inventory_path, logger)

    results["inventory_path"] = str(inventory_path)
    results["lock_file"] = str(lock_file)

    return results
