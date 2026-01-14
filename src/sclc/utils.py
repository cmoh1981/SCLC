"""
Utility functions for SCLC pipeline.
Logging, manifest writing, checksums, and common operations.
"""

import logging
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import platform


def setup_logging(
    stage_name: str,
    log_dir: Path,
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up logging for a pipeline stage.

    Args:
        stage_name: Name of the stage (e.g., "stage0", "stage1")
        log_dir: Directory for log files
        level: Logging level

    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{stage_name}.log"

    # Create logger
    logger = logging.getLogger(stage_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, level.upper()))
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(file_format)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized for {stage_name}")
    logger.info(f"Log file: {log_file}")

    return logger


def get_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """
    Calculate hash of a file.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm (md5, sha256, etc.)

    Returns:
        Hex digest of file hash
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    hasher = hashlib.new(algorithm)

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_software_versions() -> Dict[str, str]:
    """Get versions of key software packages."""
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    # Try to get package versions
    packages = [
        "numpy", "pandas", "scipy", "scanpy", "anndata",
        "gseapy", "seaborn", "matplotlib", "requests", "yaml"
    ]

    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"

    return versions


def write_manifest(
    output_path: Path,
    stage_name: str,
    parameters: Dict[str, Any],
    input_files: List[Path],
    output_files: List[Path],
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Write a manifest JSON file documenting stage execution.

    Args:
        output_path: Directory or file path for manifest
        stage_name: Name of the pipeline stage
        parameters: Parameters used in this stage
        input_files: List of input file paths
        output_files: List of output file paths
        logger: Optional logger for messages

    Returns:
        Path to written manifest file
    """
    output_path = Path(output_path)

    # Determine manifest path
    if output_path.is_dir():
        manifest_path = output_path / f"{stage_name}_manifest.json"
    else:
        manifest_path = output_path.with_suffix('.manifest.json')

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest = {
        "stage": stage_name,
        "timestamp": datetime.now().isoformat(),
        "software_versions": get_software_versions(),
        "parameters": parameters,
        "inputs": {},
        "outputs": {}
    }

    # Add input file info
    for inp in input_files:
        inp = Path(inp)
        if inp.exists():
            manifest["inputs"][str(inp.name)] = {
                "path": str(inp),
                "hash": get_file_hash(inp) if inp.is_file() else "directory"
            }

    # Add output file info
    for out in output_files:
        out = Path(out)
        if out.exists():
            manifest["outputs"][str(out.name)] = {
                "path": str(out),
                "hash": get_file_hash(out) if out.is_file() else "directory"
            }

    # Write manifest
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)

    if logger:
        logger.info(f"Manifest written to {manifest_path}")

    return manifest_path


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as filename."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name


def read_tsv(filepath: Path) -> 'pd.DataFrame':
    """Read a TSV file into DataFrame."""
    import pandas as pd
    return pd.read_csv(filepath, sep='\t')


def write_tsv(df: 'pd.DataFrame', filepath: Path) -> None:
    """Write DataFrame to TSV file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, sep='\t', index=False)


def save_figure(
    fig,
    filepath: Path,
    dpi: int = 300,
    formats: List[str] = None
) -> List[Path]:
    """
    Save figure in specified formats.

    Args:
        fig: Matplotlib figure
        filepath: Base path for figure (extension determines format)
        dpi: Resolution for raster formats
        formats: List of formats to save (default: from extension)

    Returns:
        List of saved file paths
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = [filepath.suffix.lstrip('.') or 'png']

    saved_paths = []

    for fmt in formats:
        save_path = filepath.with_suffix(f'.{fmt}')
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        saved_paths.append(save_path)

    return saved_paths
