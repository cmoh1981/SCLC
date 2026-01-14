"""
SCLC Chemo-IO Resistance Analysis Pipeline

A reproducible, config-driven pipeline for investigating primary resistance
to platinum-etoposide + PD-L1 chemo-immunotherapy in small cell lung cancer.

Core hypothesis: Immune-state stratification explains resistance across
SCLC A/N/P/I subtypes; nominate add-on repurposed drugs.
"""

__version__ = "1.0.0"
__author__ = "SCLC Research Team"

from .config import load_config, load_cohorts, load_signatures
from .utils import setup_logging, write_manifest, get_file_hash
