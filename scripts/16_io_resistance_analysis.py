#!/usr/bin/env python
"""
Stage 16: Immunotherapy Resistance Mechanism Analysis for SCLC.

This script analyzes molecular mechanisms of IO resistance:
1. Antigen presentation defects
2. T-cell exhaustion signatures
3. Immunosuppressive microenvironment
4. Subtype-specific resistance patterns
5. Therapeutic strategies to overcome resistance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.io_resistance import run_io_resistance_analysis


def main():
    """Run IO resistance mechanism analysis."""
    root = Path(__file__).parent.parent

    # Use full expression data
    expression_path = root / 'data/processed/bulk/expression_matrix.tsv'
    subtype_path = root / 'results/subtypes/subtype_calls.tsv'
    output_dir = root / 'results/io_resistance'

    results = run_io_resistance_analysis(
        expression_path=expression_path,
        subtype_path=subtype_path,
        output_dir=output_dir
    )

    print(f"\nSignature scores shape: {results['signatures'].shape}")
    print(f"Resistance mechanisms identified for {len(results['mechanisms'])} subtypes")
    print(f"Therapeutic strategies mapped for {len(results['therapeutics'])} subtypes")


if __name__ == "__main__":
    main()
