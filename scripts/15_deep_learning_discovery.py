#!/usr/bin/env python
"""
Stage 15: Deep Learning-Based Novel Target and Drug Discovery for SCLC.

This script performs:
1. Novel target gene discovery using VAE and attention mechanisms
2. Novel drug candidate prediction and ranking
3. In silico validation (ADMET, binding affinity, selectivity)
4. Subtype-specific novel therapeutic recommendations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sclc.deep_learning import run_deep_learning_analysis


def main():
    """Run deep learning analysis for novel target and drug discovery."""
    root = Path(__file__).parent.parent

    expression_path = root / 'results/subtypes/subtype_scores.tsv'
    subtype_path = root / 'results/subtypes/subtype_calls.tsv'
    output_dir = root / 'results/deep_learning'

    # Check for full expression data
    full_expr_path = root / 'data/processed/bulk/expression_matrix.tsv'
    if full_expr_path.exists():
        expression_path = full_expr_path
        print(f"Using full expression data: {expression_path}")

    results = run_deep_learning_analysis(
        expression_path=expression_path,
        subtype_path=subtype_path,
        output_dir=output_dir
    )

    print(f"\nTotal novel targets discovered: {len(results['novel_targets'])}")
    print(f"Validated targets: {len(results['validated_targets'])}")
    print(f"Novel drug candidates evaluated: {len(results['validated_drugs'])}")


if __name__ == "__main__":
    main()
