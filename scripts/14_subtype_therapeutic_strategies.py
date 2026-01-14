#!/usr/bin/env python
"""
Stage 14: Subtype-Specific Therapeutic Strategy Analysis for SCLC.

Maps SCLC molecular subtypes to tailored therapeutic recommendations based on:
1. Subtype-specific gene expression patterns
2. Known biological vulnerabilities
3. Drug-gene interactions from DGIdb
4. Metabolic dependencies
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import json
from collections import defaultdict

# Subtype-specific therapeutic knowledge base
SUBTYPE_BIOLOGY = {
    'SCLC_A': {
        'name': 'SCLC-A (ASCL1-high)',
        'biology': 'Classical neuroendocrine, high DLL3, MYC-driven proliferation',
        'key_genes': ['ASCL1', 'DLL3', 'MYC', 'AURKA', 'BCL2', 'NOTCH1', 'SOX2'],
        'vulnerabilities': [
            'DLL3 surface expression → Rovalpituzumab tesirine (Rova-T)',
            'BCL2 overexpression → BCL2 inhibitors (venetoclax)',
            'Aurora kinase dependency → AURKA inhibitors (alisertib)',
            'Notch pathway inactive → DLL3-targeting ADCs',
            'High proliferation → Topoisomerase inhibitors (lurbinectedin)',
        ],
        'recommended_drugs': [
            ('Rovalpituzumab tesirine', 'DLL3-targeting ADC', 'Phase III completed'),
            ('Alisertib', 'Aurora A inhibitor', 'Phase II'),
            ('Venetoclax', 'BCL2 inhibitor', 'Phase I/II'),
            ('Lurbinectedin', 'Transcription inhibitor', 'FDA approved'),
            ('Tarlatamab', 'DLL3xCD3 bispecific', 'FDA approved 2024'),
        ],
        'metabolic_targets': ['OXPHOS', 'Glutaminolysis'],
        'io_sensitivity': 'Low - poor immune infiltration',
    },
    'SCLC_N': {
        'name': 'SCLC-N (NEUROD1-high)',
        'biology': 'Neuroendocrine variant, MYCN amplification, neural differentiation',
        'key_genes': ['NEUROD1', 'MYCN', 'AURKA', 'AURKB', 'MYC', 'IGF1R', 'SOX2'],
        'vulnerabilities': [
            'MYCN amplification → Aurora kinase inhibitors',
            'MYC/MYCN dependency → BET inhibitors (JQ1)',
            'IGF1R signaling → IGF1R inhibitors',
            'High mitotic rate → PLK1 inhibitors',
            'DNA damage response defects → PARP inhibitors',
        ],
        'recommended_drugs': [
            ('Alisertib', 'Aurora A inhibitor', 'Stabilizes MYCN'),
            ('Olaparib', 'PARP inhibitor', 'DDR defects'),
            ('Volasertib', 'PLK1 inhibitor', 'Phase II'),
            ('JQ1/OTX015', 'BET inhibitor', 'Preclinical'),
            ('Linsitinib', 'IGF1R inhibitor', 'Preclinical'),
        ],
        'metabolic_targets': ['OXPHOS', 'Lipid synthesis'],
        'io_sensitivity': 'Low - neuroendocrine phenotype',
    },
    'SCLC_P': {
        'name': 'SCLC-P (POU2F3-high)',
        'biology': 'Tuft cell-like, non-neuroendocrine, chemoresistant',
        'key_genes': ['POU2F3', 'SOX9', 'TRPM5', 'IGF1R', 'FGFR1', 'NOTCH1', 'REST'],
        'vulnerabilities': [
            'IGF1R overexpression → IGF1R inhibitors',
            'FGFR1 amplification → FGFR inhibitors (erdafitinib)',
            'Notch pathway active → Notch inhibitors',
            'Chemoresistant → Alternative mechanisms needed',
            'Unique tuft cell biology → Novel targets',
        ],
        'recommended_drugs': [
            ('Erdafitinib', 'FGFR inhibitor', 'FDA approved other'),
            ('Linsitinib', 'IGF1R inhibitor', 'Phase II'),
            ('Nirogacestat', 'Gamma-secretase inhibitor', 'Notch pathway'),
            ('Trilaciclib', 'CDK4/6 inhibitor', 'Myeloprotection'),
            ('Temozolomide', 'Alkylating agent', 'Alternative chemo'),
        ],
        'metabolic_targets': ['Glycolysis', 'One-carbon metabolism'],
        'io_sensitivity': 'Moderate - variable immune infiltration',
    },
    'SCLC_I': {
        'name': 'SCLC-I (Inflamed)',
        'biology': 'Low neuroendocrine, high immune infiltration, T-cell inflamed',
        'key_genes': ['CD274', 'PDCD1LG2', 'IDO1', 'STAT1', 'IRF1', 'HLA-A', 'CXCL10'],
        'vulnerabilities': [
            'High PD-L1/PD-L2 → Immune checkpoint inhibitors',
            'T-cell infiltration → Enhance with anti-CTLA4',
            'IFN-γ signature → Biomarker for IO response',
            'Antigen presentation intact → Tumor vaccines',
            'LAG3/TIGIT expression → Next-gen checkpoints',
        ],
        'recommended_drugs': [
            ('Atezolizumab', 'PD-L1 inhibitor', 'FDA approved SCLC'),
            ('Durvalumab', 'PD-L1 inhibitor', 'FDA approved SCLC'),
            ('Ipilimumab', 'CTLA-4 inhibitor', 'Combination'),
            ('Tiragolumab', 'TIGIT inhibitor', 'Phase III'),
            ('Relatlimab', 'LAG-3 inhibitor', 'Phase II'),
        ],
        'metabolic_targets': ['IDO1', 'Tryptophan metabolism'],
        'io_sensitivity': 'High - best IO responders',
    },
}


def calculate_subtype_gene_expression(expression_df, subtype_df):
    """Calculate mean expression of key genes per subtype."""
    # Get subtype assignments
    subtype_calls = subtype_df.set_index(subtype_df.columns[0])['subtype'].to_dict()

    results = {}
    for subtype in ['SCLC_A', 'SCLC_N', 'SCLC_P', 'SCLC_I']:
        subtype_samples = [s for s, st in subtype_calls.items() if st == subtype]
        if subtype_samples:
            key_genes = SUBTYPE_BIOLOGY[subtype]['key_genes']
            available_genes = [g for g in key_genes if g in expression_df.index]

            if available_genes and subtype_samples:
                valid_samples = [s for s in subtype_samples if s in expression_df.columns]
                if valid_samples:
                    expr_subset = expression_df.loc[available_genes, valid_samples]
                    results[subtype] = expr_subset.mean(axis=1).to_dict()

    return results


def map_drugs_to_subtypes(drug_df, interactions_df):
    """Map drugs to subtypes based on target genes."""
    subtype_drugs = defaultdict(list)

    for subtype, info in SUBTYPE_BIOLOGY.items():
        key_genes = set(info['key_genes'])

        for _, row in drug_df.iterrows():
            drug_name = row['drug_name']
            target_genes = set(str(row['target_genes']).split(','))

            overlap = key_genes & target_genes
            if overlap:
                subtype_drugs[subtype].append({
                    'drug': drug_name,
                    'targets': list(overlap),
                    'n_targets': len(overlap),
                    'score': row.get('composite_score', row.get('target_score', 0)),
                })

    # Sort by number of targets and score
    for subtype in subtype_drugs:
        subtype_drugs[subtype].sort(key=lambda x: (-x['n_targets'], -x['score']))

    return dict(subtype_drugs)


def create_therapeutic_strategy_table(subtype_drugs):
    """Create summary table of therapeutic strategies."""
    rows = []

    for subtype, info in SUBTYPE_BIOLOGY.items():
        # Get curated recommendations
        for drug, mechanism, status in info['recommended_drugs']:
            rows.append({
                'Subtype': info['name'],
                'Drug': drug,
                'Mechanism': mechanism,
                'Development Status': status,
                'Source': 'Literature-curated',
                'Rationale': info['biology'],
            })

        # Add DGIdb-discovered drugs (top 3)
        if subtype in subtype_drugs:
            for drug_info in subtype_drugs[subtype][:3]:
                rows.append({
                    'Subtype': info['name'],
                    'Drug': drug_info['drug'],
                    'Mechanism': f"Targets: {', '.join(drug_info['targets'])}",
                    'Development Status': 'DGIdb match',
                    'Source': 'Computational',
                    'Rationale': f"Targets {drug_info['n_targets']} subtype-specific genes",
                })

    return pd.DataFrame(rows)


def create_strategy_summary():
    """Create overall strategy summary."""
    summary = []

    for subtype, info in SUBTYPE_BIOLOGY.items():
        summary.append({
            'Subtype': info['name'],
            'Biology': info['biology'],
            'IO_Sensitivity': info['io_sensitivity'],
            'Key_Vulnerabilities': '; '.join(info['vulnerabilities'][:3]),
            'Primary_Drugs': ', '.join([d[0] for d in info['recommended_drugs'][:3]]),
            'Metabolic_Targets': ', '.join(info['metabolic_targets']),
        })

    return pd.DataFrame(summary)


def main():
    """Run subtype-specific therapeutic strategy analysis."""
    root = Path(__file__).parent.parent
    output_dir = root / 'results' / 'therapeutic_strategies'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 14: Subtype-Specific Therapeutic Strategies")
    print("=" * 60)

    # Load data
    try:
        drug_df = pd.read_csv(root / 'results/drugs/top_drugs_summary.tsv', sep='\t')
        interactions_df = pd.read_csv(root / 'results/drugs/dgidb_interactions.tsv', sep='\t')
        subtype_df = pd.read_csv(root / 'results/subtypes/subtype_calls.tsv', sep='\t')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Map drugs to subtypes
    print("\n1. Mapping drugs to SCLC subtypes...")
    subtype_drugs = map_drugs_to_subtypes(drug_df, interactions_df)

    for subtype, drugs in subtype_drugs.items():
        print(f"   {subtype}: {len(drugs)} candidate drugs")

    # Create therapeutic strategy table
    print("\n2. Creating therapeutic strategy table...")
    strategy_table = create_therapeutic_strategy_table(subtype_drugs)
    strategy_table.to_csv(output_dir / 'subtype_therapeutic_strategies.tsv', sep='\t', index=False)
    print(f"   Saved {len(strategy_table)} drug-subtype recommendations")

    # Create summary
    print("\n3. Creating strategy summary...")
    summary_df = create_strategy_summary()
    summary_df.to_csv(output_dir / 'strategy_summary.tsv', sep='\t', index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("SUBTYPE-SPECIFIC THERAPEUTIC STRATEGIES")
    print("=" * 60)

    for _, row in summary_df.iterrows():
        print(f"\n{row['Subtype']}")
        print(f"  Biology: {row['Biology']}")
        print(f"  IO Sensitivity: {row['IO_Sensitivity']}")
        print(f"  Primary Drugs: {row['Primary_Drugs']}")
        print(f"  Metabolic Targets: {row['Metabolic_Targets']}")

    # Save detailed JSON
    output_json = {
        'subtype_biology': SUBTYPE_BIOLOGY,
        'drug_mappings': subtype_drugs,
    }

    with open(output_dir / 'therapeutic_strategies.json', 'w') as f:
        json.dump(output_json, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Stage 14 Complete")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
