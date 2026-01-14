#!/usr/bin/env python
"""
Immunotherapy Resistance Mechanism Analysis for SCLC.

This module analyzes molecular mechanisms of IO resistance:
1. Antigen presentation defects (HLA, B2M, TAP1/2)
2. T-cell exhaustion signatures
3. Immunosuppressive microenvironment
4. Interferon signaling defects
5. WNT/β-catenin pathway activation
6. Metabolic immune suppression
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
import json

# IO Resistance Gene Signatures
IO_RESISTANCE_SIGNATURES = {
    'antigen_presentation': {
        'genes': ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-E', 'HLA-F', 'HLA-G',
                  'B2M', 'TAP1', 'TAP2', 'TAPBP', 'PSMB8', 'PSMB9', 'PSMB10',
                  'CALR', 'CANX', 'HSPA5', 'HSP90B1'],
        'description': 'MHC class I antigen presentation machinery',
        'resistance_direction': 'low',  # Low = resistance
    },
    'hla_class_ii': {
        'genes': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1',
                  'CIITA', 'CD74', 'CTSS'],
        'description': 'MHC class II presentation',
        'resistance_direction': 'low',
    },
    't_cell_exhaustion': {
        'genes': ['PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'CTLA4', 'BTLA', 'CD160',
                  'CD244', 'ENTPD1', 'TOX', 'TOX2', 'NR4A1', 'NR4A2', 'NR4A3'],
        'description': 'T-cell exhaustion markers',
        'resistance_direction': 'high',  # High exhaustion = resistance
    },
    'treg_signature': {
        'genes': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'TNFRSF18', 'CCR8',
                  'LAYN', 'MAGEH1', 'CCR4', 'IL10'],
        'description': 'Regulatory T-cell markers',
        'resistance_direction': 'high',
    },
    'mdsc_signature': {
        'genes': ['S100A8', 'S100A9', 'S100A12', 'ARG1', 'ARG2', 'NOS2',
                  'IL4I1', 'PTGS2', 'IDO1', 'CD33', 'ITGAM'],
        'description': 'Myeloid-derived suppressor cells',
        'resistance_direction': 'high',
    },
    'tam_m2': {
        'genes': ['CD163', 'MRC1', 'CD68', 'MSR1', 'MARCO', 'SIGLEC1',
                  'CD209', 'CLEC7A', 'IL10', 'TGFB1', 'VEGFA'],
        'description': 'M2 tumor-associated macrophages',
        'resistance_direction': 'high',
    },
    'tgfb_signaling': {
        'genes': ['TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2', 'SMAD2',
                  'SMAD3', 'SMAD4', 'SMAD7', 'SERPINE1', 'COL1A1', 'ACTA2'],
        'description': 'TGF-beta immunosuppressive signaling',
        'resistance_direction': 'high',
    },
    'wnt_bcatenin': {
        'genes': ['CTNNB1', 'APC', 'AXIN1', 'AXIN2', 'GSK3B', 'WNT1', 'WNT2',
                  'WNT3A', 'WNT5A', 'LEF1', 'TCF7', 'MYC', 'CCND1'],
        'description': 'WNT/beta-catenin pathway (immune exclusion)',
        'resistance_direction': 'high',
    },
    'ifn_signaling': {
        'genes': ['IFNG', 'IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'STAT2',
                  'IRF1', 'IRF9', 'NLRC5', 'GBP1', 'GBP2', 'IDO1', 'CXCL9', 'CXCL10'],
        'description': 'Interferon-gamma signaling',
        'resistance_direction': 'low',  # Low IFN = resistance
    },
    'metabolic_immune_suppression': {
        'genes': ['IDO1', 'IDO2', 'TDO2', 'ADORA2A', 'NT5E', 'ENTPD1',
                  'ARG1', 'ARG2', 'NOS2', 'PTGS2', 'SLC7A11', 'GPX4'],
        'description': 'Metabolic pathways suppressing immunity',
        'resistance_direction': 'high',
    },
    'caf_exclusion': {
        'genes': ['FAP', 'PDPN', 'ACTA2', 'COL1A1', 'COL1A2', 'COL3A1',
                  'FN1', 'POSTN', 'CXCL12', 'TGFB1', 'TGFB2'],
        'description': 'Cancer-associated fibroblasts (T-cell exclusion)',
        'resistance_direction': 'high',
    },
    'angiogenesis': {
        'genes': ['VEGFA', 'VEGFB', 'VEGFC', 'FLT1', 'KDR', 'FLT4',
                  'ANGPT1', 'ANGPT2', 'TEK', 'PDGFA', 'PDGFB'],
        'description': 'Tumor angiogenesis (immunosuppressive)',
        'resistance_direction': 'high',
    },
}

# Subtype-specific expected resistance mechanisms
SUBTYPE_RESISTANCE_PROFILES = {
    'SCLC_A': {
        'primary_mechanisms': ['antigen_presentation', 'ifn_signaling', 'neuroendocrine'],
        'description': 'Low MHC-I, poor IFN signaling, neuroendocrine immune evasion',
        'io_sensitivity': 'Low',
    },
    'SCLC_N': {
        'primary_mechanisms': ['antigen_presentation', 'wnt_bcatenin', 'metabolic_immune_suppression'],
        'description': 'Low antigen presentation, WNT activation, metabolic suppression',
        'io_sensitivity': 'Low',
    },
    'SCLC_P': {
        'primary_mechanisms': ['tgfb_signaling', 'caf_exclusion', 'tam_m2'],
        'description': 'TGF-beta driven, T-cell exclusion, M2 macrophage infiltration',
        'io_sensitivity': 'Moderate',
    },
    'SCLC_I': {
        'primary_mechanisms': ['t_cell_exhaustion', 'treg_signature', 'adaptive_resistance'],
        'description': 'T-cell exhaustion, Treg infiltration, adaptive PD-L1',
        'io_sensitivity': 'High (but can develop resistance)',
    },
}


@dataclass
class ResistanceMechanism:
    """Represents an IO resistance mechanism."""
    name: str
    score: float
    genes_detected: List[str]
    subtype: str
    description: str
    therapeutic_target: str
    drugs_to_overcome: List[str]


def calculate_signature_scores(expression_df: pd.DataFrame,
                               signatures: Dict) -> pd.DataFrame:
    """Calculate signature scores for each sample."""
    scores = {}

    # Convert index to uppercase for matching
    expr_genes = set(expression_df.index.str.upper())

    for sig_name, sig_info in signatures.items():
        genes = sig_info['genes']
        # Match genes case-insensitively
        available_genes = []
        for g in genes:
            g_upper = g.upper()
            if g_upper in expr_genes:
                # Find the actual gene name in the index
                for idx_gene in expression_df.index:
                    if idx_gene.upper() == g_upper:
                        available_genes.append(idx_gene)
                        break
            # Also try with hyphen removed (e.g., HLA-A vs HLAA)
            g_nohyphen = g_upper.replace('-', '')
            if g_nohyphen != g_upper and g_nohyphen in expr_genes:
                for idx_gene in expression_df.index:
                    if idx_gene.upper().replace('-', '') == g_nohyphen:
                        available_genes.append(idx_gene)
                        break

        if len(available_genes) >= 2:
            # Z-score normalize and average
            sig_expr = expression_df.loc[available_genes]
            # Z-score across samples for each gene
            z_scores = sig_expr.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1)
            scores[sig_name] = z_scores.mean(axis=0)
        else:
            scores[sig_name] = pd.Series(0, index=expression_df.columns)

    return pd.DataFrame(scores)


def identify_resistance_mechanisms(expression_df: pd.DataFrame,
                                   subtype_df: pd.DataFrame) -> Dict:
    """Identify IO resistance mechanisms by subtype."""
    # Calculate signature scores
    sig_scores = calculate_signature_scores(expression_df, IO_RESISTANCE_SIGNATURES)

    # Clean sample names
    expression_df.columns = expression_df.columns.str.strip()
    subtype_df['sample'] = subtype_df['sample'].str.strip()

    # Match samples
    if sig_scores.index[0].startswith('SAMPLE_'):
        n_samples = min(len(sig_scores), len(subtype_df))
        sig_scores = sig_scores.iloc[:n_samples]
        subtype_labels = subtype_df['subtype'].iloc[:n_samples].values
    else:
        common = set(sig_scores.index) & set(subtype_df['sample'])
        if len(common) > 0:
            sig_scores = sig_scores.loc[list(common)]
            subtype_labels = subtype_df.set_index('sample').loc[list(common), 'subtype'].values
        else:
            subtype_labels = subtype_df['subtype'].iloc[:len(sig_scores)].values

    results = {}
    subtypes = ['SCLC_A', 'SCLC_N', 'SCLC_P', 'SCLC_I']

    for subtype in subtypes:
        mask = subtype_labels == subtype
        if mask.sum() == 0:
            continue

        subtype_scores = sig_scores[mask].mean()
        other_scores = sig_scores[~mask].mean()

        mechanisms = []
        for sig_name, sig_info in IO_RESISTANCE_SIGNATURES.items():
            score = subtype_scores[sig_name]
            other_score = other_scores[sig_name]
            diff = score - other_score

            # Determine if this is a resistance mechanism
            is_resistance = False
            if sig_info['resistance_direction'] == 'high' and diff > 0.3:
                is_resistance = True
            elif sig_info['resistance_direction'] == 'low' and diff < -0.3:
                is_resistance = True

            mechanisms.append({
                'signature': sig_name,
                'score': score,
                'vs_others': diff,
                'description': sig_info['description'],
                'is_resistance_mechanism': is_resistance,
                'direction': sig_info['resistance_direction'],
            })

        # Sort by absolute difference
        mechanisms.sort(key=lambda x: abs(x['vs_others']), reverse=True)
        results[subtype] = mechanisms

    return results


def map_resistance_to_therapeutics(mechanisms: Dict) -> Dict:
    """Map resistance mechanisms to therapeutic strategies."""
    therapeutic_map = {
        'antigen_presentation': {
            'target': 'Enhance antigen presentation',
            'drugs': ['Decitabine (epigenetic)', 'IFN-gamma', 'STING agonists'],
            'clinical_trials': ['NCT03233724', 'NCT02675439'],
        },
        'hla_class_ii': {
            'target': 'Restore MHC-II expression',
            'drugs': ['HDAC inhibitors', 'Entinostat', 'Panobinostat'],
            'clinical_trials': ['NCT02805660'],
        },
        't_cell_exhaustion': {
            'target': 'Reinvigorate exhausted T-cells',
            'drugs': ['Anti-LAG3 (relatlimab)', 'Anti-TIGIT (tiragolumab)', 'Anti-TIM3'],
            'clinical_trials': ['NCT03311412', 'SKYSCRAPER-02'],
        },
        'treg_signature': {
            'target': 'Deplete/inhibit Tregs',
            'drugs': ['Anti-CTLA4 (ipilimumab)', 'Anti-CCR8', 'Anti-CD25'],
            'clinical_trials': ['NCT03739931'],
        },
        'mdsc_signature': {
            'target': 'Eliminate MDSCs',
            'drugs': ['Cabozantinib', 'ATRA', 'PDE5 inhibitors'],
            'clinical_trials': ['NCT03170960'],
        },
        'tam_m2': {
            'target': 'Reprogram TAMs to M1',
            'drugs': ['CSF1R inhibitors', 'CD40 agonists', 'PI3Kgamma inhibitors'],
            'clinical_trials': ['NCT02323191'],
        },
        'tgfb_signaling': {
            'target': 'Block TGF-beta signaling',
            'drugs': ['Galunisertib', 'Bintrafusp alfa', 'Fresolimumab'],
            'clinical_trials': ['NCT02423343', 'NCT02517398'],
        },
        'wnt_bcatenin': {
            'target': 'Inhibit WNT/beta-catenin',
            'drugs': ['DKN-01', 'Foxy-5', 'ICG-001'],
            'clinical_trials': ['NCT03395080'],
        },
        'ifn_signaling': {
            'target': 'Restore IFN signaling',
            'drugs': ['STING agonists (ADU-S100)', 'IFN-alpha', 'Oncolytic viruses'],
            'clinical_trials': ['NCT03172936', 'NCT03937895'],
        },
        'metabolic_immune_suppression': {
            'target': 'Block immunometabolic pathways',
            'drugs': ['Epacadostat (IDO1)', 'CB-839 (GLS)', 'A2AR antagonists'],
            'clinical_trials': ['NCT02178722', 'NCT03381274'],
        },
        'caf_exclusion': {
            'target': 'Disrupt CAF-mediated exclusion',
            'drugs': ['FAP-targeting BiTEs', 'CXCR4 inhibitors', 'TGF-beta inhibitors'],
            'clinical_trials': ['NCT03386721'],
        },
        'angiogenesis': {
            'target': 'Normalize tumor vasculature',
            'drugs': ['Bevacizumab', 'Lenvatinib', 'Axitinib'],
            'clinical_trials': ['NCT03976375', 'IMpower150'],
        },
    }

    therapeutic_strategies = {}
    for subtype, mech_list in mechanisms.items():
        strategies = []
        for mech in mech_list:
            if mech['is_resistance_mechanism']:
                sig = mech['signature']
                if sig in therapeutic_map:
                    strategies.append({
                        'mechanism': sig,
                        'description': mech['description'],
                        'score': mech['score'],
                        **therapeutic_map[sig]
                    })
        therapeutic_strategies[subtype] = strategies

    return therapeutic_strategies


def calculate_io_resistance_score(expression_df: pd.DataFrame,
                                  subtype_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite IO resistance score per sample."""
    sig_scores = calculate_signature_scores(expression_df, IO_RESISTANCE_SIGNATURES)

    # Resistance score: high for resistance-promoting, low for protection
    resistance_components = []
    for sig_name, sig_info in IO_RESISTANCE_SIGNATURES.items():
        if sig_name in sig_scores.columns:
            if sig_info['resistance_direction'] == 'high':
                resistance_components.append(sig_scores[sig_name])
            else:
                resistance_components.append(-sig_scores[sig_name])

    resistance_score = pd.concat(resistance_components, axis=1).mean(axis=1)

    # Normalize to 0-1
    resistance_score = (resistance_score - resistance_score.min()) / (resistance_score.max() - resistance_score.min())

    return pd.DataFrame({
        'sample': resistance_score.index,
        'io_resistance_score': resistance_score.values
    })


def analyze_resistance_by_immune_state(expression_df: pd.DataFrame,
                                        subtype_df: pd.DataFrame,
                                        immune_scores: pd.DataFrame) -> Dict:
    """Analyze resistance mechanisms by immune state."""
    sig_scores = calculate_signature_scores(expression_df, IO_RESISTANCE_SIGNATURES)

    # Cluster into immune states if not provided
    if 'immune_state' not in immune_scores.columns:
        # Use hierarchical clustering on signature scores
        linkage_matrix = linkage(sig_scores.values, method='ward')
        immune_states = fcluster(linkage_matrix, t=4, criterion='maxclust')
        immune_scores = pd.DataFrame({
            'sample': sig_scores.index,
            'immune_state': immune_states
        })

    results = {}
    for state in sorted(immune_scores['immune_state'].unique()):
        mask = immune_scores['immune_state'] == state
        state_scores = sig_scores[mask.values].mean()

        results[f'State_{state}'] = {
            'n_samples': mask.sum(),
            'top_resistance': state_scores.nlargest(3).to_dict(),
            'top_protection': state_scores.nsmallest(3).to_dict(),
        }

    return results


def run_io_resistance_analysis(expression_path: Path,
                                subtype_path: Path,
                                output_dir: Path) -> Dict:
    """Run complete IO resistance mechanism analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Immunotherapy Resistance Mechanism Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading expression and subtype data...")
    expression_df = pd.read_csv(expression_path, sep='\t', index_col=0)
    subtype_df = pd.read_csv(subtype_path, sep='\t')

    # Calculate signature scores
    print("\n2. Calculating IO resistance signature scores...")
    sig_scores = calculate_signature_scores(expression_df, IO_RESISTANCE_SIGNATURES)
    sig_scores.to_csv(output_dir / 'io_resistance_signatures.tsv', sep='\t')
    print(f"   Calculated {len(IO_RESISTANCE_SIGNATURES)} resistance signatures")

    # Identify mechanisms by subtype
    print("\n3. Identifying resistance mechanisms by subtype...")
    mechanisms = identify_resistance_mechanisms(expression_df, subtype_df)

    # Save mechanisms
    mech_rows = []
    for subtype, mech_list in mechanisms.items():
        for mech in mech_list:
            mech_rows.append({
                'subtype': subtype,
                **mech
            })
    mech_df = pd.DataFrame(mech_rows)
    mech_df.to_csv(output_dir / 'resistance_mechanisms.tsv', sep='\t', index=False)

    # Map to therapeutics
    print("\n4. Mapping resistance mechanisms to therapeutic strategies...")
    therapeutics = map_resistance_to_therapeutics(mechanisms)

    # Save therapeutic strategies
    ther_rows = []
    for subtype, strategies in therapeutics.items():
        for strat in strategies:
            ther_rows.append({
                'subtype': subtype,
                'mechanism': strat['mechanism'],
                'description': strat['description'],
                'target': strat['target'],
                'drugs': ', '.join(strat['drugs']),
                'clinical_trials': ', '.join(strat.get('clinical_trials', [])),
            })
    ther_df = pd.DataFrame(ther_rows)
    ther_df.to_csv(output_dir / 'resistance_therapeutics.tsv', sep='\t', index=False)

    # Calculate resistance scores
    print("\n5. Calculating IO resistance scores...")
    resistance_scores = calculate_io_resistance_score(expression_df, subtype_df)
    resistance_scores.to_csv(output_dir / 'io_resistance_scores.tsv', sep='\t', index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("IO RESISTANCE MECHANISMS BY SUBTYPE")
    print("=" * 60)

    for subtype in ['SCLC_A', 'SCLC_N', 'SCLC_P', 'SCLC_I']:
        if subtype in mechanisms:
            print(f"\n{subtype} ({SUBTYPE_RESISTANCE_PROFILES[subtype]['io_sensitivity']} IO sensitivity):")
            print(f"  Expected: {SUBTYPE_RESISTANCE_PROFILES[subtype]['description']}")
            print("  Detected mechanisms:")
            for mech in mechanisms[subtype][:5]:
                direction = "HIGH" if mech['vs_others'] > 0 else "LOW"
                resist = " [RESISTANCE]" if mech['is_resistance_mechanism'] else ""
                print(f"    - {mech['signature']}: {direction} (diff={mech['vs_others']:.2f}){resist}")

    print("\n" + "=" * 60)
    print("THERAPEUTIC STRATEGIES TO OVERCOME RESISTANCE")
    print("=" * 60)

    for subtype, strategies in therapeutics.items():
        if strategies:
            print(f"\n{subtype}:")
            for strat in strategies[:3]:
                print(f"  - {strat['mechanism']}: {strat['target']}")
                print(f"    Drugs: {', '.join(strat['drugs'][:3])}")

    # Save summary JSON
    summary = {
        'subtype_profiles': SUBTYPE_RESISTANCE_PROFILES,
        'mechanisms': {k: v[:5] for k, v in mechanisms.items()},
        'therapeutics': therapeutics,
    }
    with open(output_dir / 'io_resistance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return {
        'signatures': sig_scores,
        'mechanisms': mechanisms,
        'therapeutics': therapeutics,
        'resistance_scores': resistance_scores,
    }
