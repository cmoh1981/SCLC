"""
Stage 11: Figures and Tables Generation

Publication-ready figures at 300 dpi:
- Fig1: Subtype landscape
- Fig2: Immune-state map
- Fig3: Module-trait heatmap
- Fig4: DepMap validation (if available)
- Fig5: Drug repositioning triangle

Tables:
- Table1: Cohort summary
- Table2: Immune states
- Table3: Top drugs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def create_fig1_subtype_landscape(
    subtype_scores_path: Path,
    subtype_calls_path: Path,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Create Figure 1: SCLC Subtype Landscape.

    Panel A: Stacked barplot of subtype distribution
    Panel B: Heatmap of subtype scores
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    try:
        # Load data
        scores = pd.read_csv(subtype_scores_path, sep='\t', index_col=0)
        calls = pd.read_csv(subtype_calls_path, sep='\t')

        # Panel A: Subtype distribution
        subtype_counts = calls['subtype'].value_counts()
        colors = {'SCLC_A': '#E64B35', 'SCLC_N': '#4DBBD5',
                  'SCLC_P': '#00A087', 'SCLC_I': '#3C5488'}

        ax = axes[0]
        bars = ax.bar(subtype_counts.index, subtype_counts.values,
                     color=[colors.get(s, 'gray') for s in subtype_counts.index])
        ax.set_xlabel('SCLC Subtype')
        ax.set_ylabel('Number of Samples')
        ax.set_title('A. SCLC Subtype Distribution')

        # Add percentage labels
        total = subtype_counts.sum()
        for bar, count in zip(bars, subtype_counts.values):
            pct = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        # Panel B: Heatmap of subtype scores
        ax = axes[1]
        score_cols = [c for c in scores.columns if 'SCLC' in c]
        if score_cols:
            sns.heatmap(scores[score_cols].T, cmap='RdBu_r', center=0,
                       ax=ax, cbar_kws={'label': 'Score (z-scaled)'})
            ax.set_title('B. Subtype Signature Scores')
            ax.set_xlabel('Samples')
            ax.set_ylabel('Subtype Signature')

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        if logger:
            logger.info(f"Created Fig1: {output_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to create Fig1: {e}")
        plt.close(fig)

    return output_path


def create_fig2_immune_states(
    immune_scores_path: Path,
    immune_states_path: Path,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Create Figure 2: Immune State Map.

    Panel A: 2D embedding (PCA/UMAP) colored by immune state
    Panel B: Key immune axis boxplots by state
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    try:
        from sklearn.decomposition import PCA

        # Load data
        scores = pd.read_csv(immune_scores_path, sep='\t', index_col=0)
        states = pd.read_csv(immune_states_path, sep='\t')

        # Merge
        data = scores.copy()
        data['immune_state'] = states.set_index('sample')['immune_state']

        # Panel A: PCA plot
        ax = axes[0]
        score_cols = [c for c in scores.columns if not c.startswith('immune')]

        if len(score_cols) >= 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(scores[score_cols].dropna())

            scatter_data = pd.DataFrame({
                'PC1': coords[:, 0],
                'PC2': coords[:, 1],
                'state': data.loc[scores[score_cols].dropna().index, 'immune_state']
            })

            for state in scatter_data['state'].unique():
                mask = scatter_data['state'] == state
                ax.scatter(scatter_data.loc[mask, 'PC1'],
                          scatter_data.loc[mask, 'PC2'],
                          label=state, alpha=0.7, s=50)

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_title('A. Immune State Map')
            ax.legend(title='Immune State', bbox_to_anchor=(1.05, 1))

        # Panel B: Key axis boxplots
        ax = axes[1]
        if 't_effector' in score_cols:
            plot_data = data[['t_effector', 'immune_state']].dropna()
            plot_data.boxplot(column='t_effector', by='immune_state', ax=ax)
            ax.set_title('B. T-Effector Score by Immune State')
            ax.set_xlabel('Immune State')
            ax.set_ylabel('T-Effector Score')
            plt.suptitle('')  # Remove automatic title

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        if logger:
            logger.info(f"Created Fig2: {output_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to create Fig2: {e}")
        plt.close(fig)

    return output_path


def create_fig3_modules(
    module_trait_path: Path,
    hub_genes_path: Path,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Create Figure 3: Module-Trait Heatmap.

    Panel A: Module-trait correlation heatmap
    Panel B: Hub gene network (simplified)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    try:
        # Panel A: Module-trait heatmap
        ax = axes[0]

        if Path(module_trait_path).exists():
            corr = pd.read_csv(module_trait_path, sep='\t', index_col=0)
            sns.heatmap(corr.astype(float), cmap='RdBu_r', center=0, ax=ax,
                       annot=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
            ax.set_title('A. Module-Trait Associations')
        else:
            ax.text(0.5, 0.5, 'Module-trait data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('A. Module-Trait Associations')

        # Panel B: Top hub genes table
        ax = axes[1]
        ax.axis('off')

        if Path(hub_genes_path).exists():
            hub = pd.read_csv(hub_genes_path, sep='\t')
            top_hub = hub.head(15)[['gene', 'module', 'module_eigengene_corr']]
            top_hub.columns = ['Gene', 'Module', 'ME Correlation']

            table = ax.table(
                cellText=top_hub.values,
                colLabels=top_hub.columns,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax.set_title('B. Top Hub Genes', pad=20)
        else:
            ax.text(0.5, 0.5, 'Hub genes not available',
                   ha='center', va='center')

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        if logger:
            logger.info(f"Created Fig3: {output_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to create Fig3: {e}")
        plt.close(fig)

    return output_path


def create_fig5_drug_triangle(
    drug_rank_path: Path,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Create Figure 5: Drug Repositioning Evidence Triangle.

    Shows top drugs with 3-leg evidence.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    try:
        if Path(drug_rank_path).exists():
            drugs = pd.read_csv(drug_rank_path, sep='\t')
            top_drugs = drugs.head(20)

            # Horizontal bar plot
            colors = ['#E64B35' if n == 3 else '#4DBBD5' if n == 2 else '#00A087'
                     for n in top_drugs['n_evidence_legs']]

            bars = ax.barh(range(len(top_drugs)), top_drugs['composite_score'],
                          color=colors)
            ax.set_yticks(range(len(top_drugs)))
            ax.set_yticklabels(top_drugs['drug_name'])
            ax.set_xlabel('Composite Score')
            ax.set_title('Top Drug Candidates (3-Leg Evidence Rule)')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#E64B35', label='3-leg evidence'),
                Patch(facecolor='#4DBBD5', label='2-leg evidence'),
                Patch(facecolor='#00A087', label='1-leg evidence')
            ]
            ax.legend(handles=legend_elements, loc='lower right')

            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'Drug ranking not available',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        if logger:
            logger.info(f"Created Fig5: {output_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to create Fig5: {e}")
        plt.close(fig)

    return output_path


def create_table1_cohorts(
    config_path: Path,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """Create Table 1: Cohort Summary."""
    import yaml

    try:
        with open(config_path, 'r') as f:
            cohorts_config = yaml.safe_load(f)

        rows = []
        for data_type, datasets in cohorts_config.get('cohorts', {}).items():
            if data_type == 'controlled':
                for ds in datasets:
                    rows.append({
                        'Data Type': 'Controlled',
                        'Dataset': ds.get('name', ''),
                        'Portal': ds.get('portal', ''),
                        'Accession': ds.get('accession', ''),
                        'Samples': ds.get('samples', ''),
                        'Access': ds.get('access', ''),
                        'Reference': ds.get('reference', '')
                    })
            else:
                for ds in datasets:
                    rows.append({
                        'Data Type': data_type,
                        'Dataset': ds.get('name', ''),
                        'Portal': ds.get('portal', ''),
                        'Accession': ds.get('accession', ''),
                        'Samples': ds.get('samples', ''),
                        'Access': ds.get('access', ''),
                        'Reference': ds.get('reference', '')
                    })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, sep='\t', index=False)

        if logger:
            logger.info(f"Created Table1: {output_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to create Table1: {e}")

    return output_path


def run_figures_tables(
    results_dir: Path,
    config_dir: Path,
    output_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Generate all figures and tables.

    Args:
        results_dir: Directory with analysis results
        config_dir: Directory with configs
        output_dir: Output directory for figures/tables
        logger: Optional logger

    Returns:
        Results dictionary
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "stage": "figures_tables",
        "timestamp": datetime.now().isoformat(),
        "figures": [],
        "tables": [],
        "success": False
    }

    try:
        # Figure 1: Subtype landscape
        fig1_path = figures_dir / "Fig1_subtype_landscape.png"
        create_fig1_subtype_landscape(
            results_dir / "subtypes" / "subtype_scores.tsv",
            results_dir / "subtypes" / "subtype_calls.tsv",
            fig1_path,
            logger
        )
        results["figures"].append(str(fig1_path))

        # Figure 2: Immune states
        fig2_path = figures_dir / "Fig2_immune_states.png"
        create_fig2_immune_states(
            results_dir / "immune" / "immune_scores.tsv",
            results_dir / "immune" / "immune_states.tsv",
            fig2_path,
            logger
        )
        results["figures"].append(str(fig2_path))

        # Figure 3: Modules
        fig3_path = figures_dir / "Fig3_modules.png"
        create_fig3_modules(
            results_dir / "modules" / "module_trait_correlation.tsv",
            results_dir / "modules" / "hub_genes.tsv",
            fig3_path,
            logger
        )
        results["figures"].append(str(fig3_path))

        # Figure 5: Drug triangle
        fig5_path = figures_dir / "Fig5_drug_triangle.png"
        create_fig5_drug_triangle(
            results_dir / "drugs" / "drug_rank.tsv",
            fig5_path,
            logger
        )
        results["figures"].append(str(fig5_path))

        # Table 1: Cohorts
        table1_path = tables_dir / "Table1_cohorts.tsv"
        create_table1_cohorts(
            config_dir / "cohorts.yaml",
            table1_path,
            logger
        )
        results["tables"].append(str(table1_path))

        results["success"] = True

        if logger:
            logger.info(f"Created {len(results['figures'])} figures, {len(results['tables'])} tables")

    except Exception as e:
        results["error"] = str(e)
        if logger:
            logger.error(f"Figures/tables generation failed: {e}")

    return results
