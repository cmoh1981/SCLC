#!/usr/bin/env python
"""
Generate publication-quality figures for SCLC manuscript.
Target journal: Signal Transduction and Targeted Therapy (Nature)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore

# Set publication style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palettes
SUBTYPE_COLORS = {
    'SCLC_A': '#E64B35',  # Red
    'SCLC_N': '#4DBBD5',  # Cyan
    'SCLC_P': '#00A087',  # Teal
    'SCLC_I': '#3C5488',  # Blue
}

IMMUNE_STATE_COLORS = {
    'ImmuneState_1': '#F39B7F',  # Light coral
    'ImmuneState_2': '#8491B4',  # Blue-gray
    'ImmuneState_3': '#91D1C2',  # Mint
    'ImmuneState_4': '#DC0000',  # Red
}


def load_data():
    """Load all results data."""
    root = Path(__file__).parent.parent

    # Load subtype scores
    subtype_scores = pd.read_csv(root / 'results/subtypes/subtype_scores.tsv', sep='\t', index_col=0)

    # Load immune scores
    immune_scores = pd.read_csv(root / 'results/immune/immune_scores.tsv', sep='\t', index_col=0)

    # Load immune states
    immune_states = pd.read_csv(root / 'results/immune/immune_states.tsv', sep='\t', index_col=0)

    # Load drug data
    drug_rank = pd.read_csv(root / 'results/drugs/drug_rank.tsv', sep='\t')
    top_drugs = pd.read_csv(root / 'results/drugs/top_drugs_summary.tsv', sep='\t')

    # Load expression data
    expr = pd.read_csv(root / 'data/processed/bulk/bulk_expression_matrix.tsv', sep='\t', index_col=0)

    return {
        'subtype_scores': subtype_scores,
        'immune_scores': immune_scores,
        'immune_states': immune_states,
        'drug_rank': drug_rank,
        'top_drugs': top_drugs,
        'expression': expr
    }


def figure1_subtype_landscape(data, output_dir):
    """
    Figure 1: SCLC Transcriptional Subtype Landscape
    (A) Pie chart of subtype distribution
    (B) PCA plot
    (C) Heatmap of marker genes
    """
    fig = plt.figure(figsize=(12, 10))

    subtype_scores = data['subtype_scores']

    # Get dominant subtype for each sample
    subtypes = ['SCLC_A', 'SCLC_N', 'SCLC_P', 'SCLC_I']
    if 'dominant_subtype' not in subtype_scores.columns:
        subtype_scores['dominant_subtype'] = subtype_scores[subtypes].idxmax(axis=1)

    # Panel A: Pie chart
    ax1 = fig.add_subplot(2, 2, 1)
    counts = subtype_scores['dominant_subtype'].value_counts()
    colors = [SUBTYPE_COLORS[s] for s in counts.index]
    wedges, texts, autotexts = ax1.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.02] * len(counts)
    )
    ax1.set_title('A. SCLC Subtype Distribution (n=86)', fontweight='bold', pad=20)

    # Panel B: PCA
    ax2 = fig.add_subplot(2, 2, 2)
    from sklearn.decomposition import PCA

    # PCA on subtype scores
    pca = PCA(n_components=2)
    scores_matrix = subtype_scores[subtypes].values
    pca_coords = pca.fit_transform(scores_matrix)

    for subtype in subtypes:
        mask = subtype_scores['dominant_subtype'] == subtype
        ax2.scatter(
            pca_coords[mask, 0], pca_coords[mask, 1],
            c=SUBTYPE_COLORS[subtype],
            label=subtype,
            s=50,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('B. Principal Component Analysis', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fancybox=True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Heatmap of subtype scores
    ax3 = fig.add_subplot(2, 1, 2)

    # Sort samples by dominant subtype
    sorted_idx = subtype_scores.sort_values('dominant_subtype').index
    score_matrix = subtype_scores.loc[sorted_idx, subtypes]

    # Z-score normalize for visualization
    score_z = score_matrix.apply(zscore, axis=0)

    im = ax3.imshow(score_z.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax3.set_yticks(range(len(subtypes)))
    ax3.set_yticklabels(subtypes)
    ax3.set_xlabel('Samples (n=86)')
    ax3.set_title('C. Subtype Signature Scores (z-score)', fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, shrink=0.6, pad=0.02)
    cbar.set_label('Z-score')

    # Add subtype annotation bar using colored rectangles
    ax_bar = ax3.inset_axes([0, 1.02, 1, 0.03])
    for i, s in enumerate(sorted_idx):
        color = SUBTYPE_COLORS[subtype_scores.loc[s, 'dominant_subtype']]
        ax_bar.add_patch(mpatches.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='none'))
    ax_bar.set_xlim(0, len(sorted_idx))
    ax_bar.set_ylim(0, 1)
    ax_bar.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'Figure1_subtype_landscape.png', dpi=300)
    plt.savefig(output_dir / 'Figure1_subtype_landscape.pdf')
    plt.close()
    print("Figure 1 saved")


def figure2_immune_states(data, output_dir):
    """
    Figure 2: Immune-State Stratification
    (A) Heatmap with clustering
    (B) Stacked bar of immune states by subtype
    (C) Correlation matrix
    """
    fig = plt.figure(figsize=(14, 10))

    immune_scores = data['immune_scores']
    immune_states = data['immune_states']
    subtype_scores = data['subtype_scores']

    # Ensure dominant_subtype exists
    subtypes = ['SCLC_A', 'SCLC_N', 'SCLC_P', 'SCLC_I']
    if 'dominant_subtype' not in subtype_scores.columns:
        subtype_scores['dominant_subtype'] = subtype_scores[subtypes].idxmax(axis=1)

    # Panel A: Clustered heatmap
    ax1 = fig.add_subplot(2, 2, 1)

    # Z-score normalize
    immune_z = immune_scores.apply(zscore, axis=0)

    # Hierarchical clustering
    linkage_matrix = linkage(immune_z, method='ward')

    # Get cluster order
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(linkage_matrix)

    im = ax1.imshow(immune_z.iloc[order].T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax1.set_yticks(range(len(immune_scores.columns)))
    ax1.set_yticklabels([c.replace('_', ' ').title() for c in immune_scores.columns], fontsize=8)
    ax1.set_xlabel('Samples (n=86)')
    ax1.set_title('A. Immune Signature Clustering', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax1, shrink=0.6)
    cbar.set_label('Z-score')

    # Add immune state annotation bar using colored rectangles
    if 'immune_state' in immune_states.columns:
        ax_bar = ax1.inset_axes([0, 1.02, 1, 0.03])
        for i, idx in enumerate(order):
            state = immune_states.loc[immune_z.index[idx], 'immune_state']
            color = IMMUNE_STATE_COLORS.get(state, '#CCCCCC')
            ax_bar.add_patch(mpatches.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='none'))
        ax_bar.set_xlim(0, len(order))
        ax_bar.set_ylim(0, 1)
        ax_bar.axis('off')

    # Panel B: Stacked bar chart
    ax2 = fig.add_subplot(2, 2, 2)

    # Merge data
    merged = pd.DataFrame({
        'subtype': subtype_scores['dominant_subtype'],
        'immune_state': immune_states['immune_state']
    })

    # Count cross-tabulation
    cross = pd.crosstab(merged['subtype'], merged['immune_state'], normalize='index') * 100

    # Plot stacked bar
    bottom = np.zeros(len(cross))
    for state in ['ImmuneState_1', 'ImmuneState_2', 'ImmuneState_3', 'ImmuneState_4']:
        if state in cross.columns:
            ax2.bar(cross.index, cross[state], bottom=bottom,
                   color=IMMUNE_STATE_COLORS[state], label=state.replace('ImmuneState_', 'State '))
            bottom += cross[state].values

    ax2.set_ylabel('Percentage (%)')
    ax2.set_xlabel('SCLC Subtype')
    ax2.set_title('B. Immune States by Subtype', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Correlation matrix
    ax3 = fig.add_subplot(2, 2, 3)

    corr = immune_scores.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                annot=True, fmt='.2f', ax=ax3, square=True,
                cbar_kws={'shrink': 0.6, 'label': 'Correlation'},
                annot_kws={'size': 8})
    ax3.set_title('C. Immune Signature Correlations', fontweight='bold')
    ax3.set_xticklabels([c.replace('_', '\n') for c in corr.columns], fontsize=8, rotation=45, ha='right')
    ax3.set_yticklabels([c.replace('_', '\n') for c in corr.index], fontsize=8, rotation=0)

    # Panel D: Immune state distribution pie chart
    ax4 = fig.add_subplot(2, 2, 4)
    state_counts = immune_states['immune_state'].value_counts()
    colors = [IMMUNE_STATE_COLORS[s] for s in state_counts.index]
    labels = [f"State {s.split('_')[1]}\n(n={c})" for s, c in zip(state_counts.index, state_counts.values)]

    wedges, texts, autotexts = ax4.pie(
        state_counts.values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )
    ax4.set_title('D. Immune State Distribution', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'Figure2_immune_states.png', dpi=300)
    plt.savefig(output_dir / 'Figure2_immune_states.pdf')
    plt.close()
    print("Figure 2 saved")


def figure3_drug_repositioning(data, output_dir):
    """
    Figure 3: Drug Repositioning Analysis
    (A) Workflow diagram
    (B) Top 20 drugs bar chart
    (C) Drug-target network
    (D) Drug class breakdown
    """
    fig = plt.figure(figsize=(14, 12))

    top_drugs = data['top_drugs']

    # Panel A: Workflow (text-based)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis('off')

    # Draw workflow boxes
    workflow_text = """
    DRUG REPOSITIONING WORKFLOW

    ┌─────────────────────────────┐
    │   57 SCLC-Associated Genes  │
    │   (curated from literature) │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │      DGIdb GraphQL API      │
    │   (Drug-Gene Interactions)  │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │   1,911 Interactions Found  │
    │     1,276 Unique Drugs      │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │   Ranking by Target Score   │
    │   Score = log2(n_targets+1) │
    └─────────────────────────────┘
    """
    ax1.text(0.5, 0.5, workflow_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.set_title('A. Drug Repositioning Workflow', fontweight='bold')

    # Panel B: Top 20 drugs horizontal bar chart
    ax2 = fig.add_subplot(2, 2, 2)

    top20 = top_drugs.head(20).copy()
    top20 = top20.iloc[::-1]  # Reverse for horizontal bar

    # Color by drug class
    def get_drug_class(drug):
        drug_upper = drug.upper()
        if 'CISPLATIN' in drug_upper or 'CARBOPLATIN' in drug_upper:
            return 'Platinum', '#E64B35'
        elif 'SERTIB' in drug_upper or 'AURORA' in drug_upper.lower():
            return 'Aurora kinase', '#4DBBD5'
        elif 'PARIB' in drug_upper or 'PARP' in drug_upper:
            return 'PARP inhibitor', '#00A087'
        elif 'TINIB' in drug_upper:
            return 'Multi-kinase', '#3C5488'
        else:
            return 'Other', '#8491B4'

    colors = [get_drug_class(d)[1] for d in top20['drug_name']]

    bars = ax2.barh(range(len(top20)), top20['n_targets'], color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_yticks(range(len(top20)))
    ax2.set_yticklabels(top20['drug_name'], fontsize=8)
    ax2.set_xlabel('Number of SCLC Targets')
    ax2.set_title('B. Top 20 Drug Candidates', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add target count labels
    for i, (bar, val) in enumerate(zip(bars, top20['n_targets'])):
        ax2.text(val + 0.1, i, str(val), va='center', fontsize=8)

    # Panel C: Drug class breakdown
    ax3 = fig.add_subplot(2, 2, 3)

    # Classify all drugs
    drug_classes = {}
    for drug in data['drug_rank']['drug_name']:
        cls, _ = get_drug_class(drug)
        drug_classes[cls] = drug_classes.get(cls, 0) + 1

    classes = list(drug_classes.keys())
    counts = [drug_classes[c] for c in classes]
    class_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#8491B4']

    ax3.pie(counts, labels=classes, colors=class_colors[:len(classes)],
            autopct='%1.1f%%', startangle=90)
    ax3.set_title('C. Drug Classes (n=1,276)', fontweight='bold')

    # Panel D: Target gene frequency
    ax4 = fig.add_subplot(2, 2, 4)

    # Count target gene frequency
    target_counts = {}
    for targets in top_drugs['target_genes'].dropna():
        for gene in str(targets).split(','):
            gene = gene.strip()
            if gene:
                target_counts[gene] = target_counts.get(gene, 0) + 1

    # Top 15 targets
    sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    genes = [t[0] for t in sorted_targets]
    freqs = [t[1] for t in sorted_targets]

    bars = ax4.bar(range(len(genes)), freqs, color='#3C5488', edgecolor='white', linewidth=0.5)
    ax4.set_xticks(range(len(genes)))
    ax4.set_xticklabels(genes, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Drug Count')
    ax4.set_xlabel('Target Gene')
    ax4.set_title('D. Most Frequently Targeted Genes', fontweight='bold')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'Figure3_drug_repositioning.png', dpi=300)
    plt.savefig(output_dir / 'Figure3_drug_repositioning.pdf')
    plt.close()
    print("Figure 3 saved")


def create_table1(data, output_dir):
    """Create Table 1: Top Drug Candidates."""
    top_drugs = data['top_drugs'].head(20).copy()

    # Select and rename columns
    table = top_drugs[['drug_name', 'target_genes', 'n_targets', 'sources']].copy()
    table.columns = ['Drug Name', 'Target Genes', 'N Targets', 'Evidence Source']
    table.index = range(1, len(table) + 1)
    table.index.name = 'Rank'

    # Save as TSV
    table.to_csv(output_dir / 'Table1_top_drugs.tsv', sep='\t')

    # Save as formatted markdown manually
    with open(output_dir / 'Table1_top_drugs.md', 'w') as f:
        f.write('# Table 1. Top Drug Candidates for SCLC\n\n')
        f.write('| Rank | Drug Name | Target Genes | N Targets | Evidence Source |\n')
        f.write('|------|-----------|--------------|-----------|------------------|\n')
        for idx, row in table.iterrows():
            targets = row['Target Genes'][:50] + '...' if len(str(row['Target Genes'])) > 50 else row['Target Genes']
            f.write(f"| {idx} | {row['Drug Name']} | {targets} | {row['N Targets']} | {row['Evidence Source']} |\n")
        f.write('\n\nAbbreviations: N, number.\n')

    print("Table 1 saved")


def main():
    """Generate all manuscript figures and tables."""
    root = Path(__file__).parent.parent
    output_dir = root / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also save to manuscript folder
    manuscript_dir = root / 'manuscript' / 'figures'
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data()

    print("Generating figures...")
    figure1_subtype_landscape(data, output_dir)
    figure2_immune_states(data, output_dir)
    figure3_drug_repositioning(data, output_dir)

    # Copy to manuscript folder
    import shutil
    for fig in output_dir.glob('Figure*.png'):
        shutil.copy(fig, manuscript_dir / fig.name)
    for fig in output_dir.glob('Figure*.pdf'):
        shutil.copy(fig, manuscript_dir / fig.name)

    print("\nGenerating tables...")
    create_table1(data, output_dir)
    shutil.copy(output_dir / 'Table1_top_drugs.tsv', manuscript_dir / 'Table1_top_drugs.tsv')
    shutil.copy(output_dir / 'Table1_top_drugs.md', manuscript_dir / 'Table1_top_drugs.md')

    print(f"\nAll figures saved to {output_dir}")
    print(f"Copies saved to {manuscript_dir}")


if __name__ == "__main__":
    main()
