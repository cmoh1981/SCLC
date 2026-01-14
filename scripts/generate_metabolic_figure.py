#!/usr/bin/env python
"""
Generate Figure 4: Metabolic Reprogramming Analysis for SCLC manuscript.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

# Set publication style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Colors
SUBTYPE_COLORS = {
    'SCLC_A': '#E64B35',
    'SCLC_N': '#4DBBD5',
    'SCLC_P': '#00A087',
    'SCLC_I': '#3C5488',
}

PATHWAY_COLORS = {
    'Glucose uptake': '#E64B35',
    'Glycolysis': '#F39B7F',
    'Pyruvate oxidation': '#4DBBD5',
    'TCA cycle': '#91D1C2',
    'Glutaminolysis': '#00A087',
    'OXPHOS': '#3C5488',
    'Nucleotide synthesis': '#8491B4',
    'Fatty acid synthesis': '#DC0000',
    'One-carbon': '#7E6148',
    'Serine synthesis': '#B09C85',
}


def create_metabolic_pathway_diagram(ax):
    """Create a metabolic pathway diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. SCLC Metabolic Network', fontweight='bold', fontsize=12)

    # Draw pathway boxes
    pathways = [
        ('Glucose\nUptake', (1, 8), 'Glucose uptake'),
        ('Glycolysis', (1, 6), 'Glycolysis'),
        ('Lactate\nExport', (0, 4), 'Glycolysis'),
        ('Pyruvate\nOxidation', (3, 6), 'Pyruvate oxidation'),
        ('TCA\nCycle', (3, 4), 'TCA cycle'),
        ('OXPHOS', (3, 2), 'OXPHOS'),
        ('Glutamine\nUptake', (6, 8), 'Glutaminolysis'),
        ('Glutaminolysis', (6, 6), 'Glutaminolysis'),
        ('PPP', (1, 4), 'Nucleotide synthesis'),
        ('Nucleotide\nSynthesis', (1, 2), 'Nucleotide synthesis'),
        ('Serine\nSynthesis', (5, 4), 'Serine synthesis'),
        ('One-Carbon\nMetabolism', (5, 2), 'One-carbon'),
        ('Fatty Acid\nSynthesis', (8, 4), 'Fatty acid synthesis'),
    ]

    for name, pos, pathway in pathways:
        color = PATHWAY_COLORS.get(pathway, '#CCCCCC')
        box = FancyBboxPatch(
            (pos[0]-0.6, pos[1]-0.4), 1.2, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=0.7,
            edgecolor='black', linewidth=1
        )
        ax.add_patch(box)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=7, fontweight='bold')

    # Draw arrows (simplified)
    arrows = [
        ((1, 7.5), (1, 6.5)),  # Glucose to Glycolysis
        ((1, 5.5), (1, 4.5)),  # Glycolysis to PPP
        ((1, 5.5), (0, 4.5)),  # Glycolysis to Lactate
        ((1.6, 6), (2.4, 6)),  # Glycolysis to Pyruvate
        ((3, 5.5), (3, 4.5)),  # Pyruvate to TCA
        ((3, 3.5), (3, 2.5)),  # TCA to OXPHOS
        ((6, 7.5), (6, 6.5)),  # Glutamine to Glutaminolysis
        ((5.4, 6), (3.6, 4)),  # Glutaminolysis to TCA (aKG)
        ((1, 3.5), (1, 2.5)),  # PPP to Nucleotide
        ((3.6, 4), (4.4, 4)),  # TCA to Serine
        ((5, 3.5), (5, 2.5)),  # Serine to One-Carbon
        ((3.6, 4.3), (7.4, 4.3)),  # Citrate to FA (lipogenesis)
    ]

    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # Add key metabolites
    metabolites = [
        ('Glc', (1, 7.2)),
        ('Pyr', (2, 6)),
        ('Lac', (0, 5)),
        ('AcCoA', (3, 5)),
        ('Gln', (6, 7.2)),
        ('Glu', (6, 5.5)),
        ('aKG', (4.5, 5)),
        ('R5P', (1.5, 3.5)),
        ('Ser', (5, 5)),
        ('ATP', (3.5, 1.5)),
    ]

    for met, pos in metabolites:
        ax.text(pos[0], pos[1], met, fontsize=6, style='italic', alpha=0.7)


def create_flux_heatmap(ax, flux_data):
    """Create heatmap of metabolic fluxes by subtype."""
    # Select key reactions
    key_rxns = ['GLCt', 'GLYC', 'LDH', 'PDH', 'CS', 'IDH', 'GLS', 'GLUD',
                'PPP', 'PHGDH', 'SHMT', 'FAS', 'OXPHOS']

    available_rxns = [r for r in key_rxns if r in flux_data.index]

    if len(available_rxns) > 0:
        plot_data = flux_data.loc[available_rxns]

        # Create heatmap
        im = ax.imshow(plot_data.values, aspect='auto', cmap='YlOrRd')

        ax.set_xticks(range(len(plot_data.columns)))
        ax.set_xticklabels(plot_data.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(plot_data.index)))
        ax.set_yticklabels(plot_data.index)

        plt.colorbar(im, ax=ax, shrink=0.6, label='Flux (mmol/gDW/h)')
    else:
        ax.text(0.5, 0.5, 'No flux data available', ha='center', va='center')
        ax.axis('off')

    ax.set_title('B. Metabolic Flux by Subtype', fontweight='bold')


def create_drug_target_bar(ax, drug_data):
    """Create bar chart of metabolic drug targets."""
    # Group by pathway and drug
    if len(drug_data) > 0:
        pathway_drugs = drug_data.groupby(['pathway', 'drug']).agg({
            'vulnerability_score': 'mean'
        }).reset_index()

        # Top 10 drug-pathway combinations
        top_drugs = pathway_drugs.nlargest(10, 'vulnerability_score')

        colors = [PATHWAY_COLORS.get(p, '#CCCCCC') for p in top_drugs['pathway']]
        labels = [f"{d}\n({p})" for d, p in zip(top_drugs['drug'], top_drugs['pathway'])]

        bars = ax.barh(range(len(top_drugs)), top_drugs['vulnerability_score'],
                      color=colors, edgecolor='white', linewidth=0.5)

        ax.set_yticks(range(len(top_drugs)))
        ax.set_yticklabels(top_drugs['drug'], fontsize=8)
        ax.set_xlabel('Vulnerability Score')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'No drug targets', ha='center', va='center')
        ax.axis('off')

    ax.set_title('C. Metabolic Drug Targets', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_pathway_importance(ax, vuln_data):
    """Create pie chart of pathway vulnerability contributions."""
    # Map reactions to pathways
    rxn_to_pathway = {
        'GLCt': 'Glucose uptake', 'HK': 'Glycolysis', 'GLYC': 'Glycolysis',
        'LDH': 'Glycolysis', 'LACt': 'Glycolysis',
        'PDH': 'Pyruvate oxidation', 'CS': 'TCA cycle', 'IDH': 'TCA cycle',
        'GLS': 'Glutaminolysis', 'GLUD': 'Glutaminolysis',
        'PPP': 'Nucleotide synthesis', 'DNPS': 'Nucleotide synthesis',
        'PHGDH': 'Serine synthesis', 'SHMT': 'One-carbon',
        'FAS': 'Fatty acid synthesis', 'ACC': 'Fatty acid synthesis',
        'OXPHOS': 'OXPHOS'
    }

    vuln_data['pathway'] = vuln_data['reaction'].map(rxn_to_pathway)
    vuln_data = vuln_data.dropna(subset=['pathway'])

    if len(vuln_data) > 0:
        pathway_scores = vuln_data.groupby('pathway')['vulnerability_score'].sum()
        pathway_scores = pathway_scores[pathway_scores > 0].sort_values(ascending=False)

        if len(pathway_scores) > 0:
            colors = [PATHWAY_COLORS.get(p, '#CCCCCC') for p in pathway_scores.index]
            wedges, texts, autotexts = ax.pie(
                pathway_scores.values,
                labels=pathway_scores.index,
                colors=colors,
                autopct='%1.0f%%',
                startangle=90,
                textprops={'fontsize': 8}
            )
            ax.set_title('D. Pathway Vulnerability', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No pathway data', ha='center', va='center')
            ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'No pathway data', ha='center', va='center')
        ax.axis('off')


def main():
    """Generate Figure 4: Metabolic Analysis."""
    root = Path(__file__).parent.parent
    output_dir = root / 'results' / 'figures'
    manuscript_dir = root / 'manuscript' / 'figures'

    # Load data
    try:
        flux_data = pd.read_csv(root / 'results/metabolic/subtype_fluxes.tsv', sep='\t', index_col=0)
        vuln_data = pd.read_csv(root / 'results/metabolic/metabolic_vulnerabilities.tsv', sep='\t')
        drug_data = pd.read_csv(root / 'results/metabolic/metabolic_drug_targets.tsv', sep='\t')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create figure
    fig = plt.figure(figsize=(14, 12))

    # Panel A: Pathway diagram
    ax1 = fig.add_subplot(2, 2, 1)
    create_metabolic_pathway_diagram(ax1)

    # Panel B: Flux heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    create_flux_heatmap(ax2, flux_data)

    # Panel C: Drug targets
    ax3 = fig.add_subplot(2, 2, 3)
    create_drug_target_bar(ax3, drug_data)

    # Panel D: Pathway importance
    ax4 = fig.add_subplot(2, 2, 4)
    create_pathway_importance(ax4, vuln_data)

    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'Figure4_metabolic.png', dpi=300)
    plt.savefig(output_dir / 'Figure4_metabolic.pdf')

    import shutil
    shutil.copy(output_dir / 'Figure4_metabolic.png', manuscript_dir / 'Figure4_metabolic.png')
    shutil.copy(output_dir / 'Figure4_metabolic.pdf', manuscript_dir / 'Figure4_metabolic.pdf')

    plt.close()
    print("Figure 4 (Metabolic Analysis) saved")


if __name__ == "__main__":
    main()
