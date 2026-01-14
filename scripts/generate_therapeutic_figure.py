#!/usr/bin/env python
"""
Generate Figure 5: Subtype-Specific Therapeutic Strategies for SCLC manuscript.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import json

# Set publication style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
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

IO_COLORS = {
    'High': '#2E7D32',
    'Moderate': '#FFA000',
    'Low': '#C62828',
}


def create_subtype_overview(ax):
    """Create panel A: Subtype overview with key characteristics."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. SCLC Molecular Subtypes', fontweight='bold', fontsize=11, loc='left')

    subtypes = [
        ('SCLC-A', 'ASCL1-high', 'Neuroendocrine\nDLL3+, BCL2+', SUBTYPE_COLORS['SCLC_A'], 1.5),
        ('SCLC-N', 'NEUROD1-high', 'Neuroendocrine\nMYCN amp', SUBTYPE_COLORS['SCLC_N'], 4.0),
        ('SCLC-P', 'POU2F3-high', 'Tuft cell-like\nChemoresistant', SUBTYPE_COLORS['SCLC_P'], 6.5),
        ('SCLC-I', 'Inflamed', 'T-cell inflamed\nPD-L1+', SUBTYPE_COLORS['SCLC_I'], 9.0),
    ]

    for name, marker, desc, color, x in subtypes:
        # Circle for subtype
        circle = Circle((x, 7), 0.8, facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, 7, name.split('-')[1], ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Marker gene
        ax.text(x, 5.8, marker, ha='center', va='center', fontsize=8, style='italic')

        # Description
        ax.text(x, 4.5, desc, ha='center', va='center', fontsize=7, linespacing=1.2)

    # Add IO sensitivity bar at bottom
    ax.text(0.3, 2.5, 'IO Response:', fontsize=8, fontweight='bold', ha='left')

    io_levels = [
        (1.5, 'Low', IO_COLORS['Low']),
        (4.0, 'Low', IO_COLORS['Low']),
        (6.5, 'Moderate', IO_COLORS['Moderate']),
        (9.0, 'High', IO_COLORS['High']),
    ]

    for x, level, color in io_levels:
        rect = Rectangle((x-0.5, 1.8), 1, 0.5, facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, 2.05, level, ha='center', va='center', fontsize=7, color='white', fontweight='bold')


def create_drug_strategy_matrix(ax, strategy_df):
    """Create panel B: Drug-subtype strategy heatmap."""
    ax.set_title('B. Subtype-Specific Drug Recommendations', fontweight='bold', fontsize=11, loc='left')

    # Define key drugs for each subtype
    drug_matrix = {
        'SCLC-A': {'Tarlatamab': 3, 'Alisertib': 3, 'Venetoclax': 2, 'Lurbinectedin': 2, 'Atezolizumab': 1},
        'SCLC-N': {'Alisertib': 3, 'Olaparib': 3, 'Volasertib': 2, 'Lurbinectedin': 2, 'Atezolizumab': 1},
        'SCLC-P': {'Erdafitinib': 3, 'Linsitinib': 2, 'Temozolomide': 2, 'Trilaciclib': 2, 'Atezolizumab': 2},
        'SCLC-I': {'Atezolizumab': 3, 'Durvalumab': 3, 'Ipilimumab': 3, 'Tiragolumab': 2, 'Relatlimab': 2},
    }

    # Get unique drugs
    all_drugs = list(set(d for subtype_drugs in drug_matrix.values() for d in subtype_drugs.keys()))
    all_drugs = sorted(all_drugs)
    subtypes = list(drug_matrix.keys())

    # Create matrix
    matrix = np.zeros((len(all_drugs), len(subtypes)))
    for j, subtype in enumerate(subtypes):
        for i, drug in enumerate(all_drugs):
            matrix[i, j] = drug_matrix[subtype].get(drug, 0)

    # Plot heatmap
    cmap = plt.cm.YlOrRd
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=3)

    ax.set_xticks(range(len(subtypes)))
    ax.set_xticklabels(subtypes, fontsize=8)
    ax.set_yticks(range(len(all_drugs)))
    ax.set_yticklabels(all_drugs, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=15)
    cbar.set_label('Recommendation\nStrength', fontsize=8)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['None', 'Low', 'Medium', 'High'], fontsize=7)

    # Add value annotations
    for i in range(len(all_drugs)):
        for j in range(len(subtypes)):
            val = int(matrix[i, j])
            if val > 0:
                text = ax.text(j, i, val, ha='center', va='center', fontsize=7,
                              color='white' if val >= 2 else 'black', fontweight='bold')


def create_treatment_algorithm(ax):
    """Create panel C: Treatment decision algorithm."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('C. Subtype-Guided Treatment Algorithm', fontweight='bold', fontsize=11, loc='left')

    # Starting point
    box = FancyBboxPatch((3.5, 8.5), 3, 0.8, boxstyle="round,pad=0.1",
                         facecolor='lightgray', edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(5, 8.9, 'SCLC Diagnosis', ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrow down
    ax.annotate('', xy=(5, 7.8), xytext=(5, 8.4), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(5.3, 8.1, 'Subtype\nclassification', fontsize=7, ha='left', va='center')

    # Subtype boxes
    subtype_info = [
        ('SCLC-A', 0.5, SUBTYPE_COLORS['SCLC_A'], 'DLL3 ADC\n+ Aurora Ki'),
        ('SCLC-N', 3.0, SUBTYPE_COLORS['SCLC_N'], 'PARP/Aurora\ninhibitors'),
        ('SCLC-P', 5.5, SUBTYPE_COLORS['SCLC_P'], 'FGFR/IGF1R\ninhibitors'),
        ('SCLC-I', 8.0, SUBTYPE_COLORS['SCLC_I'], 'IO doublet\n(PD-L1+CTLA4)'),
    ]

    for name, x, color, treatment in subtype_info:
        # Subtype box
        box = FancyBboxPatch((x, 6.5), 1.8, 1.0, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=1, alpha=0.8)
        ax.add_patch(box)
        ax.text(x+0.9, 7.0, name, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

        # Arrow down
        ax.annotate('', xy=(x+0.9, 5.8), xytext=(x+0.9, 6.4), arrowprops=dict(arrowstyle='->', lw=1))

        # Treatment box
        treat_box = FancyBboxPatch((x, 4.5), 1.8, 1.2, boxstyle="round,pad=0.05",
                                   facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(treat_box)
        ax.text(x+0.9, 5.1, treatment, ha='center', va='center', fontsize=7)

    # Common backbone
    ax.text(5, 3.5, '+ Platinum-Etoposide Backbone', ha='center', va='center',
            fontsize=9, fontweight='bold', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Metabolic targeting note
    ax.text(5, 2.3, 'Consider: OXPHOS inhibitors (all subtypes)',
            ha='center', va='center', fontsize=8, style='italic', color='gray')


def create_clinical_trials_panel(ax):
    """Create panel D: Key clinical trials table."""
    ax.axis('off')
    ax.set_title('D. Key Clinical Trials by Subtype', fontweight='bold', fontsize=11, loc='left')

    trials = [
        ['Subtype', 'Drug', 'Trial', 'Phase', 'Status'],
        ['SCLC-A', 'Tarlatamab', 'DeLLphi-301', 'III', 'Enrolling'],
        ['SCLC-A', 'Alisertib', 'NCT02038647', 'II', 'Completed'],
        ['SCLC-N', 'Olaparib+TMZ', 'NCT02446704', 'I/II', 'Active'],
        ['SCLC-P', 'Erdafitinib', 'NCT03827850', 'II', 'Recruiting'],
        ['SCLC-I', 'Atezo+Tira', 'SKYSCRAPER-02', 'III', 'Results 2024'],
        ['All', 'IACS-010759', 'NCT02882321', 'I', 'OXPHOS target'],
    ]

    # Create table
    colors = ['lightgray'] + [SUBTYPE_COLORS.get(f'SCLC_{r[0].split("-")[1]}', 'white')
                               if 'SCLC' in r[0] else 'lightyellow' for r in trials[1:]]

    table = ax.table(cellText=trials[1:], colLabels=trials[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.18, 0.22, 0.25, 0.12, 0.23])

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Style header
    for j in range(5):
        table[(0, j)].set_facecolor('lightgray')
        table[(0, j)].set_text_props(fontweight='bold')

    # Color by subtype
    for i, color in enumerate(colors):
        if i > 0:
            alpha = 0.3 if color not in ['white', 'lightyellow'] else 1.0
            for j in range(5):
                if color not in ['white', 'lightyellow']:
                    table[(i, j)].set_facecolor(color)
                    table[(i, j)].set_alpha(0.3)


def main():
    """Generate Figure 5: Therapeutic Strategies."""
    root = Path(__file__).parent.parent
    output_dir = root / 'results' / 'figures'
    manuscript_dir = root / 'manuscript' / 'figures'

    # Load strategy data
    try:
        strategy_df = pd.read_csv(root / 'results/therapeutic_strategies/subtype_therapeutic_strategies.tsv', sep='\t')
    except:
        strategy_df = None

    # Create figure
    fig = plt.figure(figsize=(14, 12))

    # Panel A: Subtype overview (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    create_subtype_overview(ax1)

    # Panel B: Drug-subtype matrix (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    create_drug_strategy_matrix(ax2, strategy_df)

    # Panel C: Treatment algorithm (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    create_treatment_algorithm(ax3)

    # Panel D: Clinical trials (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    create_clinical_trials_panel(ax4)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'Figure5_therapeutic_strategies.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Figure5_therapeutic_strategies.pdf', bbox_inches='tight')

    import shutil
    shutil.copy(output_dir / 'Figure5_therapeutic_strategies.png',
                manuscript_dir / 'Figure5_therapeutic_strategies.png')
    shutil.copy(output_dir / 'Figure5_therapeutic_strategies.pdf',
                manuscript_dir / 'Figure5_therapeutic_strategies.pdf')

    plt.close()
    print("Figure 5 (Therapeutic Strategies) saved")


if __name__ == "__main__":
    main()
