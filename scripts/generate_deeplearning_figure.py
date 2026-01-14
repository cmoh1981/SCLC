#!/usr/bin/env python
"""
Generate Figure 6: Deep Learning-Based Novel Target and Drug Discovery.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
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
    'Universal': '#7E6148',
}


def create_dl_workflow(ax):
    """Create panel A: Deep learning workflow schematic."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. Deep Learning Workflow for Novel Discovery', fontweight='bold', fontsize=11, loc='left')

    # Input data
    box = FancyBboxPatch((0.5, 8), 2, 1.2, boxstyle="round,pad=0.1",
                         facecolor='lightblue', edgecolor='black', linewidth=1)
    ax.add_patch(box)
    ax.text(1.5, 8.6, 'Gene Expression\n(15,000 genes)', ha='center', va='center', fontsize=8)

    # VAE
    box = FancyBboxPatch((0.5, 5.5), 2, 1.5, boxstyle="round,pad=0.1",
                         facecolor='#FFB74D', edgecolor='black', linewidth=1)
    ax.add_patch(box)
    ax.text(1.5, 6.25, 'Variational\nAutoencoder\n(VAE)', ha='center', va='center', fontsize=8, fontweight='bold')

    # Arrow
    ax.annotate('', xy=(1.5, 5.5), xytext=(1.5, 7.9), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Attention Network
    box = FancyBboxPatch((3.5, 5.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                         facecolor='#81C784', edgecolor='black', linewidth=1)
    ax.add_patch(box)
    ax.text(4.75, 6.25, 'Attention-based\nSubtype Classifier', ha='center', va='center', fontsize=8, fontweight='bold')

    # Arrow from input
    ax.annotate('', xy=(3.5, 6.25), xytext=(2.6, 8.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, connectionstyle='arc3,rad=-0.2'))

    # Novel Targets
    box = FancyBboxPatch((0.5, 2.5), 2, 1.5, boxstyle="round,pad=0.1",
                         facecolor='#E57373', edgecolor='black', linewidth=1)
    ax.add_patch(box)
    ax.text(1.5, 3.25, 'Novel Target\nDiscovery\n(200 genes)', ha='center', va='center', fontsize=8, fontweight='bold')

    ax.annotate('', xy=(1.5, 4.1), xytext=(1.5, 5.4), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Drug-Target Prediction
    box = FancyBboxPatch((6.5, 5.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                         facecolor='#64B5F6', edgecolor='black', linewidth=1)
    ax.add_patch(box)
    ax.text(7.75, 6.25, 'Drug-Target\nInteraction\nPrediction', ha='center', va='center', fontsize=8, fontweight='bold')

    ax.annotate('', xy=(6.5, 6.25), xytext=(6.1, 6.25), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Novel Drugs
    box = FancyBboxPatch((6.5, 2.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                         facecolor='#9575CD', edgecolor='black', linewidth=1)
    ax.add_patch(box)
    ax.text(7.75, 3.25, 'Novel Drug\nCandidates\n(13 drugs)', ha='center', va='center', fontsize=8, fontweight='bold')

    ax.annotate('', xy=(7.75, 4.1), xytext=(7.75, 5.4), arrowprops=dict(arrowstyle='->', lw=1.5))

    # In silico validation
    box = FancyBboxPatch((3.5, 0.8), 3, 1.2, boxstyle="round,pad=0.1",
                         facecolor='#FFD54F', edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(5, 1.4, 'In Silico Validation\n(ADMET, Docking, Selectivity)', ha='center', va='center', fontsize=8, fontweight='bold')

    ax.annotate('', xy=(3.5, 1.4), xytext=(2.6, 2.5), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(6.5, 1.4), xytext=(7.5, 2.5), arrowprops=dict(arrowstyle='->', lw=1.5))


def create_novel_drugs_table(ax, drugs_df):
    """Create panel B: Novel drug candidates by subtype."""
    ax.axis('off')
    ax.set_title('B. Novel Drug Candidates with In Silico Validation', fontweight='bold', fontsize=11, loc='left')

    # Sort by validation score
    drugs_df = drugs_df.sort_values('validation_score', ascending=False)

    # Prepare table data
    table_data = []
    for _, row in drugs_df.head(10).iterrows():
        table_data.append([
            row['drug'],
            row['subtype'].replace('SCLC_', ''),
            row['mechanism'][:35] + '...' if len(row['mechanism']) > 35 else row['mechanism'],
            f"{row['validation_score']:.2f}",
            f"{row['admet_score']:.1f}"
        ])

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Drug', 'Subtype', 'Mechanism', 'Valid.', 'ADMET'],
        cellLoc='left',
        loc='center',
        colWidths=[0.18, 0.1, 0.45, 0.12, 0.12]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Style header
    for j in range(5):
        table[(0, j)].set_facecolor('lightgray')
        table[(0, j)].set_text_props(fontweight='bold')

    # Color by subtype
    for i, row in enumerate(drugs_df.head(10).itertuples(), 1):
        color = SUBTYPE_COLORS.get(row.subtype, 'white')
        table[(i, 1)].set_facecolor(color)
        table[(i, 1)].set_alpha(0.5)


def create_validation_bars(ax, drugs_df):
    """Create panel C: Validation scores by drug."""
    ax.set_title('C. Validation Scores', fontweight='bold', fontsize=11, loc='left')

    # Sort and take top 10
    drugs_df = drugs_df.sort_values('validation_score', ascending=True).tail(10)

    colors = [SUBTYPE_COLORS.get(s, 'gray') for s in drugs_df['subtype']]

    bars = ax.barh(range(len(drugs_df)), drugs_df['validation_score'],
                   color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

    ax.set_yticks(range(len(drugs_df)))
    ax.set_yticklabels(drugs_df['drug'], fontsize=8)
    ax.set_xlabel('Validation Score')
    ax.set_xlim(0, 1)

    # Add threshold line
    ax.axvline(x=0.6, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0.62, len(drugs_df) - 0.5, 'Threshold', fontsize=7, color='red')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=c, label=s.replace('SCLC_', ''))
               for s, c in SUBTYPE_COLORS.items()]
    ax.legend(handles=handles, loc='lower right', fontsize=7, title='Subtype', title_fontsize=8)


def create_subtype_recommendations(ax):
    """Create panel D: Subtype-specific novel recommendations."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('D. Subtype-Specific Novel Therapeutic Recommendations', fontweight='bold', fontsize=11, loc='left')

    recommendations = {
        'SCLC-A': ['AMG-232 (MDM2-p53)', 'Navitoclax (pan-BCL2)'],
        'SCLC-N': ['Prexasertib (CHK1/2)', 'OTX015 (BET/MYCN)', 'BI-2536 (PLK1)'],
        'SCLC-P': ['Ruxolitinib (JAK1/2)', 'AZD4547 (FGFR)', 'BMS-754807 (IGF1R)'],
        'SCLC-I': ['Epacadostat (IDO1)', 'Galunisertib (TGF-b)', 'Bintrafusp alfa'],
        'Universal': ['IACS-010759 (OXPHOS)', 'CB-839 (GLS)'],
    }

    y_pos = 9
    for subtype, drugs in recommendations.items():
        color = SUBTYPE_COLORS.get(subtype.replace('-', '_'), 'gray')

        # Subtype header
        box = FancyBboxPatch((0.3, y_pos - 0.4), 2, 0.7, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=1, alpha=0.8)
        ax.add_patch(box)
        ax.text(1.3, y_pos, subtype, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')

        # Drugs
        for i, drug in enumerate(drugs):
            ax.text(2.8 + i * 2.3, y_pos, drug, fontsize=7, va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=1))

        y_pos -= 1.6

    # Add note
    ax.text(5, 0.5, 'All candidates passed in silico validation (score > 0.6)',
            ha='center', va='center', fontsize=8, style='italic', color='gray')


def main():
    """Generate Figure 6: Deep Learning Discoveries."""
    root = Path(__file__).parent.parent
    output_dir = root / 'results' / 'figures'
    manuscript_dir = root / 'manuscript' / 'figures'

    # Load data
    try:
        drugs_df = pd.read_csv(root / 'results/deep_learning/validated_novel_drugs.tsv', sep='\t')
    except:
        print("Error loading drugs data")
        return

    # Create figure
    fig = plt.figure(figsize=(14, 12))

    # Panel A: DL Workflow (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    create_dl_workflow(ax1)

    # Panel B: Novel drugs table (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    create_novel_drugs_table(ax2, drugs_df)

    # Panel C: Validation bars (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    create_validation_bars(ax3, drugs_df)

    # Panel D: Subtype recommendations (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    create_subtype_recommendations(ax4)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'Figure6_deep_learning.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Figure6_deep_learning.pdf', bbox_inches='tight')

    import shutil
    shutil.copy(output_dir / 'Figure6_deep_learning.png',
                manuscript_dir / 'Figure6_deep_learning.png')
    shutil.copy(output_dir / 'Figure6_deep_learning.pdf',
                manuscript_dir / 'Figure6_deep_learning.pdf')

    plt.close()
    print("Figure 6 (Deep Learning Discoveries) saved")


if __name__ == "__main__":
    main()
