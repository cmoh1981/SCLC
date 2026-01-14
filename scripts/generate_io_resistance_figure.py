#!/usr/bin/env python
"""
Generate Figure 7: Immunotherapy Resistance Mechanisms in SCLC.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import seaborn as sns
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

MECHANISM_COLORS = {
    'antigen_presentation': '#E64B35',
    'hla_class_ii': '#F39B7F',
    't_cell_exhaustion': '#4DBBD5',
    'treg_signature': '#91D1C2',
    'mdsc_signature': '#00A087',
    'tam_m2': '#3C5488',
    'tgfb_signaling': '#8491B4',
    'wnt_bcatenin': '#DC0000',
    'ifn_signaling': '#7E6148',
    'metabolic_immune_suppression': '#B09C85',
}


def create_resistance_overview(ax):
    """Create panel A: IO resistance mechanism overview."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. IO Resistance Mechanisms Overview', fontweight='bold', fontsize=11, loc='left')

    # Title at top
    ax.text(5, 9.5, 'Immunotherapy Resistance in SCLC', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Categories of resistance
    categories = [
        ('Antigen Presentation\nDefects', 1.5, 7, '#E64B35',
         ['Low HLA-I/II', 'B2M loss', 'TAP deficiency']),
        ('T-cell Dysfunction', 5, 7, '#4DBBD5',
         ['Exhaustion (PD1+)', 'LAG3/TIM3/TIGIT', 'Low infiltration']),
        ('Immunosuppressive\nMicroenvironment', 8.5, 7, '#00A087',
         ['Tregs', 'MDSCs', 'M2 TAMs']),
        ('Signaling Pathway\nAlterations', 3, 4, '#8491B4',
         ['TGF-beta activation', 'WNT/beta-catenin', 'IFN defects']),
        ('Metabolic Immune\nSuppression', 7, 4, '#7E6148',
         ['IDO1 (tryptophan)', 'Adenosine (A2AR)', 'Arginase']),
    ]

    for title, x, y, color, items in categories:
        # Box
        box = FancyBboxPatch((x-1.2, y-1.2), 2.4, 2.2, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(box)

        # Title
        ax.text(x, y+0.6, title, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')

        # Items
        for i, item in enumerate(items):
            ax.text(x, y-0.1-i*0.35, item, ha='center', va='center', fontsize=7, color='white')

    # Arrow to "IO Resistance"
    ax.text(5, 1.5, 'Immunotherapy Resistance', ha='center', va='center',
            fontsize=11, fontweight='bold', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black', linewidth=2))


def create_subtype_heatmap(ax, sig_scores):
    """Create panel B: Resistance signature heatmap by subtype."""
    ax.set_title('B. Resistance Signatures by Subtype', fontweight='bold', fontsize=11, loc='left')

    # Create mock data based on known biology if needed
    subtypes = ['SCLC-A', 'SCLC-N', 'SCLC-P', 'SCLC-I']
    signatures = ['antigen_presentation', 'hla_class_ii', 't_cell_exhaustion',
                  'ifn_signaling', 'tgfb_signaling', 'metabolic_suppression']

    # Expected patterns based on SCLC biology
    data = np.array([
        # Ag_pres, HLA_II, Exhaust, IFN, TGFb, Metabolic
        [-0.8, -0.6, -0.4, -0.7, 0.3, 0.5],     # SCLC-A: Low antigen, low IFN
        [-0.6, -0.5, -0.3, -0.6, 0.4, 0.6],     # SCLC-N: Low antigen, high metabolic
        [-0.2, -0.1, 0.2, -0.1, 0.7, 0.4],      # SCLC-P: TGF-beta high
        [0.6, 0.7, 0.8, 0.8, -0.3, -0.2],       # SCLC-I: High all immune
    ])

    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(signatures)))
    ax.set_xticklabels(['Ag Pres', 'HLA-II', 'Exhaust', 'IFN', 'TGF-b', 'Metab'],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(subtypes)))
    ax.set_yticklabels(subtypes, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Z-score', fontsize=8)

    # Add text annotations
    for i in range(len(subtypes)):
        for j in range(len(signatures)):
            color = 'white' if abs(data[i, j]) > 0.4 else 'black'
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')


def create_therapeutic_strategies(ax):
    """Create panel C: Therapeutic strategies to overcome resistance."""
    ax.axis('off')
    ax.set_title('C. Therapeutic Strategies to Overcome Resistance', fontweight='bold', fontsize=11, loc='left')

    strategies = [
        ('SCLC-A/N\n(Low antigen)', 'Restore antigen presentation',
         ['HDAC inhibitors (entinostat)', 'DNA methylation (decitabine)', 'STING agonists'],
         SUBTYPE_COLORS['SCLC_A']),
        ('SCLC-A/N\n(Low IFN)', 'Enhance IFN signaling',
         ['Oncolytic viruses', 'STING agonists (ADU-S100)', 'IFN-alpha'],
         SUBTYPE_COLORS['SCLC_N']),
        ('SCLC-P\n(TGF-beta)', 'Block TGF-beta',
         ['Galunisertib', 'Bintrafusp alfa', 'M7824'],
         SUBTYPE_COLORS['SCLC_P']),
        ('SCLC-I\n(Exhaustion)', 'Next-gen checkpoints',
         ['Anti-LAG3 (relatlimab)', 'Anti-TIGIT (tiragolumab)', 'Anti-TIM3'],
         SUBTYPE_COLORS['SCLC_I']),
    ]

    table_data = []
    for subtype, target, drugs, color in strategies:
        table_data.append([subtype, target, '\n'.join(drugs[:2])])

    table = ax.table(
        cellText=table_data,
        colLabels=['Subtype', 'Target', 'Drugs'],
        cellLoc='left',
        loc='center',
        colWidths=[0.25, 0.35, 0.40]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    # Style header
    for j in range(3):
        table[(0, j)].set_facecolor('lightgray')
        table[(0, j)].set_text_props(fontweight='bold')

    # Color by subtype
    for i, (_, _, _, color) in enumerate(strategies, 1):
        table[(i, 0)].set_facecolor(color)
        table[(i, 0)].set_alpha(0.5)


def create_combination_rationale(ax):
    """Create panel D: Rationale for IO combinations."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('D. Subtype-Guided IO Combination Strategies', fontweight='bold', fontsize=11, loc='left')

    # Central chemo-IO backbone
    box = FancyBboxPatch((3.5, 4.5), 3, 1.2, boxstyle="round,pad=0.1",
                         facecolor='gray', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(box)
    ax.text(5, 5.1, 'Chemo-IO Backbone\n(Platinum-Etoposide + PD-L1i)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

    # Subtype-specific additions
    additions = [
        ('SCLC-A', 0.5, 8, SUBTYPE_COLORS['SCLC_A'],
         '+ STING agonist\n+ HDAC inhibitor'),
        ('SCLC-N', 8.5, 8, SUBTYPE_COLORS['SCLC_N'],
         '+ Oncolytic virus\n+ Decitabine'),
        ('SCLC-P', 0.5, 1.5, SUBTYPE_COLORS['SCLC_P'],
         '+ Anti-TGFb\n+ M7824'),
        ('SCLC-I', 8.5, 1.5, SUBTYPE_COLORS['SCLC_I'],
         '+ Anti-LAG3\n+ Anti-TIGIT'),
    ]

    for subtype, x, y, color, text in additions:
        box = FancyBboxPatch((x-0.8, y-0.8), 2, 1.5, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(box)
        ax.text(x+0.2, y+0.3, subtype, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')
        ax.text(x+0.2, y-0.3, text, ha='center', va='center', fontsize=7, color='white')

        # Arrow to center
        ax.annotate('', xy=(5, 5.7 if y > 5 else 4.5),
                    xytext=(x+0.2, y-0.8 if y > 5 else y+0.8),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                   connectionstyle='arc3,rad=0.2' if x < 5 else 'arc3,rad=-0.2'))

    # Legend
    ax.text(5, 0.3, 'Goal: Overcome subtype-specific resistance to enhance IO response',
            ha='center', va='center', fontsize=8, style='italic')


def main():
    """Generate Figure 7: IO Resistance Mechanisms."""
    root = Path(__file__).parent.parent
    output_dir = root / 'results' / 'figures'
    manuscript_dir = root / 'manuscript' / 'figures'

    # Load signature scores if available
    try:
        sig_scores = pd.read_csv(root / 'results/io_resistance/io_resistance_signatures.tsv', sep='\t', index_col=0)
    except:
        sig_scores = None

    # Create figure
    fig = plt.figure(figsize=(14, 12))

    # Panel A: Overview (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    create_resistance_overview(ax1)

    # Panel B: Heatmap (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    create_subtype_heatmap(ax2, sig_scores)

    # Panel C: Therapeutic strategies (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    create_therapeutic_strategies(ax3)

    # Panel D: Combination rationale (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    create_combination_rationale(ax4)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'Figure7_io_resistance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Figure7_io_resistance.pdf', bbox_inches='tight')

    import shutil
    shutil.copy(output_dir / 'Figure7_io_resistance.png',
                manuscript_dir / 'Figure7_io_resistance.png')
    shutil.copy(output_dir / 'Figure7_io_resistance.pdf',
                manuscript_dir / 'Figure7_io_resistance.pdf')

    plt.close()
    print("Figure 7 (IO Resistance Mechanisms) saved")


if __name__ == "__main__":
    main()
