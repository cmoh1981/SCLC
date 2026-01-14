# SCLC Precision Oncology Framework

## Immune-State Stratification Reveals Therapeutic Vulnerabilities in Chemo-Immunotherapy Resistant Small Cell Lung Cancer

[![DOI](https://img.shields.io/badge/DOI-pending-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

A comprehensive computational framework for precision oncology in Small Cell Lung Cancer (SCLC) integrating:

- **Molecular subtype classification** (SCLC-A/N/P/I)
- **Immune microenvironment profiling** (6 immune signatures)
- **Drug repositioning** via DGIdb (1,276 compounds)
- **Genome-scale metabolic modeling** (33 reactions)
- **Deep learning-based drug discovery** (13 novel candidates)
- **Immunotherapy resistance analysis** (12 IO resistance signatures)

### Key Findings

| Finding | Description |
|---------|-------------|
| **Subtype Distribution** | SCLC-P (33.7%), SCLC-I (27.9%), SCLC-N (20.9%), SCLC-A (17.4%) |
| **Immune States** | 4 distinct states orthogonal to molecular subtypes |
| **Metabolic Vulnerability** | OXPHOS conserved across all subtypes |
| **Novel Drug Candidates** | 13 validated (prexasertib, epacadostat, ruxolitinib) |

---

## Project Structure

```
SCLC/
├── data/
│   ├── processed/bulk/          # Expression matrix (35,805 genes x 86 samples)
│   └── gene_sets/               # Curated signatures
├── src/sclc/                    # Core Python modules
│   ├── subtyping.py             # Molecular subtype classification
│   ├── immune.py                # Immune signature scoring
│   ├── drug_repositioning.py   # DGIdb queries
│   ├── metabolic.py             # GEM modeling (COBRApy)
│   ├── deep_learning.py         # VAE + attention classifier
│   └── io_resistance.py         # IO resistance analysis
├── scripts/                     # 16-stage pipeline
│   ├── 01-12_*.py               # Core analysis stages
│   ├── 13_metabolic_modeling.py
│   ├── 14_subtype_therapeutic_strategies.py
│   ├── 15_deep_learning_discovery.py
│   ├── 16_io_resistance_analysis.py
│   └── generate_*_figure.py     # Figure generation
├── results/
│   ├── subtypes/                # Classification results
│   ├── immune/                  # Immune profiling
│   ├── drugs/                   # Drug repositioning
│   ├── metabolic/               # GEM results
│   ├── deep_learning/           # Novel drug discovery
│   ├── io_resistance/           # Resistance mechanisms
│   └── figures/                 # All figures (PDF/PNG)
├── manuscript/
│   ├── main.md                  # Complete manuscript
│   ├── manuscript.docx          # Word format
│   └── figures/                 # Publication-ready figures
├── tables/                      # CSV tables
├── PIPELINE_DOCUMENTATION.md    # Full pipeline documentation
└── README.md                    # This file
```

---

## Installation

### Prerequisites
- Python 3.12+
- Conda (recommended) or pip

### Setup

```bash
# Clone repository
git clone https://github.com/cmoh1981/SCLC.git
cd SCLC

# Create environment
conda create -n sclc python=3.12
conda activate sclc

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.0.0
requests>=2.28.0
cobrapy>=0.26.0
rdkit>=2023.03.1
python-docx>=0.8.11
```

---

## Pipeline Stages

### Stage 1-3: Data Acquisition & QC
```bash
python scripts/01_download_data.py
python scripts/02_preprocess.py
python scripts/03_quality_control.py
```

### Stage 4-6: Subtype & Immune Classification
```bash
python scripts/04_subtype_classification.py
python scripts/05_immune_scoring.py
python scripts/06_immune_clustering.py
```

### Stage 7-12: Analysis & Integration
```bash
python scripts/07_differential_expression.py
python scripts/08_pathway_analysis.py
python scripts/09_drug_repositioning.py
python scripts/10_integration.py
python scripts/11_visualization.py
python scripts/12_validation.py
```

### Stage 13-16: Advanced Analyses
```bash
python scripts/13_metabolic_modeling.py           # GEM + FBA
python scripts/14_subtype_therapeutic_strategies.py
python scripts/15_deep_learning_discovery.py      # VAE + attention
python scripts/16_io_resistance_analysis.py
```

### Figure Generation
```bash
python scripts/generate_subtype_figure.py         # Figure 1
python scripts/generate_immune_figure.py          # Figure 2
python scripts/generate_drug_figure.py            # Figure 3
python scripts/generate_metabolic_figure.py       # Figure 4
python scripts/generate_therapeutic_figure.py     # Figure 5
python scripts/generate_deeplearning_figure.py    # Figure 6
python scripts/generate_io_resistance_figure.py   # Figure 7
```

---

## Main Results

### Subtype-Specific Therapeutic Recommendations

| Subtype | IO Sensitivity | Primary Strategy | Key Drugs |
|---------|---------------|------------------|-----------|
| **SCLC-A** | Low | DLL3-targeting | Tarlatamab, alisertib, venetoclax |
| **SCLC-N** | Low | DNA damage response | PARP inhibitors, prexasertib |
| **SCLC-P** | Moderate | RTK inhibition | FGFR inhibitors, ruxolitinib |
| **SCLC-I** | High | IO intensification | Anti-LAG3, anti-TIGIT, epacadostat |

### Novel Drug Candidates (Deep Learning)

| Drug | Target | Subtype | Validation Score |
|------|--------|---------|------------------|
| Prexasertib | CHK1/2 | SCLC-N | 0.87 |
| Epacadostat | IDO1 | SCLC-I | 0.86 |
| Ruxolitinib | JAK1/2 | SCLC-P | 0.85 |
| CB-839 | GLS | Universal | 0.82 |
| AZD4547 | FGFR | SCLC-P | 0.82 |

### IO Resistance Mechanisms

| Subtype | Primary Resistance | Strategy to Overcome |
|---------|-------------------|----------------------|
| SCLC-A | Low antigen presentation | HDAC inhibitors, decitabine |
| SCLC-N | WNT activation | Oncolytic viruses, STING agonists |
| SCLC-P | TGF-beta signaling | Galunisertib, bintrafusp alfa |
| SCLC-I | T-cell exhaustion | Anti-LAG3, anti-TIGIT |

---

## Figures

| Figure | Description |
|--------|-------------|
| **Figure 1** | SCLC Transcriptional Subtype Landscape |
| **Figure 2** | Immune-State Stratification |
| **Figure 3** | Drug Repositioning Analysis |
| **Figure 4** | Metabolic Reprogramming & OXPHOS Vulnerability |
| **Figure 5** | Subtype-Specific Therapeutic Strategies |
| **Figure 6** | Deep Learning Novel Drug Discovery |
| **Figure 7** | IO Resistance Mechanisms |

All figures available in `results/figures/` (PNG, 300 DPI) and `manuscript/figures/` (PDF).

---

## Data Sources

### Primary Dataset
- **GSE60052**: 86 SCLC samples (George et al., Nature 2015)

### External Databases
- [DGIdb](https://dgidb.org): Drug-gene interactions
- [KEGG](https://www.kegg.jp): Metabolic pathways
- [DrugBank](https://drugbank.com): Drug properties

---

## Manuscript

The complete manuscript is available in:
- `manuscript/main.md` (Markdown)
- `manuscript/manuscript.docx` (Word format for submission)

Target journal: [Signal Transduction and Targeted Therapy](https://www.nature.com/sigtrans/) (IF: 40.8)

---

## Citation

```bibtex
@article{SCLC2024,
  title={Immune-State Stratification Reveals Therapeutic Vulnerabilities
         in Chemo-Immunotherapy Resistant Small Cell Lung Cancer},
  author={[Authors]},
  journal={Signal Transduction and Targeted Therapy},
  year={2024},
  note={Manuscript in preparation}
}
```

---

## Documentation

- **[PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md)**: Complete pipeline documentation
- **[manuscript/main.md](manuscript/main.md)**: Full manuscript text

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- George et al. for the GSE60052 dataset
- DGIdb team for drug-gene interaction database
- COBRApy developers for metabolic modeling framework

---

## Contact

- **Repository**: https://github.com/cmoh1981/SCLC
- **Issues**: https://github.com/cmoh1981/SCLC/issues
