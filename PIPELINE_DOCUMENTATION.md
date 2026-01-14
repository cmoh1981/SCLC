# SCLC Precision Oncology Pipeline

## Complete Documentation for Reproducibility and Future Projects

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Data Sources](#data-sources)
4. [Pipeline Stages](#pipeline-stages)
5. [Core Modules](#core-modules)
6. [Figure Generation](#figure-generation)
7. [Manuscript Structure](#manuscript-structure)
8. [Key Findings](#key-findings)
9. [Reproduction Guide](#reproduction-guide)
10. [Extension Ideas](#extension-ideas)

---

## Project Overview

### Objective
Develop a comprehensive precision oncology framework for Small Cell Lung Cancer (SCLC) that integrates:
- Molecular subtype classification (SCLC-A/N/P/I)
- Immune microenvironment profiling
- Drug repositioning
- Genome-scale metabolic modeling
- Deep learning-based novel drug discovery
- Immunotherapy resistance mechanism analysis

### Key Deliverables
- 16-stage analysis pipeline
- 7 publication-ready figures
- Complete manuscript (~4,000 words)
- 13 novel drug candidates with in silico validation
- Subtype-specific therapeutic recommendations

---

## Directory Structure

```
SCLC/
├── data/
│   ├── raw/                      # Original downloaded data
│   ├── processed/
│   │   └── bulk/
│   │       └── expression_matrix.tsv
│   └── gene_sets/                # Curated gene signatures
│
├── src/
│   └── sclc/
│       ├── __init__.py
│       ├── data.py               # Data loading utilities
│       ├── subtyping.py          # Molecular subtype classification
│       ├── immune.py             # Immune signature scoring
│       ├── drug_repositioning.py # DGIdb drug queries
│       ├── metabolic.py          # GEM modeling (COBRApy)
│       ├── deep_learning.py      # VAE + attention classifier
│       └── io_resistance.py      # IO resistance analysis
│
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
│   ├── 03_quality_control.py
│   ├── 04_subtype_classification.py
│   ├── 05_immune_scoring.py
│   ├── 06_immune_clustering.py
│   ├── 07_differential_expression.py
│   ├── 08_pathway_analysis.py
│   ├── 09_drug_repositioning.py
│   ├── 10_integration.py
│   ├── 11_visualization.py
│   ├── 12_validation.py
│   ├── 13_metabolic_modeling.py
│   ├── 14_subtype_therapeutic_strategies.py
│   ├── 15_deep_learning_discovery.py
│   ├── 16_io_resistance_analysis.py
│   ├── generate_subtype_figure.py
│   ├── generate_immune_figure.py
│   ├── generate_drug_figure.py
│   ├── generate_metabolic_figure.py
│   ├── generate_therapeutic_figure.py
│   ├── generate_deeplearning_figure.py
│   └── generate_io_resistance_figure.py
│
├── results/
│   ├── subtypes/                 # Subtype classification results
│   ├── immune/                   # Immune scoring results
│   ├── drugs/                    # Drug repositioning results
│   ├── metabolic/                # GEM modeling results
│   ├── deep_learning/            # Novel drug discovery results
│   ├── io_resistance/            # IO resistance analysis
│   ├── therapeutic_strategies/   # Subtype-specific strategies
│   ├── figures/                  # All generated figures
│   └── tables/                   # Summary tables
│
├── manuscript/
│   ├── main.md                   # Complete manuscript
│   └── figures/                  # Publication-ready figures
│
├── external/                     # External tools/databases
├── tests/                        # Unit tests
├── PIPELINE_DOCUMENTATION.md     # This file
├── README.md
└── requirements.txt
```

---

## Data Sources

### Primary Dataset
| Source | Accession | Description |
|--------|-----------|-------------|
| GEO | GSE60052 | 86 SCLC samples (79 tumor, 7 normal) |
| Publication | George et al., Nature 2015 | Comprehensive genomic profiles |

### External Databases
| Database | Purpose | URL |
|----------|---------|-----|
| DGIdb | Drug-gene interactions | https://dgidb.org |
| KEGG | Metabolic pathways | https://www.kegg.jp |
| Recon3D | Human metabolic network | https://vmh.life |
| DrugBank | Drug properties | https://drugbank.com |

### Gene Signatures (Curated)
```python
# Subtype markers
SCLC_A_GENES = ['ASCL1', 'DLL3', 'SOX1', 'GRP', 'CHGA', 'SYP', 'NCAM1', 'INSM1']
SCLC_N_GENES = ['NEUROD1', 'NEUROD2', 'NEUROD4', 'HES6', 'MYT1', 'MYT1L']
SCLC_P_GENES = ['POU2F3', 'ASCL2', 'AVIL', 'TRPM5', 'SOX9', 'GFI1B']
SCLC_I_GENES = ['CD274', 'PDCD1LG2', 'IDO1', 'CXCL10', 'HLA-DRA', 'STAT1']

# Immune signatures (6 total)
T_EFFECTOR = ['CD8A', 'GZMA', 'GZMB', 'PRF1', 'IFNG', 'CXCL9', 'CXCL10']
IFN_GAMMA = ['STAT1', 'IRF1', 'IDO1', 'CXCL9', 'CXCL10', 'GBP1']
ANTIGEN_PRESENTATION = ['HLA-A', 'HLA-B', 'HLA-C', 'B2M', 'TAP1', 'TAP2']
MYELOID_TAM = ['CD68', 'CD163', 'CSF1R', 'MSR1', 'MRC1']
TREG = ['FOXP3', 'IL2RA', 'CTLA4', 'TIGIT']
EXHAUSTION = ['PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX']

# IO resistance signatures (12 total)
IO_RESISTANCE_SIGNATURES = {
    'antigen_presentation': [...],
    'hla_class_ii': [...],
    't_cell_exhaustion': [...],
    'treg_signature': [...],
    'mdsc_signature': [...],
    'tam_m2': [...],
    'tgfb_signaling': [...],
    'wnt_bcatenin': [...],
    'ifn_signaling': [...],
    'metabolic_immune_suppression': [...],
    'caf_exclusion': [...],
    'angiogenesis': [...]
}
```

---

## Pipeline Stages

### Stage 1-3: Data Acquisition & QC
```bash
python scripts/01_download_data.py    # Download GSE60052
python scripts/02_preprocess.py       # Normalize, filter
python scripts/03_quality_control.py  # QC metrics
```

**Output:**
- `data/processed/bulk/expression_matrix.tsv` (35,805 genes x 86 samples)

### Stage 4: Molecular Subtype Classification
```bash
python scripts/04_subtype_classification.py
```

**Method:** ssGSEA scoring with subtype-specific gene signatures

**Output:**
- `results/subtypes/subtype_calls.tsv`
- Distribution: SCLC-P (33.7%), SCLC-I (27.9%), SCLC-N (20.9%), SCLC-A (17.4%)

### Stage 5-6: Immune Profiling
```bash
python scripts/05_immune_scoring.py
python scripts/06_immune_clustering.py
```

**Method:** 6-signature immune scoring + hierarchical clustering

**Output:**
- `results/immune/immune_scores.tsv`
- 4 immune states identified (desert, excluded, mixed, hot)

### Stage 7-8: Differential Expression & Pathways
```bash
python scripts/07_differential_expression.py
python scripts/08_pathway_analysis.py
```

**Output:**
- Subtype-specific DE genes
- Enriched pathways per subtype

### Stage 9: Drug Repositioning
```bash
python scripts/09_drug_repositioning.py
```

**Method:** DGIdb GraphQL API query of 57 SCLC genes

**Output:**
- `results/drugs/drug_rankings.tsv`
- 1,276 unique compounds, 1,911 drug-gene interactions
- Top hits: cisplatin, Aurora kinase inhibitors, PARP inhibitors

### Stage 10-12: Integration & Validation
```bash
python scripts/10_integration.py
python scripts/11_visualization.py
python scripts/12_validation.py
```

### Stage 13: Metabolic Modeling
```bash
python scripts/13_metabolic_modeling.py
```

**Method:**
- COBRApy-based GEM with 33 reactions
- GIMME-like transcriptomic integration
- Flux Balance Analysis (FBA)

**Output:**
- `results/metabolic/flux_analysis.tsv`
- `results/metabolic/metabolic_drugs.tsv`
- Key finding: OXPHOS is conserved vulnerability across all subtypes

### Stage 14: Subtype-Specific Therapeutic Strategies
```bash
python scripts/14_subtype_therapeutic_strategies.py
```

**Output:**
- `results/therapeutic_strategies/strategy_summary.json`

**Recommendations:**
| Subtype | Primary Strategy | Key Drugs |
|---------|-----------------|-----------|
| SCLC-A | DLL3-targeting | Tarlatamab, alisertib, venetoclax |
| SCLC-N | DNA damage response | PARP inhibitors, Aurora kinase inhibitors |
| SCLC-P | RTK inhibition | FGFR inhibitors, IGF1R inhibitors |
| SCLC-I | IO intensification | Anti-LAG3, anti-TIGIT, IDO1 inhibitors |

### Stage 15: Deep Learning Novel Drug Discovery
```bash
python scripts/15_deep_learning_discovery.py
```

**Architecture:**
```
Gene Expression (15,000 genes)
         │
         ▼
┌─────────────────────────────────┐
│  Variational Autoencoder (VAE)  │
│  - Encoder: 5000 → 128 → 32     │
│  - Latent space: 32 dimensions  │
│  - Decoder: 32 → 128 → 5000     │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Attention-based Classifier     │
│  - Input: gene expression       │
│  - Attention: gene importance   │
│  - Output: 4 subtypes           │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Drug-Target Interaction        │
│  - Morgan fingerprints (2048)   │
│  - Target expression features   │
│  - Efficacy prediction          │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  In Silico Validation           │
│  - Docking score (30%)          │
│  - Binding affinity (30%)       │
│  - Selectivity (20%)            │
│  - Drug-likeness (20%)          │
└─────────────────────────────────┘
```

**Output:**
- `results/deep_learning/validated_novel_drugs.tsv`
- 200 novel targets identified
- 13 novel drug candidates (validation score > 0.6)

**Novel Drug Candidates:**
| Drug | Target | Subtype | Validation Score |
|------|--------|---------|------------------|
| Prexasertib | CHK1/2 | SCLC-N | 0.87 |
| Epacadostat | IDO1 | SCLC-I | 0.86 |
| Ruxolitinib | JAK1/2 | SCLC-P | 0.85 |
| CB-839 | GLS | Universal | 0.82 |
| AZD4547 | FGFR | SCLC-P | 0.82 |
| OTX015 | BET/BRD4 | SCLC-N | 0.81 |
| Navitoclax | BCL2 family | SCLC-A | 0.80 |
| BMS-754807 | IGF1R/IR | SCLC-P | 0.80 |
| IACS-010759 | Complex I | Universal | 0.77 |
| Bintrafusp alfa | TGF-β/PD-L1 | SCLC-I | 0.77 |
| AMG-232 | MDM2 | SCLC-A | 0.72 |
| Galunisertib | TGF-βR | SCLC-I | 0.72 |
| BI-2536 | PLK1 | SCLC-N | 0.67 |

### Stage 16: IO Resistance Analysis
```bash
python scripts/16_io_resistance_analysis.py
```

**Method:** 12 IO resistance signature scoring by subtype

**Output:**
- `results/io_resistance/resistance_mechanisms.tsv`
- `results/io_resistance/resistance_therapeutics.tsv`

**Resistance Mechanisms by Subtype:**
| Subtype | Primary Resistance | Strategy |
|---------|-------------------|----------|
| SCLC-A | Low antigen presentation, IFN defects | HDAC inhibitors, STING agonists |
| SCLC-N | WNT activation, metabolic suppression | Decitabine, oncolytic viruses |
| SCLC-P | TGF-β signaling, CAF infiltration | Galunisertib, bintrafusp alfa |
| SCLC-I | T-cell exhaustion (adaptive) | Anti-LAG3, anti-TIGIT |

---

## Core Modules

### src/sclc/subtyping.py
```python
def classify_subtypes(expression_df, method='ssgsea'):
    """
    Classify SCLC samples into A/N/P/I subtypes.

    Parameters:
        expression_df: Gene expression matrix (genes x samples)
        method: 'ssgsea' or 'correlation'

    Returns:
        DataFrame with subtype calls and scores
    """
```

### src/sclc/immune.py
```python
def score_immune_signatures(expression_df):
    """
    Calculate 6 immune signature scores per sample.

    Signatures:
        - t_effector, ifn_gamma, antigen_presentation
        - myeloid_tam, treg, exhaustion

    Returns:
        DataFrame of z-scored signature values
    """

def cluster_immune_states(scores_df, n_clusters=4):
    """
    Hierarchical clustering to identify immune states.
    """
```

### src/sclc/drug_repositioning.py
```python
def query_dgidb(genes, interaction_types=['inhibitor', 'antagonist']):
    """
    Query DGIdb GraphQL API for drug-gene interactions.

    Returns:
        DataFrame with drug, gene, interaction_type, sources
    """

def rank_drugs(interactions_df):
    """
    Rank drugs by target coverage and evidence quality.

    Score = log2(n_targets + 1) * evidence_weight
    """
```

### src/sclc/metabolic.py
```python
def build_sclc_gem():
    """
    Build genome-scale metabolic model with 33 reactions.

    Pathways:
        - Glycolysis, TCA cycle, OXPHOS
        - Glutaminolysis, PPP, one-carbon
        - Serine biosynthesis, fatty acid synthesis
    """

def integrate_transcriptomics(model, expression_df, method='gimme'):
    """
    GIMME-like integration of expression data.
    Scale reaction bounds by gene expression.
    """

def run_fba(model, objective='biomass'):
    """
    Flux Balance Analysis to predict metabolic fluxes.
    """
```

### src/sclc/deep_learning.py
```python
class VAE(nn.Module):
    """Variational Autoencoder for gene expression."""
    def __init__(self, input_dim=5000, hidden_dim=128, latent_dim=32):
        ...

class AttentionClassifier(nn.Module):
    """Attention-based subtype classifier."""
    def __init__(self, input_dim, hidden_dim=64, n_classes=4):
        ...

class DrugTargetPredictor:
    """Predict drug-target interactions using molecular fingerprints."""
    def compute_fingerprint(self, smiles):
        ...
    def predict_efficacy(self, drug, target_expression):
        ...

class InSilicoValidator:
    """Validate drug candidates with ADMET and docking."""
    def validate(self, drug_info):
        # Returns: docking_score, binding_affinity, selectivity, drug_likeness
        ...
```

### src/sclc/io_resistance.py
```python
IO_RESISTANCE_SIGNATURES = {
    'antigen_presentation': {...},
    't_cell_exhaustion': {...},
    'tgfb_signaling': {...},
    # ... 12 total signatures
}

def score_resistance_signatures(expression_df):
    """Score 12 IO resistance signatures per sample."""

def identify_resistance_mechanisms(scores_df, subtype_df):
    """Map resistance mechanisms to each subtype."""

def suggest_therapeutics(mechanisms_df):
    """Recommend drugs to overcome resistance."""
```

---

## Figure Generation

### Figure 1: Subtype Landscape
```bash
python scripts/generate_subtype_figure.py
```
- Panel A: Subtype distribution pie chart
- Panel B: PCA by subtype
- Panel C: Marker gene heatmap

### Figure 2: Immune Stratification
```bash
python scripts/generate_immune_figure.py
```
- Panel A: Immune signature clustering
- Panel B: Immune state × subtype distribution
- Panel C: Signature correlation matrix

### Figure 3: Drug Repositioning
```bash
python scripts/generate_drug_figure.py
```
- Panel A: Analysis workflow
- Panel B: Top 20 drugs by target coverage
- Panel C: Drug-target network
- Panel D: Pathway enrichment

### Figure 4: Metabolic Modeling
```bash
python scripts/generate_metabolic_figure.py
```
- Panel A: Metabolic network schematic
- Panel B: Flux heatmap by subtype
- Panel C: Top metabolic drug targets
- Panel D: Pathway vulnerability

### Figure 5: Therapeutic Strategies
```bash
python scripts/generate_therapeutic_figure.py
```
- Panel A: Subtype overview with IO sensitivity
- Panel B: Drug-subtype recommendation matrix
- Panel C: Treatment algorithm
- Panel D: Key clinical trials

### Figure 6: Deep Learning Discovery
```bash
python scripts/generate_deeplearning_figure.py
```
- Panel A: Computational workflow
- Panel B: Novel drug candidates table
- Panel C: Validation scores bar chart
- Panel D: Subtype-specific recommendations

### Figure 7: IO Resistance
```bash
python scripts/generate_io_resistance_figure.py
```
- Panel A: Resistance mechanism overview
- Panel B: Resistance signature heatmap
- Panel C: Therapeutic strategies table
- Panel D: Combination strategy diagram

---

## Manuscript Structure

### Sections (~4,000 words)
1. **Abstract** (250 words)
   - Background, Methods, Results, Conclusions

2. **Introduction** (500 words)
   - SCLC epidemiology and treatment landscape
   - Chemo-IO and resistance
   - Molecular subtypes (A/N/P/I)
   - Study rationale

3. **Results** (2,000 words)
   - Patient cohort and subtype classification
   - Immune-state stratification
   - IO resistance mechanisms
   - Drug repositioning
   - Metabolic modeling
   - Subtype-specific strategies
   - Deep learning novel discovery

4. **Discussion** (1,000 words)
   - Key findings interpretation
   - Clinical implications
   - Limitations

5. **Methods** (500 words)
   - Data sources
   - Computational methods
   - Statistical analysis

6. **References** (40 citations)

### Tables
| Table | Content |
|-------|---------|
| Table 1 | Top drug candidates from DGIdb |
| Table 2 | Subtype-specific therapeutic recommendations |
| Table 3 | Novel drug candidates from deep learning |
| Table 4 | IO resistance mechanisms by subtype |

---

## Key Findings

### 1. Subtype Distribution
- SCLC-P: 33.7% (n=29)
- SCLC-I: 27.9% (n=24)
- SCLC-N: 20.9% (n=18)
- SCLC-A: 17.4% (n=15)

### 2. Immune States
- 4 distinct immune states identified
- Immune state orthogonal to molecular subtype
- 38% of "immune hot" tumors are non-SCLC-I

### 3. Drug Repositioning
- 1,276 compounds identified
- Top classes: Aurora kinase inhibitors, PARP inhibitors, multi-kinase inhibitors

### 4. Metabolic Vulnerability
- OXPHOS is conserved vulnerability across all subtypes
- Metformin, IACS-010759 as universal combinations

### 5. Novel Drug Candidates
- 13 candidates validated in silico
- All passed validation threshold (>0.6)
- Best: Prexasertib (0.87), Epacadostat (0.86), Ruxolitinib (0.85)

### 6. IO Resistance Mechanisms
- SCLC-A/N: Antigen presentation defects
- SCLC-P: TGF-β signaling
- SCLC-I: T-cell exhaustion

---

## Reproduction Guide

### Environment Setup
```bash
# Create conda environment
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
```

### Run Full Pipeline
```bash
# Stage 1-3: Data
python scripts/01_download_data.py
python scripts/02_preprocess.py
python scripts/03_quality_control.py

# Stage 4-6: Classification & Immune
python scripts/04_subtype_classification.py
python scripts/05_immune_scoring.py
python scripts/06_immune_clustering.py

# Stage 7-8: DE & Pathways
python scripts/07_differential_expression.py
python scripts/08_pathway_analysis.py

# Stage 9: Drug Repositioning
python scripts/09_drug_repositioning.py

# Stage 10-12: Integration
python scripts/10_integration.py
python scripts/11_visualization.py
python scripts/12_validation.py

# Stage 13-16: Advanced Analyses
python scripts/13_metabolic_modeling.py
python scripts/14_subtype_therapeutic_strategies.py
python scripts/15_deep_learning_discovery.py
python scripts/16_io_resistance_analysis.py

# Generate Figures
python scripts/generate_subtype_figure.py
python scripts/generate_immune_figure.py
python scripts/generate_drug_figure.py
python scripts/generate_metabolic_figure.py
python scripts/generate_therapeutic_figure.py
python scripts/generate_deeplearning_figure.py
python scripts/generate_io_resistance_figure.py
```

---

## Extension Ideas

### For Future Projects

1. **Single-cell Integration**
   - Integrate scRNA-seq data for cell-type deconvolution
   - Use CellTypist or scType for annotation
   - Spatial transcriptomics for TME architecture

2. **Multi-omics Integration**
   - Add proteomics (mass spec)
   - Add methylation data
   - Integrate with CRISPR screens

3. **Clinical Validation**
   - Match with treatment response data
   - Survival analysis by subtype
   - Biomarker validation cohort

4. **Drug Synergy Prediction**
   - Predict synergistic combinations
   - Network-based synergy scoring
   - Validate with drug screen data

5. **Resistance Evolution**
   - Longitudinal sampling analysis
   - Subtype switching detection
   - Clonal evolution modeling

6. **Deep Learning Extensions**
   - Graph Neural Networks for drug-target
   - Transformer for expression embeddings
   - Reinforcement learning for treatment sequencing

### Adaptable to Other Cancers
This pipeline can be adapted for:
- NSCLC (non-small cell lung cancer)
- Pancreatic cancer
- Triple-negative breast cancer
- Any cancer with molecular subtypes and IO relevance

**Modification steps:**
1. Replace subtype gene signatures
2. Update drug gene list
3. Adjust metabolic model for tissue type
4. Retrain deep learning models

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-XX | Initial pipeline (Stages 1-12) |
| 1.1 | 2024-XX | Added metabolic modeling (Stage 13) |
| 1.2 | 2024-XX | Added therapeutic strategies (Stage 14) |
| 1.3 | 2024-XX | Added deep learning discovery (Stage 15) |
| 1.4 | 2024-XX | Added IO resistance analysis (Stage 16) |

---

## Citation

If using this pipeline, please cite:
```
[Author names]. Immune-State Stratification Reveals Therapeutic
Vulnerabilities in Chemo-Immunotherapy Resistant Small Cell Lung Cancer.
[Journal]. 2024.
```

---

## Contact

- GitHub: https://github.com/cmoh1981/SCLC
- Issues: https://github.com/cmoh1981/SCLC/issues

---

*Generated with Claude Code assistance*
