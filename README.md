# SCLC Chemo-Immunotherapy Resistance Analysis Pipeline

A reproducible, config-driven bioinformatics pipeline for analyzing primary resistance to first-line chemotherapy-immunotherapy (Chemo-IO) in Small Cell Lung Cancer (SCLC).

## Project Overview

This pipeline integrates multi-omics public datasets to:
1. Classify SCLC transcriptional subtypes (SCLC-A, SCLC-N, SCLC-P, SCLC-I)
2. Stratify immune states (hot/cold, exhausted/functional)
3. Discover resistance-associated gene modules via WGCNA
4. Prioritize drug repositioning candidates using 3-leg evidence

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| 0 | Bootstrap tools & verify assets | Tool inventory |
| 1 | Download GEO/SRA/external data | Raw data files |
| 2-4 | Preprocess bulk/scRNA/spatial | Processed matrices |
| 5 | Score SCLC subtypes (ssGSEA) | Subtype assignments |
| 6 | Score immune axes | Immune state labels |
| 7 | WGCNA module discovery | Hub genes |
| 7b | DisGeNET evidence integration | Disease associations |
| 8 | DepMap validation (optional) | Dependency scores |
| 9 | Drug repositioning (3-leg) | Ranked drug list |
| 10 | Multi-omics validation | Validation report |
| 11 | Generate figures & tables | Fig1-Fig5, Tables |
| 12 | Auto-draft manuscript | main.md, methods.md |

## Quick Start

```bash
# 1. Clone repository
git clone <repo_url>
cd SCLC

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run pipeline
make run_all
```

## Directory Structure

```
SCLC/
├── configs/                  # YAML configuration files
│   ├── pipeline.yaml         # Stage enables, paths, thresholds
│   ├── cohorts.yaml          # Dataset definitions
│   ├── signatures.yaml       # Gene signatures
│   └── stage0_assets.yaml    # Tools and data inventory
├── data/
│   ├── raw/                  # Downloaded raw data
│   ├── processed/            # Preprocessed matrices
│   └── external/             # External databases
├── scripts/                  # Stage execution scripts
│   ├── 00_stage0_bootstrap.py
│   ├── 01_download_data.py
│   └── ...
├── src/sclc/                 # Core Python modules
│   ├── config.py             # Configuration management
│   ├── utils.py              # Logging, manifests
│   ├── scoring.py            # ssGSEA implementation
│   ├── modules.py            # WGCNA wrapper
│   └── ...
├── results/                  # Analysis outputs
│   ├── figures/              # Publication-ready figures
│   ├── tables/               # Data tables
│   └── logs/                 # Execution logs
├── manuscript/               # Auto-generated drafts
├── tools/                    # Cloned external tools
├── Makefile                  # Build targets
└── README.md
```

## Configuration

All parameters are controlled via YAML files in `configs/`:

- **pipeline.yaml**: Enable/disable stages, set paths and thresholds
- **cohorts.yaml**: Define datasets (GEO IDs, access type, modality)
- **signatures.yaml**: Gene signatures for subtype/immune scoring

Example stage enable/disable:
```yaml
stages:
  stage8:
    enabled: false  # Skip DepMap if data not available
```

## Key Outputs

### Figures (300 DPI)
- **Fig1**: SCLC subtype UMAP with immune overlay
- **Fig2**: Immune-stratified survival curves
- **Fig3**: Module-trait heatmap + DisGeNET evidence
- **Fig4**: DepMap dependency waterfall (if enabled)
- **Fig5**: Drug repositioning evidence triangle

### Tables
- **Table1**: Dataset summary and sample sizes
- **Table2**: Hub gene annotations
- **Table3**: Top 20 repositioning candidates

### Manuscript
- **main.md**: Main text (Introduction, Results, Discussion)
- **methods.md**: Detailed methods section
- **supplement.md**: Supplementary information

## Requirements

- Python 3.9+
- Key dependencies: pandas, scanpy, gseapy, networkx, matplotlib
- Optional: R (for WGCNA if using native implementation)

## Data Access

### Open Access
- GSE60052, GSE138267, GSE138474, GSE267310 (GEO)
- PRJNA1014231, PRJNA575243 (SRA)
- ST000220 (Metabolomics Workbench)

### Controlled Access (requires application)
- EGAS00001007634 (EGA)
- NGDC datasets (NGDC)
- PDC000127 (PDC - CPTAC)

See `configs/controlled_access_datasets.md` for access instructions.

## Reproducibility

Every stage generates:
1. **Manifest file**: Input/output checksums, parameters, timestamps
2. **Log file**: Detailed execution log with warnings/errors

Re-run any stage:
```bash
make stage5  # Re-run only Stage 5
```

## Citation

If you use this pipeline, please cite:
- [Publication pending]
- Individual dataset citations as listed in their GEO/SRA pages

## License

MIT License - see LICENSE file
