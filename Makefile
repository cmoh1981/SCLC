# SCLC Chemo-IO Resistance Pipeline
# Makefile for reproducible execution
#
# Usage:
#   make stage0       - Bootstrap tools and verify assets
#   make download     - Download all datasets (Stage 1)
#   make preprocess   - Run preprocessing (Stages 2-4)
#   make analyze      - Run analysis (Stages 5-7b)
#   make validate     - Run validation (Stages 8-10)
#   make figures      - Generate figures and tables (Stage 11)
#   make manuscript   - Generate manuscript drafts (Stage 12)
#   make run_all      - Run complete pipeline
#   make clean        - Remove intermediate files
#
# Note: Requires Python 3.9+ with dependencies installed
#       Run `pip install -r requirements.txt` first

PYTHON := python
SCRIPTS := scripts

.PHONY: all stage0 download preprocess analyze validate figures manuscript run_all clean help

# Default target
all: help

help:
	@echo "SCLC Chemo-IO Resistance Pipeline"
	@echo "================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make stage0       Bootstrap tools and verify assets"
	@echo "  make download     Download all datasets (Stage 1)"
	@echo "  make preprocess   Run preprocessing (Stages 2-4)"
	@echo "  make analyze      Run analysis (Stages 5-7b)"
	@echo "  make validate     Run validation (Stages 8-10)"
	@echo "  make figures      Generate figures and tables (Stage 11)"
	@echo "  make manuscript   Generate manuscript drafts (Stage 12)"
	@echo "  make run_all      Run complete pipeline"
	@echo "  make clean        Remove intermediate files"
	@echo ""
	@echo "Requirements:"
	@echo "  - Python 3.9+"
	@echo "  - pip install -r requirements.txt"
	@echo "  - Copy .env.example to .env and fill in API keys"

# Stage 0: Bootstrap
stage0:
	@echo "[Stage 0] Bootstrapping tools and assets..."
	$(PYTHON) $(SCRIPTS)/00_stage0_bootstrap.py

# Stage 1: Download data
download: stage0
	@echo "[Stage 1] Downloading datasets..."
	$(PYTHON) $(SCRIPTS)/01_download_data.py

# Stages 2-4: Preprocessing
preprocess: download
	@echo "[Stage 2] Preprocessing bulk RNA-seq..."
	$(PYTHON) $(SCRIPTS)/02_preprocess_bulk.py
	@echo "[Stage 3] Preprocessing scRNA-seq..."
	$(PYTHON) $(SCRIPTS)/03_preprocess_scrna.py
	@echo "[Stage 4] Preprocessing spatial data..."
	$(PYTHON) $(SCRIPTS)/04_preprocess_spatial.py

# Stages 5-7b: Analysis
analyze: preprocess
	@echo "[Stage 5] Scoring SCLC subtypes..."
	$(PYTHON) $(SCRIPTS)/05_score_subtypes.py
	@echo "[Stage 6] Scoring immune states..."
	$(PYTHON) $(SCRIPTS)/06_score_immune.py
	@echo "[Stage 7] Discovering gene modules..."
	$(PYTHON) $(SCRIPTS)/07_discover_modules.py
	@echo "[Stage 7b] Integrating DisGeNET evidence..."
	$(PYTHON) $(SCRIPTS)/07b_disgenet_evidence.py

# Stages 8-10: Validation
validate: analyze
	@echo "[Stage 8] DepMap validation..."
	$(PYTHON) $(SCRIPTS)/08_depmap_validation.py
	@echo "[Stage 9] Drug repositioning..."
	$(PYTHON) $(SCRIPTS)/09_drug_repositioning.py
	@echo "[Stage 10] Multi-omics validation..."
	$(PYTHON) $(SCRIPTS)/10_multiomics_validation.py

# Stage 11: Figures and tables
figures: validate
	@echo "[Stage 11] Generating figures and tables..."
	$(PYTHON) $(SCRIPTS)/11_figures_tables.py

# Stage 12: Manuscript
manuscript: figures
	@echo "[Stage 12] Generating manuscript drafts..."
	$(PYTHON) $(SCRIPTS)/12_manuscript.py

# Run all stages
run_all: manuscript
	@echo ""
	@echo "Pipeline complete!"
	@echo "Results in: results/"
	@echo "Manuscript in: manuscript/"

# Individual stage shortcuts (skip dependencies)
stage1:
	$(PYTHON) $(SCRIPTS)/01_download_data.py

stage2:
	$(PYTHON) $(SCRIPTS)/02_preprocess_bulk.py

stage3:
	$(PYTHON) $(SCRIPTS)/03_preprocess_scrna.py

stage4:
	$(PYTHON) $(SCRIPTS)/04_preprocess_spatial.py

stage5:
	$(PYTHON) $(SCRIPTS)/05_score_subtypes.py

stage6:
	$(PYTHON) $(SCRIPTS)/06_score_immune.py

stage7:
	$(PYTHON) $(SCRIPTS)/07_discover_modules.py

stage7b:
	$(PYTHON) $(SCRIPTS)/07b_disgenet_evidence.py

stage8:
	$(PYTHON) $(SCRIPTS)/08_depmap_validation.py

stage9:
	$(PYTHON) $(SCRIPTS)/09_drug_repositioning.py

stage10:
	$(PYTHON) $(SCRIPTS)/10_multiomics_validation.py

stage11:
	$(PYTHON) $(SCRIPTS)/11_figures_tables.py

stage12:
	$(PYTHON) $(SCRIPTS)/12_manuscript.py

# Clean intermediate files
clean:
	@echo "Cleaning intermediate files..."
	rm -rf results/intermediate/*
	rm -rf results/logs/*
	rm -rf data/cache/*
	rm -rf .cache/
	@echo "Clean complete. Raw data and final results preserved."

# Deep clean (removes all results)
clean-all: clean
	@echo "Removing all results..."
	rm -rf results/
	rm -rf manuscript/
	@echo "Deep clean complete."

# Check environment
check:
	@echo "Checking Python environment..."
	$(PYTHON) --version
	$(PYTHON) -c "import pandas; print(f'pandas {pandas.__version__}')"
	$(PYTHON) -c "import scanpy; print(f'scanpy {scanpy.__version__}')"
	$(PYTHON) -c "import gseapy; print(f'gseapy {gseapy.__version__}')"
	@echo "Environment OK."
