# Methods

## Data Sources

All datasets used in this study are publicly available through indicated repositories
(see Table 1). Open-access datasets were downloaded programmatically; controlled-access
datasets require appropriate data access applications.

## Preprocessing

### Bulk RNA-seq
Raw count matrices were filtered to remove lowly-expressed genes (< 10 total counts,
detected in < 10% of samples). Expression values were log2-transformed after adding a
pseudocount of 1. Gene symbols were standardized to HGNC nomenclature.

### Single-cell RNA-seq
Single-cell data were processed using Scanpy. Cells with < 200 genes, > 8000 genes, or
> 20% mitochondrial reads were excluded. Data were normalized to 10,000 counts per cell
and log-transformed. Highly variable genes (n=2000) were selected for dimensionality
reduction.

### Spatial Transcriptomics
Spatial data (10x Visium or GeoMX DSP) were normalized per spot/ROI and log-transformed.
Quality metrics were computed and low-quality regions excluded.

## Subtype Scoring

SCLC subtypes (A, N, P, I) were assigned using single-sample GSEA (ssGSEA) with
established gene signatures. Each sample was assigned to the subtype with the highest
enrichment score.

## Immune-State Scoring

Immune axis scores were computed for:
- T-cell effector/cytotoxic activity
- IFN-gamma response
- Antigen presentation
- Myeloid/TAM infiltration
- Regulatory/immunosuppressive features

Samples were clustered into immune states using k-means clustering on z-scaled scores.

## Module Discovery

Weighted gene co-expression network analysis (WGCNA) was performed to identify
co-expression modules. Soft-thresholding power was selected to achieve approximate
scale-free topology. Module eigengenes were correlated with immune states and clinical
annotations.

## DisGeNET Evidence

Hub genes were queried against DisGeNET v7 via REST API. Associations were filtered to
SCLC, lung cancer, and immune-related disease terms.

## Drug Repositioning

Candidate drugs were identified using a three-leg evidence rule:
1. Target mapping: DGIdb drug-gene interactions for hub genes
2. Signature reversal: LINCS/CMap analysis (when available)
3. DepMap coherence: Consistency with dependency/drug response data

## Software Versions

- Python: 3.12.10
- NumPy: 2.3.5
- Pandas: 2.3.3
- Scanpy: 1.11.5
- Seaborn: 0.13.2
- Matplotlib: 3.10.8

## Reproducibility Statement

All analyses are reproducible from configuration files and scripts provided in the
repository. Random seed was set to 42 for all stochastic operations. Manifest files
with file hashes and parameters are provided for each analysis stage.

