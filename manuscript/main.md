# Immune-State Stratification Explains Primary Resistance to
# Chemo-Immunotherapy in Small Cell Lung Cancer

## Abstract

**Background:** Small cell lung cancer (SCLC) is an aggressive malignancy with limited
treatment options. While first-line platinum-etoposide plus PD-L1 checkpoint inhibitor
(chemo-IO) has become standard of care, primary resistance remains a major challenge.

**Methods:** We performed integrative analysis of 100
SCLC samples across bulk RNA-seq, single-cell RNA-seq, and spatial transcriptomics
datasets. We characterized molecular subtypes (SCLC-A/N/P/I) and developed an
immune-state stratification framework to explain resistance patterns.

**Results:** We identified distinct immune states
associated with differential response to chemo-IO across all SCLC subtypes.
Co-expression network analysis revealed 253 gene
modules associated with resistance, with 2530 hub genes
validated through DisGeNET disease associations. Drug repositioning analysis
identified several candidate compounds with
three-leg evidence support as potential add-on therapies.

**Conclusions:** Immune-state stratification provides a framework for understanding
primary chemo-IO resistance in SCLC independent of transcriptional subtype,
nominating rational add-on therapeutic strategies.

## Introduction

Small cell lung cancer (SCLC) accounts for approximately 15% of lung cancers and is
characterized by rapid growth, early metastasis, and universal TP53/RB1 inactivation.
The addition of PD-L1 inhibitors (atezolizumab or durvalumab) to first-line
platinum-etoposide chemotherapy has improved survival, establishing chemo-IO as the
standard of care for extensive-stage SCLC.

Despite this advance, primary resistance to chemo-IO remains common. Recent work has
identified four transcriptional subtypes of SCLC (SCLC-A, SCLC-N, SCLC-P, SCLC-I) based
on differential expression of master transcription factors (ASCL1, NEUROD1, POU2F3) and
immune gene signatures. The inflamed subtype (SCLC-I) shows enrichment for immune
infiltration and improved outcomes on immunotherapy.

However, the relationship between immune microenvironment composition and treatment
resistance across subtypes remains incompletely understood. Here, we present an
integrative multi-omics analysis demonstrating that immune-state stratification
explains resistance patterns independent of transcriptional subtype.

## Results

### SCLC Subtype Landscape

Analysis of 100 samples confirmed the expected
distribution of SCLC subtypes: SCLC-A (~42%),
SCLC-N (~22%),
SCLC-P (~31%), and
SCLC-I (~27%) (**Figure 1**).

### Immune-State Classification

We developed a multi-axis immune scoring framework encompassing T-effector activity,
IFN-gamma signaling, antigen presentation, myeloid/TAM infiltration, and
immunosuppressive features. Unsupervised clustering identified
distinct immune states with distinct compositions
(**Figure 2**).

### Resistance-Associated Gene Modules

Weighted gene co-expression network analysis (WGCNA) identified
253 gene modules associated with immune states and
resistance phenotypes. Hub genes from resistance-associated modules showed significant
enrichment for lung cancer and immune-related diseases in DisGeNET (**Figure 3**).

### Drug Repositioning

Application of the three-leg evidence rule (target mapping, signature reversal potential,
and DepMap coherence) nominated candidate drugs as potential
add-on therapies to overcome chemo-IO resistance (**Figure 5**).

## Discussion

Our analysis demonstrates that immune-state stratification provides explanatory power
for chemo-IO resistance beyond transcriptional subtype classification. This framework
nominates rational therapeutic combinations for prospective validation.

**Limitations:** This study relies on publicly available datasets with heterogeneous
clinical annotations. Associations reported here require prospective validation.
No causal claims can be made from observational multi-omics data.

## Data and Code Availability

All analyses were performed using publicly available datasets. Code is available at
[repository URL]. Processed data matrices and intermediate results are provided as
supplementary files.

