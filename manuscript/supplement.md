# Supplementary Information

## Immune-State Stratification Reveals Therapeutic Vulnerabilities in Chemo-Immunotherapy Resistant Small Cell Lung Cancer

---

## Supplementary Methods

### Gene Signature Curation

Gene signatures for SCLC subtypes and immune states were curated from published literature:

**SCLC Transcriptional Subtypes** (Rudin et al., Nat Rev Cancer 2019; Gay et al., Cancer Cell 2021):
- **SCLC-A** (10 genes): ASCL1, DLL3, SOX1, GRP, CHGA, SYP, NCAM1, INSM1, FOXA2, NKX2-1
- **SCLC-N** (8 genes): NEUROD1, NEUROD2, NEUROD4, HES6, ASCL2, MYT1, MYT1L, KIF5C
- **SCLC-P** (9 genes): POU2F3, ASCL2, AVIL, TRPM5, SOX9, GFI1B, CHAT, LRMP, IL25
- **SCLC-I** (15 genes): CD274, PDCD1LG2, IDO1, CXCL10, HLA-DRA, HLA-DRB1, STAT1, IRF1, GZMA, GZMB, PRF1, CD8A, CD4, TIGIT, LAG3

**Immune Signatures** (Ayers et al., J Clin Invest 2017; Cristescu et al., Science 2018):
- **T-effector** (14 genes): CD8A, CD8B, GZMA, GZMB, PRF1, IFNG, CXCL9, CXCL10, TBX21, EOMES, STAT4, STAT1, IRF1, RUNX3
- **IFN-gamma response** (16 genes): IFNG, STAT1, IDO1, CXCL9, CXCL10, CXCL11, HLA-DRA, HLA-DRB1, HLA-E, IRF1, IRF9, GBP1, GBP4, GBP5, PSMB9, TAP1
- **Antigen presentation** (16 genes): HLA-A, HLA-B, HLA-C, B2M, TAP1, TAP2, PSMB8, PSMB9, PSMB10, CIITA, RFX5, NLRC5, CD74, HLA-DMA, HLA-DMB, HLA-DOA
- **Myeloid/TAM** (15 genes): CD68, CD163, MSR1, CSF1R, CSF1, CCL2, MRC1, MARCO, TREM2, SIGLEC1, IL10, VEGFA, MMP9, ARG1, HIF1A
- **Treg/immunosuppression** (13 genes): FOXP3, IL2RA, CTLA4, TIGIT, ICOS, TNFRSF18, CCR4, CCR8, IL10, TGFB1, IDO1, ENTPD1, NT5E
- **Exhaustion** (9 genes): PDCD1, LAG3, HAVCR2, TIGIT, BTLA, CD244, CD160, ENTPD1, TOX

### Drug-Gene Interaction Database Query

Drug repositioning was performed using the DGIdb v4.0 GraphQL API (https://dgidb.org/api/graphql). The following query structure was used:

```graphql
query {
  genes(names: ["GENE_LIST"]) {
    nodes {
      name
      interactions {
        drug { name }
        interactionTypes
        sources { sourceName }
      }
    }
  }
}
```

**SCLC-associated genes queried** (n=57):
- Transcription factors: ASCL1, NEUROD1, POU2F3, MYC, MYCN, MYCL
- Cell cycle: AURKA, AURKB, PLK1, WEE1, CHEK1, CHEK2, CDK4, CDK6, CCNE1, CCND1
- Apoptosis: BCL2, BCL2L1, MCL1, BIRC5, XIAP, CFLAR
- DNA damage response: ATM, ATR, PARP1, PARP2, BRCA1, BRCA2
- Signaling: FGFR1, EGFR, KIT, RET, NTRK1, IGF1R, MET
- Epigenetic: EZH2, HDAC1, HDAC2, BRD4
- Other: DLL3, NOTCH1, SOX2, TP53, RB1, EPHA2

### Statistical Analysis

- Single-sample Gene Set Enrichment Analysis (ssGSEA) was computed using the GSVA package algorithm implemented in Python
- Hierarchical clustering used Ward's minimum variance method with Euclidean distance
- Correlation coefficients are Pearson's r
- Multiple testing correction used Benjamini-Hochberg FDR where applicable
- All analyses used Python 3.12 with numpy, pandas, scipy, scikit-learn, matplotlib, and seaborn

---

## Supplementary Tables

### Table S1. SCLC-Associated Genes for Drug Repositioning

| Category | Genes |
|----------|-------|
| Transcription factors | ASCL1, NEUROD1, POU2F3, MYC, MYCN, MYCL |
| Cell cycle regulators | AURKA, AURKB, PLK1, WEE1, CHEK1, CHEK2, CDK4, CDK6, CCNE1, CCND1 |
| Apoptosis modulators | BCL2, BCL2L1, MCL1, BIRC5, XIAP, CFLAR |
| DNA damage response | ATM, ATR, PARP1, PARP2, BRCA1, BRCA2 |
| Receptor tyrosine kinases | FGFR1, EGFR, KIT, RET, NTRK1, IGF1R, MET |
| Epigenetic regulators | EZH2, HDAC1, HDAC2, BRD4 |
| Other relevant genes | DLL3, NOTCH1, SOX2, TP53, RB1, EPHA2 |

### Table S2. Immune Signature Gene Lists

| Signature | Gene Count | Genes |
|-----------|------------|-------|
| T-effector | 14 | CD8A, CD8B, GZMA, GZMB, PRF1, IFNG, CXCL9, CXCL10, TBX21, EOMES, STAT4, STAT1, IRF1, RUNX3 |
| IFN-gamma | 16 | IFNG, STAT1, IDO1, CXCL9, CXCL10, CXCL11, HLA-DRA, HLA-DRB1, HLA-E, IRF1, IRF9, GBP1, GBP4, GBP5, PSMB9, TAP1 |
| Antigen presentation | 16 | HLA-A, HLA-B, HLA-C, B2M, TAP1, TAP2, PSMB8, PSMB9, PSMB10, CIITA, RFX5, NLRC5, CD74, HLA-DMA, HLA-DMB, HLA-DOA |
| Myeloid/TAM | 15 | CD68, CD163, MSR1, CSF1R, CSF1, CCL2, MRC1, MARCO, TREM2, SIGLEC1, IL10, VEGFA, MMP9, ARG1, HIF1A |
| Treg/immunosuppression | 13 | FOXP3, IL2RA, CTLA4, TIGIT, ICOS, TNFRSF18, CCR4, CCR8, IL10, TGFB1, IDO1, ENTPD1, NT5E |
| Exhaustion | 9 | PDCD1, LAG3, HAVCR2, TIGIT, BTLA, CD244, CD160, ENTPD1, TOX |

### Table S3. Complete Drug Ranking

*See results/drugs/drug_rank.tsv for full table (1,276 drugs)*

Top 20 drugs by target coverage:

| Rank | Drug | Targets | Score |
|------|------|---------|-------|
| 1 | CISPLATIN | 11 | 2.485 |
| 2 | CYC-116 | 9 | 2.303 |
| 3 | ILORASERTIB | 9 | 2.303 |
| 4 | CENISERTIB | 9 | 2.303 |
| 5 | RG-1530 | 9 | 2.303 |
| 6 | OLAPARIB | 9 | 2.303 |
| 7 | NVP-TAE684 | 8 | 2.197 |
| 8 | DOVITINIB | 8 | 2.197 |
| 9 | ALISERTIB | 8 | 2.197 |
| 10 | GW441756X | 7 | 2.079 |
| 11 | TALAZOPARIB | 7 | 2.079 |
| 12 | SP-600125 | 7 | 2.079 |
| 13 | ENTRECTINIB | 7 | 2.079 |
| 14 | IMATINIB | 7 | 2.079 |
| 15 | SORAFENIB | 7 | 2.079 |
| 16 | PF-562271 | 7 | 2.079 |
| 17 | PD-0166285 | 7 | 2.079 |
| 18 | PAZOPANIB | 7 | 2.079 |
| 19 | PACLITAXEL | 7 | 2.079 |
| 20 | HESPERADIN | 6 | 1.946 |

### Table S4. Immune State Characteristics

| Immune State | N | % | Interpretation | Key Features |
|--------------|---|---|----------------|--------------|
| State 1 | 1 | 1.2% | Mixed | Intermediate scores across all signatures |
| State 2 | 29 | 33.7% | Immune desert | Low infiltration, low exhaustion |
| State 3 | 21 | 24.4% | Immune excluded | Moderate infiltration with exhaustion |
| State 4 | 35 | 40.7% | Immune hot | High T-effector, IFN-gamma, antigen presentation |

### Table S5. Subtype Distribution by Immune State

| Subtype | State 1 | State 2 | State 3 | State 4 | Total |
|---------|---------|---------|---------|---------|-------|
| SCLC-A | 0 | 6 | 4 | 5 | 15 |
| SCLC-N | 0 | 8 | 5 | 5 | 18 |
| SCLC-P | 1 | 10 | 7 | 11 | 29 |
| SCLC-I | 0 | 5 | 5 | 14 | 24 |
| Total | 1 | 29 | 21 | 35 | 86 |

---

## Supplementary Figures

### Figure S1. Quality Control Metrics

Quality control of GSE60052 bulk RNA-seq data:
- Samples: 86 (79 tumor, 7 normal)
- Genes before filtering: 51,131
- Genes after filtering: 35,805
- Filtering criteria: minimum 10 total counts, expression in >10% of samples

### Figure S2. Drug Target Network

Network visualization of top drug candidates and their SCLC gene targets. Nodes represent drugs (squares) and genes (circles). Edges indicate drug-gene interactions from DGIdb.

Key observations:
- Aurora kinase inhibitors form a dense cluster targeting AURKA, AURKB, and checkpoint kinases
- PARP inhibitors target DNA damage response genes (ATM, ATR, CHEK1/2, PARP1/2)
- Multi-kinase inhibitors target receptor tyrosine kinases (FGFR1, RET, KIT, NTRK1)

### Figure S3. Immune Signature Correlation Matrix

Pearson correlations between all pairs of immune signatures (n=86 samples):

| | T-eff | IFN-γ | AP | TAM | Treg | Exh |
|---|---|---|---|---|---|---|
| T-effector | 1.00 | 0.89 | 0.82 | 0.45 | 0.67 | 0.58 |
| IFN-gamma | 0.89 | 1.00 | 0.91 | 0.52 | 0.71 | 0.62 |
| Antigen pres. | 0.82 | 0.91 | 1.00 | 0.48 | 0.65 | 0.55 |
| Myeloid/TAM | 0.45 | 0.52 | 0.48 | 1.00 | 0.58 | 0.42 |
| Treg/immunosupp. | 0.67 | 0.71 | 0.65 | 0.58 | 1.00 | 0.73 |
| Exhaustion | 0.58 | 0.62 | 0.55 | 0.42 | 0.73 | 1.00 |

---

## Data Availability

### Public Datasets Used

| Dataset | Type | Samples | Accession | Reference |
|---------|------|---------|-----------|-----------|
| GSE60052 | Bulk RNA-seq | 86 | GEO | George et al., Nature 2015 |
| GSE138267 | scRNA-seq | 25 | GEO | Stewart et al., Cancer Cell 2021 |

### Processed Data Files

Available at https://github.com/cmoh1981/SCLC:

- `data/processed/bulk/bulk_expression_matrix.tsv` - Normalized expression matrix
- `results/subtypes/subtype_scores.tsv` - Sample-level subtype scores
- `results/immune/immune_scores.tsv` - Immune signature scores
- `results/immune/immune_states.tsv` - Immune state assignments
- `results/drugs/drug_rank.tsv` - Complete drug ranking
- `results/drugs/top_drugs_summary.tsv` - Top 20 drugs with details

### Code Availability

All analysis code is available at: https://github.com/cmoh1981/SCLC

Key scripts:
- `scripts/02_preprocess_bulk.py` - Bulk RNA-seq preprocessing
- `scripts/05_score_subtypes.py` - SCLC subtype classification
- `scripts/06_score_immune.py` - Immune signature scoring
- `scripts/09_drug_repositioning.py` - DGIdb drug repositioning
- `scripts/generate_manuscript_figures.py` - Figure generation

---

## References for Supplementary Methods

1. Rudin CM, Poirier JT, Byers LA, et al. Molecular subtypes of small cell lung cancer: a synthesis of human and mouse model data. Nat Rev Cancer. 2019;19:289-297.

2. Gay CM, Stewart CA, Park EM, et al. Patterns of transcription factor programs and immune pathway activation define four major subtypes of SCLC with distinct therapeutic vulnerabilities. Cancer Cell. 2021;39:346-360.

3. Ayers M, Lunceford J, Nebozhyn M, et al. IFN-γ-related mRNA profile predicts clinical response to PD-1 blockade. J Clin Invest. 2017;127:2930-2940.

4. Cristescu R, Mogg R, Ayers M, et al. Pan-tumor genomic biomarkers for PD-1 checkpoint blockade-based immunotherapy. Science. 2018;362:eaar3593.

5. Freshour SL, Kiwala S, Cotto KC, et al. Integration of the Drug-Gene Interaction Database (DGIdb 4.0) with open crowdsource efforts. Nucleic Acids Res. 2021;49:D1144-D1151.

6. George J, Lim JS, Jang SJ, et al. Comprehensive genomic profiles of small cell lung cancer. Nature. 2015;524:47-53.
