# Controlled-Access Datasets for SCLC Chemo-IO Research

This document lists datasets requiring controlled access applications.

## European Genome-phenome Archive (EGA)

### 1. IMpower133 Trial Data (EGAS00001004888)
- **Description**: Bulk RNA-seq of 271 extensive-stage SCLC tumors from IMpower133 trial
- **Content**: Log2TPM expression matrices, subtype labels (SCLC-A, -N, -P, -I)
- **Datasets**:
  - EGAD00001006926
  - EGAD00001006927
  - EGAD00001006928
- **Access**: Apply via EGA DAC (Data Access Committee)
- **URL**: https://ega-archive.org/studies/EGAS00001004888
- **Reference**: Gay et al., Cancer Cell 2021

### 2. George et al. Nature 2015 (EGAS00001000925)
- **Description**: Multi-omics of 110 SCLC tumors
- **Content**:
  - Whole Genome Sequencing (110 tumors + matched normals)
  - RNA-seq (59 tumors)
  - SNP array (54 tumors)
- **Key Findings**: Universal TP53/RB1 loss, NOTCH mutations
- **Access**: Apply via EGA DAC
- **URL**: https://ega-archive.org/studies/EGAS00001000925
- **Reference**: George et al., Nature 2015

---

## Chinese National Genomics Data Center (NGDC)

### 3. Tian et al. STTT 2022 (PRJCA006026)
- **Description**: Single-cell RNA-seq of ~5,000 cells from 11 SCLC tumors
- **Content**:
  - scRNA-seq from treatment-naive and relapsed tumors
  - Whole-genome sequencing
  - NE vs non-NE cell state classification
- **Access**: GSA database (controlled)
- **URL**: https://ngdc.cncb.ac.cn/bioproject/browse/PRJCA006026
- **Reference**: Tian et al., Signal Transduct Target Ther 2022

### 4. Jin et al. Cell Discovery 2024 (HRA004312)
- **Description**: Digital Spatial Profiling of 44 SCLC tumors
- **Content**:
  - Nanostring GeoMX data (132 tissue cores)
  - ~18,000 mRNA transcripts
  - ~60 proteins
  - Spatially-resolved TIME (Tumor Immune Microenvironment)
- **Access**: Genome Sequence Archive (Human) - collaboration request
- **URL**: https://ngdc.cncb.ac.cn/gsa-human/browse/HRA004312
- **Reference**: Jin et al., Cell Discovery 2024

---

## NIH Proteomic Data Commons (PDC)

### 5. Liu et al. Cell 2024 - CPTAC SCLC
- **Description**: Proteogenomic profiling of 112 treatment-naive SCLC tumors
- **Content**:
  - Whole Exome Sequencing (WES)
  - RNA-seq
  - Total proteomics
  - Phosphoproteomics
  - Paired normal lung tissues
- **Key Findings**: Subtype-specific signaling networks, therapeutic targets
- **Access**: CPTAC portal (released 2024)
- **URL**: https://proteomic.datacommons.cancer.gov/
- **Reference**: Liu et al., Cell 2024

---

## Access Application Guide

### EGA (European Genome-phenome Archive)
1. Create an EGA account: https://ega-archive.org/register
2. Find the study and identify the Data Access Committee (DAC)
3. Submit a Data Access Request (DAR) with:
   - Research proposal
   - IRB/Ethics approval
   - Data security measures
4. Wait for DAC approval (typically 2-4 weeks)
5. Use EGA download client or pyega3 to download

### NGDC (Chinese databases)
1. Register at https://ngdc.cncb.ac.cn/
2. Submit collaboration request to data generators
3. Provide institutional affiliation and research purpose
4. Follow institution-specific data sharing agreements

### NIH/CPTAC
1. Access via Proteomic Data Commons portal
2. Some processed data is open access
3. Raw data may require dbGaP application:
   - https://dbgap.ncbi.nlm.nih.gov/
   - Requires institutional signing official

---

## Data Transfer Tools

```bash
# EGA download client
pip install pyega3
pyega3 -cf <credential_file> fetch <EGAD_ID>

# NCBI SRA toolkit (for dbGaP data)
prefetch --ngc <NGC_file> <SRR_accession>
fasterq-dump <SRR_accession>

# Aspera (fast transfer)
ascp -QT -l 300m -P33001 -i asperaweb_id_dsa.openssh <source> <destination>
```

---

## Notes

- **Wang et al. 2024 (Gut Microbiome)**: Data may be available upon request from authors
  - 16S rRNA + stool metabolomics from 49 ES-SCLC patients
  - Reference: Wang et al., J Thorac Dis 2024

- Processing these datasets may require significant compute resources
- Consider using cloud platforms (AWS, GCP) for large-scale analysis
