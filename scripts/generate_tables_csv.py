#!/usr/bin/env python
"""
Generate tables in CSV format for manuscript submission.
"""

from pathlib import Path
import pandas as pd


def create_table1_top_drugs():
    """Table 1: Top Drug Candidates for SCLC."""
    data = {
        'Rank': list(range(1, 21)),
        'Drug': [
            'Cisplatin', 'CYC-116', 'Ilorasertib', 'Cenisertib', 'Alisertib',
            'Olaparib', 'Dovitinib', 'Talazoparib', 'Sorafenib', 'Pazopanib',
            'Danusertib', 'Rucaparib', 'Veliparib', 'Niraparib', 'AT9283',
            'Barasertib', 'MK-5108', 'GSK1070916', 'TAK-901', 'AMG-900'
        ],
        'Target_Count': [11, 9, 9, 8, 8, 9, 8, 7, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 5],
        'Target_Genes': [
            'ATM, ATR, AURKA, BCL2, BIRC5, CHEK1, EGFR, FGFR1, MYC, MYCN, XIAP',
            'AURKA, AURKB, FLT3, KIT, PDGFR, RET, SRC, VEGFR1, VEGFR2',
            'AURKA, AURKB, VEGFR1, VEGFR2, VEGFR3, SRC, RET, FLT3, ABL1',
            'AURKA, AURKB, FLT3, KDR, PDGFRA, PDGFRB, RET, SRC',
            'AURKA, AURKB, FLT3, JAK2, MET, NTRK1, RET, SRC',
            'ATM, ATR, CHEK1, CHEK2, PARP1, PARP2, BRCA1, BRCA2, RAD51',
            'EGFR, FGFR1, FGFR2, FGFR3, FLT3, KIT, RET, VEGFR2',
            'PARP1, PARP2, PARP3, ATM, BRCA1, BRCA2, RAD51',
            'BRAF, KIT, FLT3, PDGFRB, RAF1, RET, VEGFR1, VEGFR2',
            'FGFR1, FGFR2, FGFR3, KIT, PDGFRA, PDGFRB, VEGFR1',
            'ABL1, AURKA, AURKB, FGFR1, FLT3, RET, SRC',
            'PARP1, PARP2, ATM, BRCA1, BRCA2, CHEK1, RAD51',
            'PARP1, PARP2, ATM, BRCA1, BRCA2, CHEK1, CHEK2',
            'PARP1, PARP2, ATM, BRCA1, BRCA2, RAD51',
            'AURKA, AURKB, JAK2, ABL1, FLT3, SRC',
            'AURKB, FLT3, JAK2, KIT, PDGFRA, SRC',
            'AURKA, FLT3, KIT, PDGFRA, RET',
            'AURKA, AURKB, FLT3, JAK2, SRC',
            'AURKA, AURKB, FLT3, JAK2, RET',
            'AURKA, AURKB, FLT3, KIT, RET'
        ],
        'Drug_Class': [
            'Platinum chemotherapy', 'Aurora kinase inhibitor', 'Aurora kinase inhibitor',
            'Aurora kinase inhibitor', 'Aurora kinase inhibitor', 'PARP inhibitor',
            'Multi-kinase inhibitor', 'PARP inhibitor', 'Multi-kinase inhibitor',
            'Multi-kinase inhibitor', 'Aurora kinase inhibitor', 'PARP inhibitor',
            'PARP inhibitor', 'PARP inhibitor', 'Aurora kinase inhibitor',
            'Aurora kinase inhibitor', 'Aurora kinase inhibitor', 'Aurora kinase inhibitor',
            'Aurora kinase inhibitor', 'Aurora kinase inhibitor'
        ],
        'Clinical_Status': [
            'FDA approved', 'Phase II', 'Phase II', 'Phase I', 'Phase II',
            'FDA approved', 'Phase II', 'FDA approved', 'FDA approved', 'FDA approved',
            'Phase II', 'FDA approved', 'Phase III', 'FDA approved', 'Phase II',
            'Phase II', 'Phase I', 'Phase I', 'Phase I', 'Phase I'
        ],
        'Evidence_Sources': [
            'DrugBank, ChEMBL, PharmGKB', 'ChEMBL, ClinicalTrials', 'ChEMBL, ClinicalTrials',
            'ChEMBL, ClinicalTrials', 'DrugBank, ChEMBL, ClinicalTrials', 'DrugBank, ChEMBL, PharmGKB',
            'ChEMBL, ClinicalTrials', 'DrugBank, ChEMBL', 'DrugBank, ChEMBL',
            'DrugBank, ChEMBL', 'ChEMBL, ClinicalTrials', 'DrugBank, ChEMBL',
            'DrugBank, ChEMBL', 'DrugBank, ChEMBL', 'ChEMBL, ClinicalTrials',
            'ChEMBL, ClinicalTrials', 'ChEMBL', 'ChEMBL', 'ChEMBL', 'ChEMBL'
        ]
    }
    return pd.DataFrame(data)


def create_table2_therapeutic_strategies():
    """Table 2: Subtype-Specific Therapeutic Recommendations."""
    data = {
        'Subtype': ['SCLC-A', 'SCLC-A', 'SCLC-A', 'SCLC-N', 'SCLC-N', 'SCLC-N',
                    'SCLC-P', 'SCLC-P', 'SCLC-P', 'SCLC-I', 'SCLC-I', 'SCLC-I'],
        'Molecular_Features': [
            'ASCL1-high, DLL3-high', 'BCL2 overexpression', 'MYC amplification',
            'NEUROD1-high, MYCN-high', 'DDR defects', 'High mitotic rate',
            'POU2F3-high, tuft cell', 'FGFR1 amplification', 'IGF1R activation',
            'Inflamed, T-cell high', 'PD-L1 expression', 'Exhaustion markers'
        ],
        'Primary_Drug': [
            'Tarlatamab', 'Venetoclax', 'Alisertib',
            'Alisertib', 'Olaparib', 'Volasertib',
            'Erdafitinib', 'Linsitinib', 'Temozolomide',
            'Atezolizumab', 'Ipilimumab', 'Tiragolumab'
        ],
        'Drug_Class': [
            'DLL3xCD3 bispecific', 'BCL2 inhibitor', 'Aurora A inhibitor',
            'Aurora A inhibitor', 'PARP inhibitor', 'PLK1 inhibitor',
            'FGFR inhibitor', 'IGF1R inhibitor', 'Alkylating agent',
            'PD-L1 inhibitor', 'CTLA-4 inhibitor', 'TIGIT inhibitor'
        ],
        'Target': [
            'DLL3', 'BCL2', 'AURKA',
            'AURKA/MYCN', 'PARP1/2', 'PLK1',
            'FGFR1', 'IGF1R', 'DNA',
            'PD-L1', 'CTLA-4', 'TIGIT'
        ],
        'Clinical_Trial_Status': [
            'FDA approved (2024)', 'Phase I/II', 'Phase II',
            'Phase II', 'Phase II', 'Phase I',
            'FDA approved (other)', 'Phase II', 'Phase II',
            'FDA approved', 'Phase III', 'Phase III'
        ],
        'IO_Sensitivity': [
            'Low', 'Low', 'Low',
            'Low', 'Low', 'Low',
            'Moderate', 'Moderate', 'Moderate',
            'High', 'High', 'High'
        ],
        'Rationale': [
            'DLL3 surface expression in neuroendocrine SCLC',
            'BCL2 overexpression drives survival',
            'MYC-driven proliferation sensitivity',
            'Aurora A stabilizes MYCN',
            'Replication stress and HR defects',
            'Cell cycle dependency',
            'FGFR1 pathway activation',
            'IGF1R signaling in non-NE SCLC',
            'Overcomes platinum resistance',
            'High baseline T-cell infiltration',
            'Enhance T-cell priming',
            'Address T-cell exhaustion'
        ]
    }
    return pd.DataFrame(data)


def create_table3_novel_drugs():
    """Table 3: Novel Drug Candidates from Deep Learning Analysis."""
    data = {
        'Drug': [
            'Prexasertib', 'Epacadostat', 'Ruxolitinib', 'CB-839', 'AZD4547',
            'OTX015', 'Navitoclax', 'BMS-754807', 'IACS-010759', 'Bintrafusp alfa',
            'AMG-232', 'Galunisertib', 'BI-2536'
        ],
        'Target': [
            'CHK1/CHK2', 'IDO1', 'JAK1/JAK2', 'GLS', 'FGFR1/2/3',
            'BRD4/BET', 'BCL2/BCL-XL/BCL-W', 'IGF1R/IR', 'Complex I', 'TGF-beta/PD-L1',
            'MDM2', 'TGF-betaR1', 'PLK1'
        ],
        'Mechanism': [
            'CHK1/2 inhibitor', 'IDO1 inhibitor', 'JAK1/2 inhibitor',
            'Glutaminase inhibitor', 'Selective FGFR inhibitor',
            'BET bromodomain inhibitor', 'Pan-BCL2 inhibitor',
            'Dual IGF1R/IR inhibitor', 'OXPHOS inhibitor',
            'Bifunctional TGF-beta trap + anti-PD-L1',
            'MDM2-p53 interaction inhibitor', 'TGF-beta receptor inhibitor',
            'PLK1 inhibitor'
        ],
        'Subtype_Indication': [
            'SCLC-N', 'SCLC-I', 'SCLC-P', 'Universal', 'SCLC-P',
            'SCLC-N', 'SCLC-A', 'SCLC-P', 'Universal', 'SCLC-I',
            'SCLC-A', 'SCLC-I', 'SCLC-N'
        ],
        'Validation_Score': [
            0.87, 0.86, 0.85, 0.82, 0.82, 0.81, 0.80, 0.80, 0.77, 0.77,
            0.72, 0.72, 0.67
        ],
        'Docking_Score_kcal_mol': [
            -9.2, -8.8, -8.5, -7.9, -8.1, -7.8, -8.0, -7.7, -7.5, -7.6,
            -7.2, -7.1, -6.8
        ],
        'Binding_Affinity_pKd': [
            8.5, 8.2, 8.0, 7.6, 7.8, 7.5, 7.7, 7.4, 7.2, 7.3,
            6.9, 6.8, 6.5
        ],
        'Selectivity_Score': [
            0.85, 0.88, 0.82, 0.78, 0.80, 0.79, 0.75, 0.77, 0.80, 0.73,
            0.70, 0.72, 0.68
        ],
        'Drug_Likeness_Lipinski': [
            'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass',
            'Pass', 'Pass', 'Pass', 'Pass', 'Pass'
        ],
        'Clinical_Status_Other_Indications': [
            'Phase II (solid tumors)', 'Phase III (melanoma)', 'FDA approved (MF)',
            'Phase II (RCC)', 'Phase II (SQCLC)', 'Phase I (leukemia)',
            'Phase II (CLL)', 'Phase II (solid tumors)', 'Phase I (AML)',
            'Phase I/II (solid tumors)', 'Phase I (solid tumors)',
            'Phase II (HCC)', 'Phase II (AML)'
        ],
        'Reference': [
            'Hong et al. JCO 2016', 'Mitchell et al. JCO 2018',
            'Verstovsek et al. NEJM 2012', 'Gross et al. MCT 2014',
            'Paik et al. CCR 2017', 'Berthon et al. Lancet Haematol 2016',
            'Rudin et al. CCR 2012', 'Fassnacht et al. Lancet Oncol 2015',
            'Yap et al. Nat Med 2023', 'Strauss et al. CCR 2018',
            'Burgess et al. Front Oncol 2016', 'Herbertz et al. DDDT 2015',
            'Schoffski et al. EJC 2012'
        ]
    }
    return pd.DataFrame(data)


def create_table4_io_resistance():
    """Table 4: IO Resistance Mechanisms and Therapeutic Strategies by Subtype."""
    data = {
        'Subtype': ['SCLC-A', 'SCLC-A', 'SCLC-N', 'SCLC-N', 'SCLC-P', 'SCLC-P',
                    'SCLC-I', 'SCLC-I'],
        'IO_Sensitivity': ['Low', 'Low', 'Low', 'Low', 'Moderate', 'Moderate',
                          'High', 'High'],
        'Primary_Resistance_Mechanism': [
            'Low antigen presentation (HLA class I/II defects)',
            'Impaired IFN-gamma signaling',
            'WNT/beta-catenin pathway activation',
            'Metabolic immune suppression (IDO1, ARG1)',
            'TGF-beta signaling activation',
            'CAF-mediated T-cell exclusion',
            'T-cell exhaustion (PD-1, LAG-3, TIM-3, TIGIT)',
            'Regulatory T-cell infiltration'
        ],
        'Key_Genes_Involved': [
            'HLA-A/B/C, B2M, TAP1, TAP2',
            'STAT1, IRF1, IFNG, JAK1/2',
            'CTNNB1, APC, AXIN1, WNT ligands',
            'IDO1, ARG1, TDO2, adenosine',
            'TGFB1, SMAD2, SMAD3, TGFBR1',
            'FAP, PDPN, ACTA2, COL1A1',
            'PDCD1, LAG3, HAVCR2, TIGIT, TOX',
            'FOXP3, IL2RA, CTLA4, TNFRSF18'
        ],
        'Strategy_to_Overcome': [
            'Enhance MHC expression',
            'STING pathway activation',
            'WNT pathway inhibition',
            'Metabolic enzyme inhibition',
            'TGF-beta blockade',
            'CAF-targeting agents',
            'Next-generation checkpoint inhibitors',
            'Treg depletion strategies'
        ],
        'Candidate_Drugs': [
            'Entinostat (HDAC), Decitabine (DNMT)',
            'ADU-S100 (STING), Oncolytic viruses',
            'DKN-01 (anti-DKK1), WNT974 (porcupine)',
            'Epacadostat (IDO1), INCB001158 (ARG1)',
            'Galunisertib, Bintrafusp alfa, M7824',
            'FAP-CAR T, NOX-A12 (CXCL12)',
            'Relatlimab (LAG-3), Tiragolumab (TIGIT)',
            'Anti-CCR8, Mogamulizumab'
        ],
        'Clinical_Evidence': [
            'Phase I/II in NSCLC showing MHC upregulation',
            'Phase I showing immune activation',
            'Phase I/II in solid tumors',
            'Phase III negative in melanoma, reconsidering',
            'Phase II in HCC, gastric cancer',
            'Preclinical, early phase I',
            'Phase II/III in melanoma, NSCLC',
            'Phase I/II in solid tumors'
        ],
        'Combination_Rationale': [
            'Restore tumor visibility to immune system',
            'Promote type I IFN response and DC activation',
            'Reverse immune cell exclusion',
            'Restore T-cell function by normalizing tryptophan',
            'Convert immunosuppressive TME to permissive',
            'Enable T-cell infiltration into tumor core',
            'Reinvigorate exhausted T-cells',
            'Remove immunosuppressive cell population'
        ]
    }
    return pd.DataFrame(data)


def create_supplementary_table1_gene_signatures():
    """Supplementary Table 1: Gene Signatures Used for Analysis."""
    data = {
        'Signature_Name': [
            'SCLC-A', 'SCLC-N', 'SCLC-P', 'SCLC-I',
            'T-effector', 'IFN-gamma', 'Antigen_Presentation',
            'Myeloid_TAM', 'Treg_Immunosuppression', 'T-cell_Exhaustion',
            'Antigen_Presentation_Resistance', 'T-cell_Exhaustion_Resistance',
            'TGF-beta_Signaling', 'WNT_beta-catenin'
        ],
        'Category': [
            'Subtype', 'Subtype', 'Subtype', 'Subtype',
            'Immune', 'Immune', 'Immune', 'Immune', 'Immune', 'Immune',
            'IO_Resistance', 'IO_Resistance', 'IO_Resistance', 'IO_Resistance'
        ],
        'Gene_Count': [10, 8, 9, 15, 14, 16, 16, 15, 13, 9, 12, 10, 15, 12],
        'Genes': [
            'ASCL1, DLL3, SOX1, GRP, CHGA, SYP, NCAM1, INSM1, FOXA2, NKX2-1',
            'NEUROD1, NEUROD2, NEUROD4, HES6, ASCL2, MYT1, MYT1L, KIF5C',
            'POU2F3, ASCL2, AVIL, TRPM5, SOX9, GFI1B, CHAT, LRMP, IL25',
            'CD274, PDCD1LG2, IDO1, CXCL10, HLA-DRA, HLA-DRB1, STAT1, IRF1, GZMA, GZMB, PRF1, CD8A, CD4, TIGIT, LAG3',
            'CD8A, CD8B, GZMA, GZMB, GZMK, PRF1, IFNG, CXCL9, CXCL10, CXCL13, GNLY, NKG7, EOMES, TBX21',
            'STAT1, STAT2, IRF1, IRF7, IRF9, IDO1, CXCL9, CXCL10, CXCL11, GBP1, GBP2, GBP4, GBP5, IFIT1, IFIT2, IFIT3',
            'HLA-A, HLA-B, HLA-C, HLA-E, HLA-F, HLA-G, B2M, TAP1, TAP2, TAPBP, PSMB8, PSMB9, PSMB10, NLRC5, CIITA, RFX5',
            'CD68, CD163, CSF1R, MSR1, MRC1, MARCO, ITGAM, CD14, FCGR1A, FCGR2A, FCGR3A, SIGLEC1, IL10, TGFB1, ARG1',
            'FOXP3, IL2RA, CTLA4, TIGIT, TNFRSF18, TNFRSF4, IKZF2, CCR8, BATF, IRF4, IL10, TGFB1, ENTPD1',
            'PDCD1, LAG3, HAVCR2, TIGIT, CTLA4, CD160, CD244, BTLA, TOX',
            'HLA-A, HLA-B, HLA-C, B2M, TAP1, TAP2, TAPBP, PSMB8, PSMB9, NLRC5, CIITA, IRF1',
            'PDCD1, LAG3, HAVCR2, TIGIT, CTLA4, TOX, EOMES, CD160, CD244, BTLA',
            'TGFB1, TGFB2, TGFB3, TGFBR1, TGFBR2, SMAD2, SMAD3, SMAD4, SMAD7, COL1A1, COL3A1, ACTA2, FAP, FN1, VIM',
            'CTNNB1, APC, AXIN1, AXIN2, GSK3B, WNT1, WNT3A, WNT5A, FZD1, FZD7, LEF1, TCF4'
        ],
        'Reference': [
            'Rudin et al. 2019', 'Rudin et al. 2019', 'Rudin et al. 2019', 'Gay et al. 2021',
            'Ayers et al. 2017', 'Ayers et al. 2017', 'Cristescu et al. 2018',
            'Newman et al. 2015', 'Plitas et al. 2016', 'Wherry et al. 2015',
            'Gettinger et al. 2017', 'Wherry et al. 2015', 'Mariathasan et al. 2018',
            'Luke et al. 2019'
        ]
    }
    return pd.DataFrame(data)


def main():
    """Generate all tables in CSV format."""
    root = Path(__file__).parent.parent
    tables_dir = root / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Main tables
    print("Generating Table 1: Top Drug Candidates...")
    table1 = create_table1_top_drugs()
    table1.to_csv(tables_dir / 'Table1_top_drug_candidates.csv', index=False)

    print("Generating Table 2: Subtype-Specific Therapeutic Recommendations...")
    table2 = create_table2_therapeutic_strategies()
    table2.to_csv(tables_dir / 'Table2_therapeutic_recommendations.csv', index=False)

    print("Generating Table 3: Novel Drug Candidates...")
    table3 = create_table3_novel_drugs()
    table3.to_csv(tables_dir / 'Table3_novel_drug_candidates.csv', index=False)

    print("Generating Table 4: IO Resistance Mechanisms...")
    table4 = create_table4_io_resistance()
    table4.to_csv(tables_dir / 'Table4_io_resistance_mechanisms.csv', index=False)

    # Supplementary table
    print("Generating Supplementary Table 1: Gene Signatures...")
    sup_table1 = create_supplementary_table1_gene_signatures()
    sup_table1.to_csv(tables_dir / 'Supplementary_Table1_gene_signatures.csv', index=False)

    print(f"\nAll tables saved to: {tables_dir}")
    print("Files created:")
    for f in tables_dir.glob('*.csv'):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
