"""
Genome-scale Metabolic (GEM) Modeling Module

Functions for:
- Loading and adapting metabolic models (Human1, Recon3D)
- Transcriptomic integration (iMAT, GIMME-like algorithms)
- Flux Balance Analysis (FBA) for subtype-specific metabolism
- Metabolic vulnerability identification
- Drug-metabolite target mapping
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import json
import warnings

# Suppress solver warnings
warnings.filterwarnings('ignore')


def check_cobrapy():
    """Check if COBRApy is installed."""
    try:
        import cobra
        return True
    except ImportError:
        return False


def download_human1_model(dest_dir: Path, logger: Optional[logging.Logger] = None) -> Path:
    """
    Download Human1 GEM model.

    Args:
        dest_dir: Destination directory
        logger: Optional logger

    Returns:
        Path to downloaded model
    """
    import urllib.request

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Human1 model URL (from Human-GEM repository)
    model_url = "https://github.com/SysBioChalmers/Human-GEM/raw/main/model/Human-GEM.xml"
    model_path = dest_dir / "Human-GEM.xml"

    if model_path.exists():
        if logger:
            logger.info(f"Human-GEM model already exists at {model_path}")
        return model_path

    if logger:
        logger.info("Downloading Human-GEM model (this may take a few minutes)...")

    try:
        urllib.request.urlretrieve(model_url, str(model_path))
        if logger:
            logger.info(f"Downloaded Human-GEM to {model_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to download Human-GEM: {e}")
        # Try alternative: use Recon3D or create minimal model
        return None

    return model_path


def load_metabolic_model(model_path: Optional[Path] = None, logger: Optional[logging.Logger] = None):
    """
    Load a genome-scale metabolic model.

    Args:
        model_path: Path to model file (SBML/JSON)
        logger: Optional logger

    Returns:
        COBRApy model object
    """
    import cobra

    if model_path and model_path.exists():
        if logger:
            logger.info(f"Loading metabolic model from {model_path}")

        if str(model_path).endswith('.xml') or str(model_path).endswith('.sbml'):
            model = cobra.io.read_sbml_model(str(model_path))
        elif str(model_path).endswith('.json'):
            model = cobra.io.load_json_model(str(model_path))
        else:
            model = cobra.io.read_sbml_model(str(model_path))

        if logger:
            logger.info(f"Loaded model: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites, {len(model.genes)} genes")

        return model
    else:
        # Create a minimal cancer metabolism model
        if logger:
            logger.info("Creating minimal SCLC metabolism model...")
        return create_sclc_metabolic_model(logger)


def create_sclc_metabolic_model(logger: Optional[logging.Logger] = None):
    """
    Create a minimal metabolic model focused on SCLC-relevant pathways.

    Includes:
    - Glycolysis / Warburg effect
    - Glutaminolysis
    - One-carbon metabolism
    - Nucleotide biosynthesis
    - Fatty acid metabolism
    - Oxidative phosphorylation
    """
    import cobra

    model = cobra.Model('SCLC_metabolism')

    # Define metabolites
    metabolites = {
        # Glycolysis
        'glc_e': cobra.Metabolite('glc_e', name='Glucose', compartment='e'),
        'glc_c': cobra.Metabolite('glc_c', name='Glucose', compartment='c'),
        'g6p_c': cobra.Metabolite('g6p_c', name='Glucose-6-phosphate', compartment='c'),
        'f6p_c': cobra.Metabolite('f6p_c', name='Fructose-6-phosphate', compartment='c'),
        'pyr_c': cobra.Metabolite('pyr_c', name='Pyruvate', compartment='c'),
        'lac_c': cobra.Metabolite('lac_c', name='Lactate', compartment='c'),
        'lac_e': cobra.Metabolite('lac_e', name='Lactate', compartment='e'),

        # TCA cycle
        'accoa_c': cobra.Metabolite('accoa_c', name='Acetyl-CoA', compartment='c'),
        'cit_c': cobra.Metabolite('cit_c', name='Citrate', compartment='c'),
        'akg_c': cobra.Metabolite('akg_c', name='Alpha-ketoglutarate', compartment='c'),
        'succ_c': cobra.Metabolite('succ_c', name='Succinate', compartment='c'),
        'oaa_c': cobra.Metabolite('oaa_c', name='Oxaloacetate', compartment='c'),

        # Glutamine metabolism
        'gln_e': cobra.Metabolite('gln_e', name='Glutamine', compartment='e'),
        'gln_c': cobra.Metabolite('gln_c', name='Glutamine', compartment='c'),
        'glu_c': cobra.Metabolite('glu_c', name='Glutamate', compartment='c'),

        # Energy
        'atp_c': cobra.Metabolite('atp_c', name='ATP', compartment='c'),
        'adp_c': cobra.Metabolite('adp_c', name='ADP', compartment='c'),
        'nadh_c': cobra.Metabolite('nadh_c', name='NADH', compartment='c'),
        'nad_c': cobra.Metabolite('nad_c', name='NAD+', compartment='c'),

        # Nucleotide biosynthesis
        'r5p_c': cobra.Metabolite('r5p_c', name='Ribose-5-phosphate', compartment='c'),
        'prpp_c': cobra.Metabolite('prpp_c', name='PRPP', compartment='c'),
        'imp_c': cobra.Metabolite('imp_c', name='IMP', compartment='c'),
        'ump_c': cobra.Metabolite('ump_c', name='UMP', compartment='c'),

        # One-carbon metabolism
        'ser_c': cobra.Metabolite('ser_c', name='Serine', compartment='c'),
        'gly_c': cobra.Metabolite('gly_c', name='Glycine', compartment='c'),
        'thf_c': cobra.Metabolite('thf_c', name='THF', compartment='c'),
        'methf_c': cobra.Metabolite('methf_c', name='5,10-methylene-THF', compartment='c'),

        # Fatty acid metabolism
        'palm_c': cobra.Metabolite('palm_c', name='Palmitate', compartment='c'),
        'malcoa_c': cobra.Metabolite('malcoa_c', name='Malonyl-CoA', compartment='c'),

        # OXPHOS
        'o2_e': cobra.Metabolite('o2_e', name='Oxygen', compartment='e'),
        'o2_c': cobra.Metabolite('o2_c', name='Oxygen', compartment='c'),
        'h2o_c': cobra.Metabolite('h2o_c', name='Water', compartment='c'),
        'co2_c': cobra.Metabolite('co2_c', name='CO2', compartment='c'),

        # Biomass precursors
        'biomass': cobra.Metabolite('biomass', name='Biomass', compartment='c'),
    }

    # Add all metabolites
    model.add_metabolites(list(metabolites.values()))

    # Define reactions with gene associations
    reactions = []

    # Glucose uptake (GLUT1 - SLC2A1)
    rxn = cobra.Reaction('GLCt')
    rxn.name = 'Glucose transport'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({metabolites['glc_e']: -1, metabolites['glc_c']: 1})
    rxn.gene_reaction_rule = 'SLC2A1'
    reactions.append(rxn)

    # Hexokinase (HK2)
    rxn = cobra.Reaction('HK')
    rxn.name = 'Hexokinase'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['glc_c']: -1, metabolites['atp_c']: -1,
        metabolites['g6p_c']: 1, metabolites['adp_c']: 1
    })
    rxn.gene_reaction_rule = 'HK2 or HK1'
    reactions.append(rxn)

    # Glycolysis lumped (G6P -> Pyruvate)
    rxn = cobra.Reaction('GLYC')
    rxn.name = 'Glycolysis'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['g6p_c']: -1, metabolites['nad_c']: -2, metabolites['adp_c']: -2,
        metabolites['pyr_c']: 2, metabolites['nadh_c']: 2, metabolites['atp_c']: 2
    })
    rxn.gene_reaction_rule = 'GAPDH and PKM'
    reactions.append(rxn)

    # Lactate dehydrogenase (LDHA - Warburg effect)
    rxn = cobra.Reaction('LDH')
    rxn.name = 'Lactate dehydrogenase'
    rxn.lower_bound = -1000
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['pyr_c']: -1, metabolites['nadh_c']: -1,
        metabolites['lac_c']: 1, metabolites['nad_c']: 1
    })
    rxn.gene_reaction_rule = 'LDHA or LDHB'
    reactions.append(rxn)

    # Lactate export
    rxn = cobra.Reaction('LACt')
    rxn.name = 'Lactate transport'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({metabolites['lac_c']: -1, metabolites['lac_e']: 1})
    rxn.gene_reaction_rule = 'SLC16A3'
    reactions.append(rxn)

    # Pyruvate dehydrogenase
    rxn = cobra.Reaction('PDH')
    rxn.name = 'Pyruvate dehydrogenase'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['pyr_c']: -1, metabolites['nad_c']: -1,
        metabolites['accoa_c']: 1, metabolites['nadh_c']: 1, metabolites['co2_c']: 1
    })
    rxn.gene_reaction_rule = 'PDHA1 and PDHB'
    reactions.append(rxn)

    # Citrate synthase
    rxn = cobra.Reaction('CS')
    rxn.name = 'Citrate synthase'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['accoa_c']: -1, metabolites['oaa_c']: -1,
        metabolites['cit_c']: 1
    })
    rxn.gene_reaction_rule = 'CS'
    reactions.append(rxn)

    # Citrate to aKG (lumped IDH)
    rxn = cobra.Reaction('IDH')
    rxn.name = 'Isocitrate dehydrogenase'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['cit_c']: -1, metabolites['nad_c']: -1,
        metabolites['akg_c']: 1, metabolites['nadh_c']: 1, metabolites['co2_c']: 1
    })
    rxn.gene_reaction_rule = 'IDH2 or IDH3A'
    reactions.append(rxn)

    # Succinate to OAA (lumped)
    rxn = cobra.Reaction('SUCC_OAA')
    rxn.name = 'Succinate to OAA'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['succ_c']: -1, metabolites['nad_c']: -2,
        metabolites['oaa_c']: 1, metabolites['nadh_c']: 2
    })
    rxn.gene_reaction_rule = 'SDHA and FH and MDH2'
    reactions.append(rxn)

    # Glutamine uptake (SLC1A5/ASCT2)
    rxn = cobra.Reaction('GLNt')
    rxn.name = 'Glutamine transport'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({metabolites['gln_e']: -1, metabolites['gln_c']: 1})
    rxn.gene_reaction_rule = 'SLC1A5'
    reactions.append(rxn)

    # Glutaminase (GLS)
    rxn = cobra.Reaction('GLS')
    rxn.name = 'Glutaminase'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['gln_c']: -1, metabolites['h2o_c']: -1,
        metabolites['glu_c']: 1
    })
    rxn.gene_reaction_rule = 'GLS or GLS2'
    reactions.append(rxn)

    # Glutamate dehydrogenase (GLUD1)
    rxn = cobra.Reaction('GLUD')
    rxn.name = 'Glutamate dehydrogenase'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['glu_c']: -1, metabolites['nad_c']: -1,
        metabolites['akg_c']: 1, metabolites['nadh_c']: 1
    })
    rxn.gene_reaction_rule = 'GLUD1'
    reactions.append(rxn)

    # aKG to succinate (reductive carboxylation pathway)
    rxn = cobra.Reaction('AKGtoSUCC')
    rxn.name = 'aKG oxidation'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['akg_c']: -1, metabolites['nad_c']: -1,
        metabolites['succ_c']: 1, metabolites['nadh_c']: 1, metabolites['co2_c']: 1
    })
    rxn.gene_reaction_rule = 'OGDH'
    reactions.append(rxn)

    # Pentose phosphate pathway
    rxn = cobra.Reaction('PPP')
    rxn.name = 'Pentose phosphate pathway'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['g6p_c']: -1,
        metabolites['r5p_c']: 1, metabolites['co2_c']: 1
    })
    rxn.gene_reaction_rule = 'G6PD and PGD'
    reactions.append(rxn)

    # PRPP synthesis
    rxn = cobra.Reaction('PRPPS')
    rxn.name = 'PRPP synthetase'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['r5p_c']: -1, metabolites['atp_c']: -1,
        metabolites['prpp_c']: 1, metabolites['adp_c']: 1
    })
    rxn.gene_reaction_rule = 'PRPS1 or PRPS2'
    reactions.append(rxn)

    # Purine biosynthesis (de novo)
    rxn = cobra.Reaction('DNPS')
    rxn.name = 'De novo purine synthesis'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['prpp_c']: -1, metabolites['gln_c']: -2, metabolites['gly_c']: -1,
        metabolites['atp_c']: -5, metabolites['methf_c']: -2,
        metabolites['imp_c']: 1, metabolites['adp_c']: 5, metabolites['glu_c']: 2, metabolites['thf_c']: 2
    })
    rxn.gene_reaction_rule = 'PPAT and GART and PFAS and ATIC'
    reactions.append(rxn)

    # Pyrimidine biosynthesis
    rxn = cobra.Reaction('PYRS')
    rxn.name = 'Pyrimidine synthesis'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['gln_c']: -1, metabolites['atp_c']: -2, metabolites['r5p_c']: -1,
        metabolites['ump_c']: 1, metabolites['adp_c']: 2, metabolites['glu_c']: 1
    })
    rxn.gene_reaction_rule = 'CAD and DHODH and UMPS'
    reactions.append(rxn)

    # Serine biosynthesis (PHGDH pathway)
    rxn = cobra.Reaction('PHGDH')
    rxn.name = 'Serine biosynthesis'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['g6p_c']: -1, metabolites['glu_c']: -1, metabolites['nad_c']: -1,
        metabolites['ser_c']: 1, metabolites['akg_c']: 1, metabolites['nadh_c']: 1
    })
    rxn.gene_reaction_rule = 'PHGDH and PSAT1 and PSPH'
    reactions.append(rxn)

    # Serine hydroxymethyltransferase (one-carbon)
    rxn = cobra.Reaction('SHMT')
    rxn.name = 'Serine hydroxymethyltransferase'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['ser_c']: -1, metabolites['thf_c']: -1,
        metabolites['gly_c']: 1, metabolites['methf_c']: 1
    })
    rxn.gene_reaction_rule = 'SHMT1 or SHMT2'
    reactions.append(rxn)

    # THF cycle (methylene-THF to THF)
    rxn = cobra.Reaction('MTHFD')
    rxn.name = 'MTHFD cycle'
    rxn.lower_bound = -1000
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['methf_c']: -1, metabolites['nadh_c']: -1,
        metabolites['thf_c']: 1, metabolites['nad_c']: 1
    })
    rxn.gene_reaction_rule = 'MTHFD1 or MTHFD2'
    reactions.append(rxn)

    # Fatty acid synthesis (FASN)
    rxn = cobra.Reaction('FAS')
    rxn.name = 'Fatty acid synthesis'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['accoa_c']: -1, metabolites['malcoa_c']: -7, metabolites['nadh_c']: -14,
        metabolites['palm_c']: 1, metabolites['nad_c']: 14
    })
    rxn.gene_reaction_rule = 'FASN'
    reactions.append(rxn)

    # ACC (acetyl-CoA carboxylase)
    rxn = cobra.Reaction('ACC')
    rxn.name = 'Acetyl-CoA carboxylase'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['accoa_c']: -1, metabolites['atp_c']: -1, metabolites['co2_c']: -1,
        metabolites['malcoa_c']: 1, metabolites['adp_c']: 1
    })
    rxn.gene_reaction_rule = 'ACACA or ACACB'
    reactions.append(rxn)

    # Oxidative phosphorylation
    rxn = cobra.Reaction('OXPHOS')
    rxn.name = 'Oxidative phosphorylation'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['nadh_c']: -1, metabolites['o2_c']: -0.5, metabolites['adp_c']: -2.5,
        metabolites['nad_c']: 1, metabolites['atp_c']: 2.5, metabolites['h2o_c']: 0.5
    })
    rxn.gene_reaction_rule = 'MT-ND1 and MT-CYB and MT-CO1 and ATP5F1A'
    reactions.append(rxn)

    # Oxygen uptake
    rxn = cobra.Reaction('O2t')
    rxn.name = 'Oxygen transport'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({metabolites['o2_e']: -1, metabolites['o2_c']: 1})
    reactions.append(rxn)

    # ATP maintenance
    rxn = cobra.Reaction('ATPM')
    rxn.name = 'ATP maintenance'
    rxn.lower_bound = 1
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['atp_c']: -1, metabolites['h2o_c']: -1,
        metabolites['adp_c']: 1
    })
    reactions.append(rxn)

    # Biomass reaction (simplified)
    rxn = cobra.Reaction('BIOMASS')
    rxn.name = 'Biomass synthesis'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({
        metabolites['atp_c']: -10,
        metabolites['nadh_c']: -1,
        metabolites['glu_c']: -0.5,
        metabolites['ser_c']: -0.2,
        metabolites['r5p_c']: -0.1,
        metabolites['adp_c']: 10,
        metabolites['nad_c']: 1,
        metabolites['biomass']: 1
    })
    reactions.append(rxn)

    # Exchange reactions
    for met_id in ['glc_e', 'gln_e', 'o2_e', 'lac_e']:
        rxn = cobra.Reaction(f'EX_{met_id}')
        rxn.name = f'Exchange {met_id}'
        rxn.lower_bound = -1000 if met_id in ['glc_e', 'gln_e', 'o2_e'] else 0
        rxn.upper_bound = 1000
        rxn.add_metabolites({metabolites[met_id]: -1})
        reactions.append(rxn)

    # CO2 sink
    rxn = cobra.Reaction('DM_co2')
    rxn.name = 'CO2 sink'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({metabolites['co2_c']: -1})
    reactions.append(rxn)

    # H2O sink (for balance)
    rxn = cobra.Reaction('DM_h2o')
    rxn.name = 'H2O sink'
    rxn.lower_bound = -1000
    rxn.upper_bound = 1000
    rxn.add_metabolites({metabolites['h2o_c']: -1})
    reactions.append(rxn)

    # Demand for biomass
    rxn = cobra.Reaction('DM_biomass')
    rxn.name = 'Biomass demand'
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({metabolites['biomass']: -1})
    reactions.append(rxn)

    # Add all reactions
    model.add_reactions(reactions)

    # Set objective
    model.objective = 'DM_biomass'

    if logger:
        logger.info(f"Created SCLC metabolic model: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites, {len(model.genes)} genes")

    return model


def integrate_transcriptomics(
    model,
    expression_data: pd.DataFrame,
    sample_ids: List[str],
    method: str = 'gimme',
    threshold_percentile: float = 25,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Integrate transcriptomic data with metabolic model.

    Args:
        model: COBRApy model
        expression_data: Gene expression matrix (genes x samples)
        sample_ids: Sample IDs to analyze
        method: Integration method ('gimme', 'imat', 'eflux')
        threshold_percentile: Expression threshold percentile
        logger: Optional logger

    Returns:
        Dictionary with context-specific models and flux predictions
    """
    import cobra

    results = {
        'method': method,
        'samples': sample_ids,
        'fluxes': {},
        'objectives': {}
    }

    # Get model genes
    model_genes = set(g.id for g in model.genes)

    # Find overlapping genes
    expr_genes = set(expression_data.index)
    overlap = model_genes & expr_genes

    if logger:
        logger.info(f"Gene overlap: {len(overlap)} / {len(model_genes)} model genes")

    if len(overlap) == 0:
        if logger:
            logger.warning("No gene overlap found. Using default model bounds.")
        # Run FBA with default bounds
        for sample in sample_ids:
            solution = model.optimize()
            results['fluxes'][sample] = {rxn.id: solution.fluxes[rxn.id] for rxn in model.reactions}
            results['objectives'][sample] = solution.objective_value
        return results

    # Calculate expression threshold
    all_expr = expression_data.loc[list(overlap)].values.flatten()
    threshold = np.percentile(all_expr[~np.isnan(all_expr)], threshold_percentile)

    if logger:
        logger.info(f"Expression threshold ({threshold_percentile}th percentile): {threshold:.2f}")

    # Process each sample
    for sample in sample_ids:
        if sample not in expression_data.columns:
            continue

        sample_expr = expression_data[sample]

        # Create sample-specific model
        sample_model = model.copy()

        if method == 'gimme' or method == 'eflux':
            # GIMME-like: reduce bounds for lowly expressed genes
            for rxn in sample_model.reactions:
                if rxn.gene_reaction_rule:
                    # Get genes for this reaction
                    rxn_genes = [g.id for g in rxn.genes]

                    # Calculate reaction expression (max of associated genes)
                    gene_expr = [sample_expr.get(g, 0) for g in rxn_genes if g in sample_expr.index]

                    if gene_expr:
                        max_expr = max(gene_expr)

                        if max_expr < threshold:
                            # Reduce flux bounds for lowly expressed reactions
                            if rxn.lower_bound < 0:
                                rxn.lower_bound = rxn.lower_bound * 0.1
                            if rxn.upper_bound > 0:
                                rxn.upper_bound = rxn.upper_bound * 0.1

        # Run FBA
        try:
            solution = sample_model.optimize()
            if solution.status == 'optimal':
                results['fluxes'][sample] = {rxn.id: solution.fluxes[rxn.id] for rxn in sample_model.reactions}
                results['objectives'][sample] = solution.objective_value
            else:
                results['fluxes'][sample] = {}
                results['objectives'][sample] = 0
        except Exception as e:
            if logger:
                logger.warning(f"FBA failed for {sample}: {e}")
            results['fluxes'][sample] = {}
            results['objectives'][sample] = 0

    return results


def run_subtype_fba(
    model,
    expression_data: pd.DataFrame,
    subtype_assignments: pd.Series,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run FBA for each SCLC subtype.

    Args:
        model: COBRApy model
        expression_data: Gene expression matrix
        subtype_assignments: Series mapping samples to subtypes
        logger: Optional logger

    Returns:
        Dictionary with subtype-specific flux profiles
    """
    results = {
        'subtypes': {},
        'differential_fluxes': None
    }

    # Get unique subtypes
    subtypes = subtype_assignments.unique()

    for subtype in subtypes:
        # Get samples for this subtype
        samples = subtype_assignments[subtype_assignments == subtype].index.tolist()

        if logger:
            logger.info(f"Running FBA for {subtype} ({len(samples)} samples)...")

        # Integrate transcriptomics
        subtype_results = integrate_transcriptomics(
            model,
            expression_data,
            samples,
            method='gimme',
            logger=logger
        )

        # Calculate mean flux for subtype
        if subtype_results['fluxes']:
            flux_df = pd.DataFrame(subtype_results['fluxes']).T
            mean_flux = flux_df.mean()
            std_flux = flux_df.std()

            results['subtypes'][subtype] = {
                'mean_flux': mean_flux.to_dict(),
                'std_flux': std_flux.to_dict(),
                'n_samples': len(samples),
                'mean_objective': np.mean(list(subtype_results['objectives'].values()))
            }

    # Calculate differential fluxes between subtypes
    if len(results['subtypes']) > 1:
        flux_comparison = {}
        subtype_list = list(results['subtypes'].keys())

        for rxn_id in model.reactions:
            rxn_id = rxn_id.id
            fluxes = {st: results['subtypes'][st]['mean_flux'].get(rxn_id, 0)
                     for st in subtype_list}
            flux_comparison[rxn_id] = fluxes

        results['differential_fluxes'] = pd.DataFrame(flux_comparison).T

    return results


def identify_metabolic_vulnerabilities(
    fba_results: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Identify subtype-specific metabolic vulnerabilities.

    Args:
        fba_results: Results from run_subtype_fba
        logger: Optional logger

    Returns:
        DataFrame of vulnerability scores by reaction and subtype
    """
    if fba_results['differential_fluxes'] is None:
        return pd.DataFrame()

    diff_flux = fba_results['differential_fluxes']

    # Calculate vulnerability score (high flux = high dependency)
    vulnerabilities = []

    for rxn_id in diff_flux.index:
        for subtype in diff_flux.columns:
            flux = diff_flux.loc[rxn_id, subtype]
            other_fluxes = [diff_flux.loc[rxn_id, st] for st in diff_flux.columns if st != subtype]
            mean_other = np.mean(other_fluxes) if other_fluxes else 0

            # Specificity score: how much higher is this subtype vs others
            if mean_other != 0:
                specificity = (flux - mean_other) / abs(mean_other)
            else:
                specificity = flux

            vulnerabilities.append({
                'reaction': rxn_id,
                'subtype': subtype,
                'flux': flux,
                'specificity': specificity,
                'vulnerability_score': abs(flux) * (1 + abs(specificity))
            })

    vuln_df = pd.DataFrame(vulnerabilities)

    # Rank vulnerabilities
    vuln_df = vuln_df.sort_values('vulnerability_score', ascending=False)

    if logger:
        logger.info(f"Identified {len(vuln_df)} metabolic vulnerabilities")

    return vuln_df


def map_metabolic_drugs(
    vulnerabilities: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Map metabolic reactions to potential drug targets.

    Args:
        vulnerabilities: DataFrame from identify_metabolic_vulnerabilities
        logger: Optional logger

    Returns:
        DataFrame with drug-reaction mappings
    """
    # Known metabolic drug targets
    drug_targets = {
        'GLCt': {'drugs': ['2-DG', 'WZB117', 'BAY-876'], 'target': 'GLUT1/SLC2A1', 'pathway': 'Glucose uptake'},
        'HK': {'drugs': ['2-DG', 'Lonidamine', '3-BP'], 'target': 'HK2', 'pathway': 'Glycolysis'},
        'GLYC': {'drugs': ['Koningic acid', '3-BP'], 'target': 'GAPDH', 'pathway': 'Glycolysis'},
        'LDH': {'drugs': ['Oxamate', 'FX11', 'Galloflavin'], 'target': 'LDHA', 'pathway': 'Lactate production'},
        'PDH': {'drugs': ['DCA', 'CPI-613'], 'target': 'PDK1', 'pathway': 'Pyruvate oxidation'},
        'GLNt': {'drugs': ['V-9302', 'GPNA'], 'target': 'SLC1A5/ASCT2', 'pathway': 'Glutamine uptake'},
        'GLS': {'drugs': ['CB-839 (Telaglenastat)', 'BPTES', 'DON'], 'target': 'GLS', 'pathway': 'Glutaminolysis'},
        'GLUD': {'drugs': ['EGCG', 'R162'], 'target': 'GLUD1', 'pathway': 'Glutaminolysis'},
        'PPP': {'drugs': ['6-AN', 'DHEA'], 'target': 'G6PD', 'pathway': 'Pentose phosphate'},
        'DNPS': {'drugs': ['Methotrexate', 'Pemetrexed', '6-MP'], 'target': 'GART/ATIC', 'pathway': 'Purine synthesis'},
        'PYRS': {'drugs': ['Leflunomide', 'Brequinar'], 'target': 'DHODH', 'pathway': 'Pyrimidine synthesis'},
        'PHGDH': {'drugs': ['NCT-503', 'CBR-5884'], 'target': 'PHGDH', 'pathway': 'Serine synthesis'},
        'SHMT': {'drugs': ['SHIN1', 'AGF347'], 'target': 'SHMT2', 'pathway': 'One-carbon metabolism'},
        'FAS': {'drugs': ['TVB-2640', 'C75', 'Cerulenin'], 'target': 'FASN', 'pathway': 'Fatty acid synthesis'},
        'ACC': {'drugs': ['ND-646', 'TOFA'], 'target': 'ACC', 'pathway': 'Fatty acid synthesis'},
        'OXPHOS': {'drugs': ['Metformin', 'IACS-010759', 'Oligomycin'], 'target': 'Complex I/V', 'pathway': 'OXPHOS'},
    }

    drug_mappings = []

    for _, row in vulnerabilities.iterrows():
        rxn = row['reaction']
        if rxn in drug_targets:
            info = drug_targets[rxn]
            for drug in info['drugs']:
                drug_mappings.append({
                    'reaction': rxn,
                    'subtype': row['subtype'],
                    'flux': row['flux'],
                    'vulnerability_score': row['vulnerability_score'],
                    'drug': drug,
                    'target': info['target'],
                    'pathway': info['pathway']
                })

    drug_df = pd.DataFrame(drug_mappings)

    if logger and len(drug_df) > 0:
        logger.info(f"Mapped {len(drug_df)} drug-reaction pairs")

    return drug_df


def run_metabolic_analysis(
    expression_path: Path,
    subtype_path: Path,
    output_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run complete metabolic analysis pipeline.

    Args:
        expression_path: Path to expression matrix
        subtype_path: Path to subtype assignments
        output_dir: Output directory
        logger: Optional logger

    Returns:
        Complete results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'stage': 'metabolic_analysis',
        'timestamp': datetime.now().isoformat(),
        'success': False
    }

    try:
        # Check COBRApy
        if not check_cobrapy():
            raise ImportError("COBRApy not installed. Run: pip install cobra")

        # Load data
        if logger:
            logger.info("Loading expression data...")
        expr = pd.read_csv(expression_path, sep='\t', index_col=0)

        subtypes = pd.read_csv(subtype_path, sep='\t', index_col=0)
        if 'dominant_subtype' in subtypes.columns:
            subtype_assignments = subtypes['dominant_subtype']
        else:
            # Assign based on max score
            subtype_cols = [c for c in subtypes.columns if c.startswith('SCLC_')]
            subtype_assignments = subtypes[subtype_cols].idxmax(axis=1)

        if logger:
            logger.info(f"Loaded {expr.shape[0]} genes x {expr.shape[1]} samples")
            logger.info(f"Subtype distribution: {subtype_assignments.value_counts().to_dict()}")

        # Create metabolic model
        if logger:
            logger.info("Creating SCLC metabolic model...")
        model = create_sclc_metabolic_model(logger)

        # Run subtype-specific FBA
        if logger:
            logger.info("Running subtype-specific flux analysis...")
        fba_results = run_subtype_fba(model, expr, subtype_assignments, logger)

        # Identify vulnerabilities
        if logger:
            logger.info("Identifying metabolic vulnerabilities...")
        vulnerabilities = identify_metabolic_vulnerabilities(fba_results, logger)

        # Map to drugs
        if logger:
            logger.info("Mapping to metabolic drugs...")
        drug_mappings = map_metabolic_drugs(vulnerabilities, logger)

        # Save results
        if fba_results['differential_fluxes'] is not None:
            fba_results['differential_fluxes'].to_csv(
                output_dir / 'subtype_fluxes.tsv', sep='\t'
            )

        vulnerabilities.to_csv(output_dir / 'metabolic_vulnerabilities.tsv', sep='\t', index=False)
        drug_mappings.to_csv(output_dir / 'metabolic_drug_targets.tsv', sep='\t', index=False)

        # Summary statistics
        summary = {
            'n_reactions': len(model.reactions),
            'n_metabolites': len(model.metabolites),
            'n_genes': len(model.genes),
            'subtypes_analyzed': list(fba_results['subtypes'].keys()),
            'top_vulnerabilities': vulnerabilities.head(20).to_dict('records') if len(vulnerabilities) > 0 else [],
            'top_drug_targets': drug_mappings.head(20).to_dict('records') if len(drug_mappings) > 0 else []
        }

        with open(output_dir / 'metabolic_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        results['fba_results'] = {k: v for k, v in fba_results.items() if k != 'differential_fluxes'}
        results['n_vulnerabilities'] = len(vulnerabilities)
        results['n_drug_targets'] = len(drug_mappings)
        results['output_dir'] = str(output_dir)
        results['success'] = True

        if logger:
            logger.info(f"Metabolic analysis complete. Results saved to {output_dir}")

    except Exception as e:
        results['error'] = str(e)
        if logger:
            logger.error(f"Metabolic analysis failed: {e}")

    return results
