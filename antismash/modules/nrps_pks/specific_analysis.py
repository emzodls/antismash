#  License: GNU Affero General Public License v3 or later
#  A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.

'''
In-depth analysis and annotation of NRPS/PKS gene clusters.
'''

import logging

from antismash.common.secmet import Record
from antismash.config import ConfigType

from .orderfinder import analyse_biosynthetic_order
from .parsers import calculate_consensus_prediction, modify_monomer_predictions
from .results import NRPS_PKS_Results
from .structure_drawer import generate_chemical_structure_preds
from .substrates import run_pks_substr_spec_predictions


def generate_structure_images(record: Record, results: NRPS_PKS_Results, options: ConfigType) -> None:
    """ Generate the structure images based on monomers prediction for all
        cluster features
    """
    compound_predictions = {key: val[0] for key, val in results.cluster_predictions.items()}
    if compound_predictions:
        generate_chemical_structure_preds(compound_predictions, record, options)


def specific_analysis(record: Record, results: NRPS_PKS_Results, options: ConfigType) -> NRPS_PKS_Results:
    """ Runs the various NRPS/PKS analyses on a record and returns their results """
    nrps_pks_genes = record.get_nrps_pks_cds_features()

    if not nrps_pks_genes:
        logging.debug("No NRPS or PKS genes found, skipping analysis")
        return results

    logging.critical("no NRPS prediction methods to use")

    results.pks = run_pks_substr_spec_predictions(nrps_pks_genes)
    results.consensus, results.consensus_transat = calculate_consensus_prediction(nrps_pks_genes,
                                                         results.pks.method_results)

    modify_monomer_predictions(nrps_pks_genes, results.consensus)

    results.cluster_predictions = analyse_biosynthetic_order(nrps_pks_genes, results.consensus, record)
    generate_structure_images(record, results, options)
    return results
