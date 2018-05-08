# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.


""" Classifies gene functions according to a curated set of HMM profiles.
    Phylogenetic trees of the gene functions and classification of a gene are
    possible, though must be enabled specifically in the options.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from antismash.common import path, subprocessing, hmmscan_refinement
from antismash.common.module_results import ModuleResults
from antismash.common.secmet import Record
from antismash.config import ConfigType
from antismash.config.args import ModuleArgs

from .trees import generate_trees
from .classify import classify_genes, load_cog_annotations, write_smcogs_file

NAME = "smcogs"
SHORT_DESCRIPTION = "gene function classification via smCOG"


class SMCOGResults(ModuleResults):
    """ Results for the smcogs module. Tracks the location of the tree images
        generated but does not keep a full copy of the tree.
    """
    schema_version = 1

    def __init__(self, record_id: str) -> None:
        super().__init__(record_id)
        self.tree_images = {}  # type: Dict[str, str] # gene_id -> tree filename
        self.best_hits = {}  # type: Dict[str, hmmscan_refinement.HMMResult] # gene_id -> best HMM result
        self.relative_tree_path = None  # type: str # path where tree images are saved

    def to_json(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version,
                "record_id": self.record_id,
                "tree_paths": self.tree_images,
                "best_hits": {key: [getattr(val, attr) for attr in val.__slots__]
                              for key, val in self.best_hits.items()},
                "image_dir": self.relative_tree_path}

    @staticmethod
    def from_json(json: Dict[str, Any], record: Record) -> "SMCOGResults":
        if json.get("schema_version") != SMCOGResults.schema_version:
            logging.debug("Schema version mismatch, discarding SMCOGs results")
            return None
        if record.id != json.get("record_id"):
            logging.debug("Record ID mismatch, discarding SMCOGs results")
            return None
        results = SMCOGResults(json["record_id"])
        for hit, parts in json["best_hits"].items():
            results.best_hits[hit] = hmmscan_refinement.HMMResult(*parts)

        if json.get("image_dir"):
            results.relative_tree_path = json["image_dir"]
            results.tree_images = json["tree_paths"]
        return results

    def add_to_record(self, record: Record) -> None:
        """ Annotate smCOGS in CDS features """
        functions = load_cog_annotations()
        logging.debug("annotating genes with SMCOGS info: %d genes", len(self.best_hits))
        for feature in record.get_cds_features_within_clusters():
            gene_id = feature.get_name()
            result = self.best_hits.get(gene_id)
            if result:  # TODO convert to qualifier like SecMetQualifier
                smcog_id, _ = result.hit_id.split(':', 1)
                feature.gene_functions.add(functions[smcog_id],
                             "smcogs", "%s (Score: %g; E-value: %g)" % (
                             result.hit_id, result.bitscore, result.evalue))
            if gene_id in self.tree_images:
                feature.notes.append("smCOG tree PNG image: smcogs/%s" % self.tree_images[gene_id])


def check_options(options: ConfigType) -> List[str]:
    """ Checks options for problems. """
    if options.smcogs_trees and (options.minimal and not options.smcogs_enabled):
        logging.debug("SMCOG trees enabled, but not classifications, running both anyway")
    return []


def get_arguments() -> ModuleArgs:
    """ Construct the arguments.
        Classification is enabled by default, but an extra option for generating
        trees is also required.
    """
    group = ModuleArgs("Basic analysis options", "smcogs", basic_help=True,
                       enabled_by_default=True)
    group.add_analysis_toggle('--smcogs-trees',
                              dest='smcogs_trees',
                              action='store_true',
                              default=False,
                              help="Generate phylogenetic trees of sec. "
                                   "met. cluster orthologous groups.")
    return group


def is_enabled(options: ConfigType) -> bool:
    """ Enabled if tree generation is requested or classification not disabled """
    return not options.minimal or options.smcogs_enabled or options.smcogs_trees


def regenerate_previous_results(results: Dict[str, Any], record: Record, options: ConfigType
                                ) -> Optional[SMCOGResults]:
    """ Reconstructs the previous results, unless the trees weren't generated
        previously or a previously generated tree output file is missing.
    """
    if not results:
        return None
    if options.smcogs_trees and not results["tree_paths"]:
        # trees have to be regenerated, so don't reuse
        logging.debug("Trees require recalculation")
        return None
    parsed = SMCOGResults.from_json(results, record)
    for tree_filename in parsed.tree_images.values():
        if not os.path.exists(os.path.join(parsed.relative_tree_path, tree_filename)):
            logging.debug("Tree image files missing and must be regenerated")
            return None
    return parsed


def check_prereqs() -> List[str]:
    "Check if all required applications are around"
    failure_messages = []
    for binary_name in ['muscle', 'hmmscan', 'hmmpress', 'fasttree', 'java']:
        if path.locate_executable(binary_name) is None:
            failure_messages.append("Failed to locate file: %r" % binary_name)

    for hmm in ['smcogs.hmm']:
        hmm = path.get_full_path(__file__, 'data', hmm)
        if path.locate_file(hmm) is None:
            failure_messages.append("Failed to locate file %r" % hmm)
            continue
        for ext in ['.h3f', '.h3i', '.h3m', '.h3p']:
            binary = "%s%s" % (hmm, ext)
            if path.locate_file(binary) is None:
                # regenerate them
                result = subprocessing.run_hmmpress(hmm)
                if not result.successful():
                    failure_messages.append("Failed to hmmpress %s: %s" % (hmm, result.stderr.rstrip()))
                break
    return failure_messages


def run_on_record(record: Record, results: Optional[SMCOGResults], options: ConfigType) -> SMCOGResults:
    """ Classifies gene functions and, if requested, generates phylogeny trees
        of the classifications
    """
    relative_output_dir = os.path.relpath(os.path.join(options.output_dir, "smcogs"), os.getcwd())
    smcogs_dir = os.path.abspath(relative_output_dir)
    if not os.path.exists(smcogs_dir):
        os.mkdir(smcogs_dir)

    if not results:
        results = SMCOGResults(record.id)

        genes = record.get_cds_features_within_clusters()
        hmm_results = classify_genes(genes)
        for gene in genes:
            gene_name = gene.get_name()
            hits = hmm_results.get(gene_name)
            if not hits:
                continue
            results.best_hits[gene.get_name()] = hits[0]
        write_smcogs_file(hmm_results, genes, record.get_nrps_pks_cds_features(), options)

    if not results.tree_images and options.smcogs_trees:
        # create the smcogs output directory if required
        results.relative_tree_path = relative_output_dir
        original_dir = os.getcwd()
        os.chdir(smcogs_dir)  # TODO make a context manager
        nrpspks_genes = record.get_nrps_pks_cds_features()
        nrpspks_genes = []
        results.tree_images = generate_trees(smcogs_dir, hmm_results, genes, nrpspks_genes)

        os.chdir(original_dir)

    return results
