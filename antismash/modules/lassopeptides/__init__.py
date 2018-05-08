# vim: set fileencoding=utf-8 :
#
# Copyright (C) 2010-2012 Marnix H. Medema
# University of Groningen
# Department of Microbial Physiology / Groningen Bioinformatics Centre
#
# Copyright (C) 2011,2012 Kai Blin
# University of Tuebingen
# Interfaculty Institute of Microbiology and Infection Medicine
# Div. of Microbiology/Biotechnology
#
# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.

"""Lassopeptides detection module

"""

from typing import Any, Dict, List

from antismash.common import path
from antismash.common.secmet import Record
<<<<<<< .merge_file_hE2nxZ
=======
from antismash.config import ConfigType
>>>>>>> .merge_file_RSb14C
from antismash.config.args import ModuleArgs

from .config import get_config
from .specific_analysis import specific_analysis, LassoResults
from .html_output import generate_details_div, generate_sidepanel, will_handle

NAME = "lassopeptides"
SHORT_DESCRIPTION = "lassopeptide precursor prediction"


def check_prereqs() -> List[str]:
    """ Checks if the required external programs are available """
    failure_messages = []
    for binary_name, optional in [('hmmpfam2', False), ('fimo', True)]:
        present = True
        if path.locate_executable(binary_name) is None:
            present = False
            if not optional:
                failure_messages.append("Failed to locate executable for %r" %
                                        binary_name)
        if binary_name == "fimo":
            get_config().fimo_present = present

    return failure_messages


def get_arguments() -> ModuleArgs:
    """ Runs by default, but add minimal's --enable option """
    args = ModuleArgs('Advanced options', 'lasso', enabled_by_default=True)
    return args


<<<<<<< .merge_file_hE2nxZ
def check_options(_options) -> List[str]:
=======
def check_options(_options: ConfigType) -> List[str]:
>>>>>>> .merge_file_RSb14C
    """ No options here to check, so just return """
    return []


<<<<<<< .merge_file_hE2nxZ
def is_enabled(options):
=======
def is_enabled(options: ConfigType) -> bool:
>>>>>>> .merge_file_RSb14C
    """ Will the module run with the given options """
    return not options.minimal or options.lassopeptides_enabled


<<<<<<< .merge_file_hE2nxZ
def regenerate_previous_results(previous: Dict[str, Any], record: Record, _options) -> LassoResults:
=======
def regenerate_previous_results(previous: Dict[str, Any], record: Record, _options: ConfigType) -> LassoResults:
>>>>>>> .merge_file_RSb14C
    """ Regenerate a results object from the given data """
    return LassoResults.from_json(previous, record)


<<<<<<< .merge_file_hE2nxZ
def run_on_record(record: Record, results: LassoResults, _options) -> LassoResults:
=======
def run_on_record(record: Record, results: LassoResults, _options: ConfigType) -> LassoResults:
>>>>>>> .merge_file_RSb14C
    """ Finds all precursors within lassopeptide clusters """
    if results and isinstance(results, LassoResults):
        return results
    return specific_analysis(record)
