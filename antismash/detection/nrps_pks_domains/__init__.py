# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.

""" Detection of NRPS/PKS domains within genes
"""

import logging
from typing import List

from antismash.common import path, subprocessing
from antismash.config.args import ModuleArgs

from .domain_identification import generate_domains, NRPSPKSDomains

NAME = "nrps_pks_domains"
SHORT_DESCRIPTION = "NRPS/PKS domain identification"


def get_arguments() -> ModuleArgs:
    """ Constructs commandline arguments and options for this module
    """
    args = ModuleArgs('Advanced options', '', override_safeties=True)
    return args


def check_options(options) -> List[str]:
    """ Checks the options to see if there are any issues before
        running any analyses
    """
    return []


def is_enabled(options) -> bool:
    """  Uses the supplied options to determine if the module should be run
    """
    # in this case, yes, always
    return True


def regenerate_previous_results(results, record, _options) -> NRPSPKSDomains:
    return NRPSPKSDomains.from_json(results, record)


def run_on_record(record, _previous_results, options) -> NRPSPKSDomains:
    logging.debug('Marking NRPS/PKS genes and domains in clusters')
    results = generate_domains(record)
    results.add_to_record(record)
    return results


def check_prereqs() -> List[str]:
    failure_messages = []
    for binary_name, optional in [('hmmscan', False), ('hmmpress', False)]:
        if path.locate_executable(binary_name) is None and not optional:
            failure_messages.append("Failed to locate executable for %r" %
                                    binary_name)

    markov_models = [path.get_full_path(__file__, 'data', filename) for filename in [
                                'abmotifs.hmm', 'dockingdomains.hmm',
                                'ksdomains.hmm', 'nrpspksdomains.hmm']]

    binary_extensions = ['.h3f', '.h3i', '.h3m', '.h3p']

    for hmm in markov_models:
        if path.locate_file(hmm) is None:
            failure_messages.append("Failed to locate file %r" % hmm)
            continue
        for ext in binary_extensions:
            binary = "{}{}".format(hmm, ext)
            if path.locate_file(binary) is None:
                result = subprocessing.run_hmmpress(hmm)
                if not result.successful():
                    failure_messages.append('Failed to hmmpress {!r}: {}'.format(hmm, result.stderr))
                break

    return failure_messages
