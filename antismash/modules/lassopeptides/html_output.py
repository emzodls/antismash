# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.

""" Manages HTML construction for the Lassopeptide module
"""

from typing import List

from jinja2 import FileSystemLoader, Environment, StrictUndefined

from antismash.common import path
from antismash.common.layers import RecordLayer, ClusterLayer, OptionsLayer

from .specific_analysis import LassoResults


def will_handle(products: List[str]) -> bool:
    """ Returns True if the products provided are relevant to the module """
    return 'lassopeptide' in products


def generate_details_div(cluster_layer: ClusterLayer, results: LassoResults,
                         _record_layer: RecordLayer, _options_layer: OptionsLayer) -> str:
    """ Generates a HTML div for the main page of results """
    if not results:
        return ""
    env = Environment(loader=FileSystemLoader(path.get_full_path(__file__, "templates")),
                      autoescape=True, undefined=StrictUndefined)
    template = env.get_template('details.html')
    motifs_in_cluster = {}
    for locus in results.clusters.get(cluster_layer.get_cluster_number(), []):
        motifs_in_cluster[locus] = results.motifs_by_locus[locus]
    details_div = template.render(results=motifs_in_cluster)
    return details_div


def generate_sidepanel(cluster_layer: ClusterLayer, results: LassoResults,
                       _record_layer: RecordLayer, _options_layer: OptionsLayer) -> str:
    """ Generates a div for the sidepanel results """
    if not results:
        return ""
    env = Environment(loader=FileSystemLoader(path.get_full_path(__file__, "templates")),
                      autoescape=True, undefined=StrictUndefined)
    template = env.get_template('sidepanel.html')
    motifs_in_cluster = {}
    for locus in results.clusters.get(cluster_layer.get_cluster_number(), []):
        motifs_in_cluster[locus] = results.motifs_by_locus[locus]
    sidepanel = template.render(results=motifs_in_cluster)
    return sidepanel
