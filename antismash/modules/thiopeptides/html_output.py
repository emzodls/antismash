# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.

""" HTML generation for the thiopeptides module """

from typing import List

from jinja2 import FileSystemLoader, Environment, StrictUndefined

from antismash.common import path
from antismash.common.layers import ClusterLayer


def will_handle(products: List[str]) -> bool:
    """ HTML generation only occurs if this function reutrns True """
    return 'thiopeptide' in products


class ThiopeptideLayer(ClusterLayer):
    """ A wrapper of ClusterLayer to allow for tracking the ThiopeptideMotifs """
    def __init__(self, record, results, cluster_feature):
        ClusterLayer.__init__(self, record, cluster_feature)
        self.motifs = []
        for motif in results.motifs:
            if motif.is_contained_by(self.cluster_feature):
                self.motifs.append(motif)


def generate_details_div(cluster_layer, results, record_layer, options_layer) -> str:
    """ Generates the HTML details section from the ThioResults instance """
    env = Environment(loader=FileSystemLoader(path.get_full_path(__file__, "templates")),
                      autoescape=True, undefined=StrictUndefined)
    template = env.get_template('details.html')
    details_div = template.render(record=record_layer,
                                  cluster=ThiopeptideLayer(record_layer, results, cluster_layer.cluster_feature),
                                  options=options_layer)
    return details_div


def generate_sidepanel(cluster_layer, results, record_layer, options_layer) -> str:
    """ Generates the HTML sidepanel section from the ThioResults instance """
    env = Environment(loader=FileSystemLoader(path.get_full_path(__file__, "templates")),
                      autoescape=True, undefined=StrictUndefined)
    template = env.get_template('sidepanel.html')
    cluster = ThiopeptideLayer(record_layer, results, cluster_layer.cluster_feature)
    record = record_layer
    sidepanel = template.render(record=record,
                                cluster=cluster,
                                options=options_layer)
    if cluster.motifs:
        return sidepanel
    return ""
