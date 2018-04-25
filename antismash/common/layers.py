# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.

""" A collection of classes for simplification of HTML templating.

    Each layer type is a different construct with helper functions that don't
    really belong in the original objects.
"""

import os
from types import ModuleType
from typing import Any, List, Optional

from antismash.config import ConfigType
from antismash.common.secmet import Record, Cluster
from antismash.common.module_results import ModuleResults


class RecordLayer:
    """ A layer for Record instances """
    def __init__(self, record: Record, results: ModuleResults, options: ConfigType) -> None:
        self.results = results
        self.seq_record = record
        self.options = options
        self.clusters = []  # type: List[ClusterLayer]
        for cluster in record.get_clusters():
            self.clusters.append(ClusterLayer(self, cluster))

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dict__:
            return super().__getattribute__(attr)
        return getattr(self.seq_record, attr)

    def get_from_record(self) -> str:
        " returns the text to be displayed in the overview table > separator-text "

        current_id = self.seq_record.id

        orig_id = self.seq_record.original_id

        if orig_id:
            if len(orig_id) < 40:
                current_id += " (original name was: %s)" % orig_id
            else:
                current_id += " (original name was: %s...)" % orig_id[:60]

        return 'The following clusters are from record %s:' % current_id


class ClusterLayer:
    """ A layer for Cluster instances, contains special members for result of
        the clusterblast and clusterfinder modules
    """
    def __init__(self, record: RecordLayer, cluster_feature: Cluster) -> None:
        assert isinstance(cluster_feature, Cluster), type(cluster_feature)
        assert cluster_feature.parent_record
        self.record = record  # type: RecordLayer
        self.anchor_id = "r{}c{}".format(cluster_feature.parent_record.record_index,
                                         cluster_feature.get_cluster_number())
        self.handlers = []  # type: List[ModuleType]
        self.cluster_feature = cluster_feature  # type: Cluster
        self.cluster_blast = []
        self.knowncluster_blast = []
        self.subcluster_blast = []
        if self.cluster_feature.knownclusterblast:
            self.knowncluster_blast_generator()
        if self.cluster_feature.subclusterblast:
            self.subcluster_blast_generator()
        if self.cluster_feature.clusterblast:
            self.cluster_blast_generator()

        self.find_plugins_for_cluster()
        self.has_details = self.determine_has_details()
        self.has_sidepanel = self.determine_has_sidepanel()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return super().__getattribute__(attr)
        return getattr(self.cluster_feature, attr)

    @property
    def best_knowncluster_name(self) -> str:
        """ The name of the best hit from knownclusterblast, if it was run """
        if not self.knownclusterblast:
            return "-"
        full = self.knownclusterblast[0][0]
        return full.split(": ")[1].replace("_", " ").replace(
                        " biosynthetic gene cluster", "")

    @property
    def bgc_id(self) -> str:
        """ The BGC id of the best hit from knownclusterblast, if it was run """
        if not self.cluster_feature.knownclusterblast:
            return "-"
        return format(self.cluster_feature.knownclusterblast[0][1])

    @property
    def detection_rules(self) -> List[str]:
        """ The details of rules that caused the cluster to be defined """
        return ["%s: %s" % (product, rules) for product, rules in
                zip(self.cluster_feature.products, self.cluster_feature.detection_rules)]

    def description_text(self) -> str:
        """ returns the gene cluster description """
        description_text = self.record.name \
            + ' - Gene Cluster %s. Type = %s. Location: %s - %s nt. ' % (
                        self.get_cluster_number(), self.get_product_string(),
                        self.location.start + 1, self.location.end)
        if self.probability != "BROKEN":  # TODO: real value check
            description_text += 'ClusterFinder probability: %s. ' % self.probability
        description_text += 'Click on genes for more information.'

        return description_text

    def cluster_blast_generator(self) -> None:  # TODO: deduplicate
        """ Generates the details to use for clusterblast results """
        assert self.cluster_feature.clusterblast
        top_hits = self.cluster_feature.clusterblast[:self.record.options.cb_nclusters]
        for i, label in enumerate(top_hits):
            i += 1  # 1-indexed
            svg_file = os.path.join('svg', 'clusterblast_r%dc%d_%s.svg' % (self.record.record_index, self.get_cluster_number(), i))
            self.cluster_blast.append((label, svg_file))

    def knowncluster_blast_generator(self) -> None:
        """ Generates the details to use for knownclusterblast results """
        assert self.cluster_feature.knownclusterblast
        top_hits = self.cluster_feature.knownclusterblast[:self.record.options.cb_nclusters]
        for i, label_pair in enumerate(top_hits):
            i += 1  # 1-indexed
            label = label_pair[0]
            svg_file = os.path.join('svg', 'knownclusterblast_r%dc%d_%s.svg' % (self.record.record_index, self.get_cluster_number(), i))
            self.knowncluster_blast.append((label, svg_file))

    def subcluster_blast_generator(self) -> None:
        """ Generates the details to use for subclusterblast results """
        assert self.cluster_feature.subclusterblast
        assert self.cluster_feature.subclusterblast is not None, self.cluster_feature.location
        top_hits = self.cluster_feature.subclusterblast[:self.record.options.cb_nclusters]
        for i, label in enumerate(top_hits):
            i += 1  # since one-indexed
            svg_file = os.path.join('svg', 'subclusterblast_r%dc%d_%s.svg' % (self.record.record_index, self.get_cluster_number(), i))
            self.subcluster_blast.append((label, svg_file))

    def find_plugins_for_cluster(self) -> List[ModuleType]:
        "Find a specific plugin responsible for a given gene cluster type"
        for plugin in self.record.options.plugins:
            if not hasattr(plugin, 'will_handle'):
                continue
            if plugin.will_handle(self.products):
                self.handlers.append(plugin)
        return self.handlers

    def determine_has_details(self) -> bool:
        """ Sets has_details to be True if at least one plugin might create
            detail output for the cluster
        """
        for handler in self.handlers:
            if hasattr(handler, "generate_details_div"):
                return True
        return False

    def determine_has_sidepanel(self) -> bool:
        """ Sets has_details to be True if at least one plugin might create
            output for the cluster sidepanel
        """
        for handler in self.handlers:
            if hasattr(handler, "generate_sidepanel"):
                return True
        return False


class OptionsLayer:
    """ A layer for the global Config options. """
    def __init__(self, options):
        self.options = options

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return super().__getattribute__(attr)
        return getattr(self.options, attr)

    @property
    def plugins(self) -> List[ModuleType]:
        """ A list of all modules """
        from antismash.main import get_all_modules
        return get_all_modules()

    @property
    def smcogs(self) -> bool:
        """ Whether smcogs was enabled or not """
        return (not self.options.minimal
                or self.options.smcogs_enabled
                or self.options.smcogs_trees)  # TODO work out a better way of doing this

    def download_logfile(self) -> Optional[str]:
        """ Returns the path of the logfile, if it was created (otherwise None) """
        logfile_path = os.path.abspath(self.logfile)
        if os.path.dirname(logfile_path).startswith(self.output_dir):
            return logfile_path[len(self.output_dir) + 1:]
        return None

    @property
    def base_url(self) -> str:
        """ Returns the 'home' URL for fungismash/antismash """
        if self.options.taxon == "fungi":
            return self.options.urls.fungi_baseurl
        return self.options.urls.bacteria_baseurl
