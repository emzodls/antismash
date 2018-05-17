# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.

""" The knownclusterblast variant of clusterblast, comparing clusters to MIBiG
    clusters.
"""
from pickle import dump
import logging,os
import numpy as np
from typing import Dict, List
from glob import glob
from helperlibs.wrappers.io import TemporaryDirectory
from collections import Counter
from scipy.optimize import linear_sum_assignment

from antismash.common import path,pfamdb
from antismash.common.secmet import Record,Cluster
from antismash.config import ConfigType
from pickle import load

from .core import  run_hmmalign,write_raw_bigscape_output
from .results import ClusterResult, GeneralResults, write_clusterblast_output



def _get_datafile_path(filename: str) -> str:
    """ A helper to construct absolute paths to files in the knownclusterblast
        data directory.

        Arguments:
            filename: the name only of the file

        Returns:
            the absolute path of the file
    """
    return path.get_full_path(__file__, 'data', 'known-bs', filename)


def check_known_prereqs(_options: ConfigType) -> List[str]:
    """ Determines if any prerequisite data files or executables are missing

        Arguments:
            options: antismash Config

        Returns:
            a list of error messages, one for each failing prequisite check
    """
    failure_messages = []
    for binary_name, optional in [('hmmalign', False),
                                  ('hmmfetch', False)]:
        if path.locate_executable(binary_name) is None and not optional:
            failure_messages.append("Failed to locate file: %r" % binary_name)

    for file_name, optional in [('pfam_bgc_dict.pkl', False),
                                ('mibigDomCts.pkl', False),
                                ('mibigPairs.pkl', False),
                                ('anchor.pkl', False)]:
        if path.locate_file(_get_datafile_path(file_name)) is None and not optional:
            failure_messages.append("Failed to locate file: %r" % file_name)

    return failure_messages

def load_mibig_bigscape_files():
    # specify paths
    data_dir = path.get_full_path(__file__, "data", "known_bs")
    mibig_pfam_dict_path = os.path.join(data_dir,'pfam_bgc_dict.pkl')
    mibig_dom_cts_path = os.path.join(data_dir,'mibigDomCts.pkl')
    mibig_pairs_path = os.path.join(data_dir,'mibigPairs.pkl')
    anchor_domains_path = os.path.join(data_dir,'anchor.pkl')

    #load objects
    mibig_pfam_dict = load(open(mibig_pfam_dict_path,'rb'))
    mibig_dom_cts = load(open(mibig_dom_cts_path,'rb'))
    mibig_pairs = load(open(mibig_pairs_path,'rb'))
    anchor_domains = load(open(anchor_domains_path,'rb'))

    return mibig_pfam_dict,mibig_dom_cts,mibig_pairs,anchor_domains


def run_bigscape_on_record(record: Record, options: ConfigType) -> GeneralResults:
    """ Run knownclusterblast on the given record

        Arguments:
            record: the record to analyse
            options: antismash Config

        Returns:
            an instance of GeneralResults with a result for each cluster in the record
    """
    logging.info('Running known cluster search with BiG-SCAPE distances')
    return perform_knowncluster_bigscape(options, record)

def perform_knowncluster_bigscape(options: ConfigType, record: Record) -> GeneralResults:

    logging.debug("Running BiG-SCAPE Distance Calculations")
    results = GeneralResults(record.id, search_type="knownclusterblast-bigscape")

    pfam_db_path = pfamdb.get_db_path_from_version('31.0',options.database_dir)

    logging.debug("Loading MiBIG Database")
    mibig_pfam_dict, mibig_dom_cts, mibig_pairs, anchor_domains = load_mibig_bigscape_files()
    for cluster in record.get_clusters():
        cluster_mibig_distances = compare_cluster_to_mibig(cluster,record,pfam_db_path,mibig_pfam_dict,
                                                 mibig_dom_cts, mibig_pairs, anchor_domains)

        write_raw_bigscape_output(options.output_dir,cluster_mibig_distances,
                                  prefix='bigscape-{}-cluster-{}'.format(record.id,cluster.get_cluster_number()))
    return results







def compare_cluster_to_mibig(query_cluster: Cluster,record: Record,pfam_db_path: str,
                             mibig_pfam_dict: Dict, mibig_dom_cts,mibig_pairs,anchor_domains):
    bgcs_to_test = set()
    cluster_pfam_dict = dict()
    cluster_domain_counts = Counter()
    cluster_domains = set()
    cluster_dom_strs = list()

    ## First see what pfam domains are actually in the cluster and make a dictionary with their locations and sequences

    # start at zero, will initialize with the first direction
    current_strand = 0
    current_dom_str = []
    for idx,cds in enumerate(query_cluster.cds_children):
        for pfam_domain in record.get_pfam_domains_in_cds(cds):
            pfamID = pfam_domain.domain

            # get domain counts
            cluster_domain_counts[pfamID] += 1
            cluster_domains.add(pfamID)

            # get domain str
            if current_strand == 0:
                current_strand = pfam_domain.location.strand
                current_dom_str.append(pfamID)
            elif pfam_domain.location.strand == current_strand:
                current_dom_str.append(pfamID)
            else:
                cluster_dom_strs.append(current_dom_str)
                current_dom_str = []
                current_strand = pfam_domain.location.strand

            if pfamID in mibig_pfam_dict:
                sequence = pfam_domain._translation
                dom_dict = cluster_pfam_dict.setdefault(pfamID,{})
                query_entries = dom_dict.setdefault('query',{})
                query_entries[idx,(pfam_domain.location.start,pfam_domain.location.end)] = sequence


    ## for the pfam domains seen, extract the corresponding mibig sequences that contain those pfam domains and add
    ## them to the cluster pfam dictionary
    for pfam_domain in cluster_pfam_dict.keys():
        for mibig_bgc in mibig_pfam_dict[pfam_domain]:
            bgcs_to_test.add(mibig_bgc)
            cluster_pfam_dict[pfam_domain].setdefault(mibig_bgc,{})
            for (idx,(start,stop)),seq in mibig_pfam_dict[pfam_domain][mibig_bgc]:
                cluster_pfam_dict[pfam_domain][mibig_bgc][(idx,(start,stop))] = seq
    ## perform the domain alignments
    with TemporaryDirectory(change=True) as tempdir:
        for domain, pfam_bgc_hits in cluster_pfam_dict.items():
            with open('{}.fa'.format(domain),'w') as domain_fasta_file:
                logging.debug('Writing fasta file for Domain: {}'.format(domain))
                for bgc in pfam_bgc_hits.keys():
                    for (idx,(start,stop)),seq in cluster_pfam_dict[domain][bgc].items():
                        domain_fasta_file.write('>{}%{}%{}-{}\n{}\n'.format(bgc,idx,start,stop,seq))

        logging.debug('Done writing fasta files.')
        ### search for all of the domain fasta files written and use hmmalign to generate alignments, then update
        ### pfam_dictionary

        domain_fastas = glob('*.fa')
        domains = [os.path.splitext(os.path.split(x)[1])[0] for x in domain_fastas]
        for domain_fasta, domain in zip(domain_fastas, domains):
            pfam_dict = cluster_pfam_dict[domain]
            aligned_pfam_dict = run_hmmalign(domain_fasta,domain,pfam_db_path,pfam_dict)
            cluster_pfam_dict[domain] = aligned_pfam_dict
    ## now that the sequences are aligned can use this dictionary to compare cluster
    distances = []
    logging.debug('Comparing to {} MiBiG Clusters'.format(len(bgcs_to_test)))
    for mibig_id in bgcs_to_test:
        distances.append((mibig_id,calculate_distance(cluster_pfam_dict,cluster_domain_counts,cluster_domains,cluster_dom_strs,
                                                      mibig_id,mibig_dom_cts,mibig_pairs,
                                                      anchor_domains=anchor_domains)))

    return distances

def calculate_distance(cluster_pfam_dict, query_cluster_dom_cts, query_cluster_doms,query_dom_strs,
                       mibig_id,mibig_dom_cts, mibig_pairs,
                       weights = (0.2, 0.75, 0.05, 2.0), anchor_domains = set()):
    logging.debug('Comparing cluster to MiBIG {}'.format(mibig_id))
    jacc_w, dss_w, ai_w, anchorboost = weights

    ref_cluster_dom_cts = mibig_dom_cts[mibig_id]
    ref_cluster_doms = set(domain for domain in ref_cluster_dom_cts.keys())

    domDist_Anchor, domCtr_Anchor = 0, 0
    domDist_noAnchor, domCtr_noAnchor = 0, 0

    shared_doms = query_cluster_doms & ref_cluster_doms
    unshared_doms = query_cluster_doms.symmetric_difference(ref_cluster_doms)

    ### Calculate the Jaccard Score

    jacc_score = len(shared_doms)/ (len(query_cluster_doms) + len(ref_cluster_doms))

    ### Calculate the domain similarity score

    for unshared_dom in unshared_doms:
        dom_ct = ref_cluster_dom_cts.get(unshared_dom,0) + query_cluster_dom_cts.get(unshared_dom,0)
        if unshared_dom in anchor_domains:
            domDist_Anchor += dom_ct
            domCtr_Anchor += dom_ct
        else:
            domDist_noAnchor += dom_ct
            domCtr_noAnchor += dom_ct

    for shared_dom in shared_doms:
        domain_dict = cluster_pfam_dict[shared_dom]

        query_dom_list = domain_dict['query']
        ref_dom_list = domain_dict[mibig_id]

        query_dom_ct = len(query_dom_list)
        ref_dom_ct = len(ref_dom_list)

        query_locs, query_seqs = zip(*list(query_dom_list.items()))
        ref_locs, ref_seqs = zip(*list(ref_dom_list.items()))

        # Fill distance matrix between domain's A and B versions
        dist_matrix = np.ndarray((query_dom_ct, ref_dom_ct))

        for query_dom_idx, query_seq in enumerate(query_seqs):
            for ref_dom_idx, ref_seq in enumerate(ref_seqs):

                if len(query_seq) != len(ref_seq):
                    logging.warning("\tWARNING: mismatch in sequences' lengths while calculating sequence identity ({})".format(
                        shared_dom))
                    logging.debug("\t  Specific domain 1: {} len: {}".format(query_locs[query_dom_idx], str(len(query_seq))))
                    logging.debug("\t  Specific domain 2: {} len: {}".format(ref_locs[ref_dom_idx], str(len(ref_seq))))
                    seq_length = min(len(query_seq), len(ref_seq))
                else:
                    seq_length = len(query_seq)

                matches = 0
                gaps = 0
                for position in range(seq_length):
                    if query_seq[position] == ref_seq[position]:
                        if query_seq[position] != "-":
                            matches += 1
                        else:
                            gaps += 1

                dist_matrix[query_dom_idx][ref_dom_idx] = 1 - (matches / (seq_length - gaps))

        best_idxs = linear_sum_assignment(dist_matrix)
        shared_dist = dist_matrix[best_idxs].sum()

        total_dist = (abs(query_dom_ct - ref_dom_ct) + shared_dist)

        if shared_dom in anchor_domains:
            domDist_Anchor += total_dist
            domCtr_Anchor += max(query_dom_ct, ref_dom_ct)
        else:
            domDist_noAnchor += total_dist
            domCtr_noAnchor += max(query_dom_ct, ref_dom_ct)

    if domCtr_Anchor != 0 and domCtr_noAnchor != 0:
        DSS_noAnchor = domDist_noAnchor / domCtr_noAnchor
        DSS_Anchor = domDist_Anchor / domCtr_Anchor

        # Calculate proper, proportional weight to each kind of domain
        noAnchorFrac = domCtr_noAnchor / (domCtr_Anchor + domCtr_noAnchor)
        anchorFrac = domCtr_Anchor / (domCtr_Anchor + domCtr_noAnchor)

        # boost anchor subcomponent and re-normalize
        noAnchor_weight = noAnchorFrac / (anchorFrac * anchorboost + noAnchorFrac)
        anchor_weight = anchorFrac * anchorboost / (anchorFrac * anchorboost + noAnchorFrac)

        # Use anchorboost parameter to boost percieved rDSS_anchor
        DSS = (noAnchor_weight * DSS_noAnchor) + (anchor_weight * DSS_Anchor)

    elif domCtr_Anchor == 0:
        DSS_noAnchor = domDist_noAnchor / domCtr_noAnchor
        DSS_Anchor = 0.0

        DSS = DSS_noAnchor

    else:  # only anchor domains were found
        DSS_noAnchor = 0.0
        DSS_Anchor = domDist_Anchor / domCtr_Anchor

        DSS = DSS_Anchor

    DSS = 1 - DSS  # transform into similarity

    ## Calculate Adjacency Index
    query_pairs = set()
    ref_pairs = mibig_pairs[mibig_id]
    for query_dom_str in query_dom_strs:
        if len(query_dom_str) >= 2:
            query_pairs.update(tuple(query_dom_str[i:i+2]) for i in range(len(query_dom_str)) if
                                     len(query_dom_str[i:i+2]) > 1 )
    ### If there are no intersecting pairs between query and reference AI is 0
    if len(query_pairs| ref_pairs) == 0:
        AI = 0
    else:
        AI = len(query_pairs & ref_pairs) / len(query_pairs | ref_pairs)

    distance = 1 - (jacc_w * jacc_score) - (dss_w * DSS) - (ai_w * AI)

    return (distance, jacc_score, DSS, AI, DSS_noAnchor, DSS_Anchor, domCtr_noAnchor, domCtr_Anchor)