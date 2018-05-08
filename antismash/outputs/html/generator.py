# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.

""" Responsible for creating the single web page results """

import json
import string
import os
from typing import Any, Dict, List

import jinja2
from jinja2 import FileSystemLoader, Environment

from antismash.common import path, module_results
from antismash.common.layers import RecordLayer, OptionsLayer
from antismash.common.secmet import Record
from antismash.config import ConfigType
from antismash.outputs.html import js
from antismash.typing import AntismashModule


def write_geneclusters_js(records: List[Dict[str, Any]], output_dir: str,
                          js_domains: List[List[Dict[str, Any]]]) -> None:
    """ Writes out the cluster and domain JSONs to file for the javascript sections
        of code"""
    with open(os.path.join(output_dir, 'geneclusters.js'), 'w') as handle:
        geneclusters = {"order": []}  # type: Dict[str, Any]
        for record in records:
            for cluster in record['clusters']:
                geneclusters[cluster['anchor']] = cluster
                geneclusters['order'].append(cluster['anchor'])
        handle.write('var geneclusters = %s;\n' % json.dumps(geneclusters, indent=4))

        clustered_domains = {}
        for cluster in js_domains:
            clustered_domains[cluster['id']] = cluster
        handle.write('var details_data = %s;\n' % json.dumps(clustered_domains, indent=4))


def generate_webpage(records: List[Record], results: List[Dict[str, module_results.ModuleResults]],
                     options: ConfigType) -> None:
    """ Generates and writes the HTML itself """
    generate_searchgtr_htmls(records, options)

    json_records = js.convert_records(records, results, options)

    js_domains = []

    for i, record in enumerate(records):
        json_record = json_records[i]
        json_record['seq_id'] = "".join(char for char in json_record['seq_id'] if char in string.printable)
        for json_cluster in json_record['clusters']:  # json clusters
            from antismash import get_analysis_modules  # TODO break circular dependency
            handlers = find_plugins_for_cluster(get_analysis_modules(), json_cluster)
            for handler in handlers:
                # if there's no results for the module, don't let it try
                if handler.__name__ not in results[i]:
                    continue
                if "generate_js_domains" in dir(handler):
                    domains = handler.generate_js_domains(json_cluster, record,  # type: ignore
                                                          results[i][handler.__name__], options)
                    if domains:
                        js_domains.append(domains)

    write_geneclusters_js(json_records, options.output_dir, js_domains)

    with open(os.path.join(options.output_dir, 'index.html'), 'w') as result_file:
        env = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True,
                          undefined=jinja2.StrictUndefined,
                          loader=FileSystemLoader(path.get_full_path(__file__)))
        template = env.get_template('index.html')
        options_layer = OptionsLayer(options)
        record_layers = []
        for record, record_results in zip(records, results):
            record_layers.append(RecordLayer(record, record_results, options_layer))

        records_written = sum(len(record.get_clusters()) for record in records)
        aux = template.render(records=record_layers, options=options_layer,
                              version=options.version, extra_data=js_domains,
                              records_written=records_written,
                              config=options)
        result_file.write(aux)


def find_plugins_for_cluster(plugins: List[AntismashModule], cluster: Dict[str, Any]) -> List[AntismashModule]:
    "Find a specific plugin responsible for a given gene cluster type"
    products = cluster['products']
    handlers = []
    for plugin in plugins:
        if not hasattr(plugin, 'will_handle'):
            continue
        if plugin.will_handle(products):
            handlers.append(plugin)
    return handlers


def load_searchgtr_search_form_template() -> List[str]:
    """ for SEARCHGTR HTML files, load search form template """
    with open(path.get_full_path(__file__, "searchgtr_form.html"), "r") as handle:
        template = handle.read().replace("\r", "\n")
    return template.split("FASTASEQUENCE")


def generate_searchgtr_htmls(records: List[Record], options: ConfigType) -> None:
    """ Generate lists of COGs that are glycosyltransferases or transporters """
    gtrcoglist = ['SMCOG1045', 'SMCOG1062', 'SMCOG1102']
    searchgtrformtemplateparts = load_searchgtr_search_form_template()
    # TODO store somewhere sane
    js.searchgtr_links = {}
    for record in records:
        for feature in record.get_cds_features():
            smcog_functions = feature.gene_functions.get_by_tool("smcogs")
            if not smcog_functions:
                continue
            smcog = smcog_functions[0].description.split(":")[0]
            if smcog not in gtrcoglist:
                continue
            html_dir = os.path.join(options.output_dir, "html")
            if not os.path.exists(html_dir):
                os.mkdir(html_dir)
            formfileloc = os.path.join(html_dir, feature.get_name() + "_searchgtr.html")
            link_loc = os.path.join("html", feature.get_name() + "_searchgtr.html")
            gene_id = feature.get_name()
            js.searchgtr_links[record.id + "_" + gene_id] = link_loc
            with open(formfileloc, "w") as formfile:
                specificformtemplate = searchgtrformtemplateparts[0].replace("GlycTr", gene_id)
                formfile.write(specificformtemplate)
                formfile.write("%s\n%s" % (gene_id, feature.translation))
                formfile.write(searchgtrformtemplateparts[1])
