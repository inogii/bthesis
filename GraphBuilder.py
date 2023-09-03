# requires a version of python that preserves dict order (3.7+)
import argparse
import time
from collections import defaultdict
import numpy as np
import graph_tool.all as gt
import pytricia
import json
import pybgpstream
import pycountry_convert as country
import relationships


class GraphBuilder(object):
    def __init__(self, start_time = '2022-03-01 07:40:00', end_time = '2022-03-01 08:20:00', 
                 collectors = 'rrc00 route-views2', peeringdb_path = None, as2org_path = None,
                 write_paths = False, keep_origin_pfxs_set = False,
                 no_seeing_vps = False, keep_seeing_vps = False,
                 no_seeing_rcs = False, keep_seeing_rcs = False, 
                 no_advertised_pfxs = False, topcode_advertised_pfxs = False,
                 **kwargs):
        # create graph and add property with the args used to construct graph
        self.AS_graph = gt.Graph()
        args_used = 'start_time:', start_time, '| end_time:', end_time, 'collectors:', collectors, '| peeringdb_path:', peeringdb_path, '| as2org_path:', as2org_path
        self.AS_graph.gp["args_used"] = self.AS_graph.new_graph_property("string")
        self.AS_graph.gp["args_used"] = args_used
        # init stream, set of ixp asns to be filtered, and AS-to-org mapping
        self.stream = self.get_BGPstream(start_time, end_time, collectors)
        self.ixp_asns = self.get_ixp_asn_set(peeringdb_path)
        self.business_info = self.get_business_info(peeringdb_path)
        self.AS_org_info = self.get_as_info_dict(as2org_path)
        self.VPs = set() # set of asns that act as vantage points (VPs)
        self.transits = defaultdict(set) # asn x -> set of asns x forwards traffic for
        self.AS_origin_pfxs = defaultdict(set) # asn x -> set of prefixes x originates
        
        ### OPTIONS THAT MODIFY OUTPUT:
        self.write_paths = write_paths
        # node attribute options
        self.keep_origin_pfxs_set = keep_origin_pfxs_set
        # edge attribute options
        self.no_seeing_vps = no_seeing_vps
        self.keep_seeing_vps = keep_seeing_vps
        self.no_seeing_rcs = no_seeing_rcs
        self.keep_seeing_rcs = keep_seeing_rcs
        self.no_advertised_pfxs = no_advertised_pfxs
        self.topcode_advertised_pfxs = topcode_advertised_pfxs
        
        
    # Initialize and return a BGPStream instance
    def get_BGPstream(self, start_time, end_time, collectors):
        kwargs = {"from_time": start_time,
                "until_time": end_time,
                "collectors": collectors.split(" "),
                "record_type": "ribs",
                # filter paths with non-singleton AS Sets, by filtering commas ("{6339, 179}")
                "filter": "ipversion 4 and aspath !," 
                }
        if collectors.lower() == "all": kwargs.pop("collectors")
        
        print("Arguments used to construct BGPStream: ", kwargs)
        stream = pybgpstream.BGPStream(**kwargs)
        # stream.set_data_interface_option("broker", "cache-dir", CACHE_PATH) # enable caching of data
        # Set minimum time interval between two consecutive RIB files from the same collector
        stream.stream.add_rib_period_filter(86300) #86400s = 24h
        return stream
    
    
    # Extracts Route Server ASNs from PeeringDB
    # snippet taken from ProbLink code 
    # (https://github.com/YuchenJin/ProbLink/blob/master/bgp_path_parser.py)
    def get_ixp_asn_set(self, peeringdb_path):
        if peeringdb_path is None:
            return {}
        ixps = set()
        with open(peeringdb_path) as f:
            data = json.load(f)
            info_types = {''}
            for i in data['net']['data']:
                info_types.add(i['info_type'])
                if i['info_type'] == 'Route Server':
                    ixps.add(str(i['asn']))
            print(info_types)
        return ixps
    
    #returns dict {key:asn value:business type}
    def get_business_info(self, peeringdb_path):
        if peeringdb_path is None:
            return {}
        business = {}
        with open(peeringdb_path) as f:
            data = json.load(f)
            for i in data['net']['data']:
                if i['info_type'] == "":
                    business[str(i['asn'])] = 'Not Disclosed'
                else:
                    business[str(i['asn'])] = i['info_type']
        return business
    
    
    def get_as_info_dict(self , as2org_path):
        if as2org_path is None:
            return {}
        asn2orgID_dict = {}
        orgID_to_info_dict = {}
        asn_info_dict = {}
        with open(as2org_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry['type'] == 'ASN':
                    asn2orgID_dict[entry['asn']] = entry['organizationId']
                if entry['type'] == 'Organization':
                    orgID_to_info_dict[entry['organizationId']] = {'hq_country': entry.get('country', "NOT_AVAILABLE"), 
                                                                'name': entry.get('name', "NOT_AVAILABLE"), 
                                                                "rir/nir": entry.get('source', "NOT_AVAILABLE")}
                    if orgID_to_info_dict[entry['organizationId']]['hq_country'] == 'EU':
                        orgID_to_info_dict[entry['organizationId']]['hq_country'] = 'BE'
                    if orgID_to_info_dict[entry['organizationId']]['hq_country'] == 'VA':
                        orgID_to_info_dict[entry['organizationId']]['hq_country'] = 'IT'
                    if orgID_to_info_dict[entry['organizationId']]['hq_country'] == 'TL':
                        orgID_to_info_dict[entry['organizationId']]['hq_country'] = 'ID'
                    if orgID_to_info_dict[entry['organizationId']]['hq_country'] == 'SX':
                        orgID_to_info_dict[entry['organizationId']]['hq_country'] = 'PR'
        for asn, orgID in asn2orgID_dict.items():
            if asn not in asn_info_dict:
                asn_info_dict[asn] = orgID_to_info_dict[orgID]
        return asn_info_dict
    
    
    def build_graph(self):
        edges = dict() # (src, target) -> {edge_attribute: value, ...}
        if self.write_paths:
            paths_for_rel_inference = open("bgp_paths_out.txt", "w+")
        elem_count = 0
        for elem in self.stream:  
            elem_count +=1
            # if elem_count >= 1000000:
            #     print('why only 100k?')
            #     break
            ### AS PATH CLEANING AND FILTERING
            # 1. filter paths containing non-singleton AS Sets (in BGPStream regex) and unwrap singleton AS sets
            # 2. filter IXP ASNs from path
            # 3. remove prepended ASNs from path
            # 4. skip paths containing bogon ASNs
            raw_path = elem.fields['as-path'].replace('{', '').replace('}', '').split() #1
            clean_path = []
            bogon_flag = False
            for i, asn in enumerate(raw_path): 
                if asn in self.ixp_asns: #2
                    continue
                if i == 0 or asn != raw_path[i-1]: #3
                    clean_path.append(asn)
                if self.is_bogon_asn(asn): #4
                    bogon_flag = True
                    break
            if bogon_flag: #4
                continue
            
            try:
                origin_asn = raw_path[-1] # use raw AS path because sometimes a filtered IXP is origin
            except IndexError: # in exceedingly rare cases a peer sends an empty path default route
                continue
            collector = elem.collector
            vantage_point = elem.peer_asn
            pfx = elem.fields["prefix"]
            
            self.VPs.add(vantage_point)
            self.AS_origin_pfxs[origin_asn].add(pfx) 
            
            if self.write_paths: # write paths file for relationship inference algos
                paths_for_rel_inference.write('|'.join(clean_path) + "\n") 
            
            for i in range(len(clean_path) - 1): # iterate  through as-path hops
                if i != 0: # for calculation of transit degrees
                    self.transits[clean_path[i]].add(clean_path[i - 1]) # left AS
                    self.transits[clean_path[i]].add(clean_path[i + 1]) # right AS               
                edge = (clean_path[i], clean_path[i+1])
                if edge not in edges: # add edge
                    edges[edge] = dict() 
                    if not self.no_seeing_vps:
                        edges[edge]["seeing_VPs"] = {vantage_point}
                    if not self.no_seeing_rcs:
                        edges[edge]["seeing_RCs"] = {collector}
                    if not self.no_advertised_pfxs:
                        edges[edge]["advertised_prefixes"] = {pfx}
                else: # add attributes to existing edge
                    if not self.no_seeing_vps:
                        edges[edge]["seeing_VPs"].add(vantage_point)
                    if not self.no_seeing_rcs:
                        edges[edge]["seeing_RCs"].add(collector)
                    if not self.no_advertised_pfxs:
                        # if top-coding, only add pfx to set if set is not yet aggregated due to top-coding
                        if self.topcode_advertised_pfxs:
                            if not isinstance(edges[edge]["advertised_prefixes"], int):
                                edges[edge]["advertised_prefixes"].add(pfx)
                                if len(edges[edge]["advertised_prefixes"]) >= 250000:
                                    edges[edge]["advertised_prefixes"] = 250000
                        else:
                            edges[edge]["advertised_prefixes"].add(pfx)

        if self.write_paths:
            paths_for_rel_inference.close()
            
        ### initialize property maps
        # vertex property maps 
        business_type = self.AS_graph.new_vertex_property("string")
        org_name = self.AS_graph.new_vertex_property("string")
        hq_country = self.AS_graph.new_vertex_property("string")
        hq_continent =  self.AS_graph.new_vertex_property("string")
        rir = self.AS_graph.new_vertex_property("string")
        is_VP = self.AS_graph.new_vertex_property("int")
        transit_degree = self.AS_graph.new_vertex_property("int") 
        origin_pfxs_set = self.AS_graph.new_vertex_property("vector<string>") # set of the origin prefixes
        pfxs_originating = self.AS_graph.new_vertex_property("int") # length of the set of origin prefixes
        pfxs_originating_raw = self.AS_graph.new_vertex_property("int") # number of non-overlapping origin prefixes
        ip_space_originating = self.AS_graph.new_vertex_property("long") # number of IPs originated
        # edge property maps
        seeing_VPs = self.AS_graph.new_edge_property("vector<long>") # list of vantage points that see this edge
        seeing_RCs = self.AS_graph.new_edge_property("vector<string>") # list of RCs that see this edge
        vp_visibility = self.AS_graph.new_edge_property("int") # aggregated version of seeing_VPs
        rc_visibility = self.AS_graph.new_edge_property("int") # aggregated version of seeing_RCs
        advertised_pfxs_count = self.AS_graph.new_edge_property("int") # number of pfxs propagated across link
        transit_degree_ratio = self.AS_graph.new_edge_property("float") # transit_deg(src) / transit_deg(tgt)
        
        ### add edges to graph, yielding vertex-to-ASN property map
        vertex_asn = self.AS_graph.add_edge_list((edge for edge in edges), hashed=True) 
        
        ### fill vertex property maps
        for v in self.AS_graph.vertices():
            #asn number
            v_asn = vertex_asn[v]
            #get the business type
            if str(v_asn) in self.business_info:
                business_type[v] = self.business_info[str(v_asn)]
            else:
                business_type[v] = 'Not Disclosed'

            if v_asn in self.AS_org_info:
                org_name[v] = self.AS_org_info[v_asn]['name'] 
                hq_country[v] = self.AS_org_info[v_asn]['hq_country']
                rir[v] = self.AS_org_info[v_asn]['rir/nir']
                hq_continent[v] = country.country_alpha2_to_continent_code(self.AS_org_info[v_asn]['hq_country'])
            else:
                org_name[v] = "NOT_AVAILABLE"
                hq_country[v] = "NOT_AVAILABLE"
                rir[v] = "NOT_AVAILABLE"
            is_VP[v] = int(v_asn) in self.VPs
            transit_degree[v] = len(self.transits[v_asn])
            if self.keep_origin_pfxs_set:
                origin_pfxs_set[v] = self.AS_origin_pfxs[v_asn]
            pfx_count, ip_count = self.get_trie_based_pfx_and_ip_count(self.AS_origin_pfxs[v_asn])
            pfxs_originating[v] = pfx_count
            pfxs_originating_raw[v] = len(self.AS_origin_pfxs[v_asn])
            ip_space_originating[v] = ip_count        
        
        ### fill edge property maps
        for edge in self.AS_graph.edges():
            edge_tuple = (vertex_asn[edge.source()], vertex_asn[edge.target()])
            if not self.no_seeing_vps:
                vp_visibility[edge] = len(edges[edge_tuple]['seeing_VPs'])
                if self.keep_seeing_vps:
                    seeing_VPs[edge] = edges[edge_tuple]['seeing_VPs']
            if not self.no_seeing_rcs:
                rc_visibility[edge] = len(edges[edge_tuple]['seeing_RCs'])
                if self.keep_seeing_rcs:
                    seeing_RCs[edge] = edges[edge_tuple]['seeing_RCs']
            if not self.no_advertised_pfxs:
                if not isinstance(edges[edge_tuple]["advertised_prefixes"], int): # check this because of topcoding
                    edges[edge_tuple]["advertised_prefixes"] = len(edges[edge_tuple]["advertised_prefixes"])
                    advertised_pfxs_count[edge] = edges[edge_tuple]["advertised_prefixes"]
            transit_degree_ratio[edge] = self.safe_zero_div(transit_degree[edge.source()], transit_degree[edge.target()])
        

        
        ### internalize property maps
        # vertex property maps
        self.AS_graph.vp["ASN"] = vertex_asn
        self.AS_graph.vp["org_name"] = org_name
        self.AS_graph.vp["hq_country"] = hq_country
        #NEWLY ADDED : hq_continent
        self.AS_graph.vp["hq_continent"] = hq_continent
        #NEWLY ADDED : business_type
        self.AS_graph.vp["business_type"] = business_type
        self.AS_graph.vp["rir"] = rir
        self.AS_graph.vp["is_VP"] = is_VP
        self.AS_graph.vp["transit_degree"] = transit_degree
        if self.keep_origin_pfxs_set:
            self.AS_graph.vp["origin_pfxs_set"] = origin_pfxs_set
        self.AS_graph.vp["pfxs_originating"] = pfxs_originating
        self.AS_graph.vp["pfxs_originating_raw"] = pfxs_originating_raw
        self.AS_graph.vp["ip_space_originating"] = ip_space_originating
        # edge property maps
        if not self.no_seeing_vps:
            self.AS_graph.ep["vp_visibility"] = vp_visibility
            if self.keep_seeing_vps:
                self.AS_graph.ep["seeing_VPs"] = seeing_VPs
        if not self.no_seeing_rcs:
            self.AS_graph.ep["seeing_RCs"] = seeing_RCs
            if self.keep_seeing_rcs:
                self.AS_graph.ep["rc_visibility"] = rc_visibility
        if not self.no_advertised_pfxs:
            self.AS_graph.ep["advertised_pfxs_count"] = advertised_pfxs_count
        self.AS_graph.ep["transit_degree_ratio"] = transit_degree_ratio
        # add graph property maps with meta info about graph
        self.AS_graph.gp["vertex_count"] = self.AS_graph.new_graph_property("double", self.AS_graph.num_vertices())
        self.AS_graph.gp["edge_count"] = self.AS_graph.new_graph_property("double", self.AS_graph.num_edges())
        self.AS_graph.gp["rib_entries_processed"] = self.AS_graph.new_graph_property("double", elem_count)
        #self.AS_graph = relationships.add_relationship_data_to_graph(self.AS_graph, "asrel_toposcope.txt")

    def add_topological_features(self):
        print('Starting calculation of topological features...')
        g = self.AS_graph
        ## DEGREES
        g.vp['node_degree'] = g.degree_property_map("total")
        g.vp['in_degree'] = g.degree_property_map("in")
        g.vp['out_degree'] = g.degree_property_map("out")
        ## CENTRALITY MEASURES DIRECTED
        g.vp['betweenness_d'], g.ep['betweenness_d'] = gt.betweenness(g, norm=False)
        g.vp['closeness_d'] = gt.closeness(g)
        g.vp['harmonic_closeness_d'] = gt.closeness(g, harmonic=True)
        g.vp['pagerank_d'] = gt.pagerank(g)
        _, g.vp['eigenvector_vmap'] = gt.eigenvector(g)
        ## CENTRALITY MEASURES UNDIRECTED
        g.set_directed(False)
        g.vp['betweenness_ud'], g.ep['betweenness_ud'] = gt.betweenness(g, norm=False)
        g.vp['closeness_ud'] = gt.closeness(g)
        g.vp['harmonic_closeness_ud'] = gt.closeness(g, harmonic=True)
        g.vp['pagerank_ud'] = gt.pagerank(g)
        _, g.vp['eigenvector_ud'] = gt.eigenvector(g)
        g.set_directed(True)
        ## LOCAL CLUSTERING
        g.vp['local_clustering_d'] = gt.local_clustering(g, undirected=False)
        g.vp['local_clustering_ud'] = gt.local_clustering(g, undirected=True)
        ## AVERAGE NEIGHBOR DEGREE
        g.vp['avg_neighbor_degree'] = g.new_vertex_property('int')
        for v in g.vertices():
            neigh_degs = [g.vp['node_degree'][neigh] for neigh in g.iter_all_neighbors(v)]
            g.vp['avg_neighbor_degree'][v] = np.mean(neigh_degs)
        print('Topological features have been calculated.')
            
    
    def is_bogon_asn(self, asn):
        asn = int(asn)
        if asn == 0 or asn == 23456: # AS_TRANS
            return True
        if 64496 <= asn <= 131071: # docs and sample code, private use, reserved
            return True
        if asn >= 401309: # not assigned, private use
            return True
        return False

            
    # takes a set of prefixes as input and calculates the number of unique IP addresses
    # in that set as well as the number of non-overlapping prefixes
    def get_trie_based_pfx_and_ip_count(self, pfx_set):
        pfxtrie = pytricia.PyTricia()
        for pfx in pfx_set:
            net_mask = int(pfx.split("/")[1])
            if net_mask >= 8: # skip default routes
                pfxtrie.insert(pfx, None)
        pfx_count = 0 # counts non-overlapping prefixes
        ip_count = 0
        for pfx in pfxtrie:
            net_mask = int(pfx.split("/")[1])
            if pfxtrie.parent(pfx) is None: # count if not a subprefix
                pfx_count += 1
                ip_count += 2 ** (32 - net_mask)      
        return (pfx_count, ip_count)
    
    def safe_zero_div(self, x, y):
        if y == 0:
            return 0
        else:
            return x / y
    

# Function responsible for command line input
def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "-p", "--peeringdb_path", type=str,
        help='Path to PeeringDB json file'
    )
    parser.add_argument(
        "-o", "--as2org_path", type=str, 
        help='Path to CAIDA AS-to-Organization Mapping Dataset json file'
    )
    parser.add_argument(
        "-s", "--start_time",
        help='String specifying start datetime of analyzed BGP data. Defaults to "2022-03-01 07:40:00"'
    )
    parser.add_argument(
        "-e","--end_time",
        help='String specifying end date of analyzed BGP data. Defaults to "2022-03-01 08:20:00"'
    )
    parser.add_argument(
        "-c", "--collectors",
        help='String specifying which route collectors to use, separated by whitespace. ' +
              'Input "all" for all collectors. Defaults to "rrc00 route-views2"'
    )
    parser.add_argument(
        "-f", "--file", default="AS_graph",
        help='String specifying file name of output graph. Defaults to "AS_graph"'
    )    
    parser.add_argument(
        "--write_paths", action="store_true",
        help='Enables writing of a file containing the cleaned BGP paths (for later relationship inference)'
    )    
    parser.add_argument(
        "--keep_origin_pfxs_set", action="store_true",
        help='Flag to keep an actual set attribute of the origin prefixes for each AS ' +
              '(will increase file size of resulting graph significantly)'
    ) 
    parser.add_argument(
        "--no_seeing_vps", action="store_true",
        help='Disable link attribute that tracks the vantage point ASes that see each link'
    )  
    parser.add_argument(
        "--keep_seeing_vps", action="store_true",
        help='Flag to keep an actual set attribute of the vantage point ASes that see each link ' +
        '(No effect if no_seeing_vps flag is set)'
    )  
    parser.add_argument(
        "--no_seeing_rcs", action="store_true",
        help='Disable link attribute that tracks the route collector names that see each link'
    )   
    parser.add_argument(
        "--keep_seeing_rcs", action="store_true",
        help='Flag to keep an actual set attribute of the route collector names that see each link ' +
              '(No effect if no_seeing_rcs flag is set)'
    )  
    parser.add_argument(
        "--no_advertised_pfxs", action="store_true",
        help='Flag to disable link attribute that counts number of prefixes propagated across link'
    )    
    parser.add_argument(
        "--topcode_advertised_pfxs", action="store_true",
        help='Whether the computation of the advertised_pfxs attribute should use top-coding to conserve RAM'
    )   
    parser.add_argument(
        "--no_topo_features", action="store_true",
        help='Disables computation of topological features'
    )   
    return parser.parse_args()


# Read args, create graph.
def main():
    args = parse_args()
    kwargs = vars(args) # returns dict of command line arguments and corresponding values
    builder = GraphBuilder(**kwargs) 
    builder.build_graph()
    if "no_topo_features" not in kwargs:
        builder.add_topological_features()
    #builder.AS_graph = relationships.add_relationship_data_to_graph(builder.AS_graph, 'asrel_toposcope.txt')
    builder.AS_graph.save(kwargs["file"] + ".graphml")
    builder.AS_graph.save(kwargs["file"] + ".gt")
   
   
# Execute program if not imported
if __name__ == "__main__":
    main()
