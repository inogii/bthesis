import graph_tool as gt

# turns an AS relationship file (ASRank, ProbLink, TopoScope) into a dict
def rel_file_to_dict(rel_file):
    relationships = {}
    with open(rel_file) as rf:
        for line in rf:
            if line.startswith('#'): # skip comments
                continue
            if (line.count('|') == 3): ## if caida serial-2 format (src|tgt|rel|data)
                p, c, rel, _ = line.split('|')
            else: ## caida serial-1 format (src|tgt|rel)
                p, c, rel = line.split('|') 
            relationships[(p, c)] = int(rel)
    return relationships
            
# given a graph_tool graph, add relationships to it from a rel_file
def add_relationship_data_to_graph(gt_graph, rel_file, prop_name = 'relationship'):
    rel_dict = rel_file_to_dict(rel_file)
    as_graph = gt_graph
    relationship = as_graph.new_edge_property('string')
    rel_code_to_string = {-1: 'p2c', 0: 'p2p', 1: 's2s'}
    for edge in as_graph.edges():
        src_AS = as_graph.vp['ASN'][edge.source()]
        tgt_AS = as_graph.vp['ASN'][edge.target()]
        if (src_AS, tgt_AS) in rel_dict:
            src_tgt_rel = rel_dict[(src_AS, tgt_AS)]
            relationship[edge] = rel_code_to_string[src_tgt_rel]
        elif (tgt_AS, src_AS) in rel_dict:
            tgt_src_rel = rel_dict[(tgt_AS, src_AS)]
            relationship[edge] = rel_code_to_string[tgt_src_rel][::-1]
        else:
            relationship[edge] = 'NA'
    as_graph.ep[prop_name] = relationship # internalize property map
    return as_graph
            
# create a graph_tool graph from a relationship file (such as CAIDA serial-1.txt or serial-2.txt)
def create_graph_from_rel_file(rel_file, prop_name = 'relationship'):
    as_graph = gt.Graph()
    rel_dict = rel_file_to_dict(rel_file)
    vertex_asn = as_graph.add_edge_list(rel_dict.keys(), hashed=True)
    as_graph.vp["ASN"] = vertex_asn
    as_graph = add_relationship_data_to_graph(as_graph, rel_file, prop_name=prop_name)
    return as_graph


if __name__ == "__main__":
    new_graph = gt.Graph()
    new_graph.load('AS_graph.gt')
    graph_rels = add_relationship_data_to_graph(new_graph, 'asrel_toposcope.txt')
    new_graph.save('AS_graph.gt')