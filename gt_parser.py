import enum
import networkx as nx
import numpy as np
from graphnn.parser import Parser
from graphnn.utils import plot_internal_graph
import time
import json
import pandas as pd
import pprint

class DeepTMAParser(Parser):

    def process_data(self, data, graph_type: str = 'roles'):

        if graph_type not in ['roles', 'default', 'classic']:
            print(graph_type + ' is an invalid graph_type')
            return None
            
        G = nx.Graph()

        parameter_template = {k: float('nan') for k in self.node_parameters}
        # Initialize all values with NaN, this allows to skip unset values during normalization

        def get_params(kwargs):
            params = dict(parameter_template)
            params.update((k, kwargs[k]) for k in set(kwargs).intersection(params))
            return params
        
        def get_link_parameters(kwargs):
            link_template = {k: float('nan') for k in self.link_parameters}
            params = dict(link_template)
            params.update((k, kwargs[k]) for k in set(kwargs).intersection(params))
            return params

        if graph_type == 'roles':
            print('\tBuilding role graph')

            for node in data["node"]:
                G.add_node(node["node"], **get_params({'ntype': self.node_types.node, **node}))

            for link in data["link"]:
                G.add_node(link["link"], **get_params({'ntype': self.node_types.link, **link}))
            
            for role in data["role"]:
                G.add_node(role["role_number"], **get_params({'ntype': self.node_types.role, **role}))
            
            for link in data["link"]:
                for i, p in enumerate(link["link_nodes"]):
                    G.add_edge(i, link["link"])
                    G.add_edge(link["link"], p)
                    G.add_edge(i, p)
            
            for role in data['role']:
                G.add_edge(role['role_node'], role['role_number'])
                G.add_edge(role['role_number'], role['role_link'])
        
        elif graph_type == 'default':
            print('\tBuilding default graph')
            for node in data["node"]:
                G.add_node(node["node"], **get_params({'ntype': self.node_types.node, **node}))

            for link in data["link"]:
                G.add_node(link["link"], **get_params({'ntype': self.node_types.link, **link}))
                #nx.add_path(G, link['link_nodes'])
            
            for link in data["link"]:
                [i, p] = link["link_nodes"]
                G.add_edge(i, link["link"])
                G.add_edge(link["link"], p)
                G.add_edge(i, p)

        elif graph_type == 'classic':
            #TODO
            print('\tBuilding classic graph')
            for node in data["node"]:
                G.add_node(node["node"], **get_params({'ntype': self.node_types.node, **node}))

            for link in data["link"]:
                [i, p] = link["link_nodes"]
                G.add_edge(i, p, **get_link_parameters({'ntype': self.node_types.link, **link}))

        return G


def build_networkx(graph_type: str = 'default', filename='test.gml', debug=False):
    
    with open('properties.json', 'r') as io_str:
        readfile = io_str.read()
        properties = json.loads(readfile)
    
    country_encoding = properties['countries_len']
    continent_encoding = properties['continents_len']
    business_encoding = properties['business_len']
    relationship_encoding = properties['relationships_len']
    rir_encoding = properties['rirs_len']
    role_encoding = properties['roles_len']

    
    NodeType = enum.IntEnum("NodeType", [
        "node",
        "link",
        "role"
    ])

    # Node attributes for all node types
    # encoding: values = 0 are encoded as scalar, values > 0 are one-hot encoded to the length of value 
    # is_y: set to true if we want to predict this value
    # mask: only required if is_y, sets a list of node types as mask fpr the loss function
    node_parameters = {

        'ntype': {'encoding': len(NodeType) + 1, 'is_y': False},
        # as properties
        'node_hq_country': {'encoding': country_encoding, 'is_y': False},
        'node_hq_continent': {'encoding': continent_encoding, 'is_y': False},
        'node_business_type' : {'encoding': business_encoding, 'is_y': True, 'mask': [NodeType.node]},
        'node_rir' : {'encoding': rir_encoding, 'is_y': False},
        'node_is_VP': {'encoding': 0, 'is_y': False},
        'node_transit_degree': {'encoding': 0, 'is_y': False},
        'node_pfxs_originating': {'encoding': 0, 'is_y': False},
        'node_pfxs_originating_raw': {'encoding': 0, 'is_y': False},
        'node_ip_space_originating': {'encoding': 0, 'is_y': False},
        'node_node_degree': {'encoding': 0, 'is_y': False},
        'node_in_degree': {'encoding': 0, 'is_y': False},
        'node_out_degree': {'encoding': 0, 'is_y': False},
        'node_betweenness_d': {'encoding': 0, 'is_y': False},
        'node_closeness_d': {'encoding': 0, 'is_y': False},
        'node_harmonic_closeness_d': {'encoding': 0, 'is_y': False},
        'node_pagerank_d': {'encoding': 0, 'is_y': False},
        'node_eigenvector_vmap': {'encoding': 0, 'is_y': False},
        'node_betweenness_ud': {'encoding': 0, 'is_y': False},
        'node_closeness_ud': {'encoding': 0, 'is_y': False},
        'node_harmonic_closeness_ud': {'encoding': 0, 'is_y': False},
        'node_pagerank_ud': {'encoding': 0, 'is_y': False},
        'node_eigenvector_ud': {'encoding': 0, 'is_y': False},
        'node_local_clustering_d': {'encoding': 0, 'is_y': False},
        'node_local_clustering_ud': {'encoding': 0, 'is_y': False},
        'node_avg_neighbor_degree': {'encoding': 0, 'is_y': False},
        # link properties
        'link_relationship': {'encoding': relationship_encoding, 'is_y': False},
        'link_vp_visibility': {'encoding': 0, 'is_y': False},
        'link_advertised_pfxs_count': {'encoding': 0, 'is_y': False},
        'link_transit_degree_ratio': {'encoding': 0, 'is_y': False},
        'link_betweenness_d': {'encoding': 0, 'is_y': False},
        'link_betweenness_ud': {'encoding': 0, 'is_y': False},
        #role properties
        'role_role': {'encoding': role_encoding, 'is_y': False}
    }

    link_parameters = {
        'link_relationship': {'encoding': relationship_encoding, 'is_y': False},
        'link_vp_visibility': {'encoding': 0, 'is_y': False},
        'link_advertised_pfxs_count': {'encoding': 0, 'is_y': False},
        'link_transit_degree_ratio': {'encoding': 0, 'is_y': False},
        'link_betweenness_d': {'encoding': 0, 'is_y': False},
        'link_betweenness_ud': {'encoding': 0, 'is_y': False}
    }

    parser = DeepTMAParser(
            node_types=NodeType,
            node_parameters=node_parameters
        )

    if graph_type == 'classic':
        NodeType = enum.IntEnum("NodeType", [
        "node",
        "link"
        ])
        node_parameters = {
            'ntype': {'encoding': 1, 'is_y': False},
            'node_hq_country': {'encoding': country_encoding, 'is_y': False},
            'node_hq_continent': {'encoding': continent_encoding, 'is_y': False},
            'node_business_type' : {'encoding': business_encoding, 'is_y': True, 'mask': [NodeType.node]},
            'node_rir' : {'encoding': rir_encoding, 'is_y': False},
            'node_is_VP': {'encoding': 0, 'is_y': False},
            'node_transit_degree': {'encoding': 0, 'is_y': False},
            'node_pfxs_originating': {'encoding': 0, 'is_y': False},
            'node_pfxs_originating_raw': {'encoding': 0, 'is_y': False},
            'node_ip_space_originating': {'encoding': 0, 'is_y': False},
            'node_node_degree': {'encoding': 0, 'is_y': False},
            'node_in_degree': {'encoding': 0, 'is_y': False},
            'node_out_degree': {'encoding': 0, 'is_y': False},
            'node_betweenness_d': {'encoding': 0, 'is_y': False},
            'node_closeness_d': {'encoding': 0, 'is_y': False},
            'node_harmonic_closeness_d': {'encoding': 0, 'is_y': False},
            'node_pagerank_d': {'encoding': 0, 'is_y': False},
            'node_eigenvector_vmap': {'encoding': 0, 'is_y': False},
            'node_betweenness_ud': {'encoding': 0, 'is_y': False},
            'node_closeness_ud': {'encoding': 0, 'is_y': False},
            'node_harmonic_closeness_ud': {'encoding': 0, 'is_y': False},
            'node_pagerank_ud': {'encoding': 0, 'is_y': False},
            'node_eigenvector_ud': {'encoding': 0, 'is_y': False},
            'node_local_clustering_d': {'encoding': 0, 'is_y': False},
            'node_local_clustering_ud': {'encoding': 0, 'is_y': False},
            'node_avg_neighbor_degree': {'encoding': 0, 'is_y': False},
        }

        parser = DeepTMAParser(
            link_parameters=link_parameters,
            node_types=NodeType,
            node_parameters=node_parameters
        )

    with open('graph_nodes.json', 'r') as io_str:
        readfile = io_str.read()
        nodes = json.loads(readfile)
    
    with open('graph_links.json', 'r') as io_str:
        readfile = io_str.read()
        links = json.loads(readfile)
    
    with open('graph_roles.json', 'r') as io_str:
        readfile = io_str.read()
        roles = json.loads(readfile)

    data = {'node' : nodes['node'], 'link': links['link'], 'role': roles['role']}
    processed = parser.import_raw(data, graph_type=graph_type)
    
    if debug:
    #matrix = parser.graph2matrix(processed, lambda x: x)
    
    #print(matrix['x'][3408])
    #print(matrix['x'][1])
    #code for checking if the one hot encoding is working fine
    # positions = [0, len(NodeType) + 1, country_encoding, continent_encoding, rir_encoding, 21, relationship_encoding, 5]
    # data_type = ['nodetype', 'country', 'continent', 'rir', 'node properties', 'link relationship', 'link properties']
    # pos0 = 0
    # pos1 = 0
    # for i in range(len(positions)-1):
    #     pos0 += positions[i]
    #     pos1 += positions[i+1]
    #     print(data_type[i])
    #     print(matrix['x'][19606][pos0:pos1])

    #print(len(matrix['x'][19614]))
    #print(matrix['x'][19614][-5:])

        plot_internal_graph(processed, node_types=NodeType)
        # print(nx.get_node_attributes(processed, 'ntype'))
        
        # print(nx.get_edge_attributes(processed, 'link_relationship'))
        # print(nx.get_edge_attributes(processed, 'link_vp_visibility'))
        # print(nx.get_edge_attributes(processed, 'link_advertised_pfxs_count'))
        # print(nx.get_edge_attributes(processed, 'link_transit_degree_ratio'))
        # print(nx.get_edge_attributes(processed, 'link_betweenness_d'))
        # print(nx.get_edge_attributes(processed, 'link_betweenness_ud'))
        # print(nx.get_edge_attributes(processed, 'role_role'))
        print(processed.nodes[0])
        print(processed.nodes[1])
        print(processed.edges[(0,1)])
        
    nx.write_gml(processed, filename)
    # parser.export_data(np.array([matrix, matrix], dtype=object), './dataset.npz')
    # matrix2 = parser.import_npz('./dataset.npz')[0]


if __name__ == "__main__":
    build_networkx(graph_type='classic', debug=True)