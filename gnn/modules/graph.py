import dgl
import torch
import networkx as nx
import pandas as pd
from random import sample
import numpy as np


class Graph:
    def __init__(self):
        self.graph = None
        self.node_properties = [
        #'node_id',
        'node_hq_country',
        'node_hq_continent',
        'node_business_type',
        'node_rir',
        'node_is_VP',
        'node_transit_degree',
        'node_pfxs_originating',
        'node_pfxs_originating_raw',
        'node_ip_space_originating',
        'node_node_degree',
        'node_in_degree',
        'node_out_degree',
        'node_betweenness_d',
        'node_closeness_d',
        'node_harmonic_closeness_d',
        'node_pagerank_d',
        'node_eigenvector_vmap',
        'node_betweenness_ud',
        'node_closeness_ud',
        'node_harmonic_closeness_ud',
        'node_pagerank_ud',
        'node_eigenvector_ud',
        'node_local_clustering_d',
        'node_local_clustering_ud',
        'node_avg_neighbor_degree',
        ]
        self.link_properties = [
            'link_relationship',
            'link_vp_visibility',
            'link_advertised_pfxs_count',
            'link_transit_degree_ratio',
            'link_betweenness_d',
            'link_betweenness_ud'
        ]

        self.properties=[
            #'node_id',
            'ntype',
            'node_hq_country',
            'node_hq_continent',
            'node_business_type',
            'node_rir',
            'node_is_VP',
            'node_transit_degree',
            'node_pfxs_originating',
            'node_pfxs_originating_raw',
            'node_ip_space_originating',
            'node_node_degree',
            'node_in_degree',
            'node_out_degree',
            'node_betweenness_d',
            'node_closeness_d',
            'node_harmonic_closeness_d',
            'node_pagerank_d',
            'node_eigenvector_vmap',
            'node_betweenness_ud',
            'node_closeness_ud',
            'node_harmonic_closeness_ud',
            'node_pagerank_ud',
            'node_eigenvector_ud',
            'node_local_clustering_d',
            'node_local_clustering_ud',
            'node_avg_neighbor_degree',
            'link_relationship',
            'link_vp_visibility',
            'link_advertised_pfxs_count',
            'link_transit_degree_ratio',
            'link_betweenness_d',
            'link_betweenness_ud',
            'role_role'
        ]

    def create_dataset_dgl(self, file, label_name='node_business_type', mode='default'):
        #imports the graph created by the gt_parser.py
        nx_graph = nx.read_gml('dataset/' + file)
        
        if mode=='classic':
            #print('\t\tusing classic mode')
            self.graph = dgl.from_networkx(nx_graph, node_attrs=self.node_properties)
            property_set = self.node_properties
        else:
            self.graph = dgl.from_networkx(nx_graph, node_attrs=self.properties)
            num_feats = len(self.properties) - 1
            property_set = self.properties
        #gets a list with all the unique values for the selected label
        #print(property_set)
        labels = torch.unique(self.graph.ndata[label_name])
        labels_np = labels.numpy()
        labels_np = labels_np.astype(int)
        labels_np  = list(set(labels_np))
        #preallocates the 'feat' and 'label' sizes
        num_nodes = self.graph.number_of_nodes()
        num_labels = len(labels_np)-1
        num_feats = len(property_set) - 1
        label_tensor = torch.zeros((num_nodes, num_labels))
        feat_tensor = torch.zeros((num_nodes, num_feats))  # Subtract 1 for excluding node_business_type
        #iterates over all nodes, to add the values for 'feat' and 'label' to each node
        for node in self.graph.nodes():
            try:
                label_number = int(self.graph.nodes[node].data[label_name].item())
                label_one_hot = torch.zeros(num_labels)
                try:
                    label_one_hot[label_number] = 1
                except IndexError:
                    #in this case the label is -1
                    pass
                label_tensor[node] = label_one_hot
                feat_list = []
                for feature in self.graph.nodes[node].data:
                    if feature != label_name and feature != 'feat':  # Exclude label_name and 'feat'
                        try: 
                            nan = int(self.graph.nodes[node].data[feature].item())
                            feat_list.append(self.graph.nodes[node].data[feature].item())
                        except ValueError:
                            feat_list.append(0)
                feat_tensor[node] = torch.tensor(feat_list)

            except ValueError:
                pass
        #adds the calculated 'feat' and 'label' to all nodes
        self.graph.ndata['label'] = label_tensor
        self.graph.ndata['feat'] = feat_tensor
        #removes all other features from the nodes, in order to save memory
        for prop in property_set:
            if prop!='ntype' and prop!='label' and prop!='feat':
                del self.graph.ndata[prop]


    def read_from_edgelist(self, filename):
        self.graph = nx.read_edgelist(filename, delimiter=',', nodetype=int, comments='src_id,dst_id')
        print(self.graph)

    def create_subgraph(self, percentage):
        nodes_subset = sample(self.graph.nodes(), round(len(self.graph.nodes()) * percentage))
        self.graph = self.graph.subgraph(nodes_subset)
        print(self.graph)

    def delete_nodes_with_degree(self, degree):
        remove_set = [node for node, deg in dict(self.graph.degree()).items() if deg <= degree]
        self.graph.remove_nodes_from(remove_set)
        print(self.graph)

    def get_connected_components(self):
        self.graph = self.graph.subgraph(max(nx.connected_components(self.graph), key=len))

    def get_graph(self):
        return self.graph

    @staticmethod
    def write_edgelist(graph, filename):
        def line_prepender(file, line):
            with open(file, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(line.rstrip('\r\n') + '\n' + content)

        nx.write_edgelist(graph, filename, delimiter=',', data=False)
        line_prepender(filename, 'src_id,dst_id')

    @staticmethod
    def write_nodelist(graph, path, features_filename, label=None, to_merge=False, embeddings_filename=None):
        if embeddings_filename is None:
            features = pd.read_csv(features_filename)
        else:
            embeddings = pd.read_csv(embeddings_filename)
            col = ['ASN']
            col.extend(label)
            labels = pd.read_csv(features_filename).loc[:, col]
            features = pd.merge(embeddings, labels, on="ASN")
        if label is not None:
            col = ['ASN']
            col.extend(label)
            labels = features.loc[:, col]
            if to_merge:
                col_to_merge = []
                for col in label:
                    if 'None' not in col and labels[col].value_counts()[1] < 500:
                        col_to_merge.append(col)
                new_col = []
                for index, row in labels.iterrows():
                    merged = False
                    for col in col_to_merge:
                        if row[col] == 1:
                            new_col.append(1.0)
                            merged = True
                    if not merged:
                        new_col.append(0.0)
                labels = labels.drop(col_to_merge, axis=1)
                labels['_'.join(col_to_merge)] = new_col
            features = features.drop(label, axis=1)
        f = open(path + '/node_feature_export.csv', 'w')
        fn = open(path + '/node_featureless_export.csv', 'w')
        if label is not None:
            w = 'node_id,feat,label\n'
        else:
            w = 'node_id,feat\n'
        f.write(w)
        fn.write(w)
        for node in graph.nodes():
            node_features = features.loc[features['ASN'] == node].fillna(0).to_numpy()[0].tolist()[1:]
            node_features = ', '.join([str(feature) for feature in node_features])
            w = f'{str(node)},"{node_features}"\n'
            wn = f'{str(node)},"1.0,0.0"\n'
            if label is not None:
                node_labels = labels.loc[labels['ASN'] == node].fillna(0).to_numpy()[0].tolist()[1:]
                node_labels = ', '.join([str(n_label) for n_label in node_labels])
                w = f'{str(node)},"{node_features}","{node_labels}"\n'
                wn = f'{str(node)},"1.0,0.0","{node_labels}"\n'
            f.write(w)
            fn.write(wn)

    @staticmethod
    def write_nodelist_with_subset(graph, path, features_filename, label=None, to_merge=False, embeddings_filename=None):
        if embeddings_filename is None:
            features = pd.read_csv(features_filename)
        else:
            embeddings = pd.read_csv(embeddings_filename)
            col = ['ASN']
            col.extend(label)
            labels = pd.read_csv(features_filename).loc[:, col]
            features = pd.merge(embeddings, labels, on="ASN")
        if label is not None:
            col = ['ASN']
            col.extend(label)
            labels = features.loc[:, col]
            if to_merge:
                col_to_merge = []
                for col in label:
                    if 'None' not in col and labels[col].value_counts()[1] < 500:
                        col_to_merge.append(col)
                new_col = []
                for index, row in labels.iterrows():
                    merged = False
                    for col in col_to_merge:
                        if row[col] == 1:
                            new_col.append(1.0)
                            merged = True
                    if not merged:
                        new_col.append(0.0)
                labels = labels.drop(col_to_merge, axis=1)
                labels['_'.join(col_to_merge)] = new_col
            features = features.drop(label, axis=1)
        f = open(path + '/node_feature_export.csv', 'w')
        fn = open(path + '/node_featureless_export.csv', 'w')
        if label is not None:
            w = 'node_id,feat,label\n'
        else:
            w = 'node_id,feat\n'
        f.write(w)
        fn.write(w)
        new_graph = graph.copy()
        nodes_to_remove = []
        for node in graph.nodes():
            try:
                exists = features.loc[features['ASN'] == node].fillna(0).to_numpy()[0].tolist()[1:]
            except:
                nodes_to_remove.append(node)
        new_graph.remove_nodes_from(nodes_to_remove)
        for node in new_graph.nodes():
            node_features = features.loc[features['ASN'] == node].fillna(0).to_numpy()[0].tolist()[1:]
            node_features = ', '.join([str(feature) for feature in node_features])
            w = f'{str(node)},"{node_features}"\n'
            wn = f'{str(node)},"1.0,0.0"\n'
            if label is not None:
                node_labels = labels.loc[labels['ASN'] == node].fillna(0).to_numpy()[0].tolist()[1:]
                node_labels = ', '.join([str(n_label) for n_label in node_labels])
                w = f'{str(node)},"{node_features}","{node_labels}"\n'
                wn = f'{str(node)},"1.0,0.0","{node_labels}"\n'
            f.write(w)
            fn.write(wn)
        nx.write_edgelist(new_graph, path+'edges_export.csv', data=False, delimiter=',')
