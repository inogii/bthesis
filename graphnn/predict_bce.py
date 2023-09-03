import os
import csv
import sys
import yaml
import copy
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from pprint import pprint
from torch_geometric.data import Data
from collections import defaultdict

import neural_network
from neural_network import import_npz
from graphnn.parser import Parser


def is_forest(G : nx.DiGraph):
    comps = nx.weakly_connected_components(G)
    for comp in comps:
        sub_g = G.subgraph(comp)
        if not is_sink_tree(sub_g):
            return False
    return True

def is_sink_tree(G : nx.DiGraph):
    return nx.is_tree(G) and max(d for n, d in G.out_degree()) <= 1

def predict_all(args):
    print('Current device before model import:', torch.cuda.current_device())
    model_import = neural_network.initialize_model(args)
    dataset = import_npz(args.dataset)

    correct_count = defaultdict(int)
    correct_count_all = defaultdict(int)
    tree_count = defaultdict(int)
    pred_0_label_0 = 0
    pred_1_label_0 = 0
    pred_0_label_1 = 0
    pred_1_label_1 = 0
    for data_idx, data in enumerate(tqdm(dataset, total=len(dataset))):
        device = torch.device(args.device)
        model = model_import.to(device)
        x, edge_index = data.x, data.edge_index
        
        pred = neural_network.predict(model, data, device=args.device)
        mask = data.mask
        #print('Pred:', pred)
        #print('x:', x)
        #print('Networkx node names:', data.meta_data)
        #print('FF nodes  :', data.ff_net.nodes)
        #print('FF edges  :', data.ff_net.edges)
        correct = 1
        num_links = 0
        T = nx.DiGraph()
        for mask_idx, elem in enumerate(mask):
            if elem==1.0:
                link = data.meta_data[mask_idx].split('-')[1:]
                T.add_nodes_from(link)
                num_links += 1
                prediction = torch.argmax(pred[0][mask_idx]).cpu().detach().numpy()
                label = torch.argmax(data.y[mask_idx]).cpu().detach().numpy()
                #print(mask_idx, 'Pred :', prediction)
                #print(mask_idx, 'Label:', label)
                if prediction == 0 and prediction == label:
                    pred_0_label_0 += 1
                if prediction == 1 and prediction == label:
                    pred_1_label_1 += 1
                if prediction == 0 and prediction != label:
                    pred_0_label_1 += 1
                    correct = 0
                if prediction == 1 and prediction != label:
                    pred_1_label_0 += 1
                    correct = 0
                if prediction == 0:
                    T.add_edge(*link)
                #if prediction == 1:
                #    print('\t cutting edge', link)
        correct_count[num_links] += correct
        correct_count_all[num_links] += 1

        #print('Tree nodes:', T.nodes)
        #print('Tree edges:', T.edges)
        is_tree = is_forest(T)
        #print(is_tree)
        if is_tree:
            tree_count[num_links] += 1

        all_delays = data.all_delays
        all_edges_to_keep = data.all_edges_to_keep
        for idx, edges in enumerate(all_edges_to_keep):
            trees_equal = len(edges) == len(T.edges)
            for edge in edges:
                trees_equal &= (f's{edge[0]}', f's{edge[1]}') in T.edges
            if trees_equal:
                break
        if trees_equal:
            print('trees equal for index/edges: ', idx, all_edges_to_keep[idx], all_delays[idx], min(all_delays))
        else:
            print('trees NOT equal', min(all_delays))
                    

        #print(data.y.detach().cpu().numpy())
        #print("")

        #dataset[data_idx].result = pred[0].cpu().detach().numpy()

    print("pred_0_label_0:", pred_0_label_0)
    print("pred_1_label_0:", pred_1_label_0)
    print("pred_0_label_1:", pred_0_label_1)
    print("pred_1_label_1:", pred_1_label_1)

    for k,v in sorted(correct_count_all.items()):
        print('Num links: {: 3d} -> {}/{} \t ({}%) correct. {}/{} trees'.format(k, correct_count[k], v, correct_count[k]/v*100, tree_count[k], v))
    return None

def main(args):
    res = predict_all(args)
    #Parser.export_data(res, args.output)
    #np.savez(args.output, *res)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, help="Dataset file (npz)")
    p.add_argument("--output", type=str, default="res.csv", help="File to write results to")
    p.add_argument("--input-model", type=str, default="model.out", help="Model file")
    p.add_argument("--num-input-params", type=int, default=10, help="Number of columns in the input data (x)")
    p.add_argument("--num-classes", type=int, default=1, help="Number of columns in the label data (y)")
    p.add_argument("--submission", action="store_true", help="Set to generate submission")
    p.add_argument("--device", choices=['cpu', 'cuda'], help="Set the device", default='cuda')
    p.add_argument("--hidden-size", type=int, default=64, help="Hidden size")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    p.add_argument("--nunroll", type=int, default=10, help="Number of unrolls")
    p.add_argument("--model-architecture", choices=['GatedGraphConv', 'ResGatedGraphConv'], help="Type of torch geometric model acrchitecture to use", default='GatedGraphConv')
    p.add_argument("--no-linear-layer-input", action="store_true", help="Set to NOT use a linear layer before the GRU")
    args = p.parse_args()
    main(args)

