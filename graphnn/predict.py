import os
import csv
import sys
import yaml
import copy
import time
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
import sklearn

import neural_network
from neural_network import import_npz
from graphnn.parser import Parser
from models import implemented_models

def predict_all(args):
    dataset = import_npz(args.dataset)
    all_labels = []
    all_preds = []
    print('Current device before model import:', torch.cuda.current_device())
    base_dir = os.path.dirname(args.input_model)
    with open(os.path.join(base_dir, 'config.yml'), 'r') as f:
        model_config = yaml.safe_load(f)
        for k,v in model_config.items():
            # Do not overwrite the model we want to use
            if k == 'input_model': continue
            setattr(args, k, v)
    model_import = neural_network.initialize_model(args)

    rel_err = []
    for data_idx, data in enumerate(tqdm(dataset, total=len(dataset))):
        device = torch.device(args.device)
        model = model_import.to(device)
        x, edge_index = data.x, data.edge_index

        pred = neural_network.predict(model, data, device=args.device)
        mask = data.mask
        for mask_idx, elem in enumerate(mask):
            if elem==1.0:
                # TODO: De-normalization (can be done with mean and stddev in case of z-normalization)
                prediction = pred[0][mask_idx].cpu().detach().numpy()# * args.std + args.mean
                label = data.y[mask_idx].cpu().detach().numpy()# * args.std + args.mean
                rel_err.append(abs(prediction/label - 1))
                all_labels.append(label)
                all_preds.append(prediction)

    print(f'Mean absolute relative error: {np.mean(rel_err)}, median absolute relative error: {np.median(rel_err)}')
    print(f'MAPE: {sklearn.metrics.mean_absolute_percentage_error(all_labels, all_preds)*100.0}%')
    return None

def main(args):
    res = predict_all(args)

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
    p.add_argument("--model-architecture", choices=[x.__name__ for x in implemented_models], help="Type of torch geometric model acrchitecture to use", default=[x.__name__ for x in implemented_models][0])
    p.add_argument("--no-linear-layer-input", action="store_true", help="Set to NOT use a linear layer before the GRU")
    p.add_argument("--num-layers", type=int, default=1, help="Number of GRU layers")
    p.add_argument("--dropout-gru", type=float, default=.0, help="Dropout used between the GRUs")
    p.add_argument("--mean", type=float, default=.0, help="Mean for z score normalization")
    p.add_argument("--std", type=float, default=.0, help="Standard Deviation for z score normalization")
    args = p.parse_args()
    main(args)

