#!/usr/bin/env python3
"""
Copyright (c) 2019 Fabien Geyer
Copyright (c) 2021 Benedikt Jaeger
Copyright (c) 2021 Max Helm

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys

import copy
import yaml
import git
import csv
import argparse
import numpy as np
from tqdm import tqdm
import datetime
import glob
import time
import uuid

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.utils.negative_sampling import negative_sampling
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from models import implemented_models


def import_multiple_npz(directory):
    res = []
    filenames = glob.glob(os.path.join(directory, '*.npz'))
    for filename in tqdm(filenames, total=len(filenames)):
        res.extend(import_npz(filename))
    return res

def import_npz(dataset_path):
    data = np.load(dataset_path, allow_pickle=True)
    res = []
    num_edges = []
    num_nodes = []
    for data_point in data:
        data_dict = {}
        for elem in data[data_point]:
            try:
                if elem[0] == 'edge_index':
                    num_edges.append(2*int(elem[1].shape[1]))
                if elem[0] == 'num_nodes':
                    num_nodes.append(int(elem[1]))
                data_dict[elem[0]] = torch.from_numpy(elem[1])
            except Exception as e:
                data_dict[elem[0]] = elem[1]
        data_tmp = Data.from_dict(data_dict)
        res.append(data_tmp)
        data_dict = None
    return res

def predict(model, data, device):
    model.eval()
    data.x = torch.Tensor(data.x.float())
    data.y = torch.Tensor(data.y.float())
    data.mask = torch.Tensor(data.mask.float())
    data.edge_index = torch.Tensor(data.edge_index.float())
    device = torch.device(device)
    data = data.to(device)
    data_loader_var = DataLoader([data], batch_size=16)
    output = []
    for data in data_loader_var:
        out = model(data)
        output.append(out)
    return output


def data_loader(dataset, train_test_split, batch_size):
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_test_split)
    test_size = dataset_size - train_size

    dataset_train, dataset_test = random_split(dataset, [train_size, test_size])

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    print(f"Dataset size {dataset_size}: split to train={train_size} test={test_size}")
    return loader_train, loader_test


def train_model(model, dataset, criterion, args):
    train_test_split = args.train_test_split
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    model_output_dir = args.model_output_dir
    seed = args.seed
    regression = args.regression
    gradient_clipping = args.gradient_clipping

    loader_train, loader_eval = data_loader(dataset, train_test_split, batch_size)

    device = torch.device(args.device)
    cuda_available = not torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(torch.cuda.current_device()) if args.device == 'cuda' else 'CPU'
    print(f'Running on {device_name}. CUDA is {"not " if not cuda_available else ""}available.')

    model = model.to(device)

    torch.cuda.manual_seed(seed)

    # Adam optimizer with decaying learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_scheduler_factor)

    # NNI
    default_nni_score = 'f1_score'

    # Main loop
    training_results = []
    best_val_loss = float('inf')
    best_f1_score = 0.0
    best_precision_score = 0.0
    best_recall_score = 0.0
    for epoch in range(epochs):

        f1_score_eval = None
        precision_score_eval = None
        recall_score_eval = None
        losses_train = []
        losses_eval = []
        f1_scores = []
        precision_scores = []
        recall_scores = []

        # Train model on training data
        model.train()

        pbar = tqdm(total=len(loader_train), desc='Epoch {:3d}'.format(epoch), ncols=0)

        for data in loader_train:
            optimizer.zero_grad()
            output = model(data.to(device))

            # Select only the relevant nodes for the loss function
            idxmask = torch.where(data.mask)[0]
            mlabels = torch.index_select(data.y, 0, idxmask)
            moutput = torch.index_select(output, 0, idxmask)

            loss = criterion(moutput, mlabels)
            losses_train.append(loss.item())
            loss.backward()
            optimizer.step()
            pbar.update()

        # Use model on eval data
        model.eval()
        for data in loader_eval:
            with torch.no_grad():
                output = model(data.to(device))

                idxmask = torch.where(data.mask)[0]
                mlabels = torch.index_select(data.y, 0, idxmask)
                moutput = torch.index_select(output, 0, idxmask)

                val_loss = criterion(moutput, mlabels)
                losses_eval.append(val_loss.item())
            
                labels_argmax = torch.argmax(mlabels, axis=1)
                pred_argmax = torch.argmax(moutput, axis=1)

                if args.f1_score:
                    f1_score_data = f1_score(labels_argmax.cpu(), pred_argmax.cpu())
                    f1_scores.append(f1_score_data)
                    precision_score_data = precision_score(labels_argmax.cpu(), pred_argmax.cpu())
                    precision_scores.append(precision_score_data)
                    recall_score_data = recall_score(labels_argmax.cpu(), pred_argmax.cpu())
                    recall_scores.append(recall_score_data)
                
                pbar.update()

        if args.f1_score:
            f1_score_eval = np.mean(f1_scores)
            precision_score_eval = np.mean(precision_scores)
            recall_score_eval = np.mean(recall_scores)

            best_f1_score = max([f1_score_eval, best_f1_score])
            best_precision_score = max([precision_score_eval, best_precision_score])
            best_recall_score = max([recall_score_eval, best_recall_score])
        
        loss_train = np.mean(losses_train)
        loss_eval = np.mean(losses_eval)
        best_val_loss = min([loss_eval, best_val_loss])

        scheduler.step(loss_eval)
        current_learning_rate = scheduler._last_lr
        adam_lr = optimizer.param_groups[0]['lr']

        if args.nni:
            if args.f1_score:
                scores = {
                    'eval_loss': loss_eval,
                    'f1_score': f1_score_eval,
                    'precision_score': precision_score_eval,
                    'recall_score': recall_score_eval
                }

                scores['default'] = scores[default_nni_score]
                del scores[default_nni_score]
                nni.report_intermediate_result(scores)
            else:
                nni.report_intermediate_result(loss_eval)
        pbar.set_postfix(
            training_loss=loss_train,
            validation_loss=loss_eval,
            f1_score=f1_score_eval,
            precision=precision_score_eval,
            recall=recall_score_eval,
            current_lr=current_learning_rate,
            adam_lr=adam_lr
        )
        pbar.close()

        training_results.append({
            'epoch': epoch,
            'train_loss': loss_train,
            'eval_loss': loss_eval,
            'f1_score': f1_score_eval,
            'precision': precision_score_eval,
            'recall': recall_score_eval,
            'lr': current_learning_rate[0],
            'adam_lr': adam_lr
        })

        out_file = os.path.join(model_output_dir, f'model_epoch_{epoch:04d}.out')
        torch.save(model.state_dict(), out_file)

        with open(f'{model_output_dir}/results.csv', 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, training_results[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(training_results)

    if args.nni:
        if args.f1_score:

            scores = {
                'eval_loss': best_val_loss,
                'f1_score': best_f1_score,
                'precision_score': best_precision_score,
                'recall_score': best_recall_score
            }
            scores['default'] = scores[default_nni_score]
            del scores[default_nni_score]

            nni.report_final_result(scores)
        else:
            nni.report_final_result(best_val_loss)


def initialize_model(args):

    model_class = next(x for x in implemented_models if x.__name__ == args.model_architecture)
    model = model_class(**args.__dict__)

    if args.input_model:
        model.load_state_dict(torch.load(args.input_model))
    return model


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dir_name = '{}_{}_{}'.format(
        str(os.path.abspath(args.dataset)).replace('/', '_'),
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        str(uuid.uuid4())[:8]
    )

    full_dir_path = os.path.join(args.model_output_dir, dir_name)
    os.mkdir(full_dir_path)
    # directory exists and is empty
    if os.path.isdir(full_dir_path) and not os.listdir(full_dir_path):
        params = vars(args)
        params['dataset'] = os.path.abspath(params['dataset'])
        params['model_output_dir'] = os.path.abspath(full_dir_path)
        
        if args.regression:
            #criterion = nn.L1Loss(reduction='mean')
            criterion = nn.MSELoss(reduction='mean')
            #criterion = nn.SmoothL1Loss(reduction='mean')
            #criterion = nn.HuberLoss(reduction='mean', delta=0.5)
        else:
            criterion = nn.BCEWithLogitsLoss()
        params['loss_function'] = type(criterion).__name__
        params['loss_function_params'] = {k: v for k, v in vars(criterion).items() if not k.startswith('_')}
        
        try:
            repo = git.Repo(search_parent_directories=True)
            current_hash = repo.head.object.hexsha
            params['git_commit'] = current_hash
        except:
            params['git_commit'] = None
        with open('{}/config.yml'.format(args.model_output_dir), 'w+') as f:
            yaml.dump(params, f)
    else:
        raise ValueError('Model directory either doens\'t exist or is not empty')

    dataset = import_multiple_npz(args.dataset)
    print(dataset)
    args.num_features = dataset[0].x.shape[1]
    args.num_classes = dataset[0].y.shape[1]

    model = initialize_model(args)

    train_model(model, dataset, criterion, args)

    
if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1, help="Seed used for random number generator")
    p.add_argument("--dataset", type=str, help="Directory of npz files")
    p.add_argument("--epochs", type=int, default=15, help="Number of epochs for training")
    p.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate for Adam")
    p.add_argument("--weight-decay", type=float, default=0, help="Weight decay rate for Adam")
    p.add_argument("--lr-scheduler-factor", type=float, default=0.5, help="Learning rate is multiplied by this factor when validation accuracy plateaus")
    p.add_argument("--dropout", type=float, default=.5, help="Dropout used for between the linear layers")
    p.add_argument("--dropout-gru", type=float, default=.0, help="Dropout used between the GRUs")
    p.add_argument("--train-test-split", type=float, default=.75)
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--hidden-size", type=int, default=64, help="Size of the hidden messages")
    p.add_argument("--nunroll", type=int, default=10, help="Number of loop unrolling for the Gated Graph NN")
    p.add_argument("--model-output-dir", type=str, default="/tmp", help="Output file of trained model parameters")
    p.add_argument("--input-model", type=str, help="Load model from file.")
    p.add_argument("--regression", action="store_true", help="Set to use regression, without this flag classification is assumed")
    p.add_argument("--device", choices=['cpu', 'cuda'], help="Set the device", default='cpu')
    p.add_argument("--gradient-clipping", type=float, default=float('inf'), help="Value at which to clip the gradient. Off by default.")
    p.add_argument("--model-architecture", choices=[x.__name__ for x in implemented_models], help="Type of torch geometric model acrchitecture to use", default=[x.__name__ for x in implemented_models][0])
    p.add_argument("--linear-layer-input", action="store_true", help="Set to use a linear layer before the GRU")
    p.add_argument("--num-layers", type=int, default=1, help="Number of GRU layers")
    p.add_argument("--nni", action="store_true", help="Use NNI for hyper parameter tuning")
    p.add_argument("--f1-score", action="store_true", help="Use the F1 score instead of the loss for NNI")

    args = p.parse_args()
    if args.nni:
    	# Import nni and gets the hyper-parameters
    	import nni
    	hparams = nni.get_next_parameter()
    	for k, v in hparams.items():
        	setattr(args, k, v)

    main(args)
