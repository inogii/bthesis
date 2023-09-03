import pickle
import itertools
import os
import datetime
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from modules.gnn import GNN
from modules.models import GraphSAGE, GCN, GAT, GIN
from modules.predictors import MLPPredictor
from torch.optim import Adam, SGD
import torch
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
import json
from nni.experiment import Experiment
import nni
import asyncio
warnings.filterwarnings("ignore")

#global variables
epochs = 1000
task = 'node_classification'
path = 'dataset/'
res = dict()
data = []
tests = 50 #number of tests per experiment
used_combinations_dataset = set()

def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "-e", "--experiment", type=str,
        help='experiment name'
    )
    parser.add_argument(
        "-v", "--variation", type=str, 
        help='variation name'
    )
    parser.add_argument(
        "-n", "--normalization", type=str, 
        help='normalization applied'
    )
    parser.add_argument(
        "-m", "--model", type=str, 
        help='gnn model'
    )
    parser.add_argument(
        "-t", "--timestamp", type=str, 
        help='timestamp'
    )
    
    return parser.parse_args()

def get_experiments():
    dataset_files = os.listdir(path = './'+path)
    dataset_files = [i for i in dataset_files if i.endswith('.pkl')]
    experiment_variations = [i.split('.')[0].split('-') for i in dataset_files]

    #print(dataset_files)
    #print(experiment_variations)
    experiments = {}
    experiments_normalizations = {}
    for i in experiment_variations:
        if i[0] not in experiments.keys():
            experiments[i[0]] = []
        if i[0] not in experiments_normalizations.keys():
            experiments_normalizations[i[0]] = []
        experiments[i[0]].append(i[1])
        experiments_normalizations[i[0]].append(i[2])
    return experiments, experiments_normalizations

def get_name(model):
    if str(model) == "<class 'modules.models.GraphSAGE'>":
        return 'GraphSAGE'
    elif str(model) == "<class 'modules.models.GCN'>":
        return 'GCN'
    elif str(model) == "<class 'modules.models.GAT'>":
        return 'GAT'
    elif str(model) == "<class 'modules.models.GIN'>":
        return 'GIN'

def get_model(model_name):
    if model_name == 'GraphSAGE':
        return GraphSAGE
    elif model_name == 'GCN':
        return GCN
    elif model_name == 'GAT':
        return GAT
    elif model_name == 'GIN':
        return GIN
    
def train_model(gnn, nn_model, learning_rate, learning_rate_decay, patience, epochs):
    model = nn_model(gnn.get_train_shape()[1], 32)
    try:
        predictor = MLPPredictor(32, gnn.get_label_shape()[1])
    except:
        predictor = MLPPredictor(32, 1)
    optimizer = Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=learning_rate)
    #optimizer = SGD(itertools.chain(model.parameters(), predictor.parameters()), lr=learning_rate)
    train_loss_list, val_loss_list = gnn.train_validation(model, predictor, optimizer, learning_rate_decay, patience, epochs=epochs)

    metric, (label, score) = gnn.score(model, predictor)

    accuracy = metric[0]
    f1 = metric[1]
    precission = metric[2]
    recall = metric[3]
    y_true = metric[4]
    y_pred = metric[5]

    return train_loss_list, val_loss_list, accuracy, f1, precission, recall, y_true, y_pred

def write_results():
    now = datetime.datetime.now()
    date_str = now.strftime('%Y-%m-%d-%H-%M-%S')
    filename = f'results/{task}-{date_str}'

    with open(filename+'.pickle', 'wb') as f:
        pickle.dump(res, f)
    table = tabulate(data, headers=['Model', 'Experiment', 'Variation', 'Max ACC', 'Avg ACC', 'Max macro-F1', 'Avg macro-F1', 'Max Precission', 'Avg Precission', 'Max Recall', 'Avg Recall',  'Learning Rate'])
    with open(filename+'.txt', 'w') as f:
        f.write(table)
    print(table)

def get_experiment_print_name(experiment):
    if experiment == "business_classification":
        return "Business Classification"
    elif experiment == "continent_classification":
        return "Continent Classification"
    elif experiment == "country_classification":
        return "Country Classification"
    elif experiment == "link_classification":
        return "Link Classification"
    elif experiment == "rir_classification":
        return "RIR Classification"

def plot_results(output, experiment, variation, normalization, timestamp):
    for model_name, loss_data in output.items():
        # Generate a unique filename for each plot
        with open('dataset/properties_default_dataset_'+variation+'_'+normalization+'.json', 'r') as io_str:
            readfile = io_str.read()
            properties = json.loads(readfile)
        display_labels = properties[experiment]
        exp_print = get_experiment_print_name(experiment)
        print(display_labels)
        date_str = timestamp
        filename = f'results/plots/{experiment}_{model_name}_train_{date_str}.pdf'
        print(filename)
        # Plot loss curve
        x_axis = np.arange(1, len(loss_data['train_loss_list'])+1)
        plt.plot(x_axis, loss_data['train_loss_list'], label='Train Loss')
        plt.plot(x_axis, loss_data['val_loss_list'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} {exp_print} Train Curve')
        plt.legend()
        plt.savefig(filename)  # Save the current figure to the PDF file
        plt.close()  # Close the current figure

        print(max(loss_data['y_true']))
        print(min(loss_data['y_true']))
        print(max(loss_data['y_pred']))
        print(min(loss_data['y_pred']))
        # Plot confusion matrix
        cm = confusion_matrix(loss_data['y_true'], loss_data['y_pred'], normalize='true')
        size = 10
        if experiment == 'country_classification':
            size = 30
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(size=size, cmap='Blues')
        plt.title(f'{model_name} {exp_print} Confusion Matrix')
        plt.savefig(f'results/plots/{experiment}_{model_name}_confusion_matrix_{date_str}.pdf')  # Save the current figure to the PDF file
        plt.close()  # Close the current figure

def load_dataset(experiment, variation, normalization, split_dataset, use_undersampler):

    gnn = GNN('node_classification', 'AC', debug=True)
    dataset_name = path + experiment + '-' + variation + '-' + normalization  + '.pkl'
    print('Loading: ' + dataset_name + '...')
    gnn.load_dataset(name=dataset_name, force_reload=True)
    if experiment == 'link_classification':
        gnn.split_dataset_validation(split_dataset, link_classification=True, use_undersampler=use_undersampler)
    elif variation == 'classic':
        gnn.split_dataset_validation(split_dataset, classic=True, use_undersampler=use_undersampler)
    else:
        gnn.split_dataset_validation(split_dataset, use_undersampler=use_undersampler)
    return gnn

def load_combinations_used():
    try:
        with open('dataset/combinations_used.pickle', 'rb') as f:
            return pickle.load(f)
    except:
        return set()

def save_combinations_used():
    with open('dataset/combinations_used.pickle', 'wb') as f:
        pickle.dump(used_combinations_dataset, f)

def load_combination(combination):
    with open(f'dataset/combinations/{combination}.pickle', 'rb') as f:
        gnn = pickle.load(f)
    return gnn

def save_combination(combination, gnn):
    with open(f'dataset/combinations/{combination}.pickle', 'wb') as f:
        pickle.dump(gnn, f)

def do_experiment(experiment, variation, normalization, model, timestamp):

    params = nni.get_next_parameter()
    print(params)
    split_dataset = params['split_dataset']
    use_undersampler = params['use_undersampler']

    combination = f'{experiment}-{variation}-{normalization}-{split_dataset}-{use_undersampler}'
    global used_combinations_dataset
    used_combinations_dataset = load_combinations_used()
    if combination in list(used_combinations_dataset):
        gnn = load_combination(combination)
    else:
        gnn = load_dataset(experiment, variation, normalization, split_dataset, use_undersampler)
        used_combinations_dataset.add(combination)
        save_combination(combination, gnn)
        save_combinations_used()

    nn_model = get_model(model)
    learning_rate = params['learning_rate']
    learning_rate_decay = params['learning_rate_decay']
    patience = params['patience']
    if use_undersampler:
        weight_exponent = 0
    else:
        weight_exponent = params['weight_exponent']

    gnn.weights = torch.pow(gnn.original_weights, weight_exponent)
    if torch.cuda.is_available():
        gnn.weights = gnn.weights.to('cuda')
    
    print(gnn.weights)
    train_loss_list, val_loss_list, accuracy, f1, precision, recall, y_true, y_pred = train_model(gnn, nn_model, learning_rate, learning_rate_decay, patience, epochs)
    output = {}
    output[model] = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list, 'y_true': y_true, 'y_pred': y_pred, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'learning_rate': learning_rate, 'learning_rate_decay': learning_rate_decay, 'patience': patience, 'weight_exponent': weight_exponent}
    
    data = {'Model': model, 'Experiment': experiment, 'Variation': variation, 'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall, 'learning_rate': learning_rate, 'learning_rate_decay': learning_rate_decay, 'patience': patience, 'weight_exponent': weight_exponent} 
    filename = f'results/{model}-{experiment}-{variation}-{normalization}-{timestamp}.txt'
    
    with open(filename, 'r') as f:
        best_results = json.load(f)
    
    if best_results['precision'] < precision:
        best_results = data
        with open(filename, 'w') as f:
            json.dump(best_results, f)
        with open(f'results/{model}-{experiment}-{variation}-{normalization}-{timestamp}.pickle', 'wb') as f:
            pickle.dump(output, f)
        plot_results(output, experiment, variation, normalization, timestamp)

    nni.report_final_result(precision)

    

def main():

    args = parse_args()
    kwargs = vars(args) # returns dict of command line arguments and corresponding values

    experiment = kwargs['experiment']
    variation = kwargs['variation']
    normalization = kwargs['normalization']
    model = kwargs['model']
    timestamp = kwargs['timestamp']

    print(f'Experiment: {experiment}, Variation: {variation} Normalization: {normalization}')
    
    do_experiment(experiment, variation, normalization, model, timestamp)

if __name__ == '__main__':  
    main()

