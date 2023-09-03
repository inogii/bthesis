import nni
import datetime
import argparse
import socket
import json
import pickle
from nni.experiment import Experiment

search_space = {
    'learning_rate': {'_type': 'choice', '_value': [0.001]},
    'learning_rate_decay': {'_type': 'choice', '_value': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    'weight_exponent': {'_type': 'uniform', '_value': [1, 8]},
    'patience': {'_type': 'choice', '_value': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    'split_dataset' : {'_type': 'choice', '_value': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
    'use_undersampler' : {'_type': 'choice', '_value': [True, False]},
}

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
    
    return parser.parse_args()

def find_free_port():
    with socket.socket() as s:
        s.bind(('', 0))            # Bind to a free port provided by the host.
        return s.getsockname()[1]

def create_experiment_results_file(experiment, variation, normalization, model, timestamp):
    data = {'Model': model, 'Experiment': experiment, 'Variation': variation, 'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'learning_rate': 0, 'learning_rate_decay': 0, 'patience': 0, 'weight_exponent': 0} 
    with open(f'./results/{model}-{experiment}-{variation}-{normalization}-{timestamp}.txt', 'w') as f:
        f.write(json.dumps(data))

def load_best_results(experiment, variation, normalization, model, timestamp):
    with open(f'./results/{model}-{experiment}-{variation}-{normalization}-{timestamp}.txt', 'r') as f:
        data = json.load(f)
    with open(f'./results/{model}-{experiment}-{variation}-{normalization}-{timestamp}.pickle', 'r') as f:
        output = pickle.load(f)
    return data, output

def main():

    args = parse_args()
    kwargs = vars(args) # returns dict of command line arguments and corresponding values

    experiment = kwargs['experiment']
    variation = kwargs['variation']
    normalization = kwargs['normalization']
    model = kwargs['model']
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    nni_experiment = Experiment('local')
    nni_experiment.config.experiment_name = experiment + '-' + variation + '-' + normalization
    nni_experiment.config.trial_command = f'python classify.py -e={experiment} -v={variation} -n={normalization} -m={model} -t={timestamp}'
    nni_experiment.config.trial_code_directory = '.'
    nni_experiment.config.search_space = search_space
    nni_experiment.config.tuner.name = 'TPE'
    nni_experiment.config.tuner.class_args = {
        'optimize_mode': 'maximize'
    }
    nni_experiment.config.max_trial_number = 100
    nni_experiment.config.trial_concurrency = 1              
    
    port = find_free_port()

    create_experiment_results_file(experiment, variation, normalization, model, timestamp)

    nni_experiment.run(port)

    nni_experiment.stop()
    

if __name__ == '__main__':
    main()

   
