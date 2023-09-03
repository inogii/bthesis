import graph_data
import gt_parser
import argparse

mode = 'default'
graph_types = ['default', 'classic', 'roles']
normalization = ['none', 'minmax', 'z']

def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "-m", "--mode", type=str,
        help='Running mode: default, complete or test'
    )
    parser.add_argument(
        "-g", "--graph_type", type=str, 
        help='Type of graph you would like to build: default, classic or roles'
    )
    parser.add_argument(
        "-n", "--normalization", type=str, 
        help='Normalization to be applied over the data: none, minmax or z'
    )
    
    return parser.parse_args()

def build_gml(mode, graph_type, normalization):
    graph_data.create_jsons(normalization=normalization, running_mode=mode)
    filename = 'gnn/dataset/' + mode + '_' + 'dataset_' + graph_type + '_' + normalization + '.gml'
    debug = False
    if mode == 'test':
        debug = True
    gt_parser.build_networkx(graph_type=graph_type, filename=filename, debug=debug)

def build_all():
    for norm in normalization:
        graph_data.create_jsons(normalization=norm, running_mode=mode)
        for graph_type in graph_types:
            filename = 'gnn/dataset/' + mode + '_' + 'dataset_' + graph_type + '_' + norm + '.gml'
            gt_parser.build_networkx(graph_type=graph_type, filename=filename)

def main():

    args = parse_args()
    kwargs = vars(args) # returns dict of command line arguments and corresponding values
    running_mode = kwargs['mode']
    graph_type = kwargs['graph_type']
    norm = kwargs['normalization']

    if kwargs['mode'] == 'complete':
        build_all()
    else:
        build_gml(**kwargs)
    properties = open('properties.json', 'r')
    new_properties = open(f'gnn/dataset/properties_{running_mode}_dataset_{graph_type}_{norm}.json', 'w')
    new_properties.write(properties.read())
    new_properties.close()
    properties.close()
    print('Done')
if __name__ == '__main__':  
    main()