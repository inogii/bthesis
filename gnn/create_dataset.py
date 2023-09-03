from modules.graph import Graph
import pickle
import os

graph = Graph()

dataset_path = 'dataset'

classification_tasks =  [
    'link_classification',
    'country_classification',
    'continent_classification',
    'rir_classification',
    'business_classification'
]

node_feature = [
    'link_relationship',
    'node_hq_country',
    'node_hq_continent',
    'node_rir',
    'node_business_type'   
]

graph_files = os.listdir(path = './'+dataset_path)
graph_files = [i for i in graph_files if i.endswith('.gml')]
variations = [i.split('.')[0].split('_')[-2] for i in graph_files]
normalizations = [i.split('.')[0].split('_')[-1] for i in graph_files]

for i in range(len(classification_tasks)):
    print(classification_tasks[i])
    for j in range(len(graph_files)):
        print('\t' + graph_files[j])
        if variations[j] == 'classic' and classification_tasks[i] == 'link_classification':
            continue
        else:
            filename = dataset_path + '/' + classification_tasks[i] + '-' + variations[j] + '-' + normalizations[j] + '.pkl'
            feature = node_feature[i]
            graph.create_dataset_dgl(file=graph_files[j], label_name=feature, mode=variations[j])
            f = open(filename, "wb")
            pickle.dump(graph.graph, f)
            f.close()