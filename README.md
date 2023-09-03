# Autonomous System Models using BGP Data and GNNs

## Introduction

This project was developed as my Bachelor's Thesis at the [Chair of Network Architectures and Services in TUM](https://www.net.in.tum.de/homepage/)  

In this project I combined both computer networks and machine learning. The goal is to classify properties of Autonomous Systems (ASes) and their relationships in a BGP graph by using Graph Neural Networks. A Graph Neural Network approach is used for that purpose, as it is able to learn from the graph structure and the node features. The GNN models that can be used are GraphSAGE, GAT, GCN and GIN. I recommend the usage of GraphSAGE, since it is the model that obtained the best results in my experiments, as can be seen in my Thesis: [Thesis](thesis.pdf)

The project can infer the following properties of ASes:
- Continent
- Country
- RIR (Regional Internet Registry)
- Business type (ISP, Enterprise, Content, etc.)
- Link relationship (Customer to Provider, Peer to Peer, etc.)

## Usage

### Prerequisites
A Docker image is provided to make the environment installation easier. You can find the Docker installation instructions here: [Docker](https://docs.docker.com/get-docker/)

1. Build Docker image (contains all dependencies for the project)
```sh
docker build -t autonomous_system_gnns -f Dockerfile .
```

2. Run Docker container
```sh
docker run -it autonomous_system_gnns
```

3. Run the project

Some sample commands on how to run the project are given here. You can find more details on how to run the project in the [Project overview](#project-overview) section.

```sh
python create_networkx_graphs.py -m=default -g=classic -n=minmax
cd gnn
python create_dataset.py
python nni_tuning.py -e=country_classification -v=classic -n=minmax -m=GraphSAGE
```

## Project overview

The code of the project is divided into 4 main parts:
- AS graph
- NetworkX dataset
- DGL datasets (one per classification task)
- Model training

### AS graph

To create the AS graph, the GraphBuilder.py script is used. It needs several data files from different sources to build a graph containing the AS information. 
Once the graph is built, it can be enhanced by adding information about the link relationships between the ASes. In order to do this, the TopoScope algorithm is run first, and then by using relationships.py the relationships are added to the graph. 
A sample file (AS_graph.gt) is provided, to make experimenting with the project easier. Further details on how to build your own AS graph are provided here: [AS graph](graphbuilder.md)

Credits to Pascal Henschke for the work done in his Thesis: [Henschke Thesis](https://mediatum.ub.tum.de/doc/1576026/1576026.pdf)

### NetworkX dataset

Using the AS_graph.gt file, we can create 3 different types of datasets:
- Classic dataset (only capable of AS property estimation, simpler structure, less memory usage and training time)
- Default dataset (capable of AS property and link relationship estimation, middle structural complexity)
- Roles dataset (capable of AS property and link relationship estimation, most complex structure)

To create a specific type of dataset, you can run the following script:

```sh
python create_networkx_graphs.py -m=default -g=classic -n=minmax
python create_networkx_graphs.py -m=default -g=default -n=minmax
python create_networkx_graphs.py -m=default -g=roles -n=minmax
```

As you can see the script has 3 parameters:
- m (mode): default (full graph) or test (2 node and 1 link abstraction)
- g (graph): classic, roles or default
- n (normalization technique): minmax, z or none

The output dataset is a gml file saved in gnn/dataset/dataset_name.gml

### DGL Datasets

The gnn/create_dataset.py script automatically reads which NetworkX datasets are saved in gnn/dataset and creates datasets for each specific classification task. 
This way, if we have created the classic dataset and run the script, datasets for continent, country, rir and business type classification will be created according to the classic graph structure.

```sh
python create_dataset.py
```

### Model training

Once the specific datasets are created, you can run the gnn/nni_tuning.py script in order to train a model for a specific classification task. 

In the run.sh file you can find the commands used to train the models for the different classification tasks and models. Here is one example: 

```sh
python nni_tuning.py -e=country_classification -v=classic -n=minmax -m=GraphSAGE
```

