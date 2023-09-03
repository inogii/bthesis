import enum
import torch
import torch_geometric
import numpy as np
import networkx as nx
import git
import yaml
import datetime

from typing import NoReturn
from typing import Callable
from abc import ABC, abstractmethod
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data


class Parser(ABC):

    def __init__(self, node_types: enum, node_parameters: dict, link_parameters = None):
        self.data = []
        self.node_types = node_types
        self.node_parameters = node_parameters
        self.link_parameters = link_parameters

    def import_raw(self, data, graph_type='default'):
        return self.process_data(data, graph_type)
    
    def import_npz(self, dataset_path: str):
        data = np.load(dataset_path, allow_pickle=True)
        return [Data.from_dict(dict(d)) for d in data.values()]

    def export_meta(self, output_path: str, **kwargs) -> NoReturn:
        params = kwargs
        params['node_parameter'] = self.node_parameters
        repo = git.Repo(search_parent_directories=True)
        current_hash = repo.head.object.hexsha
        params['git_commit'] = current_hash
        params['timestamp'] = datetime.datetime.now()
        with open(output_path, 'w') as f:
            yaml.dump(params, f)

    def export_data(self, data: torch_geometric.data.Data, output_path: str) -> NoReturn:
        np.savez(output_path, *data)

    @abstractmethod
    def process_data(self, data):
        """
            input: data sample
            output: networkx graph representation
        """
        pass

    def graph2matrix(
            self,
            G: nx.Graph,
            normalization_function: Callable,
            ff_net: nx.Graph = None,
            all_delays: list = None,
            all_edges_to_keep: list = None,
            elapsed_time: float = None
    ) -> torch_geometric.data.data.Data:

        try:
            data = from_networkx(G)  # TODO group_edge_attrs=all
        except Exception as e:
            print(list(G.nodes(data=True)))
            for elem in list(G.nodes(data=True)):
                print(elem[0], len(elem[1]))
            raise e

        x = torch.Tensor()
        y = torch.Tensor()
        mask = torch.Tensor()

        # Each parameter column is either one-hot encoded or normalized
        for elem_name in self.node_parameters.keys():
            elem_vector = data[elem_name]
            one_hot = self.node_parameters[elem_name]['encoding']
            # Per feature normalization function, default normalization on normalization==None
            normalization_param = self.node_parameters[elem_name].get('normalization', normalization_function)

            m = elem_vector.isnan()  # Remember all NaN values to set them to 0 after normalization
            if one_hot > 0:
                elem_vector = torch.nan_to_num(elem_vector, nan=0).long()
                norm_elem_vector = torch.nn.functional.one_hot(elem_vector, num_classes=one_hot)
            else:
                norm_elem_vector = elem_vector.double().apply_(normalization_param).unsqueeze(1)
            norm_elem_vector[m, ...] = 0  # Set NaN values to 0

            if self.node_parameters[elem_name]['is_y']:
                mask_tmp = torch.zeros(len(norm_elem_vector), dtype=bool)
                for node_id, single_type in enumerate(data['ntype']):
                    for mask_type in self.node_parameters[elem_name]['mask']:
                        if mask_type == single_type:
                            mask_tmp[node_id] = True
                mask = torch.cat((mask, mask_tmp))

                y = torch.cat((y, norm_elem_vector), 1)
            else:
                x = torch.cat((x, norm_elem_vector), 1)

        res_data = Data(
            x=x.float().cpu().numpy(),
            edge_index=data.edge_index.cpu().numpy(),
            y=y.float().cpu().numpy(),
            mask=mask.cpu().numpy(),
            num_nodes=len(G.nodes),
            meta_data=list(G.nodes),
            ff_net=ff_net,
            all_delays=all_delays,
            all_edges_to_keep=all_edges_to_keep,
            elapsed_time=elapsed_time
        )
        # TODO edge attributes
        # res_data = Data(x=x.cpu().deatch().numpy(), edge_index=data.edge_index.cpu().deatch().numpy(), y=y.cpu().deatch().numpy(), mask=mask.cpu().deatch().numpy(), edge_attr=data.edge_attr.cpu().deatch().numpy(), num_nodes=len(G.nodes), meta_data=str(G.nodes))
        return res_data
