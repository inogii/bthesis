import torch.nn as nn

from .memory_cell import MemoryCell
import torch_geometric.nn as gnn


class ResGGNN(MemoryCell):
    def __init__(self, nunroll, dropout_gru, **kwargs):
        super().__init__(**kwargs)
        self.nunroll = nunroll
        self.dropout_gru = dropout_gru
        self.cells = nn.ModuleList()

        for _ in range(self.num_layers):
            self.cells.append(
                gnn.ResGatedGraphConv(self.cell_input, self.cell_input)
            )

    def forward(self, data):
        x = data.x.float()

        if self.linear_layer_input:
            x = self.fci(x)

        for cell in self.cells:
            for _ in range(self.nunroll):
                x = cell(x, data.edge_index.long())

        x = self.fco(x)
        return x
