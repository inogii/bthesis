import torch.nn as nn

from .memory_cell import MemoryCell
import torch_geometric.nn as gnn


class GGNN(MemoryCell):
    def __init__(self, nunroll, dropout_gru, **kwargs):
        super().__init__(**kwargs)
        self.nunroll = nunroll
        self.dropout_gru = dropout_gru
        self.cell = nn.ModuleList()
        self.after_cell = nn.ModuleList()

        for i in range(self.num_layers):
            self.cell.append(gnn.GatedGraphConv(self.cell_input, nunroll))
            linear_tmp = nn.Sequential(*[
                nn.LayerNorm(self.cell_input, elementwise_affine=True),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout_gru)
            ])
            self.after_cell.append(linear_tmp)

    def forward(self, data):
        x = data.x.float()

        if self.linear_layer_input:
            x = self.fci(x)

        for cell_idx, cell_elem in enumerate(self.cell):
            x = cell_elem(x, data.edge_index.long())
            x = self.after_cell[cell_idx](x)

        x = self.fco(x)
        return x
