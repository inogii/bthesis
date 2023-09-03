import torch.nn as nn

from .base import Model


class MemoryCell(Model):
    def __init__(self, hidden_size, linear_layer_input, dropout, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.linear_layer_input = linear_layer_input
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.cell_input = self.hidden_size if self.linear_layer_input else self.num_features
        self.num_layers = num_layers

        # First layers
        self.fci = nn.Sequential(*[
            nn.Linear(self.num_features, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        ])

        # Last layers
        self.fco = nn.Sequential(*[
            nn.Linear(self.cell_input, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, self.num_classes),
        ])
