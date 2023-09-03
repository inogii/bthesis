import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_features, num_classes, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
