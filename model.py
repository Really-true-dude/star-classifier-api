import torch
from torch import nn

class StarClassifier(nn.Module):

    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()

        self.ff1 = nn.Linear(in_features, hidden_size, True)
        self.relu1 = nn.ReLU()
        self.ff2 = nn.Linear(hidden_size, hidden_size, True)
        self.relu2 = nn.ReLU()
        self.ff3 = nn.Linear(hidden_size, hidden_size, True)
        self.relu3 = nn.ReLU()
        self.final = nn.Linear(hidden_size, out_features, True)

    def forward(self, x: torch.Tensor):
        x = self.ff1(x)
        x = self.relu1(x)
        x = self.ff2(x)
        x = self.relu2(x)
        x = self.ff3(x)
        x = self.relu3(x)
        x = self.final(x)

        return x
    

