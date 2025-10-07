import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(2,4)
        self.bn1 = nn.BatchNorm1d(4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4,8)
        self.bn2 = nn.BatchNorm1d(8)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x