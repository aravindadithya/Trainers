import torch
import torch.nn as nn
import torch.nn.functional as F

class Nonlinearity(nn.Module):
    def __init__(self):
        super(Nonlinearity, self).__init__()

    def forward(self, x):
        return F.relu(x)

class Net(nn.Module):
    def __init__(self, dim, num_classes=2):
        super(Net, self).__init__()
        bias = False
        k = 1024
        self.dim = dim
        self.width = k

        self.features = nn.Sequential(
            nn.Linear(dim, k, bias=bias),
            Nonlinearity(),
            nn.Linear(k, k, bias=bias),
            Nonlinearity(),
        )

        self.classifier = nn.Sequential(           
            nn.Linear(k, num_classes, bias=bias)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x