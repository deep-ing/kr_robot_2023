
import torch 
import torch.nn as nn

class OneCNN(nn.Module):
    def __init__(self, env, out_features):
        super().__init__()
        self.out_features = out_features
        self.env = env
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 256),
            nn.ReLU(),
            nn.Linear(256, out_features),
        )
    def forward(self, x):
        return self.network(x)
