
import torch 
import torch.nn as nn
import numpy as np 
class MLP_DEEP(nn.Module):
    def __init__(self, env, out_features):
        super().__init__()
        self.out_features = out_features
        self.env = env
        self.network = nn.Sequential(
            nn.Linear(np.prod(np.array(env.single_observation_space.shape)), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, x):
        return self.network(x)
    
class MLP_SIM(nn.Module):
    def __init__(self, env, out_features):
        super().__init__()
        self.out_features = out_features
        self.env = env
        self.network = nn.Sequential(
            nn.Linear(np.prod(np.array(env.single_observation_space.shape)), 32),
            nn.ReLU(),
            nn.Linear(32, out_features),
        )

    def forward(self, x):
        return self.network(x)
