import torch 
import torch.nn as nn
import numpy as np 
from torch.distributions.categorical import Categorical


def get_activation(name):
    if name == "relu":
        ACT = nn.ReLU
    elif name == "tanh":
        ACT = nn.Tanh
    elif name =="gelu":
        ACT = nn.GELU
    elif name is None:
        ACT = None
    else:
        raise NotImplementedError()
    return ACT


def make_fc(in_features, hidden_dim, num_layers, in_activation, last_activation, out_features):
    ACT = get_activation(in_activation)
    ACT_L = get_activation(last_activation)
    
    net = []
    if num_layers == 1:
        net.append(layer_init(nn.Linear(in_features, out_features)))
    else:
        net.append(layer_init(nn.Linear(in_features, hidden_dim)))
        net.append(ACT()) 
        for idx in range(num_layers-2):
            net.append(layer_init(nn.Linear(hidden_dim, hidden_dim)))
            net.append(ACT())
        net.append(layer_init(nn.Linear(hidden_dim, out_features)))
    if ACT_L is not None:
        net.append(ACT_L())
    return nn.Sequential(*net)

def make_cnn(in_channels, channels_kernel_stride_paddings_pool, last_activation, cnn_out_features, out_features):
    
    net = []
    c_prev = in_channels 
    for c, k, s, pad, pool in channels_kernel_stride_paddings_pool:
        net.append(nn.Conv2d(in_channels=c_prev, 
                             out_channels=c, 
                             kernel_size=k, 
                             stride=s, 
                             padding=pad))
        net.append(nn.ReLU())
        if pool>0:
            net.append(nn.MaxPool2d(pool))
        c_prev=c    
    net.append(nn.Flatten(1))
    if out_features>1:  
        net.append(nn.Linear(cnn_out_features, out_features))
        ACT_L = get_activation(last_activation)
        if ACT_L is not None:
            net.append(nn.ReLU())
    return nn.Sequential(*net)
    
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        h, w, c = envs.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x.permute((0, 3, 1, 2)) / 255.0))  # "bhwc" -> "bchw"

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)



from .teacher import Teacher
from .student import Student