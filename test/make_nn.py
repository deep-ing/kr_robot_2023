
import os 
import sys 
print(os.getcwd())
sys.path.append("src")

import torch 
from models import make_fc, make_cnn 

net = make_fc(in_features=64, 
              hidden_dim=64, 
              num_layers=3, 
              in_activation='relu', 
              last_activation=None, 
              out_features=32)

print(net)
print(net(torch.rand(3, 64)).size())

net = make_cnn(
    in_channels=3, 
    channels_kernel_stride_paddings_pool=[
        [32, 3,3,1,0],
        [32, 3,1,1,2],
        [64, 3,1,1,0],
        [64, 3,1,1,2],
    ], 
    last_activation='gelu', 
    cnn_out_features=1600, 
    out_features=64
)

print(net)
print(net(torch.rand(3, 3, 64,64)).size())
