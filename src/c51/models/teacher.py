import torch 
import torch.nn as nn
from . import QNetwork
        
class Teacher():
    def __init__(self, envs, network, flags):
        self.flags = flags
        self.teacher_flags = flags.teacher          
        self.q_network = QNetwork(envs, network, n_atoms=self.flags.n_atoms, v_min=self.flags.v_min, v_max=self.flags.v_max).to(self.flags.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.teacher_flags.learning_rate, eps=0.01 / self.flags.batch_size)
        self.target_q_network = QNetwork(envs, network, n_atoms=self.flags.n_atoms, v_min=self.flags.v_min, v_max=self.flags.v_max).to(self.flags.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
    def to(self, device):
        self.q_network.to(device)
        self.target_q_network.to(device)
        return self