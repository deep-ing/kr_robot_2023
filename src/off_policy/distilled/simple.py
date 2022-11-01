import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 

class Simple():
    def __init__(self, network, flags):
        self.flags = flags 
        self.df = flags.distil.simple
        self.q_network = DQN_QNetwork(network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.df.learning_rate, eps=0.01 / self.df.batch_size)

    def to(self, device):
        self.q_network.to(device)
        return self
    
    def get_q_values(self, x):
        q_values = self.q_network(torch.Tensor(x))
        return q_values

    def get_action(self, x):
        q_values = self.q_network(torch.Tensor(x))
        actions = torch.argmax(q_values, dim=1)
        return actions, None
    

    def train_distil(self, target_model, rb, global_step, writer):
        losses = [] 
        for epoch in range(self.df.distil_epochs):
            data = rb.sample(self.df.batch_size)
            with torch.no_grad():
                targets = target_model.get_q_values(data.observations)
            q_values = self.get_q_values(data.observations)
            loss = F.mse_loss(targets, q_values)
            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())


        writer.add_scalar("losses/distil", np.mean(losses), global_step)
        print("SPS:", int(global_step / (time.time() - self.flags.start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.flags.start_time)), global_step)


            
    
     



class DQN_QNetwork(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        
    def get_action(self, x, action=None):
        q_values = self(torch.Tensor(x))
        actions = torch.argmax(q_values, dim=1)
        return actions, None
    
    def forward(self, x):
        x = x / 255.0
        return self.network(x)
    