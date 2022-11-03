import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 

class Simple():
    def __init__(self, network, flags):
        self.flags = flags 
        self.df = flags.distil.simple
        self.q_network = DQN_QNetwork(network, self.flags)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.df.learning_rate, eps=0.01 / self.df.batch_size)
        self.distil_method = self.flags.distil_method
        self.kl_tau = self.flags.kl_tau
        
        if self.distil_method == "mse":
            self.distil_loss = self.distil_mse
        elif self.distil_method == "kl":
            self.distil_loss = self.distil_kl
        else:
            raise ValueError("{0} is not implemented".format(self.distil_method))        
        

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
            
            loss = self.distil_loss(q_values=q_values, target_q_values=targets)
            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())


        writer.add_scalar("losses/distil", np.mean(losses), global_step)
        print("SPS:", int(global_step / (time.time() - self.flags.start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.flags.start_time)), global_step)


    def distil_mse(self, q_values, target_q_values):
        loss = F.mse_loss(target_q_values, q_values)
        return loss 
            


    def distil_kl(self, q_values, target_q_values):
        probs = torch.softmax(q_values, dim=-1) 
        target_probs = torch.softmax(target_q_values/self.kl_tau, dim=-1) 
        
        loss = torch.nn.functional.kl_div(probs, target_probs, reduction='batchmean')
        return loss 
    

class DQN_QNetwork(nn.Module):
    def __init__(self, network, flags):
        super().__init__()
        self.network = network
        self.divide_input = flags.divide_input
        
        
    def get_action(self, x, action=None):
        q_values = self(torch.Tensor(x))
        actions = torch.argmax(q_values, dim=1)
        return actions, None
    
    def forward(self, x):
        if self.divide_input:
            x = x / 255
        return self.network(x)
    