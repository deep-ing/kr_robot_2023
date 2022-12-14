import torch 
import torch.nn as nn
import time 
import torch.nn.functional as F 
        
class DQN():
    def __init__(self, 
                 q_network, 
                 q_t_network, 
                 flags):
        
        self.flags = flags
        self.af = flags.dqn
        self.q_network = DQN_QNetwork(q_network, flags)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.af.learning_rate, eps=0.01 / self.af.batch_size)
        self.target_q_network = DQN_QNetwork(q_t_network, flags)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_network_frequency = self.af.target_network_frequency
        
    def to(self, device):
        self.q_network.to(device)
        self.target_q_network.to(device)
        return self
    
    def train(self, rb, rb_distil, writer, global_step):
        for epoch in range(self.flags.teacher_epochs):
            data = rb.sample(self.af.batch_size)
            with torch.no_grad():
                target_max, _ = self.target_q_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + self.flags.gamma * target_max * (1 - data.dones.flatten())
            old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()

            loss = F.mse_loss(td_target, old_val)

            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            batch_loss = (td_target - old_val)**2
            upper_bound = torch.quantile(batch_loss.detach(), (1-self.flags.acceptance_ratio)) 
            indices = (batch_loss > upper_bound).nonzero()
            for idx in indices:
                # obs, next_obs, actions, rewards, dones, infos
                idx = idx.item()
                rb_distil.add(data.observations[idx].detach().cpu().numpy(),
                              data.next_observations[idx].detach().cpu().numpy(),
                              data.actions[idx].detach().cpu().numpy(),
                              data.rewards[idx].detach().cpu().numpy(),
                              data.dones[idx].detach().cpu().numpy(),
                              [{}]
                )

        writer.add_scalar("losses/loss", loss, global_step)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - self.flags.start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.flags.start_time)), global_step)


    def get_action(self, x, action=None):
        return self.q_network.get_action(x, action)


    def get_q_values(self, x):
        q_values = self.q_network(torch.Tensor(x))
        return q_values



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
            x = x / 255.0
        return self.network(x)
    

