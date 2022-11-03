import torch 
import torch.nn as nn
import time 
        
class C51():
    def __init__(self, 
                 q_network, 
                 q_t_network, 
                 flags):
        
        self.flags = flags
        self.af = flags.c51
        self.q_network = C51_Qnetwork(self.flags.action_dim, q_network, n_atoms=self.af.n_atoms, v_min=self.af.v_min, v_max=self.af.v_max, flags=self.flags)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.af.learning_rate, eps=0.01 / self.af.batch_size, flags=self.flags)
        self.target_q_network = C51_Qnetwork(self.flags.action_dim, q_t_network, n_atoms=self.af.n_atoms, v_min=self.af.v_min, v_max=self.af.v_max)
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
                _, next_pmfs = self.target_q_network.get_action(data.next_observations)
                next_atoms = data.rewards + self.flags.gamma * self.target_q_network.atoms * (1 - data.dones)
                # projection
                delta_z = self.target_q_network.atoms[1] - self.target_q_network.atoms[0]
                tz = next_atoms.clamp(self.af.v_min, self.af.v_max)

                b = (tz - self.af.v_min) / delta_z
                l = b.floor().clamp(0, self.af.n_atoms - 1)
                u = b.ceil().clamp(0, self.af.n_atoms - 1)
                # (l == u).float() handles the case where bj is exactly an integer
                # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                d_m_l = (u + (l == u).float() - b) * next_pmfs
                d_m_u = (b - l) * next_pmfs
                target_pmfs = torch.zeros_like(next_pmfs)
                for i in range(target_pmfs.size(0)):
                    target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                    target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

            _, old_pmfs = self.q_network.get_action(data.observations, data.actions.flatten())
            batch_loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1))
            loss = batch_loss.mean()
            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        writer.add_scalar("losses/loss", loss.item(), global_step)
        old_val = (old_pmfs * self.q_network.atoms).sum(1)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - self.flags.start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.flags.start_time)), global_step)

    def get_action(self, x, action=None):
        return self.q_network.get_action(x, action)
        

    def get_q_values(self, x):
        logits = self.q_network.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.q_network.n, self.q_network.n_atoms), dim=2)
        q_values = (pmfs * self.q_network.atoms).sum(2)
        return q_values


class C51_Qnetwork(nn.Module):
    def __init__(self, action_dim, network, n_atoms=101, v_min=-100, v_max=100, flags=None):
        super().__init__()
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.network = network
        self.n = action_dim
        self.n_atoms = n_atoms
        self.divide_input = flags.divide_input
        
        
    def get_action(self, x, action=None):
        if self.divide_input:
            x = x / 255.0
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]
    

