import torch 
import torch.nn as nn 

class QNetwork(nn.Module):
    def __init__(self, env, network, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.network = network  # 
        self.n = env.single_action_space.n
        self.n_atoms = n_atoms
        
    def get_action(self, x, action=None):
        # x = self.permute(x)
        logits = self.network(x / 255.0)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]

    # def permute(self, x):
    #     x = x.permute(0,3,1,2)
    #     return x


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)



from .teacher import Teacher
from .student import Student