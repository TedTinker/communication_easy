#%% 

import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torchinfo import summary as torch_summary

from utils import default_args, detach_list, attach_list, print, init_weights, episodes_steps, var, sample
from mtrnn import MTRNN
from submodules import Obs_IN, Action_IN



if __name__ == "__main__":
    
    args = default_args
    episodes = 4 ; steps = 3
        
    

class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(args)

        self.lin = nn.Sequential(
            nn.Linear(3 * args.hidden_size, args.hidden_size),
            nn.PReLU())
        self.mu = nn.Sequential(
            nn.Linear(args.hidden_size, self.args.actions + self.args.objects))
        self.std = nn.Sequential(
            nn.Linear(args.hidden_size, self.args.actions + self.args.objects),
            nn.Softplus())

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, objects, comm, pred_action, forward_hidden, action_hidden):
        if(len(forward_hidden.shape) == 2): forward_hidden = forward_hidden.unsqueeze(1)
        obs = self.obs_in(objects, comm)
        x = self.lin(torch.cat([obs, forward_hidden], dim = -1))
        mu, std = var(x, self.mu, self.std, self.args)
        x = sample(mu, std, self.args.device)
        action = torch.tanh(x)
        log_prob = Normal(mu, std).log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return action, log_prob, action_hidden
    
    
    
if __name__ == "__main__":
    
    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor,
                        ((episodes, steps, args.objects, args.shapes + args.colors),
                         (episodes, steps, args.hidden_size),
                         (episodes, steps, args.hidden_size),
                         (episodes, steps, args.hidden_size),
                         (episodes, steps, args.hidden_size))))
    
    
    
class Critic(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(self.args)
        
        self.action_in = Action_IN(self.args)
        
        self.value = nn.Sequential(
            nn.Linear(
                in_features = 4 * self.args.hidden_size,
                out_features = self.args.hidden_size),
            nn.PReLU(),
            nn.Linear(                
                in_features = self.args.hidden_size,
                out_features = 1))
        
    def forward(self, objects, comm, action, forward_hidden, critic_hidden):
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        if(len(forward_hidden.shape) == 2): forward_hidden = forward_hidden.unsqueeze(1)
        obs = self.obs_in(objects, comm)
        action = self.action_in(action)
        value = self.value(torch.cat([obs, action, forward_hidden], dim=-1))
        return(value, critic_hidden)
    


if __name__ == "__main__":
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, 
                        ((episodes, steps, args.objects, args.object_shape), 
                         (episodes, steps, args.max_comm_len, args.comm_shape), 
                         (episodes, steps, args.action_shape),
                         (episodes, steps, args.hidden_size),
                         (episodes, steps, args.hidden_size))))

# %%
