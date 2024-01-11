#%% 

import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torchinfo import summary as torch_summary

from utils import default_args, detach_list, attach_list, print, init_weights, episodes_steps, var, sample
from mtrnn import MTRNN
from submodules import Obs_IN, Obs_OUT, Action_IN
        
        

class Forward(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        self.layers = len(args.time_scales)
                
        self.obs_in = Obs_IN(args)
        self.action_in = Action_IN(args)
        
        zp_mu_layers  = []
        zp_std_layers = []
        zq_mu_layers  = []
        zq_std_layers = []
        mtrnn_layers  = []
        
        for layer in range(self.layers): 
        
            zp_mu_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size if layer == 0 else 0), args.hidden_size), 
                nn.PReLU(),
                nn.Dropout(.2),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Tanh()))
            zp_std_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size if layer == 0 else 0), args.hidden_size), 
                nn.PReLU(),
                nn.Dropout(.2),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Softplus()))
            
            zq_mu_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size * 3 if layer == 0 else args.hidden_size), args.hidden_size), 
                nn.PReLU(),
                nn.Dropout(.2),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Tanh()))
            zq_std_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size * 3 if layer == 0 else args.hidden_size), args.hidden_size), 
                nn.PReLU(),
                nn.Dropout(.2),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Softplus()))
            
            mtrnn_layers.append(MTRNN(
                input_size = args.state_size + (args.hidden_size if layer + 1 < self.layers else 0),
                hidden_size = args.hidden_size, 
                time_constant = args.time_scales[layer],
                args = args))
            
        self.zp_mu_layers  = nn.ModuleList(zp_mu_layers)
        self.zp_std_layers = nn.ModuleList(zp_std_layers)
        self.zq_mu_layers  = nn.ModuleList(zq_mu_layers)
        self.zq_std_layers = nn.ModuleList(zq_std_layers)
        self.mtrnn_layers  = nn.ModuleList(mtrnn_layers)
        
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        self.to(args.device)
        
    def p(self, prev_action, hq_m1_list = None, episodes = 1):
        if(hq_m1_list == None): hq_m1_list  = [torch.zeros(episodes, 1, self.args.hidden_size)] * self.layers
        [prev_action, hq_m1_list] = attach_list([prev_action, hq_m1_list], self.args.device)
        prev_action = self.action_in(prev_action)
        zp_mu_list = [] ; zp_std_list = [] ; zp_list = [] ; hp_list = []
        for layer in range(self.layers):
            hq_m1 = hq_m1_list[layer]
            if(len(hq_m1.shape) == 2): hq_m1 = hq_m1.unsqueeze(1)
            z_input = hq_m1 if layer != 0 else torch.cat([hq_m1, prev_action], dim = -1) 
            zp_mu, zp_std = var(z_input, self.zp_mu_layers[layer], self.zp_std_layers[layer], self.args)
            zp_mu_list.append(zp_mu) ; zp_std_list.append(zp_std) ; zp_list.append(sample(zp_mu, zp_std, self.args.device))
            h_input = zp_list[layer] if layer+1 == self.layers else torch.cat([zp_list[layer], hq_m1_list[layer+1]], dim = -1) 
            hp = self.mtrnn_layers[layer](h_input, hq_m1_list[layer]) 
            hp_list.append(hp)
        return(zp_mu_list, zp_std_list, hp_list)
    
    def q(self, prev_action, objects, comm, hq_m1_list = None):
        if(len(prev_action.shape) == 2): prev_action = prev_action.unsqueeze(1)
        if(len(objects.shape)   == 2):       objects         = objects.unsqueeze(0)
        if(len(objects.shape)   == 3):       objects         = objects.unsqueeze(1)
        if(len(comm.shape)  == 2):       comm        = comm.unsqueeze(0)
        if(len(comm.shape)  == 3):       comm        = comm.unsqueeze(1)
        episodes, steps = episodes_steps(objects)
        if(hq_m1_list == None):     hq_m1_list = [torch.zeros(episodes, steps, self.args.hidden_size)] * self.layers
        [prev_action, objects, comm, hq_m1_list] = attach_list([prev_action, objects, comm, hq_m1_list], self.args.device)
        obs = self.obs_in(objects, comm)
        prev_action = self.action_in(prev_action)
        zq_mu_list = [] ; zq_std_list = [] ; zq_list = [] ; hq_list = []
        for layer in range(self.layers):
            z_input = torch.cat((hq_m1_list[layer], obs, prev_action), dim=-1) if layer == 0 else torch.cat((hq_m1_list[layer], hq_list[layer-1]), dim=-1)
            zq_mu, zq_std = var(z_input, self.zq_mu_layers[layer], self.zq_std_layers[layer], self.args)        
            zq_mu_list.append(zq_mu) ; zq_std_list.append(zq_std) ; zq_list.append(sample(zq_mu, zq_std, self.args.device))
            h_input = zq_list[layer] if layer+1 == self.layers else torch.cat([zq_list[layer], hq_m1_list[layer+1]], dim = -1)
            hq = self.mtrnn_layers[layer](h_input, hq_m1_list[layer])
            hq_list.append(hq)
        return(zq_mu_list, zq_std_list, hq_list)
        
    def predict(self, action, h): 
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        if(len(h[0].shape) == 2):   h[0]   = h[0].unsqueeze(1)
        [action, h] = attach_list([action, h], self.args.device)
        h_w_action = torch.cat([self.action_in(action), h[0]], dim = -1)
        pred_objects, pred_comm = self.predict_obs(h_w_action)
        detach_list([h_w_action])
        return(pred_objects, pred_comm)
    
    def forward(self, prev_action, objects, comm):
        [prev_action, objects, comm] = attach_list([prev_action, objects, comm], self.args.device)
        episodes, steps = episodes_steps(objects)
        zp_mu_lists = [] ; zp_std_lists = [] ;                                                    
        zq_mu_lists = [] ; zq_std_lists = [] ; zq_object_pred_list = [] ; zq_comm_pred_list = [] ; hq_lists = [[torch.zeros(episodes, 1, self.args.hidden_size).to(self.args.device)] * self.layers]
        step = -1
        for step in range(steps-1):
            zp_mu_list, zp_std_list, hp_list = self.p(prev_action[:,step],                              hq_lists[-1], episodes = episodes)
            zq_mu_list, zq_std_list, hq_list = self.q(prev_action[:,step], objects[:,step], comm[:,step], hq_lists[-1])
            zq_object_pred, zq_comm_pred = self.predict(prev_action[:,step+1], hq_list)
            zp_mu_lists.append(zp_mu_list) ; zp_std_lists.append(zp_std_list) 
            zq_mu_lists.append(zq_mu_list) ; zq_std_lists.append(zq_std_list) ; hq_lists.append(hq_list)
            zq_object_pred_list.append(zq_object_pred) ; zq_comm_pred_list.append(zq_comm_pred)
        zp_mu_list, zp_std_list, hp_list = self.p(prev_action[:,step+1],                                hq_lists[-1], episodes = episodes)
        zq_mu_list, zq_std_list, hq_list = self.q(prev_action[:,step+1], objects[:,step+1], comm[:,step+1], hq_lists[-1])
        zp_mu_lists.append(zp_mu_list) ; zp_std_lists.append(zp_std_list) 
        zq_mu_lists.append(zq_mu_list) ; zq_std_lists.append(zq_std_list)
        hq_lists.append(hq_lists.pop(0))    
        hq_lists = [torch.cat([hq_list[layer] for hq_list in hq_lists], dim = 1) for layer in range(self.args.layers)]
        zp_mu_list  = [torch.cat([zp_mu[layer]  for zp_mu  in zp_mu_lists],  dim = 1) for layer in range(self.args.layers)]
        zp_std_list = [torch.cat([zp_std[layer] for zp_std in zp_std_lists], dim = 1) for layer in range(self.args.layers)]
        zq_mu_list  = [torch.cat([zq_mu[layer]  for zq_mu  in zq_mu_lists],  dim = 1) for layer in range(self.args.layers)]
        zq_std_list = [torch.cat([zq_std[layer] for zq_std in zq_std_lists], dim = 1) for layer in range(self.args.layers)]
        pred_objects = torch.cat(zq_object_pred_list,  dim = 1)
        pred_communications  = torch.cat(zq_comm_pred_list, dim = 1)
        #pred_communications *= create_comm_mask(pred_communications)
        return(
            (zp_mu_list, zp_std_list), 
            (zq_mu_list, zq_std_list, pred_objects, pred_communications, hq_lists))
        
    

class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(args)

        self.lin = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
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
        #x = self.obs_in(objects, comm)
        x = self.lin(forward_hidden)
        mu, std = var(x, self.mu, self.std, self.args)
        x = sample(mu, std, self.args.device)
        action = torch.tanh(x)
        log_prob = Normal(mu, std).log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return action, log_prob, action_hidden
    
    
    
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
    
    args = default_args
    episodes = 4 ; steps = 3
    
    
    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, 
                        ((episodes, steps+1, args.action_shape), 
                         (episodes, steps+1, args.observation_shape), 
                         (episodes, steps+1, 6, args.communication_shape))))
    
    
    
    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor,
                        ((episodes, steps, args.hidden_size))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, 
                        ((episodes, steps, args.observation_shape), 
                         (episodes, steps, 6, args.communication_shape), 
                         (episodes, steps, args.action_shape))))

# %%
