#%% 

# To do: Make this work for RL!

import torch
from torch import nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torchinfo import summary as torch_summary

import numpy as np
import matplotlib.pyplot as plt

from utils import default_args, args, print, init_weights, Ted_Conv1d, episodes_steps, pad_zeros, select_actions_objects, multi_hot_action, var, sample, attach_list, calculate_similarity
from task import Task
from submodules import Obs_IN, Obs_OUT, Action_IN



args.alpha = .1
args.delta = 1



def batch_of_tasks(batch_size = 64):
    tasks = []
    goals = []
    objects = []
    comms = []
    recommended_actions = []
    for _ in range(batch_size):
        tasks.append(Task(actions = 5, objects = 3, shapes = 5, colors = 6, args = args))
        tasks[-1].begin()
        goals.append(tasks[-1].goal)
        object, comm = tasks[-1].give_observation()
        objects.append(object)
        comms.append(comm)
        recommended_actions.append(tasks[-1].get_recommended_action())
    objects = torch.stack(objects, dim = 0)
    comms = torch.stack(comms, dim = 0)
    recommended_actions = torch.stack(recommended_actions, dim = 0)
    return(tasks, goals, objects, comms, recommended_actions)
        
def get_rewards(tasks, action):
    rewards = []
    wins = []
    for i in range(len(tasks)):
        reward, win = tasks[i].reward_for_action(action[i])
        if(reward != 4): reward -= 1
        rewards.append(reward)
        wins.append(win)
    rewards = torch.tensor(rewards).float().unsqueeze(-1)
    return(rewards, wins)
    
    

#"""
class Forward(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        self.layers = len(args.time_scales)
                
        self.obs_in = Obs_IN(args)
        self.between = nn.Sequential(
            nn.Linear(
                in_features = 2 * args.hidden_size,
                out_features = 1 * args.hidden_size))
        self.prev_action_in = Action_IN(self.args)
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        self.to(args.device)
    
    def forward(self, prev_action, objects, comm):
        obs = self.obs_in(objects, comm)
        hidden = self.between(obs)
        prev_action = self.prev_action_in(prev_action)
        pred_objects, pred_comm = self.predict_obs(torch.cat([prev_action, hidden], dim = -1))
        return((None, None), (None, None, pred_objects, pred_comm, hidden))

""" # I think this won't work because of just one step? Something's not being used right.

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
        #detach_list([h_w_action])
        return(pred_objects, pred_comm)
    
    def forward(self, prev_action, objects, comm):
        if(len(prev_action.shape) == 2): prev_action = prev_action.unsqueeze(1)
        if(len(objects.shape)   == 2):       objects         = objects.unsqueeze(0)
        if(len(objects.shape)   == 3):       objects         = objects.unsqueeze(1)
        if(len(comm.shape)  == 2):       comm        = comm.unsqueeze(0)
        if(len(comm.shape)  == 3):       comm        = comm.unsqueeze(1)
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
        
        # Just for 1-step stuff
        hq_lists = [hq_list]
        zq_object_pred, zq_comm_pred = self.predict(prev_action[:,step+1], hq_list)
        zq_object_pred_list.append(zq_object_pred) ; zq_comm_pred_list.append(zq_comm_pred)
        
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
            (zq_mu_list, zq_std_list, pred_objects, pred_communications, hq_lists[0]))
"""
    
    
forward = Forward(args)
forward_opt = optim.Adam(forward.parameters(), lr=args.actor_lr) 
    
print(forward)
print()
print(torch_summary(forward, 
                    ((1, 1, args.action_shape),
                     (1, 1, args.objects, args.shapes + args.colors), 
                     (1, 1, args.max_comm_len, args.communication_shape))))



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
        return action, log_prob, None
    


actor = Actor(args)
actor_opt = optim.Adam(actor.parameters(), lr=args.actor_lr) 
    
print(actor)
print()
print(torch_summary(actor, 
                    ((1, args.objects, args.shapes + args.colors), 
                     (1, args.max_comm_len, args.communication_shape),
                     (1, args.action_shape),
                     (1, args.hidden_size),
                     (1, 1))))



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
        return(value, None)



critic = Critic(args)
critic_opt = optim.Adam(critic.parameters(), lr=args.actor_lr) 
    
print(critic)
print()
print(torch_summary(critic, 
                    ((1, args.objects, args.shapes + args.colors), 
                     (1, args.max_comm_len, args.communication_shape),
                     (1, args.action_shape),
                     (1, args.hidden_size),
                     (1, 1))))



prev_action = torch.zeros((64, 1, args.action_shape))



def epoch(batch_size = 64, verbose = False):
    
    # Train forward
    tasks, goals, real_objects, real_comm, recommended_actions = batch_of_tasks(batch_size)
    (_, _), (_, _, pred_objects, pred_comm, forward_hidden) = forward(prev_action, real_objects, real_comm)
    pred_objects = pred_objects.squeeze(1)
    pred_comm = pred_comm.squeeze(1)
    object_loss = F.binary_cross_entropy(pred_objects, real_objects)
    comm_loss = F.cross_entropy(pred_comm, real_comm)
    forward_loss = object_loss + comm_loss
    forward_opt.zero_grad()
    forward_loss.backward()
    forward_opt.step()
    
    # Train critic
    tasks, goals, objects, comm, recommended_actions = batch_of_tasks(batch_size)
    (_, _), (_, _, pred_objects, pred_comm, forward_hidden) = forward(prev_action, objects, comm)
    actions, log_prob, _ = actor(objects, comm, None, forward_hidden, None)
    crit_rewards, wins = get_rewards(tasks, actions)
    crit_values, _ = critic(objects, comm, actions, forward_hidden, None)
    crit_values = crit_values.squeeze(1)
    critic_loss = 0.5*F.mse_loss(crit_values, crit_rewards)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()
    
    # Train actor
    tasks, goals, objects, comm, recommended_actions = batch_of_tasks(batch_size)
    (_, _), (_, _, pred_objects, pred_comm, forward_hidden) = forward(prev_action, objects, comm)
    actions, log_prob, _ = actor(objects, comm, None, forward_hidden, None)
    log_prob = log_prob.squeeze(1)
    rewards, wins = get_rewards(tasks, actions)
    values, _ = critic(objects, comm, actions, forward_hidden, None)
    values = values.squeeze(1)
    entropy_value = args.alpha * log_prob
    recommendation_value = args.delta * calculate_similarity(recommended_actions.unsqueeze(1), actions)
    
    actor_loss = -values.mean() + entropy_value.mean() - recommendation_value.mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    
    if(verbose):
        print("\n\nREAL OBJECTS:")
        print(real_objects[:1])
        print("PRED OBJECTS:")
        print(pred_objects[:1])
        print("REAL COMM:")
        print(real_comm[:1])
        print("PRED COMM:")
        print(pred_comm[:1])
        
        print("\nREWARDS:")
        print(rewards[:1])
        print("VALUES:")
        print(values[:1])
        
        print("\nGOALS:")
        print(goals[:1])
        print("OBJECTS:")
        print(objects[:1])
        print("RECOMMENDED ACTIONS:")
        print(recommended_actions[:1])
        print("ACTIONS:")
        print(actions[:1])
        print("ENTROPY REWARD:")
        print(entropy_value[:1])
        print("RECOMMENDATION REWARD:")
        print(recommendation_value[:1])
        print("WIN?", wins[:1])
        
    return(wins, forward_loss.detach(), actor_loss.detach(), critic_loss.detach())



wins = []
forward_losses = []
actor_losses = []
critic_losses = []
for e in range(100000):
    print("EPOCH", e, end = ", ")
    win, forward_loss, actor_loss, critic_loss = epoch(verbose = e % 100 == 0)
    wins += win
    forward_losses.append(forward_loss)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)
    if(e % 100 == 0):
        columns = 4
        fig, axs = plt.subplots(nrows=1, ncols=columns, figsize=(5 * columns, 5))
        ax_num = 0

        ax = axs[ax_num]
        ax.plot(wins)
        try:
            rolling_average = np.convolve(wins, np.ones(100)/100, mode='valid') 
            ax.plot(rolling_average)
        except: pass
        ax.set_ylabel("Wins")
        ax.set_xlabel("Episodes")
        ax.set_title("Wins")
        
        ax_num += 1
        ax = axs[ax_num]
        ax.plot(forward_losses)
        ax.set_ylabel("Value")
        ax.set_xlabel("Epochs")
        ax.set_title("Forward")
        
        ax_num += 1
        ax = axs[ax_num]
        ax.plot(actor_losses)
        ax.set_ylabel("Value")
        ax.set_xlabel("Epochs")
        ax.set_title("Actor")

        ax_num += 1
        ax = axs[ax_num]
        ax.plot(critic_losses)
        ax.set_ylabel("Value")
        ax.set_xlabel("Epochs")
        ax.set_title("Critics")
        
        plt.show()
        plt.close()

# %%