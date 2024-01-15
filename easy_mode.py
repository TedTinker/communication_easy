#%% 

# To do: Make this work for RL's PVRNN!
# Improve comm in and out.

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
from mtrnn import MTRNN
from submodules import Obs_IN, Obs_OUT, Action_IN
from pvrnn import PVRNN
from models import Actor, Critic



args.alpha = .1
args.delta = 1

actor = Actor(args)
actor_opt = optim.Adam(actor.parameters(), lr=args.actor_lr) 

critic = Critic(args)
critic_opt = optim.Adam(critic.parameters(), lr=args.actor_lr) 



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
    objects = torch.stack(objects, dim = 0).unsqueeze(1)
    comms = torch.stack(comms, dim = 0).unsqueeze(1)
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
    
    

""" This one works, but not the real one?
class PVRNN(nn.Module): 
    
    def __init__(self, args = default_args):
        super(PVRNN, self).__init__()
        
        self.args = args
        self.layers = len(args.time_scales)
                
        self.obs_in = Obs_IN(args)
        self.between = nn.Sequential(
            nn.Linear(
                in_features = 2 * args.hidden_size,
                out_features = 1 * args.hidden_size))
        self.action_in = Action_IN(self.args)
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        self.to(args.device)
    
    def forward(self, prev_hidden_states, objects, comms, prev_actions):
        obs = self.obs_in(objects, comms)
        hidden = self.between(obs)
        prev_actions = self.action_in(prev_actions)
        pred_objects, pred_comm = self.predict_obs(torch.cat([prev_actions, hidden], dim = -1))
        return((None, None), (None, None), (pred_objects, pred_comm), hidden.unsqueeze(2))
    
"""

class PVRNN(nn.Module): 
    
    def __init__(self, args = default_args):
        super(PVRNN, self).__init__()
        
        self.args = args
        self.layers = len(args.time_scales)
                
        self.obs_in = Obs_IN(args)
        
        # Prior: Previous hidden state, plus action if bottom.  
        self.zp_mu = nn.Sequential(
                nn.Linear(
                    in_features = self.args.hidden_size, 
                    out_features = self.args.state_size), 
                nn.Tanh())
        self.zp_std = nn.Sequential(
                nn.Linear(
                    in_features = self.args.hidden_size, 
                    out_features = self.args.state_size), 
                nn.Softplus())
                            
        # Posterior: Previous hidden state, plus observation and action if bottom, plus lower-layer hidden state otherwise.
        self.zq_mu = nn.Sequential(
                nn.Linear(
                    in_features = 3 * self.args.hidden_size, 
                    out_features = self.args.state_size), 
                nn.Tanh())
        self.zq_std = nn.Sequential(
                nn.Linear(
                    in_features = 3 * self.args.hidden_size, 
                    out_features = self.args.state_size), 
                nn.Softplus())
        
        self.between = nn.Sequential(
            nn.Linear(
                in_features = args.state_size,
                out_features = 1 * args.hidden_size))
        self.action_in = Action_IN(self.args)
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        self.to(args.device)
    
    def forward(self, prev_hidden_states, objects, comms, prev_actions):
        obs = self.obs_in(objects, comms)
        zp_mu, zp_std = var(prev_hidden_states, self.zp_mu, self.zp_std, self.args)
        zp = sample(zp_mu, zp_std, self.args.device)
        zq_mu, zq_std = var(torch.cat([obs, prev_hidden_states], dim = -1), self.zq_mu, self.zq_std, self.args)
        zq = sample(zq_mu, zq_std, self.args.device)
        hidden = self.between(zq)
        prev_actions = self.action_in(prev_actions)
        pred_objects, pred_comm = self.predict_obs(torch.cat([prev_actions, hidden], dim = -1))
        return((None, None), (None, None), (pred_objects, pred_comm), hidden.unsqueeze(2))
    
#"""
    
forward = PVRNN(args)
forward_opt = optim.Adam(forward.parameters(), lr=args.actor_lr) 
    
print(forward)
print()
print(torch_summary(forward, 
                    ((1, args.layers, args.hidden_size),
                     (1, 1, args.objects, args.shapes + args.colors), 
                     (1, 1, args.max_comm_len, args.comm_shape),
                     (1, 1, args.action_shape))))



prev_action = torch.zeros((64, 1, args.action_shape))
prev_hidden_states = torch.zeros((64, args.layers, args.hidden_size))



def epoch(batch_size = 64, verbose = False):
    
    # Train forward
    tasks, goals, real_objects, real_comm, recommended_actions = batch_of_tasks(batch_size)
    (_, _), (_, _), (pred_objects, pred_comm), _ = forward(prev_hidden_states, real_objects, real_comm, prev_action)
    real_comm = real_comm.reshape((real_comm.shape[0] * real_comm.shape[1], real_comm.shape[2], real_comm.shape[3]))
    pred_comm = pred_comm.reshape((pred_comm.shape[0] * pred_comm.shape[1], pred_comm.shape[2], pred_comm.shape[3]))
    object_loss = F.binary_cross_entropy(pred_objects, real_objects)
    comm_loss = F.cross_entropy(pred_comm, real_comm)
    forward_loss = object_loss + comm_loss
    forward_opt.zero_grad()
    forward_loss.backward()
    forward_opt.step()
    
    # Train critic
    tasks, goals, objects, comm, recommended_actions = batch_of_tasks(batch_size)
    (_, _), (_, _), (_, _), forward_hidden = forward(prev_hidden_states, objects, comm, prev_action)
    actions, log_prob, _ = actor(objects, comm, None, forward_hidden.squeeze(2), None)
    crit_rewards, wins = get_rewards(tasks, actions)
    crit_values, _ = critic(objects, comm, actions, forward_hidden.squeeze(2), None)
    crit_values = crit_values.squeeze(1)
    critic_loss = 0.5*F.mse_loss(crit_values, crit_rewards)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()
    
    # Train actor
    tasks, goals, objects, comm, recommended_actions = batch_of_tasks(batch_size)
    (_, _), (_, _), (_, _), forward_hidden = forward(prev_hidden_states, objects, comm, prev_action)
    actions, log_prob, _ = actor(objects, comm, None, forward_hidden.squeeze(2), None)
    log_prob = log_prob.squeeze(1)
    rewards, wins = get_rewards(tasks, actions)
    values, _ = critic(objects, comm, actions, forward_hidden.squeeze(2), None)
    values = values.squeeze(1)
    entropy_value = args.alpha * log_prob
    recommendation_value = args.delta * calculate_similarity(recommended_actions.unsqueeze(1), actions)
    
    actor_loss = -values.mean() + entropy_value.mean() - recommendation_value.mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    
    if(verbose):
        print("\n\nREAL OBJECTS:")
        print(real_objects[0])
        print("PRED OBJECTS:")
        print(pred_objects[0])
        print("REAL COMM:")
        print(real_comm[0])
        print("PRED COMM:")
        print(pred_comm[0])
        
        print("\nREWARD:")
        print(rewards[0])
        print("VALUE:")
        print(values[0])
        
        print("\nTASK:")
        print(tasks[0])
        print("ACTION:")
        print(actions[0])
        print("ENTROPY REWARD:")
        print(entropy_value[0])
        print("RECOMMENDATION REWARD:")
        print(recommendation_value[0])
        print("WIN?", wins[0])
        
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