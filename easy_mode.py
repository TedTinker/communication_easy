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
from mtrnn import MTRNN
from submodules import Action_IN



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



class Objects_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Objects_IN, self).__init__()  
        
        self.args = args
        
        self.objects_lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.shapes + self.args.colors, 
                out_features = self.args.hidden_size),
            nn.PReLU(),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.hidden_size),
            nn.PReLU())
        
        self.objects_lin_2 = nn.Sequential(
            nn.Linear(
                in_features = self.args.objects * self.args.hidden_size, 
                out_features = self.args.hidden_size),
            nn.PReLU())
                
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, objects):
        if(len(objects.shape) == 2):   objects  = objects.unsqueeze(1)
        if(len(objects.shape) == 3):   objects  = objects.unsqueeze(1)
        episodes, steps = episodes_steps(objects)
        objects = self.objects_lin(objects)
        objects = objects.reshape((episodes, steps, self.args.objects * self.args.hidden_size))
        objects = self.objects_lin_2(objects)
        return(objects)
    
    
    
class Comm_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Comm_IN, self).__init__()  
        
        self.args = args
        
        self.comm_lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.max_comm_len * self.args.communication_shape, 
                out_features = self.args.hidden_size),
            nn.PReLU(),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.hidden_size),
            nn.PReLU())
                
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, comm):
        if(len(comm.shape) == 2):  comm = comm.unsqueeze(0)
        if(len(comm.shape) == 3):  comm = comm.unsqueeze(1)
        episodes, steps = episodes_steps(comm)
        comm = pad_zeros(comm, self.args.max_comm_len)
        comm = comm.reshape((episodes, steps, self.args.max_comm_len * self.args.communication_shape))
        comm = self.comm_lin(comm)
        return(comm)
    
    
    
class Obs_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Obs_IN, self).__init__()  
                
        self.args = args
        self.objects_in = Objects_IN(self.args)
        self.comm_in = Comm_IN(self.args)
        
    def forward(self, objects, comm):
        objects = self.objects_in(objects)
        comm = self.comm_in(comm)
        return(torch.cat([objects, comm], dim = -1))
    
    
    
class Objects_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Objects_OUT, self).__init__()  
                
        self.args = args
        
        self.objects_out = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(2 * self.args.hidden_size, self.args.hidden_size), 
            nn.PReLU(),
            nn.Linear(self.args.hidden_size, self.args.hidden_size), 
            nn.PReLU(),
            nn.Linear(self.args.hidden_size, self.args.observation_shape),
            nn.Sigmoid())
        
        self.apply(init_weights)
        self.to(self.args.device)
                
    def forward(self, h_w_action):
        if(len(h_w_action.shape) == 2):   h_w_action = h_w_action.unsqueeze(1)
        episodes, steps = episodes_steps(h_w_action)
        [h_w_action] = attach_list([h_w_action], self.args.device)
        objects_pred = self.objects_out(h_w_action)
        objects_pred = objects_pred.reshape((episodes, steps, self.args.objects, self.args.shapes + self.args.colors))
        return(objects_pred)
    
    
    
class Comm_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Comm_OUT, self).__init__()  
                
        self.args = args
        
        self.comm_out = nn.Sequential(
            nn.Linear(
                in_features = 2 * self.args.hidden_size, 
                out_features = self.args.max_comm_len * self.args.communication_shape))
        
        self.apply(init_weights)
        self.to(self.args.device)
                
    def forward(self, h_w_action):
        if(len(h_w_action.shape) == 2):   h_w_action = h_w_action.unsqueeze(1)
        [h_w_action] = attach_list([h_w_action], self.args.device)
        episodes, steps = episodes_steps(h_w_action)
        comm_pred = self.comm_out(h_w_action)
        comm_pred = comm_pred.reshape(episodes, steps, self.args.max_comm_len, self.args.communication_shape)
        return(comm_pred)
    
    
    
class Obs_OUT(nn.Module):
    
    def __init__(self, args = default_args):
        super(Obs_OUT, self).__init__()  
        
        self.args = args 
        self.objects_out = Objects_OUT(self.args)
        self.comm_out = Comm_OUT(self.args)
        
    def forward(self, h_w_action):
        objects_pred = self.objects_out(h_w_action.clone())
        comm_pred = self.comm_out(h_w_action.clone())
        return(objects_pred, comm_pred)



class Forward(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        self.layers = len(args.time_scales)
                
        self.obs_in = Obs_IN(args)
        self.between = nn.Sequential(
            nn.Linear(
                in_features = 2 * args.hidden_size,
                out_features = 2 * args.hidden_size))
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        self.to(args.device)
    
    def forward(self, prev_action, objects, comm, forward_hidden):
        obs = self.obs_in(objects, comm)
        hidden = self.between(obs)
        pred_objects, pred_comm = self.predict_obs(hidden)
        return(pred_objects, pred_comm, hidden)
    
    
    
forward = Forward(args)
forward_opt = optim.Adam(forward.parameters(), lr=args.actor_lr) 
    
print(forward)
print()
print(torch_summary(forward, 
                    ((1, args.action_shape),
                     (1, args.objects, args.shapes + args.colors), 
                     (1, args.max_comm_len, args.communication_shape),
                     (1, 1))))



class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(args)

        self.lin = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size),
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
                     (1, 2 * args.hidden_size),
                     (1, 1))))



class Critic(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(self.args)
        
        self.action_in = Action_IN(self.args)
        
        self.value = nn.Sequential(
            nn.Linear(
                in_features = 5 * self.args.hidden_size,
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
                     (1, 2 * args.hidden_size),
                     (1, 1))))



def epoch(batch_size = 64, verbose = False):
    
    # Train forward
    tasks, goals, real_objects, real_comm, recommended_actions = batch_of_tasks(batch_size)
    pred_objects, pred_comm, forward_hidden = forward(None, real_objects, real_comm, None)
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
    _, _, forward_hidden = forward(None, objects, comm, None)
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
    _, _, forward_hidden = forward(None, objects, comm, None)
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