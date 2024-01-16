#%% 

# To do: 
# Improve comm in.

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary as torch_summary

import numpy as np
import matplotlib.pyplot as plt

from utils import args, print, \
    calculate_similarity, onehots_to_string
from task import Task, Task_Runner
from pvrnn import PVRNN
from models import Actor, Critic



episodes = 64 
steps = 3 

args.alpha = .1
args.delta = 1


actor = Actor(args)
actor_opt = optim.Adam(actor.parameters(), lr=args.actor_lr) 

critic = Critic(args)
critic_opt = optim.Adam(critic.parameters(), lr=args.actor_lr) 

forward = PVRNN(args)
forward_opt = optim.Adam(forward.parameters(), lr=args.actor_lr) 



def batch_of_tasks():
    tasks = []
    goals = []
    objects = []
    comms = []
    recommended_actions = []
    for _ in range(episodes):
        task = Task(actions = 5, objects = 3, shapes = 5, colors = 6, args = args)
        tasks.append(Task_Runner(task, args = args))
        tasks[-1].begin()
        goals.append(tasks[-1].task.goal)
        object, comm = tasks[-1].obs()
        objects.append(object)
        comms.append(comm)
        recommended_actions.append(tasks[-1].task.get_recommended_action())
    objects = torch.stack(objects, dim = 0).unsqueeze(1)
    comms = torch.stack(comms, dim = 0).unsqueeze(1)
    recommended_actions = torch.stack(recommended_actions, dim = 0)
    return(tasks, goals, objects, comms, recommended_actions)
        
def get_rewards(tasks, action):
    rewards = []
    dones = []
    wins = []
    for i in range(len(tasks)):
        reward, done, win = tasks[i].action(action[i])
        if(reward != 4): reward -= 1
        rewards.append(reward)
        dones.append(done)
        wins.append(win)
    rewards = torch.tensor(rewards).float().unsqueeze(1).unsqueeze(1)
    return(rewards, dones, wins)

def step_in_episode(tasks, prev_action, hq_m1, ha_m1, objects, comms):
    (_, _), (_, _), hq = forward.bottom_to_top_step(hq_m1, objects, comms, prev_action) 
    action, _, ha = actor(objects, comms, prev_action, hq[:,:,0].clone().detach(), ha_m1) 
    rewards, dones, wins = get_rewards(tasks, action)
    return(action, hq.squeeze(1), ha, rewards, dones, wins)

def episode(tasks, objects, comms):
    hq = torch.zeros((episodes, args.layers, args.hidden_size))
    ha = None
    actions = []
    rewards = []
    action = prev_action
    for step in range(steps):
        action, hq, ha, reward, dones, wins = \
            step_in_episode(tasks, action, hq, ha, objects, comms)
        actions.append(action)
        rewards.append(reward)
    actions = torch.cat(actions, dim = 1)
    actions = torch.cat([prev_action, actions], dim = 1)
    rewards = torch.cat(rewards, dim = 1)
    return(actions, hq, rewards, dones, wins)



prev_action = torch.zeros((64, 1, args.action_shape))
prev_hidden_states = torch.zeros((64, args.layers, args.hidden_size))



def epoch(verbose = False):
    
    tasks, goals, objects, comms, recommended_actions = batch_of_tasks()
    actions, hq, rewards, dones, wins = episode(tasks, objects, comms)
    objects = objects.tile((1, steps+1, 1, 1))
    comms = comms.tile((1, steps+1, 1, 1))
    (_, _), (_, _), (pred_objects, pred_comm), _ = forward(prev_hidden_states, objects, comms, actions)
    
    # Train forward
    tasks, goals, real_objects, real_comm, recommended_actions = batch_of_tasks()
    (_, _), (_, _), (pred_objects, pred_comm), _ = forward(prev_hidden_states, real_objects, real_comm, prev_action)
    
    target_comm = real_comm.reshape((real_comm.shape[0] * real_comm.shape[1] * real_comm.shape[2], real_comm.shape[3]))
    target_comm = torch.argmax(target_comm, dim = -1)
    predicted_comm = pred_comm.reshape((pred_comm.shape[0] * pred_comm.shape[1] * pred_comm.shape[2], pred_comm.shape[3]))
    real_comm = real_comm.reshape((real_comm.shape[0] * real_comm.shape[1], args.max_comm_len, args.comm_shape))
    pred_comm = pred_comm.reshape((pred_comm.shape[0] * pred_comm.shape[1], args.max_comm_len, args.comm_shape))
    
    object_loss = F.binary_cross_entropy(pred_objects, real_objects)
    comm_loss = .1 * F.cross_entropy(
        predicted_comm, target_comm)
    forward_loss = object_loss + comm_loss
    forward_opt.zero_grad()
    forward_loss.backward()
    forward_opt.step()
    
    # Train critic
    tasks, goals, objects, comm, recommended_actions = batch_of_tasks()
    (_, _), (_, _), (_, _), forward_hidden = forward(prev_hidden_states, objects, comm, prev_action)
    #print(objects.shape, comm.shape, forward_hidden.shape)
    actions, log_prob, _ = actor(objects, comm, None, forward_hidden[:,:,0], None)
    crit_rewards, dones, wins = get_rewards(tasks, actions)
    crit_values, _ = critic(objects, comm, actions, forward_hidden[:,:,0], None)
    crit_values = crit_values.squeeze(1)
    critic_loss = 0.5*F.mse_loss(crit_values, crit_rewards)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()
    
    # Train actor
    tasks, goals, objects, comm, recommended_actions = batch_of_tasks()
    (_, _), (_, _), (_, _), forward_hidden = forward(prev_hidden_states, objects, comm, prev_action)
    actions, log_prob, _ = actor(objects, comm, None, forward_hidden[:,:,0], None)
    log_prob = log_prob.squeeze(1)
    rewards, dones, wins = get_rewards(tasks, actions)
    values, _ = critic(objects, comm, actions, forward_hidden[:,:,0], None)
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
        print(onehots_to_string(real_comm[0]))
        print("PRED COMM:")
        print(onehots_to_string(pred_comm[0]))
        
        print("\nREWARD:")
        print(rewards[0])
        print("VALUE:")
        print(values[0])
        
        print("\nTASK:")
        print(tasks[0].task)
        print("ACTION:")
        print(actions[0])
        print("ENTROPY REWARD:")
        print(entropy_value[0])
        print("RECOMMENDATION REWARD:")
        print(recommendation_value[0])
        print("WIN?", wins[0])
        
    return(wins, forward_loss.detach(), actor_loss.detach(), entropy_value.mean().item(), -recommendation_value.mean().item(), critic_loss.detach())



wins = []
forward_losses = []
actor_losses = []
entropy_values = []
recommendation_values = []
critic_losses = []
for e in range(100000):
    print("EPOCH", e, end = ", ")
    win, forward_loss, actor_loss, entropy_value, recommendation_value, critic_loss = epoch(verbose = e % 100 == 0)
    wins += win
    forward_losses.append(forward_loss)
    actor_losses.append(actor_loss)
    entropy_values.append(entropy_value)
    recommendation_values.append(recommendation_value)
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
        ax.plot(entropy_values)
        ax.plot(recommendation_values)
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