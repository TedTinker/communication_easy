#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import numpy as np
from math import log
from itertools import accumulate
from copy import deepcopy
import matplotlib.pyplot as plt

from utils import default_args, dkl, print, choose_task, calculate_similarity, onehots_to_string, multihots_to_string
from task import Task, Task_Runner
from buffer import RecurrentReplayBuffer
from pvrnn import PVRNN
from models import Actor, Critic



class Agent:
    
    def __init__(self, i, args = default_args):
        
        self.agent_num = i
        self.args = args
        self.episodes = 0 ; self.epochs = 0 ; self.steps = 0
        self.task_probabilities = self.args.task_probabilities[0]
        
        self.tasks = {
            "1" : Task(actions = 5, objects = 3, shapes = 5, colors = 6, args = self.args)}
        
        self.task_runners = {task_name : Task_Runner(task) for task_name, task in self.tasks.items()}
        self.task = choose_task(self.task_probabilities)
        
        self.target_entropy = self.args.target_entropy
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=self.args.alpha_lr) 

        self.forward = PVRNN(self.args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=self.args.actor_lr)
                           
        self.actor = Actor(self.args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr) 
        
        self.critics = []
        self.critic_targets = []
        self.critic_opts = []
        for _ in range(self.args.critics):
            self.critics.append(Critic(self.args))
            self.critic_targets.append(Critic(self.args))
            self.critic_targets[-1].load_state_dict(self.critics[-1].state_dict())
            self.critic_opts.append(optim.Adam(self.critics[-1].parameters(), lr=self.args.actor_lr))
        
        self.memory = RecurrentReplayBuffer(self.args)
        self.train()
        
        self.plot_dict = {
            "args" : self.args,
            "arg_title" : args.arg_title,
            "arg_name" : args.arg_name,
            "episode_lists" : {}, 
            "agent_lists" : {"forward" : PVRNN, "actor" : Actor, "critic" : Critic},
            "wins" : [], 
            "rewards" : [], 
            "steps" : [],
            "accuracy" : [], 
            "object_loss" : [], 
            "comm_loss" : [], 
            "complexity" : [],
            "alpha" : [], 
            "actor" : [], 
            "critics" : [[] for _ in range(self.args.critics)], 
            "extrinsic" : [], 
            "q" : [], 
            "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : [], 
            "intrinsic_imitation" : [],
            "prediction_error" : [], 
            "hidden_state" : [[] for _ in range(self.args.layers)]}
        
        
        
    def training(self, q = None):        
        self.save_episodes()
        self.save_agent()
        while(True):
            cumulative_epochs = 0
            prev_task_probs = self.task_probabilities
            for i, epochs in enumerate(self.args.epochs): 
                cumulative_epochs += epochs
                if(self.epochs < cumulative_epochs): 
                    self.task_probabilities = self.args.task_probabilities[i] 
                    break
            if(prev_task_probs != self.task_probabilities): 
                self.save_episodes()
            self.training_episode()
            percent_done = str(self.epochs / sum(self.args.epochs))
            if(q != None):
                q.put((self.agent_num, percent_done))
            if(self.epochs >= sum(self.args.epochs)): break
            if(self.epochs % self.args.epochs_per_episode_list == 0): self.save_episodes()
            if(self.epochs % self.args.epochs_per_agent_list == 0): self.save_agent()
        self.plot_dict["rewards"] = list(accumulate(self.plot_dict["rewards"]))
        self.save_episodes()
        self.save_agent()
        
        self.min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in self.min_max_dict.keys():
            if(not key in ["args", "arg_title", "arg_name", "episode_lists", "agent_lists", "spot_names", "steps"]):
                minimum = None ; maximum = None 
                l = self.plot_dict[key]
                l = deepcopy(l)
                l = [_ for _ in l if _ != None]
                if(l != []):
                    if(  minimum == None):  minimum = min(l)
                    elif(minimum > min(l)): minimum = min(l)
                    if(  maximum == None):  maximum = max(l) 
                    elif(maximum < max(l)): maximum = max(l)
                self.min_max_dict[key] = (minimum, maximum)
                
    
    
    def step_in_episode(self, prev_action, hq_m1, ha_m1, push):
        with torch.no_grad():
            obj, comm = self.task.obs()
            recommended_action = self.task.task.get_recommended_action()
            (_, _), (_, _), hq = self.forward.bottom_to_top_step(hq_m1, obj, comm, prev_action) 
            action, _, ha = self.actor(obj, comm, prev_action, hq[:,:,0].clone().detach(), ha_m1) 
            reward, done, win = self.task.action(action)
            next_obj, next_comm = self.task.obs()
            if(push): 
                self.memory.push(
                    obj, 
                    comm, 
                    action, 
                    recommended_action,
                    reward, 
                    next_obj, 
                    next_comm, 
                    done)
        torch.cuda.empty_cache()
        return(action, hq.squeeze(1), ha, reward, done, win)
            
           
    
    def training_episode(self, push = True):
        done = False
        total_reward = 0
        steps = 0
        prev_action = torch.zeros((1, 1, self.args.action_shape))
        hq = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
        ha = torch.zeros((1, 1, self.args.hidden_size)) 
        
        selected_task = choose_task(self.task_probabilities)
        self.task = self.task_runners[selected_task]
        self.task.begin()        
        for step in range(self.args.max_steps):
            self.steps += 1 
            if(not done):
                steps += 1
                prev_action, hq, ha, reward, done, win = self.step_in_episode(prev_action, hq, ha, push)
                total_reward += reward
            if(self.steps % self.args.steps_per_epoch == 0):
                plot_data = self.epoch(self.args.batch_size)
                if(plot_data == False): pass
                else:
                    accuracy, object_loss, comm_loss, complexity, alpha, actor, critics, e, q, ic, ie, ii, prediction_error, hidden_state = plot_data
                    if(self.epochs == 1 or self.epochs >= sum(self.args.epochs) or self.epochs % self.args.keep_data == 0):
                        self.plot_dict["accuracy"].append(accuracy)
                        self.plot_dict["object_loss"].append(object_loss)
                        self.plot_dict["comm_loss"].append(comm_loss)
                        self.plot_dict["complexity"].append(complexity)
                        self.plot_dict["alpha"].append(alpha)
                        self.plot_dict["actor"].append(actor)
                        for layer, f in enumerate(critics):
                            self.plot_dict["critics"][layer].append(f)    
                        self.plot_dict["critics"].append(critics)
                        self.plot_dict["extrinsic"].append(e)
                        self.plot_dict["q"].append(q)
                        self.plot_dict["intrinsic_curiosity"].append(ic)
                        self.plot_dict["intrinsic_entropy"].append(ie)
                        self.plot_dict["intrinsic_imitation"].append(ii)
                        self.plot_dict["prediction_error"].append(prediction_error)
                        for layer, f in enumerate(hidden_state):
                            self.plot_dict["hidden_state"][layer].append(f)    
        self.plot_dict["steps"].append(steps)
        self.plot_dict["rewards"].append(total_reward)
        self.plot_dict["wins"].append(win)
        self.episodes += 1
        
        
        
    def save_episodes(self):
        with torch.no_grad():
            if(self.args.agents_per_episode_list != -1 and self.agent_num > self.args.agents_per_episode_list): return
            episode_lists = []
            for episode_num in range(self.args.episodes_in_episode_list):
                print("Supposed to be saving episodes! {} epochs, {} episodes, {} steps, {} agent_num, {} episode_num".format(self.epochs, self.episodes, self.steps, self.agent_num, episode_num))
                episode_dict = {
                    "objects" : [],
                    "comms" : [],
                    "actions" : [],
                    "rewards" : [],
                    "prior_predicted_objects" : [],
                    "prior_predicted_comms" : [],
                    "posterior_predicted_objects" : [],
                    "posterior_predicted_comms" : []}
                done = False
                steps = 0
                prev_action = torch.zeros((1, 1, self.args.action_shape))
                hq = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
                ha = torch.zeros((1, 1, self.args.hidden_size)) 
                selected_task = choose_task(self.task_probabilities)
                self.task = self.task_runners[selected_task]
                self.task.begin()        
                obj, comm = self.task.obs()
                episode_dict["objects"].append(obj)
                episode_dict["comms"].append(comm)
                for step in range(self.args.max_steps):
                    if(not done):
                        prev_action, hq, ha, reward, done, win = self.step_in_episode(prev_action, hq, ha, push = False)
                        episode_dict["actions"].append(prev_action)
                        obj, comm = self.task.obs()
                        episode_dict["objects"].append(obj)
                        episode_dict["comms"].append(comm)
        """
                for step in range(self.args.max_steps):
                    if(not done): 
                        o, s = self.maze.obs()
                        a, h_actor, _, _, done, action_name = self.step_in_episode(prev_a, h_actor, push = False)
                        (zp_mu, zp_std), (zq_mu, zq_std), h_q_p1 = self.forward(o, s, prev_a, h_q)
                        (rgbd_mu_pred_p, pred_rgbd_p), (spe_mu_pred_p, pred_spe_p) = self.forward.get_preds(a, zp_mu, zp_std, h_q, quantity = self.args.samples_per_pred)
                        (rgbd_mu_pred_q, pred_rgbd_q), (spe_mu_pred_q, pred_spe_q) = self.forward.get_preds(a, zq_mu, zq_std, h_q, quantity = self.args.samples_per_pred)
                        pred_rgbd_p = [pred.squeeze(0).squeeze(0) for pred in pred_rgbd_p] ; pred_rgbd_q = [pred.squeeze(0).squeeze(0) for pred in pred_rgbd_q]
                        pred_spe_p = [pred.squeeze(0).squeeze(0) for pred in pred_spe_p]   ; pred_spe_q = [pred.squeeze(0).squeeze(0) for pred in pred_spe_q]
                        o, s = self.maze.obs()
                        pred_list.append((
                            action_name, (o.squeeze(0), s.squeeze(0)), 
                            (pred_rgbd_p, pred_spe_p), 
                            (pred_rgbd_q, pred_spe_q)))
                        prev_a = a ; h_q = h_q_p1
                pred_lists.append(pred_list)
            self.plot_dict["pred_lists"]["{}_{}_{}".format(self.agent_num, self.epochs, self.maze.name)] = pred_lists
        """
        
        
        
    def save_agent(self):
        pass
        #if(self.args.agents_per_agent_list != -1 and self.agent_num > self.args.agents_per_agent_list): return
        #self.plot_dict["agent_lists"]["{}_{}".format(self.agent_num, self.epochs)] = deepcopy(self.state_dict())
    
    
    
    def epoch(self, batch_size):
                                
        batch = self.memory.sample(batch_size)
        if(batch == False): return(False)
                
        self.epochs += 1

        objects, comms, actions, recommended_actions, rewards, dones, masks = batch
        objects = torch.from_numpy(objects)
        comms = torch.from_numpy(comms)
        actions = torch.from_numpy(actions)
        recommended_actions = torch.from_numpy(recommended_actions)
        rewards = torch.from_numpy(rewards)
        dones = torch.from_numpy(dones)
        masks = torch.from_numpy(masks)
        actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape).to(self.args.device), actions], dim = 1)
        all_masks = torch.cat([torch.ones(masks.shape[0], 1, 1).to(self.args.device), masks], dim = 1)   
        episodes = rewards.shape[0]
        steps = rewards.shape[1]
        
        #print("\n\n")
        #print("objects: {}. comms: {}. actions: {}. recommended actions: {}. rewards: {}. dones: {}. masks: {}.".format(
        #    objects.shape, comms.shape, actions.shape, recommended_actions.shape, rewards.shape, dones.shape, masks.shape))
        #print("\n\n")
        
                
        
        # Train forward
        (zp_mu, zp_std), (zq_mu, zq_std), (pred_objects, pred_comms), hqs = self.forward(torch.zeros((episodes, self.args.layers, self.args.pvrnn_mtrnn_size)), objects, comms, actions)
        hqs = hqs[:,:,0]
        
        real_shapes = objects[:,1:,:,:self.args.shapes]
        real_shapes = real_shapes.reshape((episodes * steps * self.args.objects, self.args.shapes))
        real_shapes = torch.argmax(real_shapes, dim = -1)
        pred_shapes = pred_objects[:,:,:,:self.args.shapes]
        pred_shapes = pred_shapes.reshape((episodes * steps * self.args.objects, self.args.shapes))
        shape_loss = F.cross_entropy(pred_shapes, real_shapes, reduction = "none")
        shape_loss = shape_loss.reshape((episodes, steps, self.args.objects))
        
        real_colors = objects[:,1:,:,self.args.shapes:]
        real_colors = real_colors.reshape((episodes * steps * self.args.objects, self.args.colors))
        real_colors = torch.argmax(real_colors, dim = -1)
        pred_colors = pred_objects[:,:,:,self.args.shapes:]
        pred_colors = pred_colors.reshape((episodes * steps * self.args.objects, self.args.colors))
        color_loss = F.cross_entropy(pred_colors, real_colors, reduction = "none")
        color_loss = color_loss.reshape((episodes, steps, self.args.objects))
        
        object_loss = shape_loss.mean(-1).unsqueeze(-1) * masks
        object_loss += color_loss.mean(-1).unsqueeze(-1) * masks
        
        real_comms = comms[:,1:].reshape((episodes * steps * self.args.max_comm_len, self.args.comm_shape))
        real_comms = torch.argmax(real_comms, dim = -1)
        pred_comms = pred_comms.reshape((episodes * steps * self.args.max_comm_len, self.args.comm_shape))
    
        comm_loss = F.cross_entropy(pred_comms, real_comms, reduction = "none")
        comm_loss = comm_loss.reshape((episodes, steps, self.args.max_comm_len))
        comm_loss = comm_loss.mean(-1).unsqueeze(-1) * masks
        
        accuracy_for_prediction_error = object_loss + comm_loss # * self.args.comm_scaler
        accuracy           = accuracy_for_prediction_error.mean()
        
        complexity_for_hidden_state = [dkl(zq_mu[:,:,layer], zq_std[:,:,layer], zp_mu[:,:,layer], zp_std[:,:,layer]).mean(-1).unsqueeze(-1) * all_masks for layer in range(self.args.layers)] 
        complexity          = sum([self.args.beta[layer] * complexity_for_hidden_state[layer].mean() for layer in range(self.args.layers)])       
        complexity_for_hidden_state = [layer[:,1:] for layer in complexity_for_hidden_state] 
                                
        self.forward_opt.zero_grad()
        (accuracy + complexity).backward()
        self.forward_opt.step()
        
        if(self.args.beta == 0): complexity = None
        torch.cuda.empty_cache()
                        
                        
        
        # Get curiosity                  
        if(self.args.dkl_max != None):
            complexity_for_hidden_state = [torch.tanh(c) for c in complexity_for_hidden_state]
        prediction_error_curiosity = accuracy_for_prediction_error * (self.args.prediction_error_eta if self.args.prediction_error_eta != None else self.prediction_error_eta)
        hidden_state_curiosities = [complexity_for_hidden_state[layer] * (self.args.hidden_state_eta[layer] if self.args.hidden_state_eta[layer] != None else self.hidden_state_eta[layer]) for layer in range(self.args.layers)]
        hidden_state_curiosity = sum(hidden_state_curiosities)
        if(self.args.curiosity == "prediction_error"):  curiosity = prediction_error_curiosity
        elif(self.args.curiosity == "hidden_state"): curiosity = hidden_state_curiosity
        else:                                curiosity = torch.zeros(rewards.shape).to(self.args.device)
        extrinsic = torch.mean(rewards).item()
        intrinsic_curiosity = curiosity.mean().item()
        rewards += curiosity
                        
        
                
        # Train critics
        with torch.no_grad():
            new_actions, log_pis_next, _ = self.actor(objects, comms, actions, hqs.clone().detach(), None)
            Q_target_nexts = []
            for i in range(self.args.critics):
                Q_target_next, _ = self.critic_targets[i](objects, comms, new_actions, hqs.clone().detach(), None)
                Q_target_next[:,1:]
                Q_target_nexts.append(Q_target_next)
            log_pis_next = log_pis_next[:,1:]
            Q_target_nexts_stacked = torch.stack(Q_target_nexts, dim=0)
            Q_target_next, _ = torch.min(Q_target_nexts_stacked, dim=0)
            Q_target_next = Q_target_next[:,1:]
            if self.args.alpha == None: Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.alpha * log_pis_next))
            else:                       Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.args.alpha * log_pis_next))
        
        critic_losses = []
        Qs = []
        for i in range(self.args.critics):
            Q, _ = self.critics[i](objects[:,:-1], comms[:,:-1], actions[:,1:], hqs[:,:-1].clone().detach(), None)
            critic_loss = 0.5*F.mse_loss(Q*masks, Q_targets*masks)
            critic_losses.append(critic_loss)
            Qs.append(Q[0,0].item())
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()
        
            self.soft_update(self.critics[i], self.critic_targets[i], self.args.tau)
        
        torch.cuda.empty_cache()
                                
        
        
        # Train alpha
        if self.args.alpha == None:
            _, log_pis, _ = self.actor(objects[:,:-1], comms[:,:-1], actions[:,:-1], hqs[:,:-1].clone().detach(), None)
            alpha_loss = -(self.log_alpha.to(self.args.device) * (log_pis + self.target_entropy))*masks
            alpha_loss = alpha_loss.mean() / masks.mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha).to(self.args.device)
            torch.cuda.empty_cache()
        else:
            alpha_loss = None
                                    
            
        
        # Train actor
        if self.epochs % self.args.d == 0:
            if self.args.alpha == None: alpha = self.alpha 
            else:                       alpha = self.args.alpha
            new_actions, log_pis, _ = self.actor(objects[:,:-1], comms[:,:-1], actions[:,:-1], hqs[:,:-1].clone().detach(), None)
            if self.args.action_prior == "normal":
                loc = torch.zeros(self.args.action_shape, dtype=torch.float64).to(self.args.device)
                n = self.args.action_shape
                scale_tril = torch.tril(torch.ones(n, n)).to(self.args.device)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_prrgbd = policy_prior.log_prob(new_actions).unsqueeze(-1)
            elif self.args.action_prior == "uniform":
                policy_prior_log_prrgbd = 0.0
            Qs = []
            for i in range(self.args.critics):
                Q, _ = self.critics[i](objects[:,:-1], comms[:,:-1], new_actions, hqs[:,:-1].clone().detach(), None)
                Qs.append(Q)
            Qs_stacked = torch.stack(Qs, dim=0)
            Q, _ = torch.min(Qs_stacked, dim=0)
            Q = Q.mean(-1).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis)*masks).item()
            recommendation_value = calculate_similarity(recommended_actions, new_actions).unsqueeze(-1)
            intrinsic_imitation = -torch.mean((self.args.delta * recommendation_value)*masks).item() 
            actor_loss = (alpha * log_pis - policy_prior_log_prrgbd - self.args.delta * recommendation_value - Q)*masks
            actor_loss = actor_loss.mean() / masks.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
        else:
            Q = None
            intrinsic_entropy = None
            intrinsic_imitation = None
            actor_loss = None
                                
                                
                                
        if(accuracy != None):   accuracy = accuracy.item()
        if(object_loss != None):   object_loss = object_loss.mean().item()
        if(comm_loss != None):   comm_loss = comm_loss.mean().item()
        if(complexity != None): complexity = complexity.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(Q != None): Q = -Q.mean().item()
        for i in range(self.args.critics):
            if(critic_losses[i] != None): 
                critic_losses[i] = critic_losses[i].item()
                critic_losses[i] = log(critic_losses[i]) if critic_losses[i] > 0 else critic_losses[i]
        
        prediction_error_curiosity = prediction_error_curiosity.mean().item()
        hidden_state_curiosities = [hidden_state_curiosity.mean().item() for hidden_state_curiosity in hidden_state_curiosities]
        hidden_state_curiosities = [hidden_state_curiosity for hidden_state_curiosity in hidden_state_curiosities]
        
        return(accuracy, object_loss, comm_loss, complexity, alpha_loss, actor_loss, critic_losses, 
               extrinsic, Q, intrinsic_curiosity, intrinsic_entropy, intrinsic_imitation, prediction_error_curiosity, hidden_state_curiosities)
    
    
                     
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def state_dict(self):
        to_return = [self.forward.state_dict(), self.actor.state_dict()]
        for i in range(self.args.critics):
            to_return.append(self.critics[i].state_dict())
            to_return.append(self.critic_targets[i].state_dict())
        return(to_return)

    def load_state_dict(self, state_dict):
        self.forward.load_state_dict(state_dict[0])
        self.actor.load_state_dict(state_dict[1])
        for i in range(self.args.critics):
            self.critics[i].load_state_dict(state_dict[2+2*i])
            self.critic_target[i].load_state_dict(state_dict[3+2*i])
        self.memory = RecurrentReplayBuffer(self.args)

    def eval(self):
        self.forward.eval()
        self.actor.eval()
        for i in range(self.args.critics):
            self.critics[i].eval()
            self.critic_targets[i].eval()

    def train(self):
        self.forward.train()
        self.actor.train()
        for i in range(self.args.critics):
            self.critics[i].train()
            self.critic_targets[i].train()
        
        
        
if __name__ == "__main__":
    agent = Agent(default_args)
# %%