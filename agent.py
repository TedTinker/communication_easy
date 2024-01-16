#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import numpy as np
from math import log
import matplotlib.pyplot as plt

from utils import default_args, dkl, print, choose_task, calculate_similarity, onehots_to_string
from task import Task, Task_Runner
from buffer import RecurrentReplayBuffer
from pvrnn import PVRNN
from models import Actor, Critic



class Agent:
    
    def __init__(self, args = default_args):
                
        self.args = args
        self.episodes = 0 ; self.epochs = 0 ; self.steps = 0
        print(self.args.task_probabilities)
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
            "pred_lists" : {}, "pos_lists" : {}, 
            "agent_lists" : {"forward" : PVRNN, "actor" : Actor, "critic" : Critic},
            "wins" : [], "rewards" : [], "spot_names" : [], "steps" : [],
            "accuracy" : [], "complexity" : [],
            "alpha" : [], "actor" : [], 
            "critics" : [], 
            "extrinsic" : [], "q" : [], "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : [], "intrinsic_imitation" : [],
            "prediction_error" : [], "hidden_state" : [[] for _ in range(self.args.layers)]}
        
        
        
    def training(self):        
        while(True):
            cumulative_epochs = 0
            for i, epochs in enumerate(self.args.epochs): 
                cumulative_epochs += epochs
                if(self.epochs < cumulative_epochs): 
                    self.task_probabilities = self.args.task_probabilities[i] 
                    break
            if(self.episodes % 100 == 0):
                self.plot_stuff()
                verbose = True
                print("EPOCH {}, EPISODE {}, TASKS {}\n".format(self.epochs, self.episodes, self.task_probabilities))
            else:
                verbose = False
            self.training_episode(verbose = verbose)
            if(self.epochs >= sum(self.args.epochs)): break
        
    def plot_stuff(self):
        columns = 4
        fig, axs = plt.subplots(nrows=1, ncols=columns, figsize=(5 * columns, 5))

        # First Column
        ax = axs[0]
        ax.plot(self.plot_dict["wins"])
        try:
            rolling_average = np.convolve(self.plot_dict["wins"], np.ones(100)/100, mode='valid')  # Rolling average with a window of 100
            ax.plot(rolling_average)
        except: pass
        ax.set_ylabel("Wins")
        ax.set_xlabel("Epochs")
        ax.set_title("Wins")

        # Second Column
        ax = axs[1]
        ax.plot(self.plot_dict["accuracy"])
        ax.plot(self.plot_dict["complexity"])
        ax.set_ylabel("Value")
        ax.set_xlabel("Epochs")
        ax.set_title("Accuracy and Complexity")

        # Third Column
        ax = axs[2]
        ax.plot([actor_loss for actor_loss in self.plot_dict["actor"] if actor_loss != None])
        ax.plot([alpha_loss for alpha_loss in self.plot_dict["intrinsic_entropy"] if alpha_loss != None])
        ax.plot([delta_loss for delta_loss in self.plot_dict["intrinsic_imitation"] if delta_loss != None])
        ax.set_ylabel("Value")
        ax.set_xlabel("Epochs")
        ax.set_title("Actor")

        # Fourth Column
        ax = axs[3]
        for i in range(self.args.critics):
            loss_list = [sublist[i] for sublist in self.plot_dict["critics"]]
            ax.plot(loss_list)
        ax.set_ylabel("Value")
        ax.set_xlabel("Epochs")
        ax.set_title("Critics")

        plt.show()
        plt.close()
                
    
    
    def step_in_episode(self, prev_action, hq_m1, ha_m1, push, verbose):
        with torch.no_grad():
            obj, comm = self.task.obs()
            recommended_action = self.task.task.get_recommended_action()
            (_, _), (_, _), hq = self.forward.bottom_to_top_step(hq_m1, obj, comm, prev_action) 
            action, _, ha = self.actor(obj, comm, prev_action, hq[:,:,0].clone().detach(), ha_m1) 
            reward, done, win = self.task.action(action, verbose)
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
            
           
    
    def training_episode(self, push = True, verbose = False):
        done = False ; steps = 0
        prev_action = torch.zeros((1, 1, self.args.action_shape))
        hq = torch.zeros((1, self.args.layers, self.args.hidden_size)) 
        ha = torch.zeros((1, 1, self.args.hidden_size)) 
        
        selected_task = choose_task(self.task_probabilities)
        self.task = self.task_runners[selected_task]
        if(verbose):
            print("TASK:", selected_task)
        self.task.begin(verbose)        
        for step in range(self.args.max_steps):
            self.steps += 1 
            if(not done):
                steps += 1
                prev_action, hq, ha, reward, done, win = self.step_in_episode(prev_action, hq, ha, push, verbose)
            if(self.steps % self.args.steps_per_epoch == 0):
                plot_data = self.epoch(self.args.batch_size, verbose)
                if(plot_data == False): pass
                else:
                    l, e, q, ic, ie, ii, prediction_error, hidden_state = plot_data
                    if(self.epochs == 1 or self.epochs >= sum(self.args.epochs) or self.epochs % self.args.keep_data == 0):
                        self.plot_dict["accuracy"].append(l[0][0])
                        self.plot_dict["complexity"].append(l[0][1])
                        self.plot_dict["alpha"].append(l[0][2])
                        self.plot_dict["actor"].append(l[0][3])
                        self.plot_dict["critics"].append(l[0][4])
                        self.plot_dict["extrinsic"].append(e)
                        self.plot_dict["q"].append(q)
                        self.plot_dict["intrinsic_curiosity"].append(ic)
                        self.plot_dict["intrinsic_entropy"].append(ie)
                        self.plot_dict["intrinsic_imitation"].append(ii)
                        self.plot_dict["prediction_error"].append(prediction_error)
                        for layer, f in enumerate(hidden_state):
                            self.plot_dict["hidden_state"][layer].append(f)    
        self.plot_dict["steps"].append(steps)
        self.plot_dict["rewards"].append(reward)
        self.plot_dict["wins"].append(win)
        self.episodes += 1
    
    
    
    def epoch(self, batch_size, verbose):
                                
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
        (zp_mu, zp_std), (zq_mu, zq_std), (pred_objects, pred_comms), hqs = self.forward(torch.zeros((episodes, self.args.layers, self.args.hidden_size)), objects, comms, actions)
        hqs = hqs[:,:,0]
        
        object_loss   = F.binary_cross_entropy(pred_objects, objects[:,1:], reduction = "none")
        object_loss = object_loss.reshape((episodes, steps, self.args.objects * (self.args.shapes + self.args.colors)))
        object_loss = object_loss.mean(-1).unsqueeze(-1) * masks
        
        if(verbose):
            print("\nObjects:", objects[0,1])
            print("Prediciton:", pred_objects[0,0])
        
        real_comms = comms[:,1:].reshape((episodes * steps * self.args.max_comm_len, self.args.comm_shape))
        real_comms = torch.argmax(real_comms, dim = -1)
        pred_comms = pred_comms.reshape((episodes * steps * self.args.max_comm_len, self.args.comm_shape))
    
        comm_loss = F.cross_entropy(pred_comms, real_comms, reduction = "none")
        comm_loss = comm_loss.reshape((episodes, steps, self.args.max_comm_len))
        comm_loss = comm_loss.mean(-1).unsqueeze(-1) * masks
        
        if(verbose):
            print("\nReal comm:", onehots_to_string(comms[0,1]))
            print("Pred comm:", onehots_to_string(pred_comms.reshape((episodes, steps, self.args.max_comm_len, self.args.comm_shape))[0,0]))
        
        accuracy_for_prediction_error = object_loss + self.args.comm_scaler * comm_loss
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
        
        if(verbose):
            print("\nRewards:", rewards[0,0].item())
            print("Predictions:", Qs)
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
            if(verbose):
                print("\nQ:", -Q[0,0])
                print("ENTROPY:", alpha * log_pis[0,0])
                print("RECOMMENDED VALUE:", -self.args.delta * recommendation_value[0,0])
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
        if(complexity != None): complexity = complexity.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        for i in range(self.args.critics):
            if(critic_losses[i] != None): 
                critic_losses[i] = critic_losses[i].item()
                critic_losses[i] = log(critic_losses[i]) if critic_losses[i] > 0 else critic_losses[i]
        losses = np.array([[accuracy, complexity, alpha_loss, actor_loss, critic_losses]])
        
        prediction_error_curiosity = prediction_error_curiosity.mean().item()
        hidden_state_curiosities = [hidden_state_curiosity.mean().item() for hidden_state_curiosity in hidden_state_curiosities]
        hidden_state_curiosities = [hidden_state_curiosity for hidden_state_curiosity in hidden_state_curiosities]
        
        return(losses, extrinsic, Q, intrinsic_curiosity, intrinsic_entropy, intrinsic_imitation, prediction_error_curiosity, hidden_state_curiosities)
    
    
                     
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
