#%%

import numpy as np

from utils import default_args, pad_zeros



class VariableBuffer:
    def __init__(self, shape = (1,), before_and_after = False, args = default_args):
        self.args = args
        self.shape = shape
        self.data = np.zeros((self.args.capacity, self.args.max_steps + (1 if before_and_after else 0)) + self.shape, dtype='float32')

    def reset_episode(self, episode_ptr):
        self.data[episode_ptr] = 0

    def push(self, episode_ptr, time_ptr, value):
        if self.shape == (1,): self.data[episode_ptr, time_ptr]    = value
        else:                  self.data[episode_ptr, time_ptr, :] = value

    def sample(self, indices):
        return self.data[indices]



class RecurrentReplayBuffer:
    def __init__(self, args = default_args):
        self.args = args
        self.capacity = self.args.capacity
        self.max_episode_len = self.args.max_steps
        self.num_episodes = 0

        self.objects = VariableBuffer(
            shape = (self.args.objects,self.args.shapes + self.args.colors), 
            before_and_after = True, 
            args = self.args)
        self.communications_in = VariableBuffer(
            shape = (self.args.max_comm_len, self.args.comm_shape,), 
            before_and_after = True, 
            args = self.args)
        self.actions = VariableBuffer(
            shape = (self.args.action_shape,), 
            args = self.args)
        self.communications_out = VariableBuffer(
            shape = (self.args.max_comm_len, self.args.comm_shape,), 
            args = self.args)
        self.recommended_actions = VariableBuffer(
            shape = (self.args.action_shape,), 
            args = self.args)
        self.rewards = VariableBuffer(args = self.args)
        self.dones = VariableBuffer(args = self.args)
        self.masks = VariableBuffer(args = self.args)

        self.episode_ptr = 0
        self.time_ptr = 0

    def push(
            self, 
            obj, 
            communication_in, 
            action, 
            communication_out, 
            recommended_action,
            reward, 
            next_obj, 
            next_communication_in, 
            done):
        
        if self.time_ptr == 0:
            for buffer in [
                    self.objects, 
                    self.communications_in, 
                    self.actions, 
                    self.communications_out, 
                    self.recommended_actions,
                    self.rewards, 
                    self.dones, 
                    self.masks]:
                buffer.reset_episode(self.episode_ptr)

        communication_in = pad_zeros(communication_in, self.args.max_comm_len)
        next_communication_in = pad_zeros(next_communication_in, self.args.max_comm_len)
        self.objects.push(self.episode_ptr, self.time_ptr, obj)
        self.communications_in.push(self.episode_ptr, self.time_ptr, communication_in)
        self.actions.push(self.episode_ptr, self.time_ptr, action)
        self.communications_out.push(self.episode_ptr, self.time_ptr, communication_out)
        self.recommended_actions.push(self.episode_ptr, self.time_ptr, recommended_action)
        self.rewards.push(self.episode_ptr, self.time_ptr, reward)
        self.dones.push(self.episode_ptr, self.time_ptr, done)
        self.masks.push(self.episode_ptr, self.time_ptr, 1.0)

        self.time_ptr += 1
        if done or self.time_ptr >= self.max_episode_len:
            self.objects.push(self.episode_ptr, self.time_ptr,  next_obj)
            self.communications_in.push(self.episode_ptr, self.time_ptr, next_communication_in)
            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0
            self.num_episodes = min(self.num_episodes + 1, self.capacity)

    def sample(self, batch_size):
        if(self.num_episodes == 0): return(False)
        if(self.num_episodes < batch_size):
            indices = np.random.choice(self.num_episodes, self.num_episodes, replace=False)
            batch = (
                self.objects.sample(indices),
                self.communications_in.sample(indices),
                self.actions.sample(indices),
                self.communications_out.sample(indices),
                self.recommended_actions.sample(indices),
                self.rewards.sample(indices),
                self.dones.sample(indices),
                self.masks.sample(indices))
        else:
            indices = np.random.choice(self.num_episodes, batch_size, replace=False)
            batch = (
                self.objects.sample(indices),
                self.communications_in.sample(indices),
                self.actions.sample(indices),
                self.communications_out.sample(indices),
                self.recommended_actions.sample(indices),
                self.rewards.sample(indices),
                self.dones.sample(indices),
                self.masks.sample(indices))
        return batch