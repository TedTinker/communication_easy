#%% 

import builtins
import datetime 
import matplotlib
import argparse, ast
from math import exp
import torch
from torch import nn 
import platform
from random import choices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
device = "cpu"

"""
color_map = {
    0: "red", 
    1: "blue", 
    2: "yellow", 
    3: "green", 
    4: "orange", 
    5: "purple"}
shape_map = {
    0: "pole", 
    1: "T", 
    2: "L", 
    3: "cross", 
    4: "I"}
action_map = {
    0: "push", 
    1: "pull", 
    2: "lift", 
    3: "spin-L", 
    4: "spin-R"}

comm_map = {
    0: ' ', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G',
    8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N',
    15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U',
    22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: '.'}

"""

color_map = {
    0: "A", # red", 
    1: "B", #"blue", 
    2: "C", #"yellow", 
    3: "D", #"green", 
    4: "E", #"orange", 
    5: "F"} #"purple"}
shape_map = {
    0: "A", #"pole", 
    1: "B", 
    2: "C", 
    3: "D", #"cross", 
    4: "E"}
action_map = {
    0: "A", #"push", 
    1: "B", #"pull", 
    2: "C", #"lift", 
    3: "D", #"spin-L", 
    4: "E"} #"spin-R"}

comm_map = {
    0: ' ', 1: 'A', 2: 'B', 3: 'C', 4: '.'}
#"""



char_to_index = {v: k for k, v in comm_map.items()}



# Arguments to parse. 
def literal(arg_string): return(ast.literal_eval(arg_string))
parser = argparse.ArgumentParser()

    # Meta 
parser.add_argument('--device',             type=str,        default = device,
                    help='Which device to use for Torch.')

    # Training
parser.add_argument('--epochs',             type=literal,    default = 20*[500] + [10000],
                    help='List of how many epochs to train in each maze.')
parser.add_argument('--batch_size',         type=int,        default = 128, 
                    help='How many episodes are sampled for each epoch.')
parser.add_argument('--GAMMA',              type=float,      default = .9,
                    help='How heavily critics consider the future.')
parser.add_argument("--d",                  type=int,        default = 2,
                    help='Delay for training actors.')        

    # Memory buffer
parser.add_argument('--capacity',           type=int,        default = 250,
                    help='How many episodes can the memory buffer contain.')

    # Task and Agent details
parser.add_argument('--task_probabilities', type=literal,    default = [
    (("1",1), ("2",1), ("3",1)),
    (("1",1), ("2",1), ("3",1),  ("4",.1)),
    (("1",1), ("2",1), ("3",1),  ("4",.2)),
    (("1",1), ("2",1), ("3",1),  ("4",.3)),
    (("1",1), ("2",1), ("3",1),  ("4",.4)),
    (("1",1), ("2",1), ("3",1),  ("4",.5)),
    (("1",1), ("2",1), ("3",1),  ("4",.6)),
    (("1",1), ("2",1), ("3",1),  ("4",.7)),
    (("1",1), ("2",1), ("3",1),  ("4",.8)),
    (("1",1), ("2",1), ("3",1),  ("4",.9)),
    (("1",1), ("2",1), ("3",1),  ("4",1)),
    (("1",.9),("2",.9),("3",.9), ("4",1)),
    (("1",.8),("2",.8),("3",.8), ("4",1)),
    (("1",.7),("2",.7),("3",.7), ("4",1)),
    (("1",.6),("2",.6),("3",.6), ("4",1)),
    (("1",.5),("2",.5),("3",.5), ("4",1)),
    (("1",.4),("2",.4),("3",.4), ("4",1)),
    (("1",.3),("2",.3),("3",.3), ("4",1)),
    (("1",.2),("2",.2),("3",.2), ("4",1)),
    (("1",.1),("2",.1),("3",.1), ("4",1)),
    (("4", 1),)],
                    help='List of probabilities of tasks. Agent trains on each set of tasks based on epochs in epochs parameter.')
parser.add_argument('--max_steps',          type=int,        default = 3,
                    help='How many steps the agent can make in one episode.')
parser.add_argument('--step_lim_punishment',type=float,      default = -1,
                    help='Extrinsic punishment for taking max_steps steps.')
parser.add_argument('--extrinsic_reward',   type=float,     default = 1,
                    help='Extrinsic reward for choosing incorrect exit. Format: [(weight, reward), (weight, reward), ...]')  
parser.add_argument('--step_cost',          type=float,      default = .9,
                    help='How much extrinsic rewards for exiting are reduced per step.')
parser.add_argument('--actions',            type=int,        default = 3,
                    help='Maximum count of actions in one episode.')
parser.add_argument('--objects',            type=int,        default = 3,
                    help='Maximum count of objects in one episode.')
parser.add_argument('--shapes',            type=int,        default = 3,
                    help='Maximum count of shapes in one episode.')
parser.add_argument('--colors',            type=int,        default = 3,
                    help='Maximum count of colors in one episode.')
parser.add_argument('--max_comm_len',      type=int,        default = 4,
                    help='Maximum length of communication.')

    # Module 
parser.add_argument('--hidden_size',        type=int,        default = 32,
                    help='Parameters in hidden layers.')   
parser.add_argument('--state_size',         type=int,        default = 32,
                    help='Parameters in prior and posterior inner-states.')
parser.add_argument('--time_scales',        type=literal,    default = [1],
                    help='Time-scales for MTRNN.')
parser.add_argument('--forward_lr',         type=float,      default = .01,
                    help='Learning rate for forward model.')
parser.add_argument('--alpha_lr',           type=float,      default = .01,
                    help='Learning rate for alpha value.') 
parser.add_argument('--actor_lr',           type=float,      default = .01,
                    help='Learning rate for actor model.')
parser.add_argument('--critic_lr',          type=float,      default = .01,
                    help='Learning rate for critic model.')
parser.add_argument("--tau",                type=float,      default = .1,
                    help='Rate at which target-critics approach critics.')      

    # Complexity 
parser.add_argument('--std_min',            type=int,        default = exp(-20),
                    help='Minimum value for standard deviation.')
parser.add_argument('--std_max',            type=int,        default = exp(2),
                    help='Maximum value for standard deviation.')
parser.add_argument("--beta",               type=literal,    default = [0],
                    help='Relative importance of complexity in each layer.')

    # Entropy
parser.add_argument("--alpha",              type=str,        default = 1,
                    help='Nonnegative value, how much to consider entropy. Set to None to use target_entropy.')        
parser.add_argument("--target_entropy",     type=float,      default = -2,
                    help='Target for choosing alpha if alpha set to None. Recommended: negative size of action-space.')      
parser.add_argument('--action_prior',       type=str,        default = "normal",
                    help='The actor can be trained based on normal or uniform distributions.')

    # Curiosity
parser.add_argument("--curiosity",          type=str,        default = "none",
                    help='Which kind of curiosity: none, prediction_error, or hidden_state.')  
parser.add_argument("--dkl_max",            type=float,      default = 1,
                    help='Maximum value for clamping Kullback-Liebler divergence for hidden_state curiosity.')        
parser.add_argument("--prediction_error_eta",          type=float,      default = 1,
                    help='Nonnegative value, how much to consider prediction_error curiosity.')    
parser.add_argument("--hidden_state_eta",           type=literal,    default = [1],
                    help='Nonnegative valued, how much to consider hidden_state curiosity in each layer.')       

    # Saving data
parser.add_argument('--keep_data',           type=int,        default = 1,
                    help='How many epochs should pass before saving data.')

parser.add_argument('--epochs_per_pred_list',type=int,        default = 10000000,
                    help='How many epochs should pass before saving agent predictions.')
parser.add_argument('--agents_per_pred_list',type=int,        default = 1,
                    help='How many agents to save predictions.')
parser.add_argument('--episodes_in_pred_list',type=int,       default = 1,
                    help='How many episodes of predictions to save per agent.')

parser.add_argument('--epochs_per_pos_list', type=int,        default = 100,
                    help='How many epochs should pass before saving agent positions.')
parser.add_argument('--agents_per_pos_list', type=int,        default = -1,
                    help='How many agents to save positions.')
parser.add_argument('--episodes_in_pos_list',type=int,        default = 1,
                    help='How many episodes of positions to save per agent.')

parser.add_argument('--epochs_per_agent_list',type=int,       default = 100000,
                    help='How many epochs should pass before saving agent model.')
parser.add_argument('--agents_per_agent_list',type=int,       default = 1,
                    help='How many agents to save.') 



try:
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
except:
    import sys ; sys.argv=[''] ; del sys           # Comment this out when using bash
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
    
def extend_list_to_match_length(target_list, length, value):
    while len(target_list) < length:
        target_list.append(value)
    return target_list

for arg_set in [default_args, args]:
    arg_set.steps_per_epoch = arg_set.max_steps
    arg_set.observation_shape = arg_set.objects * (arg_set.shapes + arg_set.colors)
    arg_set.communication_shape = len(comm_map)
    arg_set.action_shape = arg_set.actions + arg_set.objects
    max_length = max(len(arg_set.time_scales), len(arg_set.beta), len(arg_set.hidden_state_eta))
    arg_set.time_scales = extend_list_to_match_length(arg_set.time_scales, max_length, 1)
    arg_set.beta = extend_list_to_match_length(arg_set.beta, max_length, 0)
    arg_set.hidden_state_eta = extend_list_to_match_length(arg_set.hidden_state_eta, max_length, 0)
    arg_set.layers = len(arg_set.time_scales)



def print(*args, **kwargs):
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)

font = {'family' : 'sans-serif',
        #'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

    
start_time = datetime.datetime.now()

def duration(start_time = start_time):
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)

def estimate_total_duration(proportion_completed, start_time=start_time):
    if(proportion_completed != 0): 
        so_far = datetime.datetime.now() - start_time
        estimated_total = so_far / proportion_completed
        estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    else: estimated_total = "?:??:??"
    return(estimated_total)
    
def attach_list(tensor_list, device):
    updated_list = []
    for tensor in tensor_list:
        if isinstance(tensor, list):
            updated_sublist = [t.to(device) if t.device != device else t for t in tensor]
            updated_list.append(updated_sublist)
        else:
            updated_tensor = tensor.to(device) if tensor.device != device else tensor
            updated_list.append(updated_tensor)
    return updated_list

def detach_list(l): 
    return([element.detach() for element in l])
    
def memory_usage(device):
    print(device, ":", platform.node(), torch.cuda.memory_allocated(device), "out of", torch.cuda.max_memory_allocated(device))

def string_to_onehots(s):
    s = ''.join([char.upper() if char.upper() in char_to_index else ' ' for char in s])
    onehots = []
    for char in s:
        tensor = torch.zeros(len(comm_map))
        tensor[char_to_index[char]] = 1
        onehots.append(tensor.unsqueeze(0))
    onehots = torch.cat(onehots, dim = 0)
    return onehots

def onehots_to_string(onehots):
    string = ''
    for tensor in onehots:
        index = torch.argmax(tensor).item()
        string += comm_map[index]
    return string

def pad_zeros(value, length):
    rows_to_add = length - value.size(-2)
    padding_shape = list(value.shape)
    padding_shape[-2] = rows_to_add
    padding = torch.zeros(padding_shape)
    padding[..., 0] = 1
    value = torch.cat([value, padding], dim=-2)
    return value

def create_comm_mask(comm):
    period_index = char_to_index["."]  # Index for the period character in the one-hot encoding
    mask = torch.ones_like(comm[..., 0], dtype=torch.float32)  # Create a mask with the same shape except for the last dimension
    max_indices = comm.argmax(dim=-1)
    period_mask = torch.where(max_indices == period_index, 1, 0)
    if len(comm.shape) == 2:  # Handling (sequence_length, len(comm_map))
        mask[period_mask.argmax()+1:] = 0
    elif len(comm.shape) == 3:  # Handling (steps, sequence_length, len(comm_map))
        for step in range(comm.shape[0]):
            mask[step, period_mask[step].argmax()+1:] = 0
    elif len(comm.shape) == 4:  # Handling (episodes, steps, sequence_length, len(comm_map))
        for episode in range(comm.shape[0]):
            for step in range(comm.shape[1]):
                mask[episode, step, period_mask[episode, step].argmax()+1:] = 0
    return mask

def choose_task(probabilities):
    if(len(probabilities) == 1):
        selected_task = probabilities[0][0]
    else:
        tasks, weights = zip(*probabilities)
        selected_task = choices(tasks, weights=weights, k=1)[0]
    print("TASK:", selected_task)
    return selected_task


    
def dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)



def multi_hot_action(action,  args = default_args):
    if(len(action.shape) == 1): action = action.unsqueeze(0)
    if(len(action.shape) == 2): action = action.unsqueeze(0)
    action = torch.as_tensor(action)
    multi_hot = torch.zeros_like(action, dtype=torch.float32)
    action_indices_list = []
    object_indices_list = []
    for episode in range(action.shape[0]):
        episode_action_indices = []
        episode_object_indices = []
        for step in range(action.shape[1]):
            action_index = torch.argmax(action[episode, step, :args.actions])
            object_index = torch.argmax(action[episode, step, args.actions:])
            multi_hot[episode, step, action_index] = 1
            multi_hot[episode, step, object_index] = 1
            episode_action_indices.append(action_index.item())
            episode_object_indices.append(object_index.item())
        action_indices_list.append(torch.tensor(episode_action_indices))
        object_indices_list.append(torch.tensor(episode_object_indices))
    action_indices_tensor = torch.stack(action_indices_list)
    object_indices_tensor = torch.stack(object_indices_list)
    return multi_hot, action_indices_tensor, object_indices_tensor

"""
example_action = torch.tensor([.1, .2, .3, .4, .5, -1, -2, -3])
example_action_2 = torch.tensor([[.1, .2, .3, .4, .5, -1, -2, -3], 
                                 [.1, .2, .3, .4, .5, -1, -2, -3]])
example_action_3 = torch.tensor([[[.1, .2, .3, .4, .5, -1, -2, -3],
                                  [.1, .2, .3, .4, .5, -1, -2, -3]], 
                                 [[.1, .2, .3, .4, .5, -1, -2, -3], 
                                  [.1, .2, .3, .4, .5, -1, -2, -3]]])

print(example_action)
print(multi_hot_action(example_action))
print(example_action_2)
print(multi_hot_action(example_action_2))
print(example_action_3)
print(multi_hot_action(example_action_3))
"""


class ConstrainedConv1d(nn.Conv1d):
    def forward(self, input):
        return nn.functional.conv1d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        
class Ted_Conv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernels = [1,2,3]):
        super(Ted_Conv1d, self).__init__()
        
        self.Conv1ds = nn.ModuleList()
        for kernel, out_channel in zip(kernels, out_channels):
            padding = (kernel-1)//2
            layer = nn.Sequential(
                ConstrainedConv1d(
                    in_channels = in_channels,
                    out_channels = out_channel,
                    kernel_size = kernel,
                    padding = padding,
                    padding_mode = "reflect"),
                nn.PReLU())
            self.Conv1ds.append(layer)
                
    def forward(self, x):
        y = []
        for Conv1d in self.Conv1ds: y.append(Conv1d(x)) 
        return(torch.cat(y, dim = -2))