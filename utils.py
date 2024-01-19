#%% 

# To do:
# Finish plotting. (mostly done: only one critic-loss plotted, and odd hidden-state curiosity min_maxing.)
# Save whole episodes.
# Fix forward-collapse.
# Make a "generalization" check to see if it can generalize combos it hasn't seen.
# Add double-agent. 

import os
import pickle
from time import sleep
import builtins
import datetime 
import matplotlib
import argparse, ast
from math import exp
import torch
from torch import nn 
import platform
from random import choices
from torch.distributions import Normal
import torch.distributions as dist
from torch.nn.functional import cosine_similarity

if(os.getcwd().split("/")[-1] != "communication_easy"): os.chdir("communication_easy")

torch.set_printoptions(precision=3, sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
device = "cpu"

shape_map = {
    0: "pole", 
    1: "T", 
    2: "L", 
    3: "cross", 
    4: "I"}
color_map = {
    0: "red", 
    1: "blue", 
    2: "yellow", 
    3: "green", 
    4: "orange", 
    5: "purple"}
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

char_to_index = {v: k for k, v in comm_map.items()}



# Arguments to parse. 
def literal(arg_string): return(ast.literal_eval(arg_string))
parser = argparse.ArgumentParser()

    # Meta 
parser.add_argument("--arg_title",          type=str,        default = "default",
                    help='Title of argument-set containign all non-default arguments.') 
parser.add_argument("--arg_name",           type=str,        default = "default",
                    help='Title of argument-set for human-understanding.') 
parser.add_argument("--agents",             type=int,        default = 36,
                    help='How many agents are trained in this job?')
parser.add_argument("--previous_agents",    type=int,        default = 0,
                    help='How many agents with this argument-set are trained in previous jobs?')
parser.add_argument("--init_seed",          type=float,      default = 777,
                    help='Random seed.')
parser.add_argument('--comp',               type=str,        default = "deigo",
                    help='Cluster name (deigo or saion).')
parser.add_argument('--device',             type=str,        default = device,
                    help='Which device to use for Torch.')

    # Task details
parser.add_argument('--task_probabilities', type=literal,    default = [
    (("1", 1),)],
                    help='List of probabilities of tasks. Agent trains on each set of tasks based on epochs in epochs parameter.')
parser.add_argument('--max_steps',          type=int,        default = 3,
                    help='How many steps the agent can make in one episode.')
parser.add_argument('--step_lim_punishment',type=float,      default = -1,
                    help='Extrinsic punishment for taking max_steps steps.')
parser.add_argument('--action_reward',      type=float,     default = 0,
                    help='Extrinsic reward for choosing incorrect action.') 
parser.add_argument('--shape_reward',       type=float,     default = 0,
                    help='Extrinsic reward for choosing incorrect shape.') 
parser.add_argument('--color_reward',       type=float,     default = 0,
                    help='Extrinsic reward for choosing incorrect color.')  
parser.add_argument('--correct_reward',     type=float,     default = 5,
                    help='Extrinsic reward for choosing incorrect action, shape, and color.') 
parser.add_argument('--step_cost',          type=float,      default = .9,
                    help='How much extrinsic rewards for exiting are reduced per step.')
parser.add_argument('--actions',            type=int,        default = 5,
                    help='Maximum count of actions in one episode.')
parser.add_argument('--objects',            type=int,        default = 3,
                    help='Maximum count of objects in one episode.')
parser.add_argument('--shapes',            type=int,        default = 5,
                    help='Maximum count of shapes in one episode.')
parser.add_argument('--colors',            type=int,        default = 6,
                    help='Maximum count of colors in one episode.')
parser.add_argument('--max_comm_len',      type=int,        default = 20,
                    help='Maximum length of communication.')

    # Training
parser.add_argument('--epochs',             type=literal,    default = [2000],
                    help='List of how many epochs to train in each maze.')
parser.add_argument('--batch_size',         type=int,        default = 128, 
                    help='How many episodes are sampled for each epoch.')       

    # Memory buffer
parser.add_argument('--capacity',           type=int,        default = 250,
                    help='How many episodes can the memory buffer contain.')

    # Module 
parser.add_argument('--critics',            type=int,        default = 2,
                    help='How many critics?')   
parser.add_argument('--hidden_size',        type=int,        default = 32,
                    help='Parameters in hidden layers.')   
parser.add_argument('--pvrnn_mtrnn_size',   type=int,        default = 64,
                    help='Parameters in hidden layers pf PVRNN\'s mtrnn.')   
parser.add_argument('--state_size',         type=int,        default = 128,
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
parser.add_argument('--GAMMA',              type=float,      default = .9,
                    help='How heavily critics consider the future.')
parser.add_argument("--d",                  type=int,        default = 2,
                    help='Delay for training actors.') 

    # Complexity 
parser.add_argument('--std_min',            type=int,        default = exp(-20),
                    help='Minimum value for standard deviation.')
parser.add_argument('--std_max',            type=int,        default = exp(2),
                    help='Maximum value for standard deviation.')
parser.add_argument("--beta",               type=literal,    default = [1],
                    help='Relative importance of complexity in each layer.')

    # Entropy
parser.add_argument("--alpha",              type=literal,    default = 0,
                    help='Nonnegative value, how much to consider entropy. Set to None to use target_entropy.')        
parser.add_argument("--target_entropy",     type=float,      default = 0,
                    help='Target for choosing alpha if alpha set to None. Recommended: negative size of action-space.')      
parser.add_argument('--action_prior',       type=str,        default = "uniform",
                    help='The actor can be trained based on normal or uniform distributions.')

    # Curiosity
parser.add_argument("--curiosity",          type=str,        default = "none",
                    help='Which kind of curiosity: none, prediction_error, or hidden_state.')  
parser.add_argument("--dkl_max",            type=float,      default = 1,
                    help='Maximum value for clamping Kullback-Liebler divergence for hidden_state curiosity.')        
parser.add_argument("--prediction_error_eta", type=float,    default = 1,
                    help='Nonnegative value, how much to consider prediction_error curiosity.')    
parser.add_argument("--hidden_state_eta",   type=literal,    default = [1],
                    help='Nonnegative valued, how much to consider hidden_state curiosity in each layer.')       

    # Imitation
parser.add_argument("--delta",              type=float,     default = 0,
                    help='How much to consider action\'s similarity to recommended action.')  

    # Saving data
parser.add_argument('--keep_data',           type=int,        default = 1,
                    help='How many epochs should pass before saving data.')

parser.add_argument('--epochs_per_episode_list',type=int,        default = 10000000,
                    help='How many epochs should pass before saving an episode.')
parser.add_argument('--agents_per_episode_list',type=int,        default = 1,
                    help='How many agents to save episodes.')
parser.add_argument('--episodes_in_episode_list',type=int,       default = 1,
                    help='How many episodes to save per agent.')

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
    arg_set.object_shape = arg_set.shapes + arg_set.colors
    arg_set.comm_shape = len(comm_map)
    arg_set.action_shape = arg_set.actions + arg_set.objects
    max_length = max(len(arg_set.time_scales), len(arg_set.beta), len(arg_set.hidden_state_eta))
    arg_set.time_scales = extend_list_to_match_length(arg_set.time_scales, max_length, 1)
    arg_set.beta = extend_list_to_match_length(arg_set.beta, max_length, 0)
    arg_set.hidden_state_eta = extend_list_to_match_length(arg_set.hidden_state_eta, max_length, 0)
    arg_set.layers = len(arg_set.time_scales)
    #for arg in vars(arg_set):
    #    print(arg)
    #    if(getattr(arg_set, arg) == "None"):  arg_set.arg = None
    #    if(getattr(arg_set, arg) == "True"):  arg_set.arg = True
    #    if(getattr(arg_set, arg) == "False"): arg_set.arg = False
        
args_not_in_title = ["arg_title", "id", "agents", "previous_agents", "init_seed", "keep_data", "epochs_per_pred_list", "episodes_in_pred_list", "agents_per_pred_list", "epochs_per_pos_list", "episodes_in_pos_list", "agents_per_pos_list"]
def get_args_title(default_args, args):
    if(args.arg_title[:3] == "___"): return(args.arg_title)
    name = "" ; first = True
    arg_list = list(vars(default_args).keys())
    arg_list.insert(0, arg_list.pop(arg_list.index("arg_name")))
    for arg in arg_list:
        if(arg in args_not_in_title): pass 
        else: 
            default, this_time = getattr(default_args, arg), getattr(args, arg)
            if(this_time == default): pass
            elif(arg == "arg_name"):
                name += "{} (".format(this_time)
            else: 
                if first: first = False
                else: name += ", "
                name += "{}: {}".format(arg, this_time)
    if(name == ""): name = "default" 
    else:           name += ")"
    if(name.endswith(" ()")): name = name[:-3]
    parts = name.split(',')
    name = "" ; line = ""
    for i, part in enumerate(parts):
        if(len(line) > 50 and len(part) > 2): name += line + "\n" ; line = ""
        line += part
        if(i+1 != len(parts)): line += ","
    name += line
    return(name)

args.arg_title = get_args_title(default_args, args)

try: os.mkdir("saved")
except: pass
folder = "saved/" + args.arg_name
if(args.arg_title[:3] != "___" and not args.arg_name in ["default", "finishing_dictionaries", "plotting", "plotting_predictions", "plotting_positions"]):
    try: os.mkdir(folder)
    except: pass
    try: os.mkdir("saved/thesis_pics")
    except: pass
    try: os.mkdir("saved/thesis_pics/final")
    except: pass
if(default_args.alpha == "None"): default_args.alpha = None
if(args.alpha == "None"):         args.alpha = None

if(args == default_args): print("Using default arguments.")
else:
    for arg in vars(default_args):
        default, this_time = getattr(default_args, arg), getattr(args, arg)
        if(this_time == default): pass
        else: print("{}:\n\tDefault:\t{}\n\tThis time:\t{}".format(arg, default, this_time))



def print(*args, **kwargs):
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)

# Adjusting PLT.
font = {'family' : 'sans-serif',
        #'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)



# Duration functions.
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



# Functions for task at hand.
def choose_task(probabilities):
    if(len(probabilities) == 1):
        selected_task = probabilities[0][0]
    else:
        tasks, weights = zip(*probabilities)
        selected_task = choices(tasks, weights=weights, k=1)[0]
    return selected_task

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
            multi_hot[episode, step, object_index + args.actions] = 1
            episode_action_indices.append(action_index.item())
            episode_object_indices.append(object_index.item())
        action_indices_list.append(torch.tensor(episode_action_indices))
        object_indices_list.append(torch.tensor(episode_object_indices))
    action_indices_tensor = torch.stack(action_indices_list)
    object_indices_tensor = torch.stack(object_indices_list)
    return multi_hot, action_indices_tensor, object_indices_tensor



# For printing tensors as what they represent.
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

def multihots_to_string(multihots):
    shapes = multihots[:,:len(shape_map)]
    colors = multihots[:,len(shape_map):]
    to_return = ""
    for i in range(multihots.shape[0]):
        shape_index = torch.argmax(shapes[i]).item()
        color_index = torch.argmax(colors[i]).item()
        to_return += "{} {}{}".format(color_map[color_index], shape_map[shape_index], "." if i+1 == multihots.shape[0] else ", ")
    return(to_return)

def action_to_string(action):
    while(len(action.shape) > 1): action = action.squeeze(0)
    act = action[:len(action_map)]
    act_index = torch.argmax(act).item()
    objects = action[len(action_map):]
    object_index = torch.argmax(objects).item()
    return("{} object {}.\t".format(action_map[act_index], object_index))



# PyTorch functions.
def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

def episodes_steps(this):
    return(this.shape[0], this.shape[1])

def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

def sample(mu, std, device):
    e = Normal(0, 1).sample(std.shape).to(device)
    return(mu + e * std)

def rnn_cnn(do_this, to_this):
    episodes = to_this.shape[0] ; steps = to_this.shape[1]
    this = to_this.view((episodes * steps, to_this.shape[2], to_this.shape[3], to_this.shape[4]))
    this = do_this(this)
    this = this.view((episodes, steps, this.shape[1], this.shape[2], this.shape[3]))
    return(this)

def extract_and_concatenate(tensor, expected_len):
    episodes, steps, _ = tensor.shape
    all_indices = []
    for episode in range(episodes):
        episode_indices = []
        for step in range(steps):
            ones_indices = tensor[episode, step].nonzero(as_tuple=True)[0]
            if(ones_indices.shape[0] < expected_len):
                ones_indices = torch.zeros((expected_len,)).int()
            episode_indices.append(ones_indices)
        episode_tensor = torch.stack(episode_indices)
        all_indices.append(episode_tensor)
    final_tensor = torch.stack(all_indices, dim=0)
    return final_tensor
    
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

def pad_zeros(value, length):
    rows_to_add = length - value.size(-2)
    padding_shape = list(value.shape)
    padding_shape[-2] = rows_to_add
    padding = torch.zeros(padding_shape)
    padding[..., 0] = 1
    value = torch.cat([value, padding], dim=-2)
    return value

# Not in use, but maybe should be. 
def create_comm_mask(comm):
    period_index = char_to_index["."]  # Index for the period character in the one-hot encoding
    mask = torch.ones_like(comm[..., 0], dtype=torch.float32)  # Create a mask with the same shape except for the last dimension
    max_indices = comm.argmax(dim=-1)
    period_mask = torch.where(max_indices == period_index, 1, 0)

    def apply_mask(mask, period_mask):
        if period_mask.any():
            first_period_index = period_mask.argmax()
            mask[first_period_index+1:] = 0
            return mask, first_period_index
        else:
            return mask, mask.size(0) - 1  # Return the last index if no period

    if len(comm.shape) == 2:  # Handling (sequence_length, len(comm_map))
        mask, last_index = apply_mask(mask, period_mask)
        last_indices = torch.tensor(last_index).expand_as(mask)
    elif len(comm.shape) == 3:  # Handling (steps, sequence_length, len(comm_map))
        last_indices = torch.empty(comm.shape[0], dtype=torch.long)
        for step in range(comm.shape[0]):
            mask[step], last_indices[step] = apply_mask(mask[step], period_mask[step])
    elif len(comm.shape) == 4:  # Handling (episodes, steps, sequence_length, len(comm_map))
        last_indices = torch.empty((comm.shape[0], comm.shape[1]), dtype=torch.long)
        for episode in range(comm.shape[0]):
            for step in range(comm.shape[1]):
                mask[episode, step], last_indices[episode, step] = apply_mask(mask[episode, step], period_mask[episode, step])

    return mask, last_indices

#example_comm = torch.stack([
#    pad_zeros(string_to_onehots("HELLO WORLD."), args.max_comm_len),
#    pad_zeros(string_to_onehots("GOODBYE WORLD."), args.max_comm_len),
#    pad_zeros(string_to_onehots("YOWSERS. HI.. ."), args.max_comm_len),
#    pad_zeros(string_to_onehots("YOWSERS"), args.max_comm_len),
#    pad_zeros(string_to_onehots("."), args.max_comm_len),
#    pad_zeros(string_to_onehots("ABCDEF."), args.max_comm_len)],
#    dim = 0)
#example_comm = example_comm.reshape((2,3,args.max_comm_len, args.comm_shape))
#mask, last_indices = create_comm_mask(example_comm)

def dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)

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
    
def calculate_similarity(recommended_actions, actor_actions):
    recommended_actions_flat = recommended_actions.view(recommended_actions.size(0), recommended_actions.size(1), -1)
    actor_actions_flat = actor_actions.view(actor_actions.size(0), actor_actions.size(1), -1)
    step_similarities = cosine_similarity(recommended_actions_flat, actor_actions_flat, dim=-1)
    return step_similarities



# For loading and plotting in final files.
real_names = {
    "d"  : "No Entropy, No Curiosity",
    "e"  : "Entropy",
    "n"  : "Prediction Error Curiosity",
    "f"  : "Hidden State Curiosity",
    "i"  : "Imitation",
    "en" : "Entropy and Prediction Error Curiosity",
    "ef" : "Entropy and Hidden State Curiosity",
    "ei" : "Entropy and Imitation",
    "ni" : "Prediction Error Curiosity and Imitation",
    "fi" : "Hidden State Curiosity and Imitation",
    "eni" : "Entropy, Prediction Error Curiosity, and Imitation",
    "efi" : "Entropy, Hidden State Curiosity, and Imitation",
}

def add_this(name):
    keys, values = [], []
    for key, value in real_names.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        new_key = key + "_" + name 
        real_names[new_key] = value
add_this("hard")
add_this("many")

short_real_names = {
    "d"  : "N",
    "e"  : "E",
    "n"  : "P",
    "f"  : "H",
    "i"  : "I",
    "en" : "EP",
    "ef" : "EH",
    "ei" : "EI",
    "ni" : "PI",
    "fi" : "HI",
    "eni" : "EPI",
    "efi" : "EHI",
}



def load_dicts(args):
    if(os.getcwd().split("/")[-1] != "saved"): os.chdir("saved")
    plot_dicts = [] ; min_max_dicts = []
        
    complete_order = args.arg_title[3:-3].split("+")
    order = [o for o in complete_order if not o in ["empty_space", "break"]]
    
    for name in order:
        got_plot_dicts = False ; got_min_max_dicts = False
        while(not got_plot_dicts):
            try:
                with open(name + "/" + "plot_dict.pickle", "rb") as handle: 
                    plot_dicts.append(pickle.load(handle)) ; got_plot_dicts = True
            except: print("Stuck trying to get {}'s plot_dicts...".format(name)) ; sleep(1)
        while(not got_min_max_dicts):
            try:
                with open(name + "/" + "min_max_dict.pickle", "rb") as handle: 
                    min_max_dicts.append(pickle.load(handle)) ; got_min_max_dicts = True 
            except: print("Stuck trying to get {}'s min_max_dicts...".format(name)) ; sleep(1)
    
    min_max_dict = {}
    for key in plot_dicts[0].keys():
        if(not key in ["args", "arg_title", "arg_name", "episode_lists", "agent_lists", "spot_names", "steps"]):
            minimum = None ; maximum = None
            for mm_dict in min_max_dicts:
                if(mm_dict[key] != (None, None)):
                    if(minimum == None):             minimum = mm_dict[key][0]
                    elif(minimum > mm_dict[key][0]): minimum = mm_dict[key][0]
                    if(maximum == None):             maximum = mm_dict[key][1]
                    elif(maximum < mm_dict[key][1]): maximum = mm_dict[key][1]
            min_max_dict[key] = (minimum, maximum)
            
    final_complete_order = [] ; final_plot_dicts = []

    for arg_name in complete_order: 
        if(arg_name in ["break", "empty_space"]): 
            final_complete_order.append(arg_name)
        else:
            for plot_dict in plot_dicts:
                if(plot_dict["args"].arg_name == arg_name):    
                    final_complete_order.append(arg_name) 
                    final_plot_dicts.append(plot_dict)
                    
    while(len(final_complete_order) > 0 and final_complete_order[0] in ["break", "empty_space"]): 
        final_complete_order.pop(0)              
    
    return(plot_dicts, min_max_dict, complete_order, final_plot_dicts)
# %%



