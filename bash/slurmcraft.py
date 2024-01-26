#%%
from copy import deepcopy
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument("--comp",         type=str,  default = "deigo")
parser.add_argument("--agents",       type=int,  default = 10)
parser.add_argument("--arg_list",     type=str,  default = [])
try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()

if(type(args.arg_list) != list): args.arg_list = json.loads(args.arg_list)
combined = "___{}___".format("+".join(args.arg_list))    

import os 
try:    os.chdir("communication_easy/bash")
except: pass



from itertools import product
def expand_args(name, args):
    combos = [{}]
    complex = False
    for key, value in args.items():
        if(type(value) != list):
            for combo in combos:
                combo[key] = value
        else: 
            complex = True
            if(value[0]) == "num_min_max": 
                num, min_val, max_val = value[1]
                num = int(num)
                min_val = float(min_val)
                max_val = float(max_val)
                value = [min_val + i*((max_val - min_val) / (num - 1)) for i in range(num)]
            new_combos = []
            for v in value:
                temp_combos = deepcopy(combos)
                for combo in temp_combos: 
                    combo[key] = v        
                    new_combos.append(combo)   
            combos = new_combos  
    if(complex and name[-1] != "_"): name += "_"
    return(name, combos)

slurm_dict = {
    "d"     : {}, 
    }




def add_this(name, args):
    keys, values = [], []
    for key, value in slurm_dict.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        if(key == "d"): key = ""
        between = "" if key == "" or len(name) == 1 else "_"
        new_key = key + between + name 
        new_value = deepcopy(value)
        for arg_name, arg in args.items():
            if(type(arg) != list): new_value[arg_name] = arg
            elif(type(arg[0]) != list): new_value[arg_name] = arg
            else:
                for condition in arg:
                    for if_arg_name, if_arg in condition[0].items():
                        if(if_arg_name in value and value[if_arg_name] == if_arg):
                            new_value[arg_name] = condition[1]
        slurm_dict[new_key] = new_value



add_this("e",   {"alpha" : "None"})
add_this("n",   {"curiosity" : "prediction_error"})
add_this("f",   {"curiosity" : "hidden_state"})
add_this("i",   {"delta" : 1})



new_slurm_dict = {}
for key, value in slurm_dict.items():
    key, combos = expand_args(key, value)
    if(len(combos) == 1): new_slurm_dict[key] = combos[0] 
    else:
        for i, combo in enumerate(combos): new_slurm_dict[key + str(i+1)] = combo
        
slurm_dict = new_slurm_dict

def get_args(name):
    s = "" 
    for key, value in slurm_dict[name].items(): s += "--{} {} ".format(key, value)
    return(s)

def all_like_this(this): 
    if(this in ["break", "empty_space"]): result = [this]
    elif(this[-1] != "_"):                result = [this]
    else: result = [key for key in slurm_dict.keys() if key.startswith(this) and key[len(this):].isdigit()]
    return(json.dumps(result))
            


        
if(__name__ == "__main__" and args.arg_list == []):
    for key, value in slurm_dict.items(): print(key, ":", value,"\n")
    interesting = ["efi_{}".format(i) for i in [7, 28, 35]]
    for this in interesting:
        print("{} : {}".format(this,slurm_dict[this]))

max_cpus = 36
if(__name__ == "__main__" and args.arg_list != []):
    
    if(args.comp == "deigo"):
        nv = ""
        module = "module load singularity"
        partition = \
"""
#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 5:00:00
#SBATCH --mem=50G"""

    if(args.comp == "saion"):
        nv = "--nv"
        module = "module load singularity cuda"
        partition = \
"""
#!/bin/bash -l
#SBATCH --partition=taniu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=490G
#SBATCH --gres=gpu:4"""
    for name in args.arg_list:
        if(name in ["break", "empty_space"]): pass 
        else:
            with open("main_{}.slurm".format(name), "w") as f:
                f.write(
"""
{}
#SBATCH --ntasks={}
{}
singularity exec {} maze.sif python communication_easy/main.py --comp {} --arg_name {} {} --agents $agents_per_job --previous_agents $previous_agents
""".format(partition, max_cpus, module, nv, args.comp, name, get_args(name))[2:])
            


    with open("finish_dicts.slurm", "w") as f:
        f.write(
"""
{}
{}
singularity exec {} maze.sif python communication_easy/finish_dicts.py --comp {} --arg_title {} --arg_name finishing_dictionaries
""".format(partition, module, nv, args.comp, combined)[2:])
        
    with open("plotting.slurm", "w") as f:
        f.write(
"""
{}
{}
singularity exec {} maze.sif python communication_easy/plotting.py --comp {} --arg_title {} --arg_name plotting
""".format(partition, module, nv, args.comp, combined)[2:])
        
    with open("plotting_episodes.slurm", "w") as f:
        f.write(
"""
{}
{}
singularity exec {} maze.sif python communication_easy/plotting_episodes.py --comp {} --arg_title {} --arg_name plotting_episodes
""".format(partition, module, nv, args.comp, combined)[2:])
        
    with open("plotting_p_values.slurm", "w") as f:
        f.write(
"""
{}
{}
singularity exec {} maze.sif python communication_easy/plotting_p_val.py --comp {} --arg_title {} --arg_name plotting_p_values
""".format(partition, module, nv, args.comp, combined)[2:])
        
    with open("combine_plots.slurm", "w") as f:
        f.write(
"""
{}
{}
singularity exec {} maze.sif python communication_easy/combine_plots.py --comp {} --arg_title {} --arg_name combining_plots
""".format(partition, module, nv, args.comp, combined)[2:])
# %%

