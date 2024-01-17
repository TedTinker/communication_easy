#%%

import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch.multiprocessing as multiprocessing
import pickle, torch, random
import numpy as np
from torch.multiprocessing import Process, Queue
from time import sleep 
from math import floor

from utils import args, folder, memory_usage, duration, estimate_total_duration, print, device

print("\nname:\n{}".format(args.arg_name))
print("\nagents: {}. previous_agents: {}.".format(args.agents, args.previous_agents))

from agent import Agent



def train(q, i):
    seed = args.init_seed + i
    np.random.seed(seed) ; random.seed(seed) ; torch.manual_seed(seed) ; torch.cuda.manual_seed(seed)
    #if(args.comp == "saion"):
    #    args.device = f'cuda:{i % 4}'
    #memory_usage(args.device)
    agent = Agent(i, args)
    agent.training(q)
    with open(folder + "/plot_dict_{}.pickle".format(   str(i).zfill(3)), "wb") as handle:
        pickle.dump(agent.plot_dict, handle)
    with open(folder + "/min_max_dict_{}.pickle".format(str(i).zfill(3)), "wb") as handle:
        pickle.dump(agent.min_max_dict, handle)

def main():

    queue = Queue()

    processes = []
    for worker_id in range(1 + args.previous_agents, 1 + args.agents + args.previous_agents):
        process = Process(target=train, args=(queue, worker_id))
        processes.append(process)
        process.start()

    progress_dict      = {i : "0"  for i in range(1 + args.previous_agents, 1 + args.agents + args.previous_agents)}
    prev_progress_dict = {i : None for i in range(1 + args.previous_agents, 1 + args.agents + args.previous_agents)}

    while any(process.is_alive() for process in processes) or not queue.empty():
        while not queue.empty():
            worker_id, progress_percentage = queue.get()
            progress_dict[worker_id] = progress_percentage

        if any(progress_dict[key] != prev_progress_dict[key] for key in progress_dict.keys()):
            prev_progress_dict = progress_dict.copy()
            string = "" ; hundreds = 0
            values = list(progress_dict.values()) ; values.sort()
            so_far = duration()
            lowest = float(values[0])
            estimated_total = estimate_total_duration(lowest)
            if(estimated_total == "?:??:??"): to_do = "?:??:??"
            else:                                   to_do = estimated_total - so_far
            values = [str(floor(100 * float(value))).ljust(3, " ") for value in values]
            for value in values:
                if(value != "100"): string += " " + value
                else:               hundreds += 1 
            if(hundreds > 0): string += " ##" + " 100" * hundreds
            string = "{} ({} left):".format(so_far, to_do) + string
            if(hundreds == 0): string += " ##"
            string = string.rstrip() + "."
            print(string)
        sleep(1)

    for process in processes:
        process.join()

    print("\nDuration: {}. Done!".format(duration()))
    
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
# %%