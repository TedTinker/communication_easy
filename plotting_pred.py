import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches

import torch
from torch.distributions import Normal
import torch.nn.functional as F

from utils import args, duration, load_dicts, print

print("name:\n{}".format(args.arg_name))



def easy_plotting_pred(complete_order, plot_dicts):
    epochs_maze_names = list(set(["_".join(key.split("_")[1:]) for key in plot_dicts[0]["pred_lists"].keys()]))
    epochs_maze_names.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pred_lists"].keys()])) ; agents.sort()
    first_arena_name = plot_dicts[0]["args"].maze_list[0] 
    episodes = len(plot_dicts[0]["pred_lists"]["1_0_{}".format(first_arena_name)])
    
    cmap = plt.cm.get_cmap("gray_r")
    norm = Normalize(vmin = -1, vmax = 1)
    handles = []
    for c in [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1]:
        handle = plt.scatter(0, 0, marker = "s", s = 250, facecolor = cmap(norm(c)))
        handles.append(handle)
    plt.close()
        
    for epoch_maze_name in epochs_maze_names:
        epoch, maze_name = epoch_maze_name.split("_")
        for agent in agents:
            for arg_name in complete_order:
                if(arg_name in ["break", "empty_space"]): pass 
                else:
                    for plot_dict in plot_dicts:
                        if(plot_dict["arg_name"] == arg_name): pred_lists = plot_dict["pred_lists"]["{}_{}_{}".format(agent, epoch, maze_name)] ; break 
                    obs_size = 12 + plot_dict["args"].randomness
                    for episode in range(episodes):
                        pred_list = pred_lists[episode]
                        rows = len(pred_list) ; columns = 3
                        fig, axs = plt.subplots(rows, columns, figsize = (columns * 3, rows * 1.5))
                        title = "Agent {}: Epoch {} (Maze {}), Episode {}".format(agent, epoch, maze_name, episode)
                        fig.suptitle(plot_dict["arg_title"] + "\n" + title, y = 1.1)
                        for row, (action_name, obs, zp_pred, zq_pred) in enumerate(pred_list):
                            for column in range(columns):
                                ax = axs[row, column] ; ax.axis("off")
                                if(row == 0 and column > 0): pass
                                else:                
                                    # Actual obs
                                    if(column == 0):   
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = obs, norm = norm)
                                        ax.set_title("Step {}\nAction: {}".format(row, action_name))
                                    # ZP Sample
                                    elif(column == 1): 
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = torch.tanh(zp_pred), norm = norm)
                                        ax.set_title("Prior")
                                    # ZQ Sample
                                    elif(column == 2):
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = torch.tanh(zq_pred), norm = norm)
                                        ax.set_title("Posterior")
                        plt.savefig("{}/{}.png".format(arg_name, title), format = "png", bbox_inches = "tight", dpi=300)
                        plt.close()
                        
        print("{}:\tDone with easy epoch {}.".format(duration(), epoch))
                                
                                                
        
def hard_plotting_pred(complete_order, plot_dicts):
    too_many_plot_dicts = len(plot_dicts) > 20
    epochs_maze_names = list(set(["_".join(key.split("_")[1:]) for key in plot_dicts[0]["pred_lists"].keys()]))
    epochs_maze_names.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pred_lists"].keys()])) ; agents.sort()
    first_arena_name = plot_dicts[0]["args"].maze_list[0] 
    episodes = len(plot_dicts[0]["pred_lists"]["1_0_{}".format(first_arena_name)])
        
    for epoch_maze_name in epochs_maze_names:
        epoch, maze_name = epoch_maze_name.split("_")
        for agent in agents:
            for arg_name in complete_order:
                if(arg_name in ["break", "empty_space"]): pass 
                else:
                    for plot_dict in plot_dicts:
                        if(plot_dict["arg_name"] == arg_name): pred_lists = plot_dict["pred_lists"]["{}_{}_{}".format(agent, epoch, maze_name)] ; break 
                    for episode in range(episodes):
                        pred_list = pred_lists[episode]
                        rows = len(pred_list) ; columns = 3
                        fig, axs = plt.subplots(rows, columns, figsize = (columns * 2, rows * 2.5))
                        title = "Agent {}: Epoch {} (Maze {}), Episode {}".format( agent, epoch, maze_name, episode)
                        fig.suptitle(plot_dict["arg_title"] + "\n" + title, y = 1.1)
                        for row, (action_name, (rgbd, spe), (pred_rgbd_p, pred_spe_p), (pred_rgbd_q, pred_spe_q)) in enumerate(pred_list):
                            for column in range(columns):
                                ax = axs[row, column] ; ax.axis("off")
                                if(row == 0 and column > 0): pass
                                else:                
                                    steps_per_step = plot_dict["args"].steps_per_step
                                    # Actual obs
                                    if(column == 0):   
                                        while(len(rgbd.shape) > 3): rgbd = rgbd.squeeze(0)
                                        ax.imshow(rgbd[:,:,0:3])
                                        ax.set_title("Step {}\nAction: {}\nSpeed {}".format(row, action_name, steps_per_step*round(spe.item())), fontsize = 12)
                                    # ZP Sample
                                    elif(column == 1): 
                                        while(len(pred_rgbd_p.shape) > 3): pred_rgbd_p = pred_rgbd_p.squeeze(0)
                                        ax.imshow(torch.sigmoid(pred_rgbd_p[:,:,0:3])) 
                                        ax.set_title("Prior\nSpeed {}".format(steps_per_step*round(pred_spe_p.item())), fontsize = 12)
                                    # ZQ Sample
                                    elif(column == 2):
                                        while(len(pred_rgbd_q.shape) > 3): pred_rgbd_q = pred_rgbd_q.squeeze(0)
                                        ax.imshow(torch.sigmoid(pred_rgbd_q[:,:,0:3]))
                                        ax.set_title("Posterior\nSpeed {}".format(steps_per_step*round(pred_spe_q.item())), fontsize = 12)
                                    xlim = ax.get_xlim()
                                    ylim = ax.get_ylim()

                                    # Create a rectangle patch with a black edge that matches the image axes' limits
                                    # The rectangle is positioned at (0,0) with the width and height matching the image dimensions
                                    rect = patches.Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0], linewidth=2, edgecolor='black', facecolor='none')
                                    ax.add_patch(rect)
                        plt.savefig("{}/{}.png".format(arg_name, title), format = "png", bbox_inches = "tight", dpi=300)
                        plt.close()
                        
        print("{}:\tDone with hard epoch {}.".format(duration(), epoch))
    


plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)    
if(hard): print("\nPlotting predictions in hard maze(s).\n") ; hard_plotting_pred(complete_hard_order, hard_plot_dicts)    
print("\nDuration: {}. Done!".format(duration()))