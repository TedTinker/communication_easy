import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import imageio.v2 as imageio
from io import BytesIO
import os
import numpy as np
from skimage.transform import resize
from itertools import product

from utils import args, duration, load_dicts, print, real_names, maze_real_names
from arena import arena_dict

print("name:\n{}".format(args.arg_name))



def easy_plotting_pos(complete_order, plot_dicts):
    from easy_maze import Easy_Maze
    images = []
    rows = 0 ; columns = 0 ; current_count = 0 
    for arg_name in complete_order: 
        if(arg_name == "break"): rows += 1 ; columns = max(columns, current_count) ; current_count = 0
        else: current_count += 1
    columns = max(columns, current_count)
    if(complete_order[-1] != "break"): rows += 1
        
    epochs_maze_names = list(set(["_".join(key.split("_")[1:]) for key in plot_dicts[0]["pos_lists"].keys()]))
    epochs_maze_names.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    first_arena_name = plot_dicts[0]["args"].maze_list[0] 
    steps = len(plot_dicts[0]["pos_lists"]["1_0_{}".format(first_arena_name)][0])
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pos_lists"].keys()])) ; agents.sort()
    episodes = len(plot_dicts[0]["pred_lists"]["1_0_{}".format(first_arena_name)])
    
    cmap = plt.cm.get_cmap("gray_r")
    norm = Normalize(vmin = 0, vmax = 1)
    handles = []
    for c in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
        handle = plt.scatter(0, 0, marker = "s", s = 250, facecolor = cmap(norm(c)))
        handles.append(handle)
    plt.close()
    
    for epoch_maze_name in epochs_maze_names:
        epoch, maze_name = epoch_maze_name.split("_")
        for step in range(steps):
            fig, axs = plt.subplots(rows, columns, figsize = (columns * 3, rows * 1.5))
            fig.suptitle("Epoch {} ({}) Step {}".format(epoch, maze_real_names[maze_name], step), y = 1.1)
            plot_position = (0, 0)
            for arg_name in complete_order:
                if(  arg_name == "break"): plot_position = (plot_position[0] + 1, 0)
                elif(arg_name == "empty_space"): 
                    ax = axs[plot_position[0], plot_position[1]] if rows > 1 else axs[plot_position[1]]
                    ax.axis("off")
                    plot_position = (plot_position[0], plot_position[1] + 1)
                else:
                    for plot_dict in plot_dicts:
                        if(plot_dict["arg_name"] == arg_name): break

                    ax = axs[plot_position[0], plot_position[1]] if rows > 1 else axs[plot_position[1]]
                    positions = [spot.pos for spot in Easy_Maze(maze_name, plot_dict["args"]).maze.spots]
                    for spot in positions:
                        ax.text(spot[0], spot[1], "\u25A1", fontsize = 30, ha = "center", va = "center")
                    to_plot = {}
                    for agent in agents:
                        for episode in range(episodes):
                            coordinate = plot_dict["pos_lists"]["{}_{}_{}".format(agent, epoch, maze_name)][episode][step]
                            if(not coordinate in to_plot): to_plot[coordinate] = 0
                            to_plot[coordinate] += 1
                            
                    total_points = sum(to_plot.values())
                    to_plot = {k: v / total_points for k, v in to_plot.items()}
                    coords, proportions = zip(*to_plot.items())
                    x_coords, y_coords = zip(*coords)
                    
                    if(len(proportions) == 1): ax.scatter(x_coords, y_coords, marker = "s", s = 250, cmap = cmap, c = "black")
                    else:                      ax.scatter(x_coords, y_coords, marker = "s", s = 250, cmap = cmap, c = proportions, norm = norm)
                    ax.set_ylim([-.5, 2.5])
                    ax.set_xlim([-1.5, 3.5])
                    ax.set_title("{}\n{}".format(plot_dict["arg_name"], plot_dict["arg_title"]))
                    ax.axis("off")
                
                    plot_position = (plot_position[0], plot_position[1] + 1)
                
            fig.legend(loc = "upper left", handles = handles, labels= ["{}%".format(p) for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
            buf = BytesIO()
            plt.savefig(buf, format = "png", bbox_inches = "tight")
            buf.seek(0)
            im = imageio.imread(buf)
            images.append(im)
            buf.close()
            plt.close()
            
        print("{}:\tDone with easy epoch {} (maze {}).".format(duration(), epoch, maze_real_names[maze_name]))
            
    x_max = None ; y_max = None 
    for image in images:
        x, y = image.shape[0], image.shape[1]
        x_max = x if x_max == None else max([x_max, x])
        y_max = y if y_max == None else max([y_max, y])
    resized = []
    for image in images: resized.append(resize(image, (x_max, y_max, 4)))
    imageio.mimwrite("easy_video.mp4", resized, fps = 3)
            
            
        
def hard_plotting_pos(complete_order, plot_dicts):
    too_many_plot_dicts = len(plot_dicts) > 20
    os.chdir("..")
    images = []
    rows = 0 ; columns = 0 ; current_count = 0 
    for arg_name in complete_order: 
        if(arg_name == "break"): rows += 1 ; columns = max(columns, current_count) ; current_count = 0
        else: current_count += 1
    columns = max(columns, current_count)
    if(complete_order[-1] != "break"): rows += 1
    figsize = (rows * 3, columns * 3) if too_many_plot_dicts else (rows * 10, columns * 10)
    
    epochs_maze_names = list(set(["_".join(key.split("_")[1:]) for key in plot_dicts[0]["pos_lists"].keys()]))
    epochs_maze_names.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pos_lists"].keys()])) ; agents.sort()
    first_arena_name = plot_dicts[0]["args"].maze_list[0] 
    episodes = len(plot_dicts[0]["pos_lists"]["1_0_{}".format(first_arena_name)])
            
    cmap = plt.cm.get_cmap("hsv", len(agents))
    norm = Normalize(vmin = 0, vmax = len(agents))
    handles = []
    for c in agents:
        handle = plt.scatter(c, 0, marker = "s", s = 10, color=cmap(norm(c)))
        handles.append(handle)
    plt.close()
    
    saved_paths = 0
    mazes_done = 0
    for i, epoch_maze_name in enumerate(epochs_maze_names):
        epoch, maze_name = epoch_maze_name.split("_")
        print(epoch, maze_name, mazes_done)
        if(i+1 != len(epochs_maze_names)):
            next_epoch_maze_name = epochs_maze_names[i+1]
            _, next_maze_name = next_epoch_maze_name.split("_")
        else:
            next_maze_name = None
        fig, axs = plt.subplots(columns, rows, figsize = figsize)
        fig.suptitle("Epoch {} ({})".format(epoch, maze_real_names[maze_name]), y = 1.05)
        plot_position = (0, 0)
        for arg_name in complete_order:
            print(arg_name)
            if(  arg_name == "break"): plot_position = (0, plot_position[1] + 1)
            elif(arg_name == "empty_space"): 
                ax = axs[plot_position[0], plot_position[1]] if rows > 1 else axs[plot_position[1]]
                ax.axis("off")
                plot_position = (plot_position[0] + 1, plot_position[1])
            else:
                for plot_dict in plot_dicts:
                    if(plot_dict["arg_name"] == arg_name): break

                ax = axs[plot_position[0], plot_position[1]] if rows > 1 else axs[plot_position[1]]
                arena_map = plt.imread("arenas/{}.png".format(maze_name))
                arena_map = np.flip(arena_map, 0)    
                h, w, _ = arena_map.shape
                extent = [-.5, w-.5, -h+.5, .5]
                
                def plot_pos(here, title = True):
                    here.imshow(arena_map, extent = extent, origin = "lower", zorder = 1) 
                    for c, agent in enumerate(agents):
                        for episode in range(episodes):
                            path = plot_dict["pos_lists"]["{}_{}_{}".format(agent, epoch, maze_name)][episode][1:]
                            xs = [p[1] for p in path] ; ys = [-p[0] for p in path]
                            here.plot(xs, ys, color=cmap(norm(c)), zorder = 2)
                    exits = arena_dict[maze_name + ".png"].exit_list
                    for exit in exits:
                        y, x = exit.pos
                        y = -y
                        if(exit.rew == "default"): text = "X" ; fontsize = 40 * (1.7544 if mazes_done == 0 else 1.2626 if mazes_done == 1 else 1)
                        if(exit.rew == "better"):  text = "✓" ; fontsize = 50 * (1.7544 if mazes_done == 0 else 1.2626 if mazes_done == 1 else 1)
                        here.text(x, y, text, fontsize = fontsize, ha='center', va='center', zorder = 4)
                    if(plot_dict["args"].random_by_choice):
                        traps = arena_dict[maze_name + ".png"].random_by_choice
                        text = "?" ; fontsize = 40 * (1.7544 if mazes_done == 0 else 1.2626 if mazes_done == 1 else 1)
                        for y, x in traps:
                            here.text(x, -y, text, fontsize = fontsize, ha='center', va='center', zorder = 4)
                    
                    if(title):
                        here.set_title("{}\n{}".format(plot_dict["arg_name"], plot_dict["arg_title"]))
                    here.axis("off")
                
                if(not too_many_plot_dicts):                    
                    plot_pos(ax)
                    title = real_names[arg_name] if arg_name in real_names else "with Curiosity Traps"
                    if(maze_name in ["1", "t"]): title = title
                    else:                        title = ""  
                    ax.set_title(title)  
                if(next_maze_name != maze_name):
                    print("Making thesis_pic")  
                    fig2, ax2 = plt.subplots(figsize = (3, 3) if too_many_plot_dicts else (10, 10))  
                    plot_pos(ax2, False)  
                    title = real_names[arg_name] if arg_name in real_names else "with Curiosity Traps"
                    print("\n\n")
                    print("TITLE:", title)
                    if(maze_name in ["1", "t"]): title = title
                    else:                        title = ""   
                    print("TITLE:", title)
                    if(arg_name.split("_")[1] == "hard"):                   
                        ax2.set_title(title)
                    fig2.savefig("saved/thesis_pics/paths_{}_{}.png".format(plot_dict["arg_name"], saved_paths), bbox_inches = "tight", dpi=300) 
                    plt.close(fig2)            
                plot_position = (plot_position[0] + 1, plot_position[1])
        
        if(next_maze_name != maze_name): 
            mazes_done += 1
            saved_paths += 1
                        
        #fig.legend(loc = "upper left", handles = handles, labels= ["Agent {}".format(agent) for agent in agents])
        buf = BytesIO()
        
        if(not too_many_plot_dicts):
            plt.savefig(buf, format = "png", bbox_inches = "tight")
            buf.seek(0)
            im = imageio.imread(buf)
            images.append(im)
            buf.close()
        else: 
            print("Can't save this many plots!")
        plt.close(fig)
            
        print("{}:\tDone with hard epoch {} (maze {}).".format(duration(), epoch, maze_name))
    
    x_max = None ; y_max = None 
    for image in images:
        x, y = image.shape[0], image.shape[1]
        x_max = x if x_max == None else max([x_max, x])
        y_max = y if y_max == None else max([y_max, y])
    resized = []
    for image in images: resized.append(resize(image, (x_max, y_max, 4)))
    imageio.mimwrite("saved/hard_video.mp4", resized, fps = 1/3)
    
    

plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)               
if(easy): print("\nPlotting positions in easy maze.\n")    ; easy_plotting_pos(complete_easy_order, easy_plot_dicts)
if(hard): print("\nPlotting positions in hard maze(s).\n") ; hard_plotting_pos(complete_hard_order, hard_plot_dicts)   
print("\nDuration: {}. Done!".format(duration()))