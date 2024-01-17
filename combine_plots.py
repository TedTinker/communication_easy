import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

import os
import re

from utils import args, duration, print, real_names

print("name:\n{}".format(args.arg_name))

os.chdir("saved/thesis_pics")

files = os.listdir() ; files.sort()
rewards_files = [file for file in files if file.startswith("rewards")]
exits_files = [file for file in files if file.startswith("exits")]
arg_names = ["_".join(rewards.split("_")[1:])[:-4] for rewards in rewards_files]
too_many_plot_dicts = len(arg_names) > 20

paths_files = []
for arg_name in arg_names:
    paths_files.append([file for file in files if re.match(r"paths_{}_\d+.png".format(re.escape(arg_name)), file)])



"""
for (arg_name, rewards, exits, paths_list) in zip(arg_names, rewards_files, exits_files, paths_files):
    print(arg_name)
    if(len(paths_list) == 1):
        fig, axs = plt.subplots(3, 1, figsize = (3, 9) if too_many_plot_dicts else (10, 30))
        axs[0].imshow(plt.imread(rewards))       ; axs[0].axis("off")
        axs[1].imshow(plt.imread(exits))         ; axs[1].axis("off")
        axs[2].imshow(plt.imread(paths_list[0])) ; axs[2].axis("off")
    else:
        rows = max([2, len(paths_list)])
        columns = 2
        fig, axs = plt.subplots(rows, columns, figsize = (10 * columns, 10 * rows))
        axs[0,0].imshow(plt.imread(rewards))
        axs[1,0].imshow(plt.imread(exits))   
        for i, paths in enumerate(paths_list):
            axs[i,1].imshow(plt.imread(paths_list[i]))
        for row in range(rows):
            for column in range(columns):
                axs[row, column].axis("off")

    if(arg_name in real_names.keys()): title = real_names[arg_name]
    elif(arg_name.endswith("rand")):   title = "with Curiosity Traps"
    else:                              title = arg_name
    fig.suptitle(title, fontsize=30, y=1.0)
    fig.tight_layout(pad=1.0)
    plt.savefig("{}.png".format(arg_name), bbox_inches = "tight", dpi=600)
    plt.close(fig)
    
    #os.remove(rewards)
    #os.remove(exits)
    #for paths in paths_list:
    #    os.remove(paths)
    print("Done with", arg_name)
"""



def draw_lines(fig, x_start = .15, x_end = .88, x_off_1 = .035, x_off_2 = -.0425, y_start = .13, y_end = .87, y_off = .0125):
    # Drawing black lines between every row
    line = Line2D([x_start, x_end], [1/3 + x_off_1, 1/3 + x_off_1], transform=fig.transFigure, color='black', linewidth=1)
    fig.add_artist(line)
    line = Line2D([x_start, x_end], [2/3 + x_off_2, 2/3 + x_off_2], transform=fig.transFigure, color='black', linewidth=1)
    fig.add_artist(line)

    # Drawing black lines between every other column
    line = Line2D([1/2 + y_off, 1/2 + y_off], [y_start, y_end], transform=fig.transFigure, color='black', linewidth=1)
    fig.add_artist(line)

    

names = ["d_hard", "d_hard_rand", "e_hard", "e_hard_rand", "n_hard", "n_hard_rand", "en_hard", "en_hard_rand", "f_hard", "f_hard_rand", "ef_hard", "ef_hard_rand"]
poses = [(0,0),    (0,1),         (0,2),    (0,3),         (1,0),    (1,1),         (1,2),    (1,3),           (2,0),    (2,1),         (2,2),     (2,3)]
fig, axs = plt.subplots(3, 4, figsize = (10, 7) if too_many_plot_dicts else (10, 7))
fig.suptitle("Biased T-Maze Trajectories", y = .9, fontsize = 10)

for i, arg_name in enumerate(names):
    if(arg_name in arg_names): 
        (x,y) = poses[names.index(arg_name)]
        ax = axs[x, y]
        ax.imshow(plt.imread("paths_{}_0.png".format(arg_name)))       ; ax.axis("off")
        
draw_lines(fig)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("final/paths_hard.png", bbox_inches = "tight", dpi=300)
plt.close(fig)



names = ["d_hard", "d_hard_rand", "e_hard", "e_hard_rand", "n_hard", "n_hard_rand", "en_hard", "en_hard_rand", "f_hard", "f_hard_rand", "ef_hard", "ef_hard_rand"]
poses = [(0,0),    (0,1),         (0,2),    (0,3),         (1,0),    (1,1),         (1,2),    (1,3),           (2,0),    (2,1),         (2,2),     (2,3)]
fig, axs = plt.subplots(3, 4, figsize = (10, 7) if too_many_plot_dicts else (10, 7))
fig.suptitle("Biased T-Maze Exit Choices", y = .9, fontsize = 10)

for i, arg_name in enumerate(names):
    if(arg_name in arg_names): 
        (x,y) = poses[names.index(arg_name)]
        ax = axs[x, y]
        ax.imshow(plt.imread("exits" + "_" + arg_name + ".png"))              ; ax.axis("off")
        
draw_lines(fig)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("final/exits_hard.png", bbox_inches = "tight", dpi=300)
plt.close(fig)



names = ["d_many", "d_many_rand", "e_many", "e_many_rand", "n_many", "n_many_rand", "en_many", "en_many_rand", "f_many", "f_many_rand", "ef_many", "ef_many_rand"]
poses = [(0,0),    (0,1),         (0,2),    (0,3),         (1,0),    (1,1),         (1,2),    (1,3),           (2,0),    (2,1),         (2,2),     (2,3)]
fig = plt.figure(figsize=(7, 10))
small = .57 ; med = .792
gs = gridspec.GridSpec(12, 4, height_ratios=[.05, small, med, 1, .05, small, med, 1, .05, small, med, 1], width_ratios=[1,1,1,1])
fig.suptitle("Expanding T-Maze Trajectories", y = .9, fontsize = 10)

for i, arg_name in enumerate(names):
    if(arg_name in arg_names): 
        (x,y) = poses[names.index(arg_name)]
        all_paths = ["paths_{}_{}.png".format(arg_name, j) for j in range(3)]
        for j in range(4):
            ax = fig.add_subplot(gs[4*x + j, y])
            if(j == 0): pass 
            else:
                ax.imshow(plt.imread(all_paths[j-1]))       
            if(j == 1):
                title = real_names[arg_name] if arg_name in real_names else "with Curiosity Traps"
                ax.set_title(title, fontsize = 5, y = .86)
            ax.axis("off")
            
draw_lines(fig)
            
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("final/paths_many.png", bbox_inches = "tight", dpi=300)
plt.close(fig)



names = ["d_many", "d_many_rand", "e_many", "e_many_rand", "n_many", "n_many_rand", "en_many", "en_many_rand", "f_many", "f_many_rand", "ef_many", "ef_many_rand"]
poses = [(0,0),    (0,1),         (0,2),    (0,3),         (1,0),    (1,1),         (1,2),    (1,3),           (2,0),    (2,1),         (2,2),     (2,3)]
fig, axs = plt.subplots(3, 4, figsize = (10, 7) if too_many_plot_dicts else (10, 7))
fig.suptitle("Expanding T-Maze Exit Choices", y = .9, fontsize = 10)

for i, arg_name in enumerate(names):
    if(arg_name in arg_names): 
        (x,y) = poses[names.index(arg_name)]
        ax = axs[x, y]
        image = plt.imread("exits" + "_" + arg_name + ".png")
        height, width = image.shape[:2]
        ax.imshow(image)              ; ax.axis("off")
        
draw_lines(fig)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("final/exits_many.png", bbox_inches = "tight", dpi=300)
plt.close(fig)
    
    

print("\nDuration: {}. Done!".format(duration()))