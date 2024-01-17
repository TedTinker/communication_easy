#%% 
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from statsmodels.stats.proportion import proportions_ztest as ztest
from scipy.stats import norm
import scipy.stats as stats
from copy import deepcopy
import statistics
import math

from utils import args, duration, load_dicts, print, maze_real_names, short_real_names as real_names

print("name:\n{}\n".format(args.arg_name),)



confidence = .99
prev_episodes = 10



def confidence_interval_proportions(proportions, confidence = confidence):
    mean = statistics.mean(proportions)
    std_dev = statistics.stdev(proportions)
    n = len(proportions)
    
    z = norm.ppf(1 - (1 - confidence) / 2)
    margin_of_error = z * (std_dev / math.sqrt(n))
    
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return (lower_bound, upper_bound)

def z_test_proportions(list1, list2):
    p1 = statistics.mean(list1)
    p2 = statistics.mean(list2)
    pooled_p = (sum(list1) + sum(list2)) / (len(list1) + len(list2))
    se = math.sqrt(pooled_p * (1 - pooled_p) * ((1/len(list1)) + (1/len(list2))))
    z = (p1 - p2) / se
    p_value = 1 - norm.cdf(abs(z))
    return p_value
     


def hard_p_values(complete_order, plot_dicts):
    too_many_plot_dicts = len(plot_dicts) > 20
    arg_names = [arg_name for arg_name in complete_order if not arg_name in ["break", "empty_space"]]
    
    real_arg_names = []
    for arg_name in arg_names:
        if arg_name.endswith("rand"): real_name = "w/ traps"
        elif(arg_name in real_names): real_name = real_names[arg_name]
        else:                         real_name = arg_name
        real_arg_names.append(real_name)
    reversed_names = deepcopy(real_arg_names)
    reversed_names.reverse()
    total_epochs = 0
    
    p_value_dicts = {}
    for maze_name, epochs in zip(plot_dicts[0]["args"].maze_list, plot_dicts[0]["args"].epochs):
        p_value_dicts[(maze_name, epochs)] = {}

        for (x, arg_1) in enumerate(arg_names):
            for plot_dict in plot_dicts:
                if(plot_dict["args"].arg_name == arg_1): 
                    all_agent_spots = [] 
                    for spot_names in plot_dict["spot_names"]:
                        final_spots = spot_names[epochs + total_epochs - prev_episodes - 1 : epochs + total_epochs - 1]
                        final_spots = [1 if spot in ["RIGHT", "R", "LR", "RLL"] else 0 for spot in final_spots]
                        final_spots = sum(final_spots)/prev_episodes
                        all_agent_spots.append(final_spots)

            prop = sum(all_agent_spots) / len(all_agent_spots)
            lower_bound, upper_bound = confidence_interval_proportions(all_agent_spots)
            print(lower_bound, prop, upper_bound)
            print(upper_bound - lower_bound)
            
            p_value_dicts[(maze_name, epochs)][arg_1] = [x, all_agent_spots, prop, lower_bound, upper_bound]
            
        total_epochs += epochs
    

             
    kinds = ["d", "e", "n", "f", "en", "ef"]

    total_epochs = 0
    for (maze_name, epochs), p_value_dict in p_value_dicts.items():
        all_vals = []
        for kind in kinds:
            for arg_1, (x, all_agent_spots, prop, lower_bound, upper_bound) in p_value_dict.items():
                base_arg_1 = arg_1.split("_")[0]
                if(arg_1.split("_")[-1] != "rand" and base_arg_1 == kind):
                    if(arg_1 in real_names):   real_name_1 = real_names[arg_1]
                    else:                      real_name_1 = arg_1
                    all_vals.append([real_name_1, prop, (lower_bound, upper_bound)])
                    break
        all_names = [val[0] for val in all_vals]
        all_heights = [val[1] for val in all_vals]
        all_conf = [val[2] for val in all_vals]
        all_colors = ["white" for val in all_vals]

        x = .1
        bar_width = 0.4  
        spacing = 0.5  
        
        plt.figure(figsize = (6, 5))
        fig, ax = plt.subplots()
        ax = plt.gca()

        for i in range(len(all_heights)):
            bar_center = x + bar_width / 2
            ax.add_patch(patches.Rectangle((x, 0), bar_width, all_heights[i], facecolor=all_colors[i], edgecolor="black"))
            ax.text(x + bar_width/2, -.02, all_names[i], ha='center', va='top', rotation=90, fontsize=20)
            ax.plot([bar_center, bar_center], [all_conf[i][0], all_conf[i][1]], color="black")
            ax.plot([bar_center - 0.02, bar_center + 0.02], [all_conf[i][0], all_conf[i][0]], color="black")
            ax.plot([bar_center - 0.02, bar_center + 0.02], [all_conf[i][1], all_conf[i][1]], color="black")
            if i in [0, 3]:
                ax.axvline(x = x + bar_width + (spacing - bar_width)/2, color='black', linestyle='dotted')
                # Can we also have a light-gray dotted line to show which bar in each area is bigger?
            x += spacing  

        ax.set_xlim(0, x)
        ax.set_ylim(0, 1.1)  
        ax.set_ylabel('Proportion of Correct Exits')
        plt.title("Agent Performance\n(Epoch {}, {})".format(epochs + total_epochs, maze_real_names[maze_name])) 
        ax.axes.get_xaxis().set_visible(False)  # Hide the x-axis

        plt.savefig("{}_p_values_hypothesis_1.png".format(maze_name), format = "png", bbox_inches = "tight", dpi=300)
        plt.close()
        total_epochs += epochs
                
        
        
    total_epochs = 0
    for (maze_name, epochs), p_value_dict in p_value_dicts.items():
        all_vals = []
        for arg_1, (x, all_agent_spots_1, prop_1, lower_bound_1, upper_bound_1) in p_value_dict.items():
            for arg_2, (x, all_agent_spots_2, prop_2, lower_bound_2, upper_bound_2) in p_value_dict.items():
                if(("n" in arg_1.split("_")[0] or "f" in arg_1.split("_")[0]) and arg_2 == arg_1 + "_rand"):
                    if arg_1.endswith("rand"): real_name_1 = "w/ traps"
                    elif(arg_1 in real_names): real_name_1 = real_names[arg_1]
                    else:                      real_name_1 = arg_1
                    if arg_2.endswith("rand"): real_name_2 = "w/ traps"
                    elif(arg_2 in real_names): real_name_2 = real_names[arg_2]
                    else:                      real_name_2 = arg_2
                    p = z_test_proportions(all_agent_spots_1, all_agent_spots_2)
                    print(real_name_1, real_name_2, p, 1 - confidence, p < 1-confidence, prop_1 > prop_2)
                    all_vals.append([real_name_1, prop_1, (lower_bound_1, upper_bound_1), prop_2, (lower_bound_2, upper_bound_2), p < 1-confidence])
        all_names = sum([[val[0], "w/ traps"] for val in all_vals], [])
        all_heights = sum([[val[1], val[3]] for val in all_vals], [])
        all_conf = sum([[val[2], val[4]] for val in all_vals], [])
        all_colors = sum([["white", "white"] if not val[-1] else ["green", "red"] if val[1] > val[3] else ["white", "white"] for val in all_vals], []) 
        fig, ax = plt.subplots()

        x = .1
        bar_width = 0.4  
        spacing = 0.5  
        
        plt.figure(figsize = (6, 5))
        ax = plt.gca()

        for i in range(len(all_heights)):
            bar_center = x + bar_width / 2
            ax.add_patch(patches.Rectangle((x, 0), bar_width, all_heights[i], facecolor=all_colors[i], edgecolor="black"))
            ax.text(x + bar_width/2, -.02, all_names[i], ha='center', va='top', rotation=90, fontsize=20)
            ax.plot([bar_center, bar_center], [all_conf[i][0], all_conf[i][1]], color="black")
            ax.plot([bar_center - 0.02, bar_center + 0.02], [all_conf[i][0], all_conf[i][0]], color="black")
            ax.plot([bar_center - 0.02, bar_center + 0.02], [all_conf[i][1], all_conf[i][1]], color="black")
            if i in [3]:
                ax.axvline(x = x + bar_width + (spacing - bar_width)/2, color='black', linestyle='dotted')
            if(i % 2 != 0):
                x += spacing  
            else:
                x += bar_width

        ax.set_xlim(0, x)
        ax.set_ylim(0, 1.1)  # Adding 1 for a bit of padding at the top
        ax.set_ylabel('Proportion of Correct Exits')
        plt.title("Impact of Traps\n(Epoch {}, {})".format(epochs + total_epochs, maze_real_names[maze_name]))   
        ax.axes.get_xaxis().set_visible(False)  # Hide the x-axis

        plt.savefig("{}_p_values_hypothesis_2.png".format(maze_name), format = "png", bbox_inches = "tight", dpi=300)
        plt.close()
        total_epochs += epochs
        


plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)               
if(hard): print("\nPlotting p-values in hard maze(s).\n") ; hard_p_values(complete_hard_order, hard_plot_dicts)   
print("\nDuration: {}. Done!".format(duration()))
# %%
