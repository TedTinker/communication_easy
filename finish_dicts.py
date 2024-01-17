import os, pickle

from utils import duration, args, print

print("name:\n{}".format(args.arg_name))

os.chdir("saved")
folders = os.listdir() ; folders.sort()
try: folders.remove("thesis_pics")
except: pass 
print("\n{} folders.".format(len(folders)))

plot_dicts = {}
for folder in folders:
    plot_dict = {} ; min_max_dict = {}
    files = os.listdir(folder) ; files.sort()
    print("{} files in folder {}.".format(len(files), folder))
    for file in files:
        if(file.split("_")[0] == "plot"): d = plot_dict    ; plot = True 
        if(file.split("_")[0] == "min"):  d = min_max_dict ; plot = False
        with open(folder + "/" + file, "rb") as handle: 
            saved_d = pickle.load(handle) ; os.remove(folder + "/" + file)
        for key in saved_d.keys(): 
            if(not key in d): d[key] = []
            if(key in ["args", "arg_title", "arg_name"]): d[key] = saved_d[key]
            else:  d[key].append(saved_d[key])
            
    pred_lists = {}
    for d in plot_dict["pred_lists"]: pred_lists.update(d)
    plot_dict["pred_lists"] = pred_lists
            
    pos_lists = {}
    for d in plot_dict["pos_lists"]: pos_lists.update(d)
    plot_dict["pos_lists"] = pos_lists
    
    agent_lists = {}
    for d in plot_dict["agent_lists"]: agent_lists.update(d)
    plot_dict["agent_lists"] = agent_lists
        
    for key in min_max_dict.keys():
        if(not key in ["args", "arg_title", "arg_name", "pred_lists", "pos_lists", "agent_lists", "spot_names", "steps"]):
            if(key == "hidden_state"):
                min_maxes = []
                for layer in min_max_dict[key]:
                    minimum = None ; maximum = None
                    for min_max in layer:
                        if(  minimum == None):      minimum = min_max[0]
                        elif(minimum > min_max[0]): minimum = min_max[0]
                        if(  maximum == None):      maximum = min_max[1]
                        elif(maximum < min_max[1]): maximum = min_max[1]
                    min_maxes.append((minimum, maximum))
                min_max_dict[key] = min_maxes
            else:
                minimum = None ; maximum = None
                for min_max in min_max_dict[key]:
                    if(  minimum == None):      minimum = min_max[0]
                    elif(minimum > min_max[0]): minimum = min_max[0]
                    if(  maximum == None):      maximum = min_max[1]
                    elif(maximum < min_max[1]): maximum = min_max[1]
                min_max_dict[key] = (minimum, maximum)

    plot_dicts[folder] = plot_dict
    with open(folder + "/min_max_dict.pickle", "wb") as handle:
        pickle.dump(min_max_dict, handle)
            
for folder, plot_dict in plot_dicts.items():
    print("Saving folder {}.".format(folder))
    with open(folder + "/plot_dict.pickle", "wb") as handle:
        pickle.dump(plot_dict, handle)
    
print("\nDuration: {}. Done!\n".format(duration()))