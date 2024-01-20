from utils import args, duration, load_dicts
print("name:\n{}".format(args.arg_name))

def plot_episodes(complete_order, plot_dicts):
    pass

plot_dicts, min_max_dict, complete_order = load_dicts(args)
print("\nDuration: {}. Done!".format(duration()))