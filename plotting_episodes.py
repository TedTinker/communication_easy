from utils import args, duration, load_dicts
print("name:\n{}".format(args.arg_name))

plot_dicts, min_max_dict, complete_order, plot_dicts = load_dicts(args)
print("\nDuration: {}. Done!".format(duration()))