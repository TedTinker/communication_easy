import matplotlib.pyplot as plt

from utils import print, args, duration, load_dicts, multihots_to_string, onehots_to_string, action_to_string
print("name:\n{}".format(args.arg_name))

def plot_episodes(complete_order, plot_dicts):
    for arg_name in complete_order:
        if(arg_name in ["break", "empty_space"]): pass 
        else:
            for plot_dict in plot_dicts:
                if(plot_dict["arg_name"] == arg_name):
                    episode_dicts = plot_dict["episode_dicts"]
                    for key, episode_dict in episode_dicts.items():
                        plot_episode(key, episode_dict, arg_name)
                        
def plot_episode(key, episode_dict, arg_name):
    agent_num, epoch, episode_num = key.split("_")

    steps = len(episode_dict["objects"])
    fig, axs = plt.subplots(steps, 8, figsize=(20, 2 * steps))
    fig.suptitle(f"Epoch {epoch}")

    for step in range(steps):
        for ax_idx in range(8):
            ax = axs[step, ax_idx]
            ax.axis('off')  

            if ax_idx == 0:
                objects = episode_dict["objects"][step]
                text = multihots_to_string(objects)
            elif ax_idx == 1:
                comms = episode_dict["comms"][step]
                text = onehots_to_string(comms)
            elif step + 1 != steps:
                if ax_idx == 2:
                    actions = episode_dict["actions"][step]
                    text = action_to_string(actions)
                elif ax_idx == 3:
                    rewards = episode_dict["rewards"][step]
                    text = str(rewards)
                elif ax_idx == 4:
                    objects_p = episode_dict["prior_predicted_objects"][step]
                    text = multihots_to_string(objects_p)
                elif ax_idx == 5:
                    comms_p = episode_dict["prior_predicted_comms"][step]
                    text = onehots_to_string(comms_p)
                elif ax_idx == 6:
                    objects_q = episode_dict["posterior_predicted_objects"][step]
                    text = multihots_to_string(objects_q)
                elif ax_idx == 7:
                    comms_q = episode_dict["posterior_predicted_comms"][step]
                    text = onehots_to_string(comms_q)
            else:
                text = ''
                
            text = text.replace('\t', ' ')
            text = text.replace('(', ' ')
            text = text.replace(')', ' ')

            # Add text to the subplot with a buffer for margins and word wrapping
            ax.text(0.1, 0.5 - .2 * ax_idx, text, va='center', ha='left', wrap=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent overlap
    plt.savefig(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}.png")
    plt.close()

plot_dicts, min_max_dict, complete_order = load_dicts(args)
plot_episodes(complete_order, plot_dicts)
print("\nDuration: {}. Done!".format(duration()))