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
    print("{}: agent {}, epoch {}, episode {}.".format(arg_name, agent_num, epoch, episode_num))

    steps = len(episode_dict["objects"])
    fig, axs = plt.subplots(steps, 1, figsize=(10, 5 * steps))
    fig.suptitle(f"Epoch {epoch}", fontsize=16)
    
    for step, ax in enumerate(axs):
        ax.axis('off')
        
        text_list = []
        label_list = []

        objects = episode_dict["objects"][step]
        text_list.append(multihots_to_string(objects))
        label_list.append("Objects")

        comms = episode_dict["comms"][step]
        text_list.append(onehots_to_string(comms))
        label_list.append("Comms")
        
        if step + 1 != steps:

            actions = episode_dict["actions"][step]
            text_list.append(action_to_string(actions))
            label_list.append("Actions")

            rewards = episode_dict["rewards"][step]
            text_list.append(str(rewards))
            label_list.append("Reward")

            objects_p = episode_dict["prior_predicted_objects"][step]
            text_list.append(multihots_to_string(objects_p))
            label_list.append("Prior Objects")

            comms_p = episode_dict["prior_predicted_comms"][step]
            text_list.append(onehots_to_string(comms_p))
            label_list.append("Prior Comms")

            objects_q = episode_dict["posterior_predicted_objects"][step]
            text_list.append(multihots_to_string(objects_q))
            label_list.append("Posterior Objects")

            comms_q = episode_dict["posterior_predicted_comms"][step]
            text_list.append(onehots_to_string(comms_q))
            label_list.append("Posterior Comms")
                
        for i, (text, label) in enumerate(zip(text_list, label_list)):
            text = text.replace('\t', ' ').replace('(', ' ').replace(')', ' ')
            ax.text(0.1, 1 - (0.1 * i), f"{label}:", va='center', ha='left', fontsize=12, fontweight='bold')
            ax.text(0.3, 1 - (0.1 * i), text, va='center', ha='left', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}.png")
    plt.close()

plot_dicts, min_max_dict, complete_order = load_dicts(args)
plot_episodes(complete_order, plot_dicts)
print("\nDuration: {}. Done!".format(duration()))