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

    steps = len(episode_dict["objects_1"])
    fig, axs = plt.subplots(steps, 1, figsize=(10, 5 * steps))
    fig.suptitle(f"Epoch {epoch}", fontsize=16)
    
    for step, ax in enumerate(axs):
        ax.axis('off')
        
        text_list = []
        label_list = []

        objects_1 = episode_dict["objects_1"][step]
        text_list.append(objects_1)
        label_list.append("Objects (1)")

        comms_in_1 = episode_dict["comms_in_1"][step]
        text_list.append(comms_in_1)
        label_list.append("Comms (1)")
        
        if step + 1 != steps:

            actions_1 = episode_dict["actions_1"][step]
            text_list.append(action_to_string(actions_1))
            label_list.append("Actions (1)")

            rewards = episode_dict["rewards"][step]
            text_list.append(rewards)
            label_list.append("Reward")

            objects_p_1 = episode_dict["prior_predicted_objects_1"][step]
            text_list.append(objects_p_1)
            label_list.append("Predicted Objects (Prior) (1)")

            comms_in_p_1 = episode_dict["prior_predicted_comms_in_1"][step]
            text_list.append(comms_in_p_1)
            label_list.append("Predicted Comms (Prior) (1)")

            objects_q_1 = episode_dict["posterior_predicted_objects_1"][step]
            text_list.append(objects_q_1)
            label_list.append("Predicted Objects (Posterior) (1)")

            comms_in_q_1 = episode_dict["posterior_predicted_comms_in_1"][step]
            text_list.append(comms_in_q_1)
            label_list.append("Predicted Comms (Posterior) (1)")
        
        ax.text(.1, 1, "Step {}".format(step+1) if step+1 != steps else "Done", va='center', ha='left', fontsize=20, fontweight='bold')
        for i, (text, label) in enumerate(zip(text_list, label_list)):
            text = text.replace('\t', ' ').replace('(', ' ').replace(')', ' ')
            ax.text(0.3, 1 - (0.1 * i), f"{label}:", va='center', ha='left', fontsize=12, fontweight='bold')
            ax.text(0.7, 1 - (0.1 * i), text, va='center', ha='left', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}.png")
    plt.close()

plot_dicts, min_max_dict, complete_order = load_dicts(args)
plot_episodes(complete_order, plot_dicts)
print("\nDuration: {}. Done!".format(duration()))