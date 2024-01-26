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
    agent_num, epoch, episode_num, swapping = key.split("_")
    print("{}: agent {}, epoch {}, episode {}.{}".format(arg_name, agent_num, epoch, episode_num, " Swapping!" if swapping == 1 else ""))

    steps = len(episode_dict["objects_1"])
    fig, axs = plt.subplots(steps, 1, figsize=(25, 5 * steps))
    fig.suptitle(f"Epoch {epoch}", fontsize=16)
    
    for step, ax in enumerate(axs):
        ax.axis('off')
        
        text_list = []
        label_list = []
        
        if(step == 0):
            goal = episode_dict["task"].goal_text
            text_list.append(goal)
            label_list.append("Goal:")

        objects_1 = episode_dict["objects_1"][step]
        text_list.append(objects_1)
        label_list.append("Objects (1):")

        comms_in_1 = episode_dict["comms_in_1"][step]
        text_list.append(comms_in_1)
        label_list.append("Comms In (1):")
        
        if step + 1 != steps:

            actions_1 = episode_dict["actions_1"][step]
            text_list.append(action_to_string(actions_1))
            label_list.append("Actions (1):")
            
            comms_out_1 = episode_dict["comms_out_1"][step]
            text_list.append(comms_out_1)
            label_list.append("Comms Out (1):")

            rewards = episode_dict["rewards"][step]
            text_list.append(rewards)
            label_list.append("Reward:")
            
            values_1 = episode_dict["critic_predictions_1"][step]
            values = ""
            for i, value in enumerate(values_1):
                values += "{}".format(value) + ("." if i+1 == len(values_1) else ", ")
            text_list.append(values)
            label_list.append("Predicted Values:")

            objects_p_1 = episode_dict["prior_predicted_objects_1"][step]
            text_list.append(objects_p_1)
            label_list.append("Predicted Objects (Prior) (1):")

            comms_in_p_1 = episode_dict["prior_predicted_comms_in_1"][step]
            text_list.append(comms_in_p_1)
            label_list.append("Predicted Comms (Prior) (1):")

            objects_q_1 = episode_dict["posterior_predicted_objects_1"][step]
            text_list.append(objects_q_1)
            label_list.append("Predicted Objects (Posterior) (1):")

            comms_in_q_1 = episode_dict["posterior_predicted_comms_in_1"][step]
            text_list.append(comms_in_q_1)
            label_list.append("Predicted Comms (Posterior) (1):")
        
        ax.text(0.0, 1, "Step {}".format(step+1) if step+1 != steps else "Done", va='center', ha='left', fontsize=20, fontweight='bold')
        for i, (text, label) in enumerate(zip(text_list, label_list)):
            text = text.replace('\t', ' ').replace('(', ' ').replace(')', ' ')
            ax.text(0.1, 1 - (0.1 * i), label, va='center', ha='left', fontsize=12, fontweight='bold')
            ax.text(0.3, 1 - (0.1 * i), text, va='center', ha='left', fontsize=12)
            
        if(episode_dict["task"]).parent:
            pass 
        else:
            text_list = []
            label_list = []
            
            if(step == 0):
                goal = episode_dict["task"].goal_text
                text_list.append("")
                label_list.append("")

            objects_2 = episode_dict["objects_2"][step]
            text_list.append(objects_2)
            label_list.append("Objects (2):")

            comms_in_2 = episode_dict["comms_in_2"][step]
            text_list.append(comms_in_2)
            label_list.append("Comms In (2):")
            
            if step + 1 != steps:

                actions_2 = episode_dict["actions_2"][step]
                text_list.append(action_to_string(actions_2))
                label_list.append("Actions (2):")
                
                comms_out_2 = episode_dict["comms_out_2"][step]
                text_list.append(comms_out_2)
                label_list.append("Comms Out (2):")

                rewards = episode_dict["rewards"][step]
                text_list.append(rewards)
                label_list.append("Reward:")
                
                values_2 = episode_dict["critic_predictions_2"][step]
                values = ""
                for i, value in enumerate(values_2):
                    values += "{}".format(value) + ("." if i+1 == len(values_1) else ", ")
                text_list.append(values)
                label_list.append("Predicted Values:")

                objects_p_2 = episode_dict["prior_predicted_objects_2"][step]
                text_list.append(objects_p_2)
                label_list.append("Predicted Objects (Prior) (2):")

                comms_in_p_2 = episode_dict["prior_predicted_comms_in_2"][step]
                text_list.append(comms_in_p_2)
                label_list.append("Predicted Comms (Prior) (2):")

                objects_q_2 = episode_dict["posterior_predicted_objects_2"][step]
                text_list.append(objects_q_2)
                label_list.append("Predicted Objects (Posterior) (2):")

                comms_in_q_2 = episode_dict["posterior_predicted_comms_in_2"][step]
                text_list.append(comms_in_q_2)
                label_list.append("Predicted Comms (Posterior) (2):")
            
            for i, (text, label) in enumerate(zip(text_list, label_list)):
                text = text.replace('\t', ' ').replace('(', ' ').replace(')', ' ')
                ax.text(0.5, 1 - (0.1 * i), label, va='center', ha='left', fontsize=12, fontweight='bold')
                ax.text(0.7, 1 - (0.1 * i), text, va='center', ha='left', fontsize=12)
        
    plt.savefig(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}_swapping_{swapping}.png")
    plt.close()

plot_dicts, min_max_dict, complete_order = load_dicts(args)
plot_episodes(complete_order, plot_dicts)
print("\nDuration: {}. Done!".format(duration()))