import matplotlib.pyplot as plt

from utils import print, args, duration, load_dicts, multihots_to_string, onehots_to_string
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
    print("\n\n")
    print(arg_name)
    agent_num, epoch, episode_num = key.split("_")
    
    steps = len(episode_dict["objects"])
    fig, axs = plt.subplots(steps, 8)
    fig.suptitle("Epoch {}".format(epoch))
    
    for step in range(steps):
        
        ax = axs[step, 0]
        objects = episode_dict["objects"][step]
        print("objects", objects.shape)
        print(objects)
        ax.text(0, 1, objects)
        ax.text(0, 0, multihots_to_string(objects))
        
        ax = axs[step, 1]
        comms = episode_dict["comms"][step]
        print("comms", comms.shape)
        print(comms)
        ax.text(0, 1, comms)
        ax.text(0, 0, comms)
        
        if step+1 !=  steps:
            ax = axs[step, 2]
            actions = episode_dict["actions"][step]
            print("actions", actions.shape)
            print(actions)
            ax.text(0, 0, actions)
            
            ax = axs[step, 3]
            rewards = episode_dict["rewards"][step]
            print("rewards")
            print(rewards)
            ax.text(0, 0, rewards)
            
            ax = axs[step, 4]
            objects_p = episode_dict["prior_predicted_objects"][step]
            print("objects_p", objects_p.shape)
            print(objects_p)
            ax.text(0, 1, objects_p)
            ax.text(0, 0, multihots_to_string(episode_dict["prior_predicted_objects"][step]))
            
            ax = axs[step, 5]
            comms_p = episode_dict["prior_predicted_comms"][step]
            print("comms_p", comms_p.shape)
            print(comms_p)
            ax.text(0, 1, comms_p)
            ax.text(0, 0, onehots_to_string(episode_dict["prior_predicted_comms"][step]))
            
            ax = axs[step, 6]
            objects_q = episode_dict["posterior_predicted_objects"][step]
            print("objects_q", objects_q.shape)
            print(objects_q)
            ax.text(0, 1, objects_q)
            ax.text(0, 0, multihots_to_string(episode_dict["posterior_predicted_objects"][step]))
            
            ax = axs[step, 7]
            comms_q = episode_dict["posterior_predicted_comms"][step]
            print("comms_q", comms_q.shape)
            print(comms_q)
            ax.text(0, 1, comms_q)
            ax.text(0, 0, onehots_to_string(episode_dict["posterior_predicted_comms"][step]))
                
    plt.savefig("{}/epoch_{}_episode_{}_agent_{}.png".format(arg_name, epoch, episode_num, agent_num))
    plt.close()

plot_dicts, min_max_dict, complete_order = load_dicts(args)
plot_episodes(complete_order, plot_dicts)
print("\nDuration: {}. Done!".format(duration()))