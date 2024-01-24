#%%
from random import randint, choice, randrange
import torch

from utils import default_args, color_map, shape_map, action_map, make_object, pad_zeros,\
    string_to_onehots, onehots_to_string, multi_hot_action, action_to_string, print



class Task:
    
    def __init__(
            self, 
            actions = 1, 
            objects = 1, 
            shapes = 1, 
            colors = 1, 
            parent = True, 
            args = default_args):
        self.actions = actions
        self.objects = objects 
        self.shapes = shapes
        self.colors = colors
        self.parent = parent
        self.args = args
        
    def begin(self, test = False, verbose = False):
        self.solved = False
        self.goal = []
        self.current_objects_1 = []
        obj_list_1 = []
        for i in range(self.args.objects):
            obj_list_1.append(self.make_object(nothing = i >= self.objects, test = test))
        self.current_objects_tensor_1 = torch.cat(obj_list_1, dim = 0)
        
        index = self.make_goal()

        if(not self.parent):
            self.current_objects_2 = []
            obj_list_2 = []
            for i in range(self.args.objects):
                obj_list_2.append(self.make_object(nothing = i >= self.objects, test = test, agent_1 = False))
            new_index = randrange(len(self.current_objects_2))
            obj_list_2[new_index] = obj_list_1[index]
            self.current_objects_2[new_index] = self.current_objects_1[index]
            self.current_objects_tensor_2 = torch.cat(obj_list_2, dim = 0)
                
        if(verbose):
            print(self)
        
    def make_object(self, nothing = False, test = False, agent_1 = True):
        obj = torch.zeros((1, self.args.shapes + self.args.colors))
        if(not nothing):
            shape, color = make_object(self.shapes, self.colors, test)
            if(agent_1):
                self.current_objects_1.append((shape, color))
            else:
                self.current_objects_2.append((shape, color))
            obj[0, shape] = 1
            obj[0, self.args.shapes + color] = 1
        return(obj)
    
    def make_goal(self):
        action = randint(0, self.actions - 1)
        index = randrange(len(self.current_objects_1))
        shape, color = self.current_objects_1[index]
        self.goal = (action, shape, color)
        self.goal_text = "{} {} {}.".format(action_map[action], color_map[color], shape_map[shape])
        self.goal_comm = string_to_onehots(self.goal_text)
        self.goal_comm = pad_zeros(self.goal_comm, self.args.max_comm_len)
        return(index)
        
    def give_observation(self):
        return(
            self.current_objects_tensor_1, 
            None if self.parent else self.current_objects_tensor_2, 
            pad_zeros(string_to_onehots("CORRECT."), self.args.max_comm_len) if self.solved else 
            self.goal_comm)
    
    def reward_for_action(self, action, agent_1 = True):
        if(action == None): return(0, 0)
        _, action_index, object_index = multi_hot_action(action, self.args)
        goal_action, goal_shape, goal_color = self.goal
        try:    object_shape, object_color = self.current_objects_1[object_index] if agent_1 else self.current_objects_2[object_index]
        except: object_shape, object_color = None, None
        reward = 0
        win = 0
        if(action_index == goal_action):
            reward += self.args.action_reward
        if(object_shape == goal_shape):
            reward += self.args.shape_reward
        if(object_color == goal_color):
            reward += self.args.color_reward
        if( action_index == goal_action and 
            object_shape == goal_shape and 
            object_color == goal_color):
            reward += self.args.correct_reward
            win = 1
        if(win == 1): 
            self.solved = True
        return(reward, win)
    
    def get_recommended_action(self, agent_1 = True):
        action_num = self.goal[0]
        goal_shape, goal_color = self.goal[1], self.goal[2]
        matching_indices = [i for i, (shape, color) in enumerate(self.current_objects_1 if agent_1 else self.current_objects_2)
                            if shape == goal_shape and color == goal_color]
        object_num = choice(matching_indices)
        recommended_action = torch.ones((self.args.action_shape,)) * -1
        recommended_action[action_num] = 1
        recommended_action[self.args.actions + object_num] = 1
        return(recommended_action)
    
    def __str__(self):
        to_return = "\n\nSHAPE-COLORS (1):\t{}".format(["{} {}".format(color_map[color], shape_map[shape]) for shape, color in self.current_objects_1])
        if(not self.parent):
            to_return += "\nSHAPE-COLORS (2):\t{}".format(["{} {}".format(color_map[color], shape_map[shape]) for shape, color in self.current_objects_2])
        to_return += "\nGOAL:\t{}".format(onehots_to_string(self.goal_comm))
        to_return += "\nRECOMMENDED ACTION (1):\t{}".format(self.get_recommended_action())
        if(not self.parent):
            to_return += "\nRECOMMENDED ACTION (2):\t{}".format(self.get_recommended_action(agent_1 = False))
        return(to_return)



class Task_Runner:
    
    def __init__(self, task, args = default_args):
        self.args = args
        self.task = task
        
    def begin(self, test = False, verbose = False):
        self.steps = 0 
        self.task.begin(test, verbose)
        
    def obs(self):
        return(self.task.give_observation())
        
    def action(self, action_1, action_2 = None, verbose = False):
        self.steps += 1
        done = False
        
        if(verbose):
            print("ACTION (1):\t{}".format(action_to_string(action_1)), end = " ")
            if(action_2 != None):
                print("\nACTION (2):\t{}".format(action_to_string(action_2)), end = " ")
        reward_1, win_1 = self.task.reward_for_action(action_1)
        reward_2, win_2 = self.task.reward_for_action(action_2, agent_1 = False)
        reward = max([reward_1, reward_2])
        win = max([win_1, win_2])
                    
        if(reward > 0): 
            reward *= self.args.step_cost ** (self.steps-1)
        
        end = self.steps >= self.args.max_steps
        if(end and not win): 
            reward += self.args.step_lim_punishment
            done = True
        if(win):
            done = True
            if(verbose):
                print("Correct!", end = " ")
        if(verbose):
            print("Reward:", reward)
            if(done): print("Done.")
        return(reward, done, win)
    
    
    
if __name__ == "__main__":        
    from time import sleep
    args = default_args

    task_runner = Task_Runner(Task(actions = 5, objects = 3, shapes = 5, colors = 6))
    task_runner.begin(verbose = True)
    task_runner.task.give_observation()
    done = False
    while(done == False):
        action_num = randint(0, args.actions - 1)
        object_num = randint(0, args.objects - 1)
        action = torch.zeros((args.action_shape,))
        action[action_num] = 1
        action[args.actions + object_num] = 1
        reward, done, win = task_runner.action(action, verbose = True)
        sleep(1)
        
    task_runner = Task_Runner(Task(actions = 5, objects = 3, shapes = 5, colors = 6, parent = False))
    task_runner.begin(verbose = True)
    task_runner.task.give_observation()
    done = False
    while(done == False):
        action_1_num = randint(0, args.actions - 1)
        object_1_num = randint(0, args.objects - 1)
        action_1 = torch.zeros((args.action_shape,))
        action_1[action_1_num] = 1
        action_1[args.actions + object_1_num] = 1
        action_2_num = randint(0, args.actions - 1)
        object_2_num = randint(0, args.objects - 1)
        action_2 = torch.zeros((args.action_shape,))
        action_2[action_2_num] = 1
        action_2[args.actions + object_2_num] = 1
        reward, done, win = task_runner.action(action_1, action_2, verbose = True)
        sleep(1)
# %%