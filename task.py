#%%
from random import randint, choice
import torch

from utils import default_args, color_map, shape_map, action_map, pad_zeros,\
    string_to_onehots, onehots_to_string, multi_hot_action, action_to_string, print



class Task:
    
    def __init__(
            self, 
            actions = 1, 
            objects = 1, 
            shapes = 1, 
            colors = 1, 
            parent = True, 
            test_generalizing = False,
            args = default_args):
        self.actions = actions
        self.objects = objects 
        self.shapes = shapes
        self.colors = colors
        self.parent = parent
        self.test_generalizing = test_generalizing
        self.args = args
        
    def begin(self, verbose = False):
        self.solved = False
        self.current_objects = []
        self.goal = []
        obj_list = []
        for i in range(self.args.objects):
            obj_list.append(self.make_object(nothing = i >= self.objects))
        self.current_objects_tensor = torch.cat(obj_list, dim = 0)
        self.make_goal()
        if(verbose):
            print(self)
        
    def make_object(self, nothing = False):
        obj = torch.zeros((1, self.args.shapes + self.args.colors))
        if(not nothing):
            shape = randint(0, self.shapes - 1)
            color = randint(0, self.colors - 1)
            self.current_objects.append((shape, color))
            obj[0, shape] = 1
            obj[0, self.args.shapes + color] = 1
        return(obj)
    
    def make_goal(self):
        action = randint(0, self.actions - 1)
        shape, color = choice(self.current_objects)
        self.goal = (action, shape, color)
        self.goal_text = "{} {} {}.".format(action_map[action], color_map[color], shape_map[shape])
        self.goal_comm = string_to_onehots(self.goal_text)
        self.goal_comm = pad_zeros(self.goal_comm, self.args.max_comm_len)
        
    def give_observation(self):
        return(
            self.current_objects_tensor, 
            pad_zeros(string_to_onehots("CORRECT."), self.args.max_comm_len) if self.solved else self.goal_comm if self.parent else None)
    
    def reward_for_action(self, action):
        _, action_index, object_index = multi_hot_action(action, self.args)
        goal_action, goal_shape, goal_color = self.goal
        try:    object_shape, object_color = self.current_objects[object_index]
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
    
    def get_recommended_action(self):
        action_num = self.goal[0]
        goal_shape, goal_color = self.goal[1], self.goal[2]
        matching_indices = [i for i, (shape, color) in enumerate(self.current_objects)
                            if shape == goal_shape and color == goal_color]
        object_num = choice(matching_indices)
        recommended_action = torch.ones((self.args.action_shape,)) * -1
        recommended_action[action_num] = 1
        recommended_action[self.args.actions + object_num] = 1
        return(recommended_action)
    
    def __str__(self):
        return("\n\nSHAPE-COLORS:\t{}\nGOAL:\t{}\nRECOMMENDED ACTION:\t{}".format(
            ["{} {}".format(color_map[color], shape_map[shape]) for shape, color in self.current_objects], 
            onehots_to_string(self.goal_comm), 
            action_to_string(self.get_recommended_action())))



class Task_Runner:
    
    def __init__(self, task, verbose = False, args = default_args):
        self.args = args
        self.task = task
        self.begin(verbose)
        
    def begin(self, verbose = False):
        self.steps = 0 
        self.task.begin(verbose)
        
    def obs(self):
        return(self.task.give_observation())
        
    def action(self, action, verbose = False):
        self.steps += 1
        done = False
        
        if(verbose):
            print("ACTION:\t{}".format(action_to_string(action)), end = " ")
        reward, win = self.task.reward_for_action(action)
                    
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

    task_runner = Task_Runner(Task(actions = 5, objects = 3, shapes = 5, colors = 6), verbose = True)
    task_runner.task.give_observation()
    done = False
    while(done == False):
        action_num = randint(0, args.actions - 1)
        object_num = randint(0, args.objects - 1)
        action = torch.zeros((args.action_shape,))
        action[action_num] = 1
        action[args.actions + object_num] = 1
        print("ACTION: {}, {}".format(action_num, object_num))
        reward, done, win = task_runner.action(action, verbose = True)
        sleep(1)
# %%