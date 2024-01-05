#%%
from random import randint, choice
import torch

from utils import default_args, color_map, shape_map, action_map, \
    string_to_onehots, onehots_to_string, multi_hot_action, print



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
        self.args = default_args
        
    def begin(self):
        self.current_objects = []
        self.goal = []
        obj_list = []
        for i in range(self.args.objects):
            obj_list.append(self.make_object(nothing = i >= self.objects))
        self.current_objects_tensor = torch.cat(obj_list, dim = -1)
        self.make_goal()
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
        s = "{}{}{}.".format(action_map[action], color_map[color], shape_map[shape])
        self.goal_comm = string_to_onehots(s)
        
    def give_observation(self):
        return(
            self.current_objects_tensor, 
            self.goal_comm if self.parent else None)
    
    def action_meets_goal(self, action_index, object_index):
        goal_action, goal_shape, goal_color = self.goal
        try:    object_shape, object_color = self.current_objects[object_index]
        except: object_shape, object_color = None, None
        return(
            action_index == goal_action and 
            object_shape == goal_shape and 
            object_color == goal_color)
    
    def __str__(self):
        return("SHAPE-COLORS:\t{}\nGOAL:\t{}".format(
            ["{} {}".format(color_map[color], shape_map[shape]) for shape, color in self.current_objects], 
            onehots_to_string(self.goal_comm)))



class Task_Runner:
    
    def __init__(self, task, args = default_args):
        self.args = args
        self.task = task
        self.begin()
        
    def begin(self):
        self.steps = 0 
        self.task.begin()
        
    def obs(self):
        return(self.task.give_observation())
        
    def action(self, action):
        self.steps += 1
        
        while(len(action.shape)>1): action = action.squeeze(0)
        _, action_index, object_index = multi_hot_action(action)
        action_index = action_index.item()
        object_index = object_index.item()
        print("ACTION:\t{}\tobject {}".format(action_map[action_index], object_index), end = " ")
        reward = -1
        if(self.task.action_meets_goal(action_index, object_index)):
            reward = self.args.extrinsic_reward
            print("Correct!", end = " ")
                    
        if(reward > 0): 
            reward *= self.args.step_cost ** (self.steps-1)
            end = True
            done = True
        else:
            end = False
            done = False
        
        if(not end): end = self.steps >= self.args.max_steps
        if(end and not done): 
            reward += self.args.step_lim_punishment
            done = True
        print("Reward:", reward)
        if(done): print("Done!")
        return(reward, done)
    
    
    
if __name__ == "__main__":        
    from time import sleep
    args = default_args

    task_runner = Task_Runner(Task(actions = 2, objects = 2, shapes = 2, colors = 2))
    done = False
    while(done == False):
        action = torch.zeros((1, args.actions + args.objects))
        action_num = randint(0, args.actions - 1)
        object_num = randint(0, args.objects - 1)
        action[0, action_num] = 1
        action[0, args.actions + object_num] = 1
        reward, done = task_runner.action(action)
        sleep(1)
# %%