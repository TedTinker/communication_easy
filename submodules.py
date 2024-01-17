#%%
import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary as torch_summary

from utils import default_args, init_weights, attach_list, detach_list, \
    episodes_steps, pad_zeros, Ted_Conv1d, extract_and_concatenate
from mtrnn import MTRNN



if __name__ == "__main__":
    
    args = default_args
    episodes = 4 ; steps = 3



class Objects_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Objects_IN, self).__init__()  
        
        self.args = args
        
        self.objects_in_1 = nn.Sequential(
            nn.Linear(
                in_features = self.args.object_shape, 
                out_features = self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(.2),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(.2))
        
        self.objects_in_2 = nn.Sequential(
            nn.Linear(
                in_features = self.args.objects * self.args.hidden_size, 
                out_features = self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(.2))
                
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, objects):
        if(len(objects.shape) == 2):   objects  = objects.unsqueeze(0)
        if(len(objects.shape) == 3):   objects  = objects.unsqueeze(1)
        episodes, steps = episodes_steps(objects)
        objects = self.objects_in_1(objects)
        objects = objects.reshape((episodes, steps, self.args.objects * self.args.hidden_size))
        objects = self.objects_in_2(objects)
        return(objects)
    
    
    
if __name__ == "__main__":
    
    objects_in = Objects_IN(args = args)
    
    print("\n\n")
    print(objects_in)
    print()
    print(torch_summary(objects_in, 
                        (episodes, steps, args.objects, args.object_shape)))
    
    

"""
class Comm_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Comm_IN, self).__init__()  
        
        self.args = args
        
        self.comm_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings = self.args.comm_shape,
                embedding_dim = self.args.hidden_size),
            nn.PReLU())
        
        self.comm_cnn = nn.Sequential(
            nn.Dropout(.2),
            Ted_Conv1d(
                in_channels = self.args.hidden_size, 
                out_channels = [self.args.hidden_size//4]*4, 
                kernels = [1,1,3,5]),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU())
        
        self.comm_rnn = MTRNN(
            input_size = self.args.hidden_size, 
            hidden_size = self.args.hidden_size, 
            time_constant = 1,
            args = self.args)
        
        self.comm_lin = nn.Sequential(
            nn.Dropout(.2),
            nn.PReLU(),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.hidden_size),
            nn.PReLU())
                
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, comm):
        if(len(comm.shape) == 2):  comm = comm.unsqueeze(0)
        if(len(comm.shape) == 3):  comm = comm.unsqueeze(1)
        episodes, steps = episodes_steps(comm)
        comm = pad_zeros(comm, self.args.max_comm_len)
        comm = torch.argmax(comm, dim = -1)
        comm = self.comm_embedding(comm.int())
        comm = comm.reshape((episodes*steps, self.args.max_comm_len, self.args.hidden_size))
        comm = self.comm_cnn(comm.permute((0,2,1))).permute((0,2,1))
        comm = self.comm_rnn(comm)
        comm = comm[:,-1]
        comm = comm.reshape((episodes, steps, self.args.hidden_size))
        comm = self.comm_lin(comm)
        return(comm)
"""

class Comm_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Comm_IN, self).__init__()  
        
        self.args = args
        
        self.comm_lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.max_comm_len * self.args.comm_shape, 
                out_features = self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(.2),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(.2))
                
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, comm):
        if(len(comm.shape) == 2):  comm = comm.unsqueeze(0)
        if(len(comm.shape) == 3):  comm = comm.unsqueeze(1)
        episodes, steps = episodes_steps(comm)
        comm = pad_zeros(comm, self.args.max_comm_len)
        comm = comm.reshape((episodes, steps, self.args.max_comm_len * self.args.comm_shape))
        comm = self.comm_lin(comm)
        return(comm)
#"""
    
    
if __name__ == "__main__":
    
    comm_in = Comm_IN(args = args)
    
    print("\n\n")
    print(comm_in)
    print()
    print(torch_summary(comm_in, 
                        (episodes, steps, args.max_comm_len, args.comm_shape)))
    
    
    
class Obs_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Obs_IN, self).__init__()  
                
        self.args = args
        self.objects_in = Objects_IN(self.args)
        self.comm_in = Comm_IN(self.args)
        
    def forward(self, objects, comm):
        objects = self.objects_in(objects)
        comm = self.comm_in(comm)
        return(torch.cat([objects, comm], dim = -1))
    
    
    
if __name__ == "__main__":
    
    obs_in = Obs_IN(args = args)
    
    print("\n\n")
    print(obs_in)
    print()
    print(torch_summary(obs_in, 
                        ((episodes, steps, args.objects, args.object_shape),
                         (episodes, steps, args.max_comm_len, args.comm_shape))))
    


#"""
class Objects_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Objects_OUT, self).__init__()  
                
        self.args = args
        
        self.objects_out = nn.Sequential(
            nn.Linear(self.args.pvrnn_mtrnn_size + self.args.hidden_size, self.args.hidden_size), 
            nn.PReLU(),
            nn.Dropout(.2),
            nn.Linear(self.args.hidden_size, self.args.hidden_size), 
            nn.PReLU(),
            nn.Dropout(.2),
            nn.Linear(self.args.hidden_size, self.args.objects * self.args.object_shape))
        
        self.apply(init_weights)
        self.to(self.args.device)
                
    def forward(self, h_w_action):
        if(len(h_w_action.shape) == 2):   h_w_action = h_w_action.unsqueeze(1)
        episodes, steps = episodes_steps(h_w_action)
        [h_w_action] = attach_list([h_w_action], self.args.device)
        objects_pred = self.objects_out(h_w_action)
        objects_pred = objects_pred.reshape((episodes, steps, self.args.objects, self.args.object_shape))
        return(objects_pred)
    
    
    
if __name__ == "__main__":

    objects_out = Objects_OUT(args = args)
    
    print("\n\n")
    print(objects_out)
    print()
    print(torch_summary(objects_out, 
                        (episodes, steps, args.pvrnn_mtrnn_size + args.hidden_size)))
    
    

class Comm_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Comm_OUT, self).__init__()  
                
        self.args = args
        
        self.comm_rnn = MTRNN(
            input_size = self.args.pvrnn_mtrnn_size + self.args.hidden_size, 
            hidden_size = self.args.hidden_size, 
            time_constant = True,
            args = self.args)
        
        self.comm_cnn = nn.Sequential(
            Ted_Conv1d(
                in_channels = self.args.hidden_size, 
                out_channels = [self.args.hidden_size//4]*4, 
                kernels = [1,3,5,7]),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(.4))
        
        self.comm_out = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.comm_shape))
        
        self.apply(init_weights)
        self.to(self.args.device)
                
    def forward(self, h_w_action):
        if(len(h_w_action.shape) == 2):   h_w_action = h_w_action.unsqueeze(1)
        [h_w_action] = attach_list([h_w_action], self.args.device)
        episodes, steps = episodes_steps(h_w_action)
        comm_h = None
        comm_hs = []
        for i in range(self.args.max_comm_len):
            comm = self.comm_rnn(h_w_action.clone(), comm_h)
            comm_h = comm.clone()
            comm_hs.append(comm.reshape(episodes * steps, 1, self.args.hidden_size))
        comm_h = torch.cat(comm_hs, dim = -2)
        comm_h = self.comm_cnn(comm_h.permute((0,2,1))).permute((0,2,1))
        comm_h = comm_h.reshape(episodes, steps, self.args.max_comm_len, self.args.hidden_size)
        comm_pred = self.comm_out(comm_h)
        return(comm_pred)
    
    
    
if __name__ == "__main__":
    
    comm_out = Comm_OUT(args = args)
    
    print("\n\n")
    print(comm_out)
    print()
    print(torch_summary(comm_out, 
                        (episodes, steps, args.pvrnn_mtrnn_size + args.hidden_size)))
    
    
    
class Obs_OUT(nn.Module):
    
    def __init__(self, args = default_args):
        super(Obs_OUT, self).__init__()  
        
        self.args = args 
        self.objects_out = Objects_OUT(self.args)
        self.comm_out = Comm_OUT(self.args)
        
    def forward(self, h_w_action):
        objects_pred = self.objects_out(h_w_action.clone())
        comm_pred = self.comm_out(h_w_action.clone())
        return(objects_pred, comm_pred)
    
    
    
class Action_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Action_IN, self).__init__()
        
        self.args = args 
        
        self.action_in = nn.Sequential(
            nn.Linear(self.args.action_shape, args.hidden_size),
            nn.PReLU(),
            nn.Dropout(.2),
            nn.Linear(self.args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Dropout(.2))
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, action):
        [action] = attach_list([action], self.args.device)
        if(len(action.shape) == 2):   action = action.unsqueeze(1)
        action = self.action_in(action)
        return(action)
    
    
    
if __name__ == "__main__":
    
    action_in = Action_IN(args = args)
    
    print("\n\n")
    print(action_in)
    print()
    print(torch_summary(action_in, 
                        (episodes, steps, args.actions + args.objects)))
# %%
