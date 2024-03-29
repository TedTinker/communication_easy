#%%
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchinfo import summary as torch_summary

from utils import print, default_args, init_weights, attach_list, detach_list, \
    episodes_steps, pad_zeros, Ted_Conv1d, extract_and_concatenate, create_comm_mask, var, sample, onehots_to_string
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
    
    

class Comm_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Comm_IN, self).__init__()  
        
        self.args = args
        
        self.comm_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings = self.args.comm_shape,
                embedding_dim = self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(.2))
        
        self.comm_cnn = nn.Sequential(
            Ted_Conv1d(
                in_channels = self.args.hidden_size, 
                out_channels = [self.args.hidden_size//4]*4, 
                kernels = [1,3,5,7]),
            #nn.BatchNorm1d(self.args.hidden_size),
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
        # Should use create_comm_mask to avoid unnecessary info
        comm = comm[:,-1]
        comm = comm.reshape((episodes, steps, self.args.hidden_size))
        comm = self.comm_lin(comm)
        return(comm)

    
    
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
    


class Objects_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Objects_OUT, self).__init__()  
                
        self.args = args
        
        self.objects_out = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size + 2 * self.args.hidden_size, 
                out_features = self.args.hidden_size), 
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
                        (episodes, steps, args.pvrnn_mtrnn_size + 2 * args.hidden_size)))
    
    

class Comm_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Comm_OUT, self).__init__()  
                
        self.args = args
        
        self.comm_rnn = MTRNN(
            input_size = self.args.pvrnn_mtrnn_size + 2 * self.args.hidden_size, 
            hidden_size = self.args.hidden_size, 
            time_constant = 1,
            args = self.args)
        
        self.comm_cnn = nn.Sequential(
            Ted_Conv1d(
                in_channels = self.args.hidden_size, 
                out_channels = [self.args.hidden_size//4]*4, 
                kernels = [1,3,5,7]),
            #nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(.2))
        
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
        h_w_action = h_w_action.reshape(episodes * steps, 1, self.args.pvrnn_mtrnn_size + 2 * self.args.hidden_size)
        comm_h = None
        comm_hs = []
        for i in range(self.args.max_comm_len):
            comm_h = self.comm_rnn(h_w_action.clone(), comm_h)
            comm_hs.append(comm_h.clone())
        comm_h = torch.cat(comm_hs, dim = -2)
        comm_h = self.comm_cnn(comm_h.permute((0,2,1))).permute((0,2,1))
        comm_h = comm_h.reshape(episodes, steps, self.args.max_comm_len, self.args.hidden_size)
        comm_pred = self.comm_out(comm_h)
        mask, last_indexes = create_comm_mask(comm_pred)
        comm_pred *= mask.unsqueeze(-1).tile((1,1,1,self.args.comm_shape))
        return(comm_pred)
    
    
    
if __name__ == "__main__":
    
    comm_out = Comm_OUT(args = args)
    
    print("\n\n")
    print(comm_out)
    print()
    print(torch_summary(comm_out, 
                        (episodes, steps, args.pvrnn_mtrnn_size + 2 * args.hidden_size)))
    
    
    
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
    
    
    
class Actor_Comm_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Actor_Comm_OUT, self).__init__()  
                
        self.args = args
        
        self.comm_rnn = MTRNN(
            input_size = self.args.hidden_size, 
            hidden_size = self.args.hidden_size, 
            time_constant = 1,
            args = self.args)
        
        self.comm_cnn = nn.Sequential(
            Ted_Conv1d(
                in_channels = self.args.hidden_size, 
                out_channels = [self.args.hidden_size//4]*4, 
                kernels = [1,3,5,7]),
            #nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(.2))
        
        self.comm_out_mu = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.comm_shape))
        
        self.comm_out_std = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.comm_shape),
            nn.Softplus())
        
        self.apply(init_weights)
        self.to(self.args.device)
                
    def forward(self, x):
        if(len(x.shape) == 2):   x = x.unsqueeze(1)
        [x] = attach_list([x], self.args.device)
        episodes, steps = episodes_steps(x)
        x = x.reshape(episodes * steps, 1, self.args.hidden_size)
        comm_h = None
        comm_hs = []
        for i in range(self.args.max_comm_len):
            comm_h = self.comm_rnn(x.clone(), comm_h)
            comm_hs.append(comm_h.clone())
        comm_h = torch.cat(comm_hs, dim = -2)
        comm_h = self.comm_cnn(comm_h.permute((0,2,1))).permute((0,2,1))
        comm_h = comm_h.reshape(episodes, steps, self.args.max_comm_len, self.args.hidden_size)
        mu, std = var(comm_h, self.comm_out_mu, self.comm_out_std, self.args)
        comm = sample(mu, std, self.args.device)
        comm_out = torch.tanh(comm)
        log_prob = Normal(mu, std).log_prob(comm) - torch.log(1 - comm_out.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        log_prob = log_prob.mean(-2)
        return(comm_out, log_prob)
    
    
    
if __name__ == "__main__":
    
    actor_comm_out = Actor_Comm_OUT(args = args)
    
    print("\n\n")
    print(actor_comm_out)
    print()
    print(torch_summary(actor_comm_out, 
                        (episodes, steps, args.hidden_size)))
    
# %%
