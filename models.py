#%% 

import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torchinfo import summary as torch_summary

from utils import default_args, detach_list, attach_list, print, create_comm_mask, pad_zeros, Ted_Conv1d



def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

def episodes_steps(this):
    return(this.shape[0], this.shape[1])

def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

def sample(mu, std, device):
    e = Normal(0, 1).sample(std.shape).to(device)
    return(mu + e * std)

def rnn_cnn(do_this, to_this):
    episodes = to_this.shape[0] ; steps = to_this.shape[1]
    this = to_this.view((episodes * steps, to_this.shape[2], to_this.shape[3], to_this.shape[4]))
    this = do_this(this)
    this = this.view((episodes, steps, this.shape[1], this.shape[2], this.shape[3]))
    return(this)

def extract_and_concatenate(tensor, expected_len):
    episodes, steps, _ = tensor.shape
    all_indices = []
    for episode in range(episodes):
        episode_indices = []
        for step in range(steps):
            ones_indices = tensor[episode, step].nonzero(as_tuple=True)[0]
            if(ones_indices.shape[0] < expected_len):
                ones_indices = torch.zeros((expected_len,)).int()
            episode_indices.append(ones_indices)
        episode_tensor = torch.stack(episode_indices)
        all_indices.append(episode_tensor)
    final_tensor = torch.stack(all_indices, dim=0)
    return final_tensor



class Obs_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Obs_IN, self).__init__()  
        
        self.args = args
        
        self.obs_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings = self.args.observation_shape,
                embedding_dim = self.args.hidden_size),
            nn.PReLU())
        
        self.obs_lin = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(
                in_features = 2 * self.args.objects * self.args.hidden_size, 
                out_features = self.args.hidden_size),
            nn.PReLU())
        
        self.comm_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings = self.args.communication_shape,
                embedding_dim = self.args.hidden_size),
            nn.PReLU())
        
        self.comm_cnn = nn.Sequential(
            nn.Dropout(.2),
            Ted_Conv1d(
                in_channels = self.args.hidden_size, 
                out_channels = [self.args.hidden_size//4]*4, 
                kernels = [1,3,5,7]),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU())
        
        self.comm_rnn = MTRNN(
            input_size = self.args.hidden_size, 
            hidden_size = self.args.hidden_size, 
            time_constant = True,
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
        
    def forward(self, obs, comm):
        if(len(obs.shape) == 2):   obs  = obs.unsqueeze(1)
        if(len(comm.shape) == 2):  comm = comm.unsqueeze(0)
        if(len(comm.shape) == 3):  comm = comm.unsqueeze(1)
        episodes, steps = episodes_steps(obs)

        obs = extract_and_concatenate(obs, 2 * self.args.objects)
        obs = self.obs_embedding(obs)
        obs = obs.reshape((episodes, steps, 2 * self.args.objects * self.args.hidden_size))
        obs = self.obs_lin(obs)
        
        comm = pad_zeros(comm, self.args.max_comm_len)
        comm = torch.argmax(comm, dim = -1)
        comm = self.comm_embedding(comm.int())
        comm = comm.reshape((episodes*steps, self.args.max_comm_len, self.args.hidden_size))
        comm = self.comm_cnn(comm.permute((0,2,1))).permute((0,2,1))
        comm = self.comm_rnn(comm)
        comm = comm[:,-1]
        comm = comm.reshape((episodes, steps, self.args.hidden_size))
        comm = self.comm_lin(comm)
        return(torch.cat([obs, comm], dim = -1))
    
    

class Obs_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Obs_OUT, self).__init__()  
                
        self.args = args
        
        self.obs_out = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(2 * self.args.hidden_size, self.args.hidden_size), 
            nn.PReLU(),
            nn.Linear(self.args.hidden_size, self.args.hidden_size), 
            nn.PReLU(),
            nn.Linear(self.args.hidden_size, self.args.observation_shape))
        
        self.comm_rnn = MTRNN(
            input_size = 2 * self.args.hidden_size, 
            hidden_size = self.args.hidden_size, 
            time_constant = True,
            args = self.args)
        
        self.comm_cnn = nn.Sequential(
            nn.Dropout(.2),
            nn.PReLU(),
            Ted_Conv1d(
                in_channels = self.args.hidden_size, 
                out_channels = [self.args.hidden_size//4]*4, 
                kernels = [1,3,5,7]),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU())
        
        self.comm_out = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.communication_shape))
        
        self.apply(init_weights)
        self.to(self.args.device)
                
    def forward(self, h_w_action):
        if(len(h_w_action.shape) == 2):   h_w_action = h_w_action.unsqueeze(1)
        [h_w_action] = attach_list([h_w_action], self.args.device)
        episodes, steps = episodes_steps(h_w_action)
        obs_pred = self.obs_out(h_w_action)
        comm_h = None
        comm_hs = []
        for i in range(self.args.max_comm_len):
            comm = self.comm_rnn(h_w_action.clone(), comm_h)
            comm = comm[:,-1]
            comm_h = comm.clone()
            comm_hs.append(comm.reshape(episodes * steps, 1, self.args.hidden_size))
        comm_h = torch.cat(comm_hs, dim = -2)
        comm_h = self.comm_cnn(comm_h.permute((0,2,1))).permute((0,2,1))
        comm_h = comm_h.reshape(episodes, steps, self.args.max_comm_len, self.args.hidden_size)
        comm_pred = self.comm_out(comm_h)
        return(obs_pred, comm_pred)
    
    
    
class Action_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Action_IN, self).__init__()
        
        self.args = args 
        
        self.action_in = nn.Sequential(
            nn.Linear(self.args.action_shape, args.hidden_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, action):
        [action] = attach_list([action], self.args.device)
        if(len(action.shape) == 2):   action = action.unsqueeze(1)
        action = self.action_in(action)
        return(action)
        
        

class MTRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, time_constant, args):
        super(MTRNNCell, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_constant = time_constant
        self.new = 1 / time_constant
        self.old = 1 - self.new

        self.r_x = nn.Sequential(
            nn.Linear(input_size, hidden_size))
        self.r_h = nn.Sequential(
            nn.Linear(hidden_size, hidden_size))
        
        self.z_x = nn.Sequential(
            nn.Linear(input_size, hidden_size))
        self.z_h = nn.Sequential(
            nn.Linear(hidden_size, hidden_size))
        
        self.n_x = nn.Sequential(
            nn.Linear(input_size, hidden_size))
        self.n_h = nn.Sequential(
            nn.Linear(hidden_size, hidden_size))
        
        self.apply(init_weights)
        self.to(args.device)

    def forward(self, x, h):
        attach_list([x, h], self.args.device)
        r = torch.sigmoid(self.r_x(x) + self.r_h(h))
        z = torch.sigmoid(self.z_x(x) + self.z_h(h))
        new_h = torch.tanh(self.n_x(x) + r * self.n_h(h))
        new_h = new_h * (1 - z)  + h * z
        new_h = new_h * self.new + h * self.old
        detach_list([r, z])
        return new_h.unsqueeze(1)

class MTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, time_constant, args):
        super(MTRNN, self).__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.mtrnn_cell = MTRNNCell(input_size, hidden_size, time_constant, args)
        self.apply(init_weights)

    def forward(self, input, h = None):
        if(h == None):
            h = torch.zeros((input.shape[0], 1, self.hidden_size))
        [input, h] = attach_list([input, h], self.args.device)
        episodes, steps = episodes_steps(input)
        outputs = []
        for step in range(steps):  
            h = self.mtrnn_cell(input[:, step], h[:, 0])
            outputs.append(h)
        outputs = torch.stack(outputs, dim = 1)
        return outputs
        
        

class Forward(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        self.layers = len(args.time_scales)
                
        self.obs_in = Obs_IN(args)
        self.action_in = Action_IN(args)
        
        zp_mu_layers  = []
        zp_std_layers = []
        zq_mu_layers  = []
        zq_std_layers = []
        mtrnn_layers  = []
        
        for layer in range(self.layers): 
        
            zp_mu_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size if layer == 0 else 0), args.hidden_size), 
                nn.PReLU(),
                nn.Dropout(.2),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Tanh()))
            zp_std_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size if layer == 0 else 0), args.hidden_size), 
                nn.PReLU(),
                nn.Dropout(.2),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Softplus()))
            
            zq_mu_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size * 3 if layer == 0 else args.hidden_size), args.hidden_size), 
                nn.PReLU(),
                nn.Dropout(.2),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Tanh()))
            zq_std_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size * 3 if layer == 0 else args.hidden_size), args.hidden_size), 
                nn.PReLU(),
                nn.Dropout(.2),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Softplus()))
            
            mtrnn_layers.append(MTRNN(
                input_size = args.state_size + (args.hidden_size if layer + 1 < self.layers else 0),
                hidden_size = args.hidden_size, 
                time_constant = args.time_scales[layer],
                args = args))
            
        self.zp_mu_layers  = nn.ModuleList(zp_mu_layers)
        self.zp_std_layers = nn.ModuleList(zp_std_layers)
        self.zq_mu_layers  = nn.ModuleList(zq_mu_layers)
        self.zq_std_layers = nn.ModuleList(zq_std_layers)
        self.mtrnn_layers  = nn.ModuleList(mtrnn_layers)
        
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        self.to(args.device)
        
    def p(self, prev_action, hq_m1_list = None, episodes = 1):
        if(hq_m1_list == None): hq_m1_list  = [torch.zeros(episodes, 1, self.args.hidden_size)] * self.layers
        [prev_action, hq_m1_list] = attach_list([prev_action, hq_m1_list], self.args.device)
        prev_action = self.action_in(prev_action)
        zp_mu_list = [] ; zp_std_list = [] ; zp_list = [] ; hp_list = []
        for layer in range(self.layers):
            z_input = hq_m1_list[layer] if layer != 0 else torch.cat([hq_m1_list[layer], prev_action], dim = -1) 
            zp_mu, zp_std = var(z_input, self.zp_mu_layers[layer], self.zp_std_layers[layer], self.args)
            zp_mu_list.append(zp_mu) ; zp_std_list.append(zp_std) ; zp_list.append(sample(zp_mu, zp_std, self.args.device))
            h_input = zp_list[layer] if layer+1 == self.layers else torch.cat([zp_list[layer], hq_m1_list[layer+1]], dim = -1) 
            hp = self.mtrnn_layers[layer](h_input, hq_m1_list[layer]) 
            hp_list.append(hp[:,-1])
        return(zp_mu_list, zp_std_list, hp_list)
    
    def q(self, prev_action, obs, comm, hq_m1_list = None):
        if(len(prev_action.shape) == 2): prev_action = prev_action.unsqueeze(1)
        if(len(obs.shape)   == 2):       obs         = obs.unsqueeze(1)
        if(len(comm.shape)  == 3):       comm        = comm.unsqueeze(1)
        episodes, steps = episodes_steps(obs)
        if(hq_m1_list == None):     hq_m1_list = [torch.zeros(episodes, steps, self.args.hidden_size)] * self.layers
        [prev_action, obs, comm, hq_m1_list] = attach_list([prev_action, obs, comm, hq_m1_list], self.args.device)
        obs = self.obs_in(obs, comm)
        prev_action = self.action_in(prev_action)
        zq_mu_list = [] ; zq_std_list = [] ; zq_list = [] ; hq_list = []
        for layer in range(self.layers):
            z_input = torch.cat((hq_m1_list[layer], obs, prev_action), dim=-1) if layer == 0 else torch.cat((hq_m1_list[layer], hq_list[layer-1]), dim=-1)
            zq_mu, zq_std = var(z_input, self.zq_mu_layers[layer], self.zq_std_layers[layer], self.args)        
            zq_mu_list.append(zq_mu) ; zq_std_list.append(zq_std) ; zq_list.append(sample(zq_mu, zq_std, self.args.device))
            h_input = zq_list[layer] if layer+1 == self.layers else torch.cat([zq_list[layer], hq_m1_list[layer+1]], dim = -1)
            hq = self.mtrnn_layers[layer](h_input, hq_m1_list[layer])
            hq_list.append(hq[:,-1])
        return(zq_mu_list, zq_std_list, hq_list)
        
    def predict(self, action, h): 
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        if(len(h[0].shape) == 2):   h[0]   = h[0].unsqueeze(1)
        [action, h] = attach_list([action, h], self.args.device)
        h_w_action = torch.cat([self.action_in(action), h[0]], dim = -1)
        pred_obs, pred_comm = self.predict_obs(h_w_action)
        detach_list([h_w_action])
        return(pred_obs, pred_comm)
    
    def forward(self, prev_action, obs, comm):
        [prev_action, obs, comm] = attach_list([prev_action, obs, comm], self.args.device)
        episodes, steps = episodes_steps(obs)
        zp_mu_lists = [] ; zp_std_lists = [] ;                                                    
        zq_mu_lists = [] ; zq_std_lists = [] ; zq_rgbd_pred_list = [] ; zq_speed_pred_list = [] ; hq_lists = [[torch.zeros(episodes, 1, self.args.hidden_size).to(self.args.device)] * self.layers]
        step = -1
        for step in range(steps-1):
            zp_mu_list, zp_std_list, hp_list = self.p(prev_action[:,step],                              hq_lists[-1], episodes = episodes)
            zq_mu_list, zq_std_list, hq_list = self.q(prev_action[:,step], obs[:,step], comm[:,step], hq_lists[-1])
            zq_rgbd_pred, zq_speed_pred = self.predict(prev_action[:,step+1], hq_list)
            zp_mu_lists.append(zp_mu_list) ; zp_std_lists.append(zp_std_list) 
            zq_mu_lists.append(zq_mu_list) ; zq_std_lists.append(zq_std_list) ; hq_lists.append(hq_list)
            zq_rgbd_pred_list.append(zq_rgbd_pred) ; zq_speed_pred_list.append(zq_speed_pred)
        zp_mu_list, zp_std_list, hp_list = self.p(prev_action[:,step+1],                                  hq_lists[-1], episodes = episodes)
        zq_mu_list, zq_std_list, hq_list = self.q(prev_action[:,step+1], obs[:,step+1], comm[:,step+1], hq_lists[-1])
        zp_mu_lists.append(zp_mu_list) ; zp_std_lists.append(zp_std_list) 
        zq_mu_lists.append(zq_mu_list) ; zq_std_lists.append(zq_std_list)
        hq_lists.append(hq_lists.pop(0))    
        hq_lists = [torch.cat([hq_list[layer] for hq_list in hq_lists], dim = 1) for layer in range(self.args.layers)]
        zp_mu_list  = [torch.cat([zp_mu[layer]  for zp_mu  in zp_mu_lists],  dim = 1) for layer in range(self.args.layers)]
        zp_std_list = [torch.cat([zp_std[layer] for zp_std in zp_std_lists], dim = 1) for layer in range(self.args.layers)]
        zq_mu_list  = [torch.cat([zq_mu[layer]  for zq_mu  in zq_mu_lists],  dim = 1) for layer in range(self.args.layers)]
        zq_std_list = [torch.cat([zq_std[layer] for zq_std in zq_std_lists], dim = 1) for layer in range(self.args.layers)]
        pred_observations = torch.cat(zq_rgbd_pred_list,  dim = 1)
        pred_communications  = torch.cat(zq_speed_pred_list, dim = 1)
        #pred_communications *= create_comm_mask(pred_communications)
        return(
            (zp_mu_list, zp_std_list), 
            (zq_mu_list, zq_std_list, pred_observations, pred_communications, hq_lists))
    
    

"""
class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args

        self.lin = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU())
        self.mu = nn.Sequential(
            nn.Linear(args.hidden_size, self.args.actions + self.args.objects))
        self.std = nn.Sequential(
            nn.Linear(args.hidden_size, self.args.actions + self.args.objects),
            nn.Softplus())

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, obs, comm, prev_action, hq, ha = None):
        x = self.lin(hq)
        mu, std = var(x, self.mu, self.std, self.args)
        x = sample(mu, std, self.args.device)
        action = torch.tanh(x)
        log_prob = Normal(mu, std).log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return action, log_prob, ha
"""
    
    

class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(args)
        self.prev_action_in = Action_IN(args)
        
        self.gru = nn.GRU(
            input_size =  4 * args.hidden_size,
            hidden_size = args.hidden_size,
            batch_first = True)

        self.lin = nn.Sequential(
            nn.Dropout(.2),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU())
        self.mu = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(args.hidden_size, self.args.actions + self.args.objects))
        self.std = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(args.hidden_size, self.args.actions + self.args.objects),
            nn.Softplus())

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, obs, comm, prev_action, hq, ha = None):
        [obs, comm, prev_action, hq] = attach_list([obs, comm, prev_action, hq], self.args.device)
        obs = self.obs_in(obs, comm)
        prev_action = self.prev_action_in(prev_action)
        ha, _ = self.gru(torch.cat((obs, prev_action, hq), dim=-1), ha)
        x = self.lin(ha)
        mu, std = var(x, self.mu, self.std, self.args)
        x = sample(mu, std, self.args.device)
        action = torch.tanh(x)
        log_prob = Normal(mu, std).log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return action, log_prob, ha
    
    
    
class Critic(nn.Module):

    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(args)
        self.action_in = Action_IN(args)
        
        self.gru = nn.GRU(
            input_size =  4 * args.hidden_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        
        self.lin = nn.Sequential(
            nn.Dropout(.2),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, 1))

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, obs, comm, action, hq, hc = None):
        [obs, comm, action, hq] = attach_list([obs, comm, action, hq], self.args.device)
        obs = self.obs_in(obs, comm)
        action = self.action_in(action)
        hc, _ = self.gru(torch.cat((obs, action, hq), dim=-1), hc)
        Q = self.lin(hc)
        detach_list([obs, action])
        return(Q, hc)
    


if __name__ == "__main__":
    
    args = default_args
    episodes = 32 ; steps = 10
    
    
    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, 
                        ((episodes, steps+1, args.action_shape), 
                         (episodes, steps+1, args.observation_shape), 
                         (episodes, steps+1, 6, args.communication_shape))))
    
    
    
    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor,
                        ((episodes, steps, args.hidden_size))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, 
                        ((episodes, steps, args.observation_shape), 
                         (episodes, steps, 6, args.communication_shape), 
                         (episodes, steps, args.action_shape))))

# %%
