#%%
import torch
from torch import nn
from torchinfo import summary as torch_summary

from utils import default_args, init_weights, var, sample, attach_list, detach_list, episodes_steps, pad_zeros
from mtrnn import MTRNN
from submodules import Obs_IN, Obs_OUT, Action_IN



if __name__ == "__main__":
    
    args = default_args
    episodes = 4 ; steps = 3



class PVRNN_LAYER(nn.Module):
    
    def __init__(self, time_scale = 1, bottom = False, top = False, args = default_args):
        super(PVRNN_LAYER, self).__init__()
        
        self.args = args 
        self.bottom = bottom
        self.top = top
        
        if(self.bottom):
            self.obs_in = Obs_IN(self.args)
            self.action_in = Action_IN(self.args)
            
        # Prior: Previous hidden state, plus action if bottom.  
        self.zp_mu = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (self.args.hidden_size if self.bottom else 0), 
                    out_features = self.args.state_size), 
                nn.Tanh())
        self.zp_std = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (self.args.hidden_size if self.bottom else 0), 
                    out_features = self.args.state_size), 
                nn.Softplus())
                            
        # Posterior: Previous hidden state, plus observation and action if bottom, plus lower-layer hidden state otherwise.
        self.zq_mu = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (3 * self.args.hidden_size if self.bottom else self.args.pvrnn_mtrnn_size), 
                    out_features = self.args.state_size), 
                nn.Tanh())
        self.zq_std = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (3 * self.args.hidden_size if self.bottom else self.args.pvrnn_mtrnn_size), 
                    out_features = self.args.state_size), 
                nn.Softplus())
                            
        # New hidden state: Previous hidden state, zq value, plus higher-layer hidden state if not top.
        #"""
        self.mtrnn = MTRNN(
                input_size = self.args.state_size + (self.args.pvrnn_mtrnn_size if not self.top else 0),
                hidden_size = self.args.pvrnn_mtrnn_size, 
                time_constant = time_scale,
                args = self.args)
        """
        
        self.mtrnn = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size + self.args.state_size + (self.args.pvrnn_mtrnn_size if not self.top else 0),
                out_features = self.args.pvrnn_mtrnn_size),
            nn.Tanh())
        #"""
            
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(
        self, 
        prev_hidden_states, 
        objects = None, comms = None, prev_actions = None, 
        hidden_states_below = None, 
        prev_hidden_states_above = None):
        
        if(self.bottom):
            obs = self.obs_in(objects, comms)
            prev_actions = self.action_in(prev_actions)
            zp_inputs = torch.cat([prev_hidden_states, prev_actions], dim = -1)
            zq_inputs = torch.cat([prev_hidden_states, obs, prev_actions], dim = -1)
        else:
            zp_inputs = prev_hidden_states 
            zq_inputs = torch.cat([prev_hidden_states, hidden_states_below], dim = -1)
            
        zp_mu, zp_std = var(zp_inputs, self.zp_mu, self.zp_std, self.args)
        zp = sample(zp_mu, zp_std, self.args.device)
        zq_mu, zq_std = var(zq_inputs, self.zq_mu, self.zq_std, self.args)
        zq = sample(zq_mu, zq_std, self.args.device)
            
        if(self.top):
            mtrnn_inputs = zq 
        else:
            mtrnn_inputs = torch.cat([zq, prev_hidden_states_above], dim = -1)
            
        new_hidden_states = self.mtrnn(mtrnn_inputs, prev_hidden_states)
        #new_hidden_states = self.mtrnn(torch.cat([mtrnn_inputs, prev_hidden_states], dim = -1))
        
        return(
            (zp_mu, zp_std),
            (zq_mu, zq_std),
            new_hidden_states)
        
        
    
if __name__ == "__main__":
    
    bottom_top_layer = PVRNN_LAYER(bottom = True, top = True, args = args)
    
    print("\n\nBOTTOM-TOP")
    print(bottom_top_layer)
    print()
    print(torch_summary(bottom_top_layer, 
                        ((episodes, 1, args.pvrnn_mtrnn_size), 
                         (episodes, 1, args.objects, args.object_shape), 
                         (episodes, 1, args.max_comm_len, args.comm_shape),
                         (episodes, 1, args.actions + args.objects))))
    
    bottom_layer = PVRNN_LAYER(bottom = True, args = args)
    
    print("\n\nBOTTOM")
    print(bottom_layer)
    print()
    print(torch_summary(bottom_layer, 
                        ((episodes, 1, args.pvrnn_mtrnn_size), 
                         (episodes, 1, args.objects, args.object_shape), 
                         (episodes, 1, args.max_comm_len, args.comm_shape),
                         (episodes, 1, args.actions + args.objects),
                         (1,), # No hidden_states_below 
                         (episodes, 1, args.pvrnn_mtrnn_size))))
    
    top_layer = PVRNN_LAYER(top = True, args = args)
    
    print("\n\nTOP")
    print(top_layer)
    print()
    print(torch_summary(top_layer, 
                        ((episodes, 1, args.pvrnn_mtrnn_size), 
                         (1,), # No objects
                         (1,), # No comms
                         (1,), # No actions
                         (episodes, 1, args.pvrnn_mtrnn_size))))
    
    middle_layer = PVRNN_LAYER(args = args)
    
    print("\n\nMIDDLE")
    print(middle_layer)
    print()
    print(torch_summary(middle_layer, 
                        ((episodes, 1, args.pvrnn_mtrnn_size), 
                         (1,), # No objects
                         (1,), # No comms
                         (1,), # No actions
                         (episodes, 1, args.pvrnn_mtrnn_size),
                         (episodes, 1, args.pvrnn_mtrnn_size))))
    
    
    
class PVRNN(nn.Module):
    
    def __init__(self, args = default_args):
        super(PVRNN, self).__init__()
        
        self.args = args 
        
        pvrnn_layers = []
        for layer in range(self.args.layers): 
            pvrnn_layers.append(
                PVRNN_LAYER(
                    self.args.time_scales[layer], 
                    bottom = layer == 0, 
                    top = layer + 1 == self.args.layers, 
                    args = self.args))
            
        self.pvrnn_layers = nn.ModuleList(pvrnn_layers)
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        self.to(args.device)
        
    def bottom_to_top_step(self, prev_hidden_states, objects = None, comms = None, prev_actions = None):
        if(objects != None and len(objects.shape) == 3): 
            objects = objects.unsqueeze(1)
        if(comms != None and len(comms.shape) == 3): 
            comms = comms.unsqueeze(1)
        if(prev_actions != None and len(prev_actions.shape) == 2): 
            prev_actions = prev_actions.unsqueeze(1)
        
        zp_mu_list = []
        zp_std_list = []
        zq_mu_list = []
        zq_std_list = []
        new_hidden_states_list= []
                
        for layer in range(self.args.layers):
            (zp_mu, zp_std), (zq_mu, zq_std), new_hidden_states = \
                self.pvrnn_layers[layer](
                    prev_hidden_states[:,layer].unsqueeze(1), 
                    objects, comms, prev_actions,
                    new_hidden_states_list[-1] if layer > 0 else None, 
                    prev_hidden_states[:,layer+1].unsqueeze(1) if layer + 1 < self.args.layers else None)
    
            for l, o in zip(
                [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list],
                [zp_mu, zp_std, zq_mu, zq_std, new_hidden_states]):            
                l.append(o)
                
        lists = [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list]
        for i in range(len(lists)):
            lists[i] = torch.cat(lists[i], dim=1)
        zp_mu, zp_std, zq_mu, zq_std, new_hidden_states = lists
                        
        return(
            (zp_mu.unsqueeze(1), zp_std.unsqueeze(1)),
            (zq_mu.unsqueeze(1), zq_std.unsqueeze(1)),
            new_hidden_states.unsqueeze(1))
    
    def forward(self, prev_hidden_states, objects, comms, prev_actions):
        zp_mu_list = []
        zp_std_list = []
        zq_mu_list = []
        zq_std_list = []
        new_hidden_states_list = []
                
        episodes, steps = episodes_steps(objects)
        
        for step in range(steps):
            (zp_mu, zp_std), (zq_mu, zq_std), new_hidden_states = \
                self.bottom_to_top_step(prev_hidden_states, objects[:,step], comms[:,step], prev_actions[:,step])
            
            for l, o in zip(
                [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list],
                [zp_mu, zp_std, zq_mu, zq_std, new_hidden_states]):     
                l.append(o)
                
            prev_hidden_states = new_hidden_states.squeeze(1)
            
        lists = [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list]
        for i in range(len(lists)):
            lists[i] = torch.cat(lists[i], dim=1)
        zp_mu, zp_std, zq_mu, zq_std, new_hidden_states = lists
                
        if(steps == 1):
            h_w_actions = torch.cat([new_hidden_states[:,:,0], self.pvrnn_layers[0].action_in(prev_actions)], dim = -1)
        else:
            h_w_actions = torch.cat([new_hidden_states[:,:-1,0], self.pvrnn_layers[0].action_in(prev_actions[:,1:])], dim = -1)
        pred_objects, pred_comms = self.predict_obs(h_w_actions)
        
        return(
            (zp_mu, zp_std),
            (zq_mu, zq_std),
            (pred_objects, pred_comms),
            new_hidden_states)
        
        
        
if __name__ == "__main__":
    
    args.layers = 1
    args.time_scales = [1]
    
    pvrnn = PVRNN(args = args)
    
    print("\n\nPVRNN: ONE LAYER")
    print(pvrnn)
    print()
    print(torch_summary(pvrnn, 
                        ((episodes, args.layers, args.pvrnn_mtrnn_size), 
                         (episodes, steps+1, args.objects, args.object_shape), 
                         (episodes, steps+1, args.max_comm_len, args.comm_shape),
                         (episodes, steps+1, args.actions + args.objects))))
"""
    args.layers = 5
    args.time_scales = [1, 1, 1, 1, 1]
    
    pvrnn = PVRNN(args = args)
    
    print("\n\nPVRNN: MANY LAYERS")
    print(pvrnn)
    print()
    print(torch_summary(pvrnn, 
                        ((episodes, args.layers, args.hidden_size), 
                         (episodes, steps+1, args.objects, args.object_shape), 
                         (episodes, steps+1, args.max_comm_len, args.comm_shape),
                         (episodes, steps+1, args.actions + args.objects))))
"""
            
