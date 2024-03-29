#%%
import torch
from torch import nn
from torchinfo import summary as torch_summary

from utils import default_args, init_weights, var, sample, attach_list, detach_list, episodes_steps, pad_zeros, create_comm_mask
from mtrnn import MTRNN
from submodules import Obs_IN, Obs_OUT, Action_IN, Comm_IN



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
            self.comm_in = Comm_IN(self.args)
            
        # Prior: Previous hidden state, plus action if bottom.  
        self.zp_mu = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (2 * self.args.hidden_size if self.bottom else 0), 
                    out_features = self.args.state_size), 
                nn.Tanh())
        self.zp_std = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (2 * self.args.hidden_size if self.bottom else 0), 
                    out_features = self.args.state_size), 
                nn.Softplus())
                            
        # Posterior: Previous hidden state, plus observation and action if bottom, plus lower-layer hidden state otherwise.
        self.zq_mu = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (4 * self.args.hidden_size if self.bottom else self.args.pvrnn_mtrnn_size), 
                    out_features = self.args.state_size), 
                nn.Tanh())
        self.zq_std = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (4 * self.args.hidden_size if self.bottom else self.args.pvrnn_mtrnn_size), 
                    out_features = self.args.state_size), 
                nn.Softplus())
                            
        # New hidden state: Previous hidden state, zq value, plus higher-layer hidden state if not top.
        self.mtrnn = MTRNN(
                input_size = self.args.state_size + (self.args.pvrnn_mtrnn_size if not self.top else 0),
                hidden_size = self.args.pvrnn_mtrnn_size, 
                time_constant = time_scale,
                args = self.args)
            
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(
        self, 
        prev_hidden_states, 
        objects = None, comms_in = None, prev_actions = None, prev_comms_out = None, 
        hidden_states_below = None, 
        prev_hidden_states_above = None):
        
        if(self.bottom):
            mask, last_indexes = create_comm_mask(comms_in)
            comms_in *= mask.unsqueeze(-1).tile((1,self.args.comm_shape))
            mask, last_indexes = create_comm_mask(prev_comms_out)
            prev_comms_out *= mask.unsqueeze(-1).tile((1,self.args.comm_shape))
            
            obs = self.obs_in(objects, comms_in)
            prev_actions = self.action_in(prev_actions)
            prev_comms_out = self.comm_in(prev_comms_out)
            zp_inputs = torch.cat([prev_hidden_states, prev_actions, prev_comms_out], dim = -1)
            zq_inputs = torch.cat([prev_hidden_states, obs, prev_actions, prev_comms_out], dim = -1)
        else:
            zp_inputs = prev_hidden_states 
            zq_inputs = torch.cat([prev_hidden_states, hidden_states_below], dim = -1)
            
        zp_mu, zp_std = var(zp_inputs, self.zp_mu, self.zp_std, self.args)
        zp = sample(zp_mu, zp_std, self.args.device)
        zq_mu, zq_std = var(zq_inputs, self.zq_mu, self.zq_std, self.args)
        zq = sample(zq_mu, zq_std, self.args.device)
            
            
        if(self.top):
            mtrnn_inputs_p = zp
        else:
            mtrnn_inputs_p = torch.cat([zp, prev_hidden_states_above], dim = -1)
            
        if(self.top):
            mtrnn_inputs_q = zq 
        else:
            mtrnn_inputs_q = torch.cat([zq, prev_hidden_states_above], dim = -1)
            
        new_hidden_states_p = self.mtrnn(mtrnn_inputs_p, prev_hidden_states)
        new_hidden_states_q = self.mtrnn(mtrnn_inputs_q, prev_hidden_states)
        
        return(
            (zp_mu, zp_std, new_hidden_states_p),
            (zq_mu, zq_std, new_hidden_states_q))        
        
    
if __name__ == "__main__":
    
    bottom_top_layer = PVRNN_LAYER(bottom = True, top = True, args = args)
    
    print("\n\nBOTTOM-TOP")
    print(bottom_top_layer)
    print()
    print(torch_summary(bottom_top_layer, 
                        ((episodes, 1, args.pvrnn_mtrnn_size), 
                         (episodes, 1, args.objects, args.object_shape), 
                         (episodes, 1, args.max_comm_len, args.comm_shape),
                         (episodes, 1, args.actions + args.objects),
                         (episodes, 1, args.max_comm_len, args.comm_shape))))
    
    bottom_layer = PVRNN_LAYER(bottom = True, args = args)
    
    print("\n\nBOTTOM")
    print(bottom_layer)
    print()
    print(torch_summary(bottom_layer, 
                        ((episodes, 1, args.pvrnn_mtrnn_size), 
                         (episodes, 1, args.objects, args.object_shape), 
                         (episodes, 1, args.max_comm_len, args.comm_shape),
                         (episodes, 1, args.actions + args.objects),
                         (episodes, 1, args.max_comm_len, args.comm_shape),
                         (1,), # No hidden_states_below 
                         (episodes, 1, args.pvrnn_mtrnn_size))))
    
    top_layer = PVRNN_LAYER(top = True, args = args)
    
    print("\n\nTOP")
    print(top_layer)
    print()
    print(torch_summary(top_layer, 
                        ((episodes, 1, args.pvrnn_mtrnn_size), 
                         (1,), # No objects
                         (1,), # No comms in
                         (1,), # No actions
                         (1,), # No comms out 
                         (episodes, 1, args.pvrnn_mtrnn_size))))
    
    middle_layer = PVRNN_LAYER(args = args)
    
    print("\n\nMIDDLE")
    print(middle_layer)
    print()
    print(torch_summary(middle_layer, 
                        ((episodes, 1, args.pvrnn_mtrnn_size), 
                         (1,), # No objects
                         (1,), # No comms in
                         (1,), # No actions
                         (1,), # comms out
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
        
    def predict(self, h, action, comm_out):
        h_w_actions = torch.cat([h, self.pvrnn_layers[0].action_in(action), self.pvrnn_layers[0].comm_in(comm_out)], dim = -1)
        pred_objects, pred_comms = self.predict_obs(h_w_actions)
        return(pred_objects,  pred_comms)
        
    def bottom_to_top_step(self, prev_hidden_states, objects = None, comms = None, prev_actions = None, prev_comms_out = None):
        if(objects != None and len(objects.shape) == 3): 
            objects = objects.unsqueeze(1)
        if(comms != None and len(comms.shape) == 3): 
            comms = comms.unsqueeze(1)
        if(prev_actions != None and len(prev_actions.shape) == 2): 
            prev_actions = prev_actions.unsqueeze(1)
        if(prev_comms_out != None and len(prev_comms_out.shape) == 3): 
            prev_comms_out = prev_comms_out.unsqueeze(1)
        
        zp_mu_list = []
        zp_std_list = []
        zq_mu_list = []
        zq_std_list = []
        new_hidden_states_list_p = []
        new_hidden_states_list_q = []
                
        for layer in range(self.args.layers):
            (zp_mu, zp_std, new_hidden_states_p), (zq_mu, zq_std, new_hidden_states_q) = \
                self.pvrnn_layers[layer](
                    prev_hidden_states[:,layer].unsqueeze(1), 
                    objects, comms, prev_actions, prev_comms_out,
                    new_hidden_states_list_q[-1] if layer > 0 else None, 
                    prev_hidden_states[:,layer+1].unsqueeze(1) if layer + 1 < self.args.layers else None)
    
            for l, o in zip(
                [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list_p, new_hidden_states_list_q],
                [zp_mu, zp_std, zq_mu, zq_std, new_hidden_states_p,  new_hidden_states_q]):            
                l.append(o)
                
        lists = [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list_p, new_hidden_states_list_q]
        for i in range(len(lists)):
            lists[i] = torch.cat(lists[i], dim=1)
        zp_mu, zp_std, zq_mu, zq_std, new_hidden_states_p, new_hidden_states_q = lists
                        
        return(
            (zp_mu.unsqueeze(1), zp_std.unsqueeze(1), new_hidden_states_p.unsqueeze(1)),
            (zq_mu.unsqueeze(1), zq_std.unsqueeze(1), new_hidden_states_q.unsqueeze(1)))
    
    def forward(self, prev_hidden_states, objects, comms_in, prev_actions, prev_comms_out):
        zp_mu_list = []
        zp_std_list = []
        zq_mu_list = []
        zq_std_list = []
        new_hidden_states_list_p = []
        new_hidden_states_list_q = []
                
        episodes, steps = episodes_steps(objects)
        
        for step in range(steps):
            (zp_mu, zp_std, new_hidden_states_p), (zq_mu, zq_std, new_hidden_states_q) = \
                self.bottom_to_top_step(prev_hidden_states, objects[:,step], comms_in[:,step], prev_actions[:,step], prev_comms_out[:,step])
            
            for l, o in zip(
                [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list_p, new_hidden_states_list_q],
                [zp_mu, zp_std, zq_mu, zq_std, new_hidden_states_p, new_hidden_states_q]):     
                l.append(o)
                
            prev_hidden_states = new_hidden_states_q.squeeze(1)
            
        lists = [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list_p, new_hidden_states_list_q]
        for i in range(len(lists)):
            lists[i] = torch.cat(lists[i], dim=1)
        zp_mu, zp_std, zq_mu, zq_std, new_hidden_states_p, new_hidden_states_q = lists
        
        pred_objects, pred_comms = self.predict(new_hidden_states_q[:,:-1,0], prev_actions[:,1:], prev_comms_out[:,1:])
        
        return(
            (zp_mu, zp_std, new_hidden_states_p),
            (zq_mu, zq_std, new_hidden_states_q),
            (pred_objects, pred_comms))
        
        
        
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
                         (episodes, steps+1, args.actions + args.objects),
                         (episodes, steps+1, args.max_comm_len, args.comm_shape))))
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
                         (episodes, steps+1, args.actions + args.objects),
                         (episodes, steps+1, args.max_comm_len, args.comm_shape))))
"""
            
