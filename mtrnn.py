#%%
import torch 
from torch import nn 
from torchinfo import summary as torch_summary

from utils import default_args, init_weights, attach_list, detach_list, episodes_steps



if __name__ == "__main__":
    
    args = default_args
    episodes = 4 ; steps = 3



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
        if(len(new_h.shape) == 2):
            new_h = new_h.unsqueeze(1)
        return new_h
    
    
    
if __name__ == "__main__":
    
    cell = MTRNNCell(
        input_size = 16,
        hidden_size = 32,
        time_constant = 1,
        args = args)
    
    print("\n\n")
    print(cell)
    print()
    print(torch_summary(cell, 
                        ((episodes, 1, 16), 
                         (episodes, 1, 32))))
    
    

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
        outputs = torch.cat(outputs, dim = 1)
        return outputs
    


if __name__ == "__main__":
    
    mtrnn = MTRNN(
        input_size = 16,
        hidden_size = 32,
        time_constant = 1,
        args = args)
    
    print("\n\n")
    print(mtrnn)
    print()
    print(torch_summary(mtrnn, 
                        ((episodes, steps, 16), 
                         (episodes, steps, 32))))
# %%
