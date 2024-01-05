import torch
from torch import nn 
import torch.nn.functional as F
from torchinfo import summary as torch_summary

from utils import default_args, attach_list, detach_list, ConstrainedConv1d, Ted_Conv1d



def episodes_steps(this):
    return(this.shape[0], this.shape[1])

def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass



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



class Text_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Text_IN, self).__init__()  
        
        self.args = args
        
        self.comm_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings = self.args.communication_shape,
                embedding_dim = self.args.hidden_size),
            nn.PReLU())
        
        self.comm_cnn = nn.Sequential(
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
            nn.PReLU(),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.hidden_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, comm):
        if(len(comm.shape) == 3): comm = comm.unsqueeze(1)
        episodes, steps = episodes_steps(comm)
        comm = torch.argmax(comm, dim = -1)
        comm = self.comm_embedding(comm.int())
        comm = comm.reshape((episodes*steps, self.args.max_comm_len, self.args.hidden_size))
        comm = self.comm_cnn(comm.permute((0,2,1))).permute((0,2,1))
        comm = self.comm_rnn(comm)
        comm = comm[:,-1]
        comm = comm.reshape((episodes, steps, self.args.hidden_size))
        comm = self.comm_lin(comm)
        return(comm)
    
    

class Text_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Text_OUT, self).__init__()  
                
        self.args = args
        
        self.comm_rnn = MTRNN(
            input_size = self.args.hidden_size, 
            hidden_size = self.args.hidden_size, 
            time_constant = True,
            args = self.args)
        
        self.comm_cnn = nn.Sequential(
            nn.PReLU(),
            Ted_Conv1d(
                in_channels = self.args.hidden_size, 
                out_channels = [self.args.hidden_size//4]*4, 
                kernels = [1,3,5,7]),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU())
        
        self.comm_out = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.communication_shape))
        
        self.apply(init_weights)
        self.to(self.args.device)
                
    def forward(self, h):
        episodes, steps = episodes_steps(h)
        h = h.reshape((episodes * steps, 1, self.args.hidden_size))
        comm_h = None
        comm_hs = []
        for i in range(self.args.max_comm_len):
            comm = self.comm_rnn(h.clone(), comm_h)
            comm = comm[:,-1]
            comm_h = comm.clone()
            comm_hs.append(comm.reshape(episodes * steps, 1, self.args.hidden_size))
        comm_h = torch.cat(comm_hs, dim = -2)
        comm_h = self.comm_cnn(comm_h.permute((0,2,1))).permute((0,2,1))
        comm_h = comm_h.reshape(episodes, steps, self.args.max_comm_len, self.args.hidden_size)
        comm_pred = self.comm_out(comm_h)
        return(comm_pred)
    


class Model(nn.Module):
    
    def __init__(self, args = default_args):
        super(Model, self).__init__()  
        
        self.args = args 
        self.text_in = Text_IN(args)
        self.text_out = Text_OUT(args)
        
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, comm):
        comm = self.text_in(comm)
        comm = self.text_out(comm)
        return(comm)
    
    
     
if __name__ == "__main__":
    args = default_args
    episodes = 32 ; steps = 10
    model = Model(args)
    print("\n\n")
    print(model)
    print()
    print(torch_summary(model, 
                        ((episodes, steps + 1, args.max_comm_len, args.communication_shape))))