import torch 
import torch.optim as optim
import torch.nn.functional as F

from utils import args, pad_zeros, onehots_to_string
from task import tasks
from trying_models import Model

task = tasks["1"]
model = Model(args)
opt = optim.Adam(model.parameters(), lr=.01)



def print_examples(batch, predictions, ep_max = 2, step_max = 2):
    for ep_num, (ep_1, ep_2) in enumerate(zip(predictions, batch)):
        if(ep_num < ep_max): 
            for step_num, (step_1, step_2) in enumerate(zip(ep_1, ep_2)):
                if(step_num < step_max):
                    print(onehots_to_string(step_1), "|", onehots_to_string(step_2))
    
    

def epoch(episodes = 32, steps = 11):
    texts = []
    for episode in range(episodes):
        task.begin()
        _, text = task.give_observation()
        text = pad_zeros(text, args.max_comm_len)
        texts.append(text.unsqueeze(0))
    batch = torch.cat(texts, dim = 0).unsqueeze(1)
    batch = batch.tile((1, steps, 1, 1))
    predictions = model(batch.clone())

    print_examples(batch, predictions)
    
    predictions = predictions.reshape((episodes * steps * args.max_comm_len, args.communication_shape))
    batch = batch.reshape((episodes * steps * args.max_comm_len, args.communication_shape))
    batch = torch.argmax(batch, dim=-1)
    communication_loss = F.cross_entropy(predictions, batch, reduction = "none").reshape((episodes, steps, args.max_comm_len))
    
    opt.zero_grad()
    (communication_loss.mean()).backward()
    opt.step()

e = 1
while(e < 300):
    print("\nEpoch {}:".format(e))
    epoch()
    e += 1 