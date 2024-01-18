#%%

from utils import args
from agent import Agent

args.alpha = .1
args.delta = 5

agent = Agent(1, args)
agent.training()
# %%
