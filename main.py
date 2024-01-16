#%%

from utils import args
from agent import Agent

args.alpha = .1
args.delta = 2

agent = Agent(args)
agent.training()
# %%
