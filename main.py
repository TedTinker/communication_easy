#%%

from utils import args
from agent import Agent

args.alpha = .1
args.delta = 1

agent = Agent(args)
agent.training()
# %%
