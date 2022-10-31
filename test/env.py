
import os 
import sys 
print(os.getcwd())
sys.path.append("src")
from make_env import make_env


for naem in ['procgen-coinrun-v0', 'procgen-ninja-v0', 'procgen-leaper-v0']:
    env = make_env('procgen-leaper-v0', seed=0)()
    env.reset()
    print(env.__class__.__name__)
    print(env.action_space.n)
    print(env.observation_space.shape)