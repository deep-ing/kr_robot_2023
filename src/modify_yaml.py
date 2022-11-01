import os
from omegaconf import OmegaConf

def modify_yaml(yaml, keys, value):
    target = yaml
    keys = keys.split("/")
    for i in range(len(keys)-1):
        target = target[keys[i]]
    target[keys[-1]] = value
        
def open_yaml(path):
    y = OmegaConf.load(path)
    return y 

def save_yaml(yaml, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    OmegaConf.save(yaml, path)
    
    
    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--items", type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--save-path', type=str)

args = parser.parse_args()
yaml = open_yaml(args.config)

print(args.items)

replace_keys_and_times = eval(args.items)
for k,v in replace_keys_and_times.items():
    modify_yaml(yaml, k,v)

save_yaml(yaml, args.save_path)