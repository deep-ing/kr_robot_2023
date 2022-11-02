from agents import get_agent
from distilled import get_distilled
from utils import linear_schedule
import time
import os 
import gym 
# from procgen import ProcgenEnv
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np 
import torch 
import argparse
from omegaconf import OmegaConf
from stable_baselines3.common.buffers import ReplayBuffer
from make_env import make_env

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--env-id", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--teacher-agent", type=str)
parser.add_argument("--teacher-encoder", type=str)
parser.add_argument("--distil-agent", type=str)
parser.add_argument("--distil-encoder", type=str)
parser.add_argument("--distil-method", type=str)
parser.add_argument("--total-timesteps", type=int)
parser.add_argument("--learning-starts", type=int)
parser.add_argument("--acceptance-ratio", type=float)
parser.add_argument("--postfix", type=str, default="")

args = parser.parse_args()
flags = OmegaConf.load(args.config)
for key in vars(args):
    setattr(flags, key, getattr(args, key))

run_name = f"runs/{flags.env_id}/{flags.teacher_agent}_{flags.teacher_encoder}_{flags.distil_agent}_{flags.distil_encoder}/{flags.distil_method}_{flags.postfix}_{int(time.time())}"
writer = SummaryWriter(f"{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

# TRY NOT TO MODIFY: seeding
random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.backends.cudnn.deterministic = True
device = torch.device(flags.device)

envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, run_name)])
flags.action_dim = envs.single_action_space.n

rb = ReplayBuffer(
    flags.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    optimize_memory_usage=False,
    handle_timeout_termination=True,
)

rb_distil = ReplayBuffer(
    flags.distil_buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    optimize_memory_usage=False,
    handle_timeout_termination=True,
)

flags.start_time = time.time()
OmegaConf.save(flags, f'{run_name}/config.yaml')

# ------------------------------------------------------------------------
from encoder import get_encoder 

teacher_out_features = {
    'c51' : envs.single_action_space.n * flags.c51.n_atoms,
    'dqn' : envs.single_action_space.n
}[flags.teacher_agent]
distil_out_features = envs.single_action_space.n

encoder_t = get_encoder(flags.teacher_encoder, envs, out_features=teacher_out_features)
encoder_t_target = get_encoder(flags.teacher_encoder, envs, out_features=teacher_out_features)
encoder_distil = get_encoder(flags.distil_encoder, envs, out_features=teacher_out_features)
teacher = get_agent(flags.teacher_agent, encoder_t, encoder_t_target, flags).to(device)
distil = get_distilled(flags.distil_agent ,encoder_distil, flags).to(device)
# ------------------------------------------------------------------------

# TRY NOT TO MODIFY: start the game
obs = envs.reset()
for global_step in range(flags.total_timesteps):
    
    epsilon = linear_schedule(flags.start_e, flags.end_e, flags.exploration_fraction * flags.total_timesteps, global_step)
    if random.random() < epsilon:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        with torch.no_grad():
            actions, _ = distil.get_action(torch.Tensor(obs).to(device))
        actions = actions.cpu().numpy()

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, rewards, dones, infos = envs.step(actions)
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    for info in infos:
        if "episode" in info.keys():
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
            break 
    
    rb.add(obs, next_obs, actions, rewards, dones, infos)
    obs = next_obs
    if global_step > flags.learning_starts and global_step % flags.train_teacher_frequency == 0:
        teacher.train(rb, rb_distil, writer, global_step)
    if global_step > flags.learning_starts and global_step % flags.train_distil_frequency == 0:
        distil.train_distil(teacher, rb_distil, global_step, writer)
            # update the target network
        OmegaConf.save(flags, f'{run_name}/config.yaml')
    if global_step % teacher.target_network_frequency == 0:
        for param, target_param in zip(teacher.q_network.parameters(), teacher.target_q_network.parameters()):
            target_param.data.copy_(teacher.af.tau * param.data + (1 - teacher.af.tau) * target_param.data)
    
envs.close()
writer.close()