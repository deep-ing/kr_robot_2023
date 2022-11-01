from agents import get_agent
from utils import linear_schedule
import time
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
parser.add_argument("--student-encoder", type=str)
parser.add_argument("--total-timesteps", type=int)
parser.add_argument("--learning-starts", type=int)
parser.add_argument("--postfix", type=str, default="")

args = parser.parse_args()
flags = OmegaConf.load(args.config)
for key in vars(args):
    setattr(flags, key, getattr(args, key))

run_name = f"{flags.env_id}__{flags.seed}__{flags.postfix}_{int(time.time())}"
writer = SummaryWriter(f"runs/{run_name}")
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
flags.start_time = time.time()


# ------------------------------------------------------------------------
from encoder import get_encoder 

teacher_out_features = {
    'c51' : envs.single_action_space.n * flags.agent.n_atoms,
    'dqn' : envs.single_action_space.n
}[flags.teacher_agent]

encoder_t = get_encoder(flags.teacher_encoder, envs, out_features=teacher_out_features)
encoder_t_target = get_encoder(flags.teacher_encoder, envs, out_features=teacher_out_features)
teacher = get_agent(flags.teacher_agent, encoder_t, encoder_t_target, flags).to(device)
# ------------------------------------------------------------------------

# TRY NOT TO MODIFY: start the game
obs = envs.reset()
for global_step in range(flags.total_timesteps):
    
    epsilon = linear_schedule(flags.start_e, flags.end_e, flags.exploration_fraction * flags.total_timesteps, global_step)
    if random.random() < epsilon:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        actions, _ = teacher.get_action(torch.Tensor(obs).to(device))
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
        teacher.train(rb, writer, global_step)
        
envs.close()
writer.close()