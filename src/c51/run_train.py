from models import Teacher, Student, linear_schedule
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
parser.add_argument("--teacher-encoder", type=str)
parser.add_argument("--student-encoder", type=str)
parser.add_argument("--total-timesteps", type=int)
parser.add_argument("--learning-starts", type=int)
parser.add_argument("--postfix", type=str, default="")

args = parser.parse_args()
flags = OmegaConf.load(args.config)
flags.env_id = args.env_id 
flags.seed = args.seed 
flags.total_timesteps = args.total_timesteps
flags.learning_starts = args.learning_starts
flags.teacher_encoder = args.teacher_encoder 
flags.student_encoder = args.student_encoder
flags.postfix = args.postfix


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

# # env setup
# envs = ProcgenEnv(num_envs=1, env_name=flags.env_id, num_levels=0, start_level=0, distribution_mode="easy")
# envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
# envs.single_action_space = envs.action_space
# envs.single_observation_space = envs.observation_space["rgb"]
# envs.is_vector_env = True

# envs = gym.wrappers.RecordEpisodeStatistics(envs)
# envs = gym.wrappers.NormalizeReward(envs, gamma=flags.gamma)
# envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
# assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
# assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, run_name)])
rb = ReplayBuffer(
    flags.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    optimize_memory_usage=False,
    handle_timeout_termination=True,
)
start_time = time.time()

# ------------------------------------------------------------------------
from encoder import get_encoder 
encoder_t = get_encoder(flags.teacher_encoder, envs, out_features=envs.single_action_space.n * flags.n_atoms)
encoder_s = get_encoder(flags.student_encoder, envs, out_features=envs.single_action_space.n * flags.n_atoms)
teacher = Teacher(envs, encoder_t, flags).to(device)
student = Student(envs, encoder_s, flags).to(device)
# ------------------------------------------------------------------------



# TRY NOT TO MODIFY: start the game
obs = envs.reset()
for global_step in range(flags.total_timesteps):
    
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(flags.start_e, flags.end_e, flags.exploration_fraction * flags.total_timesteps, global_step)
    if random.random() < epsilon:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        actions, pmf = teacher.q_network.get_action(torch.Tensor(obs).to(device))
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

    # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
    # real_next_obs = next_obs.copy()
    # for idx, d in enumerate(dones):
    #     if d:
    #         real_next_obs[idx] = infos[idx]["terminal_observation"]

    
    rb.add(obs, next_obs, actions, rewards, dones, infos)

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs

    # ALGO LOGIC: training.
    if global_step > flags.learning_starts and global_step % flags.train_frequency == 0:
        data = rb.sample(flags.batch_size)
        with torch.no_grad():
            _, next_pmfs = teacher.target_q_network.get_action(data.next_observations)
            next_atoms = data.rewards + flags.gamma * teacher.target_q_network.atoms * (1 - data.dones)
            # projection
            delta_z = teacher.target_q_network.atoms[1] - teacher.target_q_network.atoms[0]
            tz = next_atoms.clamp(flags.v_min, flags.v_max)

            b = (tz - flags.v_min) / delta_z
            l = b.floor().clamp(0, flags.n_atoms - 1)
            u = b.ceil().clamp(0, flags.n_atoms - 1)
            # (l == u).float() handles the case where bj is exactly an integer
            # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = teacher.q_network.get_action(data.observations, data.actions.flatten())
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

        if global_step % 100 == 0:
            writer.add_scalar("losses/loss", loss.item(), global_step)
            old_val = (old_pmfs * teacher.q_network.atoms).sum(1)
            writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # optimize the model
        teacher.optimizer.zero_grad()
        loss.backward()
        teacher.optimizer.step()

        # update the target network
        if global_step % flags.target_network_frequency == 0:
            for param, target_param in zip(teacher.q_network.parameters(), teacher.target_q_network.parameters()):
                target_param.data.copy_(flags.tau * param.data + (1 - flags.tau) * target_param.data)

envs.close()
writer.close()