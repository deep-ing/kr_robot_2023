from models import Teacher, Student


import time
import gym 
from procgen import ProcgenEnv
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np 
import torch 
import argparse
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--env-id", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--total-timesteps", type=int)

args = parser.parse_args()
flags = OmegaConf.load(args.config)
flags.env_id = args.env_id 
flags.seed = args.seed 
flags.total_timesteps = args.total_timesteps
flags.batch_size = int(flags.num_envs * flags.num_steps)
flags.minibatch_size = int(flags.batch_size // flags.num_minibatches)

run_name = f"runs/{flags.env_id}__{flags.seed}__{int(time.time())}"

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

# env setup
envs = ProcgenEnv(num_envs=flags.num_envs, env_name=flags.env_id, num_levels=0, start_level=0, distribution_mode="easy")
envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
envs.single_action_space = envs.action_space
envs.single_observation_space = envs.observation_space["rgb"]
envs.is_vector_env = True
envs = gym.wrappers.RecordEpisodeStatistics(envs)
envs = gym.wrappers.NormalizeReward(envs, gamma=flags.gamma)
envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"



# ------------------------------------------------------------------------
teacher = Teacher(envs, flags).to(device)
student = Student(envs, flags).to(device)
# ------------------------------------------------------------------------


# ALGO Logic: Storage setup
obs = torch.zeros((flags.num_steps, flags.num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((flags.num_steps, flags.num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((flags.num_steps, flags.num_envs)).to(device)
rewards = torch.zeros((flags.num_steps, flags.num_envs)).to(device)
dones = torch.zeros((flags.num_steps, flags.num_envs)).to(device)
values = torch.zeros((flags.num_steps, flags.num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
flags.global_step = 0
flags.start_time = time.time()
next_obs = torch.Tensor(envs.reset()).to(device)
next_done = torch.zeros(flags.num_envs).to(device)
num_updates = flags.total_timesteps // flags.batch_size

print("---------------")
print("---------------")

for update in range(1, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if flags.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * flags.learning_rate
        teacher.optimizer.param_groups[0]["lr"] = lrnow
        student.optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, flags.num_steps):
        flags.global_step += 1 * flags.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = teacher.agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        for item in info:
            if "episode" in item.keys():
                print(f"global_step={flags.global_step}, episodic_return={item['episode']['r']}")
                writer.add_scalar("charts/episodic_return", item["episode"]["r"], flags.global_step)
                writer.add_scalar("charts/episodic_length", item["episode"]["l"], flags.global_step)
                break

    # bootstrap value if not done
    with torch.no_grad():
        next_value = teacher.agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(flags.num_steps)):
            if t == flags.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + flags.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + flags.gamma * flags.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(flags.batch_size)
    clipfracs = []
    for epoch in range(flags.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, flags.batch_size, flags.minibatch_size):
            end = start + flags.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = teacher.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()
            teacher.train(
                    envs, 
                    obs, 
                    next_obs, 
                    actions, 
                    logprobs,
                    rewards, 
                    next_done, 
                    values, 
                    dones,
                    writer,
            )
            
            

envs.close()
writer.close()