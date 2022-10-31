import torch 
import torch.nn as nn 
from . import Agent
import time
import numpy as np 

class Student():
    def __init__(self, envs, flags):
        self.flags = flags
        self.teacher_flags = flags.teacher
        self.agent = Agent(envs)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.teacher_flags.learning_rate, eps=1e-5)
    
    def to(self, device):
        self.agent.to(device)
        return self
    
    def train(self, envs, 
                    obs, 
                    next_obs, 
                    actions, 
                    logprobs,
                    rewards, 
                    next_done, 
                    values, 
                    dones,
                    writer,
                    
            ):
        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.flags.device)
            lastgaelam = 0
            for t in reversed(range(self.flags.num_steps)):
                if t == self.flags.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.flags.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.flags.gamma * self.flags.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.flags.batch_size)
        clipfracs = []
        for epoch in range(self.flags.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.flags.batch_size, self.flags.minibatch_size):
                end = start + self.flags.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.flags.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.flags.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.flags.clip_coef, 1 + self.flags.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.flags.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.flags.clip_coef,
                        self.flags.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.flags.ent_coef * entropy_loss + v_loss * self.flags.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.flags.max_grad_norm)
                self.optimizer.step()

            if self.flags.target_kl is not None:
                if approx_kl > self.flags.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.flags.global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), self.flags.global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), self.flags.global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), self.flags.global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.flags.global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), self.flags.global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.flags.global_step)
        writer.add_scalar("losses/explained_variance", explained_var, self.flags.global_step)
        print("SPS:", int(self.flags.global_step / (time.time() - self.flags.start_time)))
        writer.add_scalar("charts/SPS", int(self.flags.global_step / (time.time() - self.flags.start_time)), self.flags.global_step)
