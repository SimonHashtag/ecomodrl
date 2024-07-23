import random
import time

import numpy as np
import torch
import torch.nn as nn
from typing import Union

class PPO:
    def __init__(
        self,
        envs,
        agent,
        optimizer,
        writer,
        stopper,
        args,        
        run_name,
        device="cpu",
        seed=1,
        floattype=torch.float64,
        ):
        self.envs = envs
        self.agent = agent
        self.optimizer = optimizer
        self.writer = writer
        self.stopper = stopper
        
        # Hyperparameters
        self.exp_name = args.exp_name
        self.num_steps = args.num_steps
        self.num_envs = args.num_envs
        self.num_iterations = args.num_iterations
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.minibatch_size = args.minibatch_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.update_epochs = args.update_epochs
        self.clip_coef = args.clip_coef
        self.norm_adv = args.norm_adv
        self.clip_vloss = args.clip_vloss
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        self.target_kl = args.target_kl
        self.anneal_lr = args.anneal_lr
        
        self.run_name = run_name
        self.device = device
        self.seed = seed
        self.floattype = floattype
        
        self.seeding()

        
    def seeding(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def train(
        self, 
        save_model=False):
        # Storage setup
        env_steps = (self.num_steps, self.num_envs)
        obs = torch.zeros(env_steps + self.envs.single_observation_space.shape).to(self.device, dtype=self.floattype)
        real_obs = torch.zeros(env_steps + self.envs.single_observation_space.shape).to(self.device, dtype=self.floattype)
        actions = torch.zeros(env_steps + self.envs.single_action_space.shape).to(self.device, dtype=self.floattype)
        logprobs = torch.zeros(env_steps).to(self.device, dtype=self.floattype)
        rewards = torch.zeros(env_steps).to(self.device, dtype=self.floattype)
        dones = torch.zeros(env_steps).to(self.device)
        truncs = torch.zeros(env_steps).to(self.device)
        values = torch.zeros(env_steps).to(self.device, dtype=self.floattype)
        
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=[self.seed]*self.num_envs) # set the same training seed for all parallel envs
        next_obs = torch.Tensor(next_obs).to(self.device, dtype=self.floattype)
        real_next_obs = next_obs.detach().clone()
        next_done = torch.zeros(self.num_envs).to(self.device)
        next_trunc = torch.zeros(self.num_envs).to(self.device)
        
        for iteration in range(1, self.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                real_obs[step] = real_next_obs
                dones[step] = next_done
                truncs[step] = next_trunc
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                
                # Deal with truncated observations
                real_next_obs = next_obs.copy()
                for idx, trunc in enumerate(truncations):
                    if trunc:
                        real_next_obs[idx] = infos["final_observation"][idx]
                
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, real_next_obs, next_done, next_trunc = torch.Tensor(next_obs).to(self.device, dtype=self.floattype), torch.Tensor(real_next_obs).to(self.device, dtype=self.floattype), torch.Tensor(terminations).to(self.device), torch.Tensor(truncations).to(self.device)      

                self.writer.log_env_stats(global_step, infos)
                
            if self.stopper(self.writer.eps_buffer):
                break
            self.writer.empty_eps_buffer()
                
            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device, dtype=self.floattype)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminated = 1.0 - next_done
                        nextnontruncated = 1.0 - next_trunc
                        nextvalues = next_value
                    else:
                        nextnonterminated = 1.0 - dones[t + 1]
                        nextnontruncated = 1.0 - truncs[t + 1]
                        nextvalues = torch.where(truncs[t + 1] == 1.0, self.agent.get_value(real_obs[t + 1]).flatten(), values[t + 1])
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminated - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * torch.logical_and(nextnonterminated, nextnontruncated) * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for _ in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            self.writer.log_training_stats(
                global_step, 
                v_loss, 
                pg_loss,
                entropy_loss, 
                old_approx_kl,
                approx_kl,
                clipfracs,
                explained_var,
                start_time,
                self.optimizer.param_groups[0]["lr"])
        if save_model:
            model_path = f"runs/{self.run_name}/{self.exp_name}.ecomodrl_model"
            torch.save(self.agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
    
    def evaluate(
        self,
        eval_seeds: Union[int, list],
        eval_steps: int):
        # Configure logging label
        self.writer.is_eval(True)
        
        eval_seeds = [eval_seeds] if isinstance(eval_seeds, int) else eval_seeds
        
        # Start evaluation for #eval_steps over #eval_seeds
        for eval_seed in eval_seeds:
            self.seed = eval_seed
            self.seeding()
            next_obs, _ = self.envs.reset(seed=[self.seed]*self.num_envs)
            next_obs = torch.Tensor(next_obs).to(self.device, dtype=self.floattype)
            for step in range(1, eval_steps + 1):
                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(next_obs)
                next_obs, _, _, _, infos = self.envs.step(action.cpu().numpy())
                        
                next_obs = torch.Tensor(next_obs).to(self.device, dtype=self.floattype)
                self.writer.log_env_stats(step, infos)
            self.writer.empty_eps_buffer()
        self.writer.is_eval(False)