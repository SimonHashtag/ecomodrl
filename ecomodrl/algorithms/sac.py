import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Union, Optional
from stable_baselines3.common.buffers import ReplayBuffer

class SAC:
    def __init__(
        self,
        envs,
        Actor,
        QNetwork,
        writer,
        stopper,
        args,
        run_name,
        device,
        seed,
        floattype):
        self.envs = envs
        self.writer = writer
        self.stopper = stopper
        
        # hyperparameters
        self.policy_lr = args.policy_lr
        self.q_lr = args.q_lr
        self.gamma = args.gamma
        self.total_timesteps = args.total_timesteps
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.learning_starts = args.learning_starts
        self.policy_frequency = args.policy_frequency
        self.alpha = args.alpha
        self.autotune = args.autotune
        self.buffer_size = args.buffer_size
        self.max_episode_steps = args.max_episode_steps
        
        self.exp_name = args.exp_name        
        self.run_name = run_name
        self.device = device
        self.seed = seed
        self.floattype = floattype
        
        # agent setup
        self.actor = Actor(self.envs).to(self.device)
        self.qf1 = QNetwork(self.envs).to(self.device)
        self.qf2 = QNetwork(self.envs).to(self.device)
        self.qf1_target = QNetwork(self.envs).to(self.device)
        self.qf2_target = QNetwork(self.envs).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr)
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)
        
        # Automatic entropy tuning
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.envs.single_action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr)
        
        # replay buffer setup
        self.rb = ReplayBuffer(
            self.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )
        
        self.seeding()

    def seeding(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def train(self, save_model=False):
        start_time = time.time()
        obs, _ = self.envs.reset(seed=self.seed)
        for global_step in range(self.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.learning_starts:
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            self.writer.log_env_stats(global_step, infos)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
            
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.learning_starts:
                data = self.rb.sample(self.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
                    qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if global_step % self.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        self.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.actor.get_action(data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        if self.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations)
                            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()
                        else:
                            alpha_loss = None

                # update the target networks
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if global_step % 100 == 0:
                    self.writer.log_training_stats(
                        global_step, 
                        qf1_a_values,
                        qf2_a_values,
                        qf1_loss,
                        qf2_loss,
                        qf_loss,
                        actor_loss,
                        self.alpha,
                        self.autotune,
                        alpha_loss,
                        start_time)
                    
                    # evaluate training
                    self.evaluate(self.seed, self.max_episode_steps, global_step)
                    if (self.stopper(self.writer.eps_buffer)):
                        break
        
        if save_model:
            model_path = f"runs/{self.run_name}/{self.exp_name}.ecomodrl_model"
            torch.save((self.actor.state_dict(), self.qf1.state_dict(), self.qf2.state_dict()), model_path)
            print(f"model saved to {model_path}")
                    
    def evaluate(self, eval_seeds: Union[int, list], eval_steps: int, global_step: Optional[int]=None):
        # Clear episode stats buffer
        self.writer.empty_eps_buffer()
        # Configure logging label
        self.writer.is_eval(True)
        
        eval_seeds = [eval_seeds] if isinstance(eval_seeds, int) else eval_seeds
        # Start evaluation for #eval_steps over #eval_seeds
        for eval_seed in eval_seeds:
            self.seed = eval_seed
            self.seeding()
            obs, _ = self.envs.reset(seed=eval_seed)
            for step in range(1, eval_steps + 1):
                with torch.no_grad():
                    actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                next_obs, _, _, _, infos = self.envs.step(actions.detach().cpu().numpy())
                step_count = step if global_step is None else global_step
                self.writer.log_env_stats(step_count, infos)
                obs = next_obs
        # Reconfigure logging label
        self.writer.is_eval(False)