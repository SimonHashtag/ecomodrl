# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from dehb import DEHB

from ecomodrl.envs.casskoopmans import CassKoopmansEnv

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    eval_steps: int = 10000
    """The number of steps run for evaluation of training run"""
    min_fidelity: int = 1e4
    """Minimum number of total steps during training"""
    max_fidelity: int = 1e5
    """Maximum number of total steps during training"""
    eta: float = 3.0
    """Parameter for Hyperband controlling early stopping aggressiveness"""
    mutation_factor: float = 0.5
    """Parameter for Hyperband controlling mutation strength between sampled configurations"""
    output_path: str = f"./tune_results/{exp_name}"
    """Directory where tune results are stored"""
    n_workers: int = 10
    """Number of DEHB workers"""
    verbose: bool = True
    """if true, DEHB tuning verbose"""
    runtime: int = 1000
    """Total budget of DEHB optimization in seconds"""

    # Algorithm specific arguments
    env_id: str = "CassKoopmans-v0"
    """the environment id of the task"""
    max_episode_steps: str = 100
    """The number of steps run in each env instance before resetting"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    learning_starts: int = min_fidelity/2
    """timestep to start learning"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""

def get_configspace(seed=None):
    cs = CS.ConfigurationSpace(seed)
    
    batch_size = CSH.CategoricalHyperparameter("batch_size", choices=[64, 128, 256, 512, 1024])
    policy_lr = CSH.UniformFloatHyperparameter("policy_lr", lower=0.000001, upper=0.01, log=True)
    q_lr = CSH.UniformFloatHyperparameter("q_lr", lower=0.00001, upper=0.1, log=True)
    tau = CSH.UniformFloatHyperparameter("tau", lower=0.0001, upper=0.05, log=False)
    cs.add_hyperparameters([batch_size, policy_lr, q_lr, tau])
    
    alpha = CSH.UniformFloatHyperparameter("alpha", lower=0.01, upper=1., log=False)
    autotune = CSH.CategoricalHyperparameter("autotune", choices=[True, False])
    cs.add_hyperparameters([alpha, autotune])
    
    return cs

def make_env(env_id, max_episode_steps, seed):
    def thunk():
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
def objective_fun(config, fidelity, **kwargs):
    envs = kwargs["envs"]
    device = kwargs["device"]
    buffer_size = kwargs["buffer_size"]
    learning_starts = kwargs["learning_starts"]
    gamma = kwargs["gamma"]
    seed = kwargs["seed"]
    policy_frequency = kwargs["policy_frequency"]
    
    batch_size = config["batch_size"]
    policy_lr = config["policy_lr"]
    q_lr = config["q_lr"]
    tau = config["tau"]
    alpha = config["alpha"]
    autotune = config["autotune"]
    
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=policy_lr)

    # Automatic entropy tuning
    if autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=q_lr)
    else:
        alpha = alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=seed)
    for global_step in range(int(fidelity)):
        # ALGO LOGIC: put action logic here
        if global_step < learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > learning_starts:
            data = rb.sample(batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        cost = time.time()-start_time
    
    # evaluate algo (test runs)   
    eps_return = list()
    eps_length = list()
    
    for s in range(seed, seed + 10):
        next_obs, _ = envs.reset(seed=s)
        next_obs = torch.Tensor(next_obs).to(device)
        for _ in range(0, args.eval_steps//10):
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                    
            next_obs = torch.Tensor(next_obs).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" in info:
                        eps_return.append(info["episode"]["r"])
                        eps_length.append(info["episode"]["l"])      
    envs.close()
    res = {
    "fitness": -np.mean(eps_return, dtype=np.float64),
    "cost": cost,
    "info": {"eps_length": np.mean(eps_length), "training_steps": fidelity-learning_starts}
    }
    return res


if __name__ == "__main__":

    args = tyro.cli(Args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.max_episode_steps, args.seed)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    cs = get_configspace(args.seed)
    dimensions = len(cs.get_hyperparameters())
    
    dehb = DEHB(f=objective_fun, 
                cs=cs, 
                dimensions=dimensions, 
                min_fidelity=args.min_fidelity,
                max_fidelity=args.max_fidelity,
                eta=args.eta,
                mutation_factor=args.mutation_factor,
                output_path=args.output_path,
                client=None,
                n_workers=args.n_workers)
    traj, runtime, history = dehb.run(total_cost=args.runtime, 
                                      verbose=args.verbose,
                                      envs=envs,
                                      device=device,
                                      buffer_size=args.buffer_size,
                                      learning_starts=args.learning_starts,
                                      gamma=args.gamma,
                                      seed=args.seed,
                                      policy_frequency=args.policy_frequency)
    print("finished")