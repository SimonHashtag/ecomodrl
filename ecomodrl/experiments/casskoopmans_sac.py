import os
import random
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

from ecomodrl.algorithms.sac import SAC
from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.utils.stopper import Stopper
from ecomodrl.utils.hyperparam_import import import_tuned_hp
from ecomodrl.utils.logger import SACLogger

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seeds: int = field(default_factory=lambda: list(range(15, 16)))
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    
    # Algorithm specific arguments
    env_id: str = "CassKoopmans-v0"
    """the environment id of the task"""
    max_episode_steps: str = 100
    """The number of steps run in each env instance before resetting"""
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 10000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    
    # Evaluation specific arguments
    eval_seeds: list = field(default_factory=lambda: list(range(100, 110)))
    """A list of seeds for evaluation"""
    eval_steps: int = 100
    """The number of environment steps per eval seed"""

def make_env(env_id, max_episode_steps, seed):
    def thunk():
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


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

if __name__ == "__main__":
    # load (tuned) hyperparameters
    args = tyro.cli(Args)
    args = import_tuned_hp(args, args.exp_name)
    
    floattype = torch.float32
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    for seed in args.seeds:
        # logger setup
        run_name = f"{args.env_id}__{args.exp_name}__{seed}__{int(time.time())}"
        writer = SACLogger(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # seeding (DO NOT MOVE THIS -> Must be before env and agent setup)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # env setup
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.max_episode_steps, seed)])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        # env setup
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.max_episode_steps, seed)])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
        
        # Early stopping mechanism
        min_iter_esm = (args.total_timesteps-args.learning_starts)//(3*args.max_episode_steps)
        stopper = Stopper(
            max_iter=args.total_timesteps, 
            min_iter=min_iter_esm,
            early_stopping=50,
            min_episode_length=args.max_episode_steps*(1-0.05),
            max_deviation=0.05,
            )
        
        sac = SAC(
            envs,
            Actor,
            SoftQNetwork,
            writer,
            stopper,
            args,
            run_name,
            device,
            seed,
            floattype,
        )
        sac.train(args.save_model)
        sac.evaluate(args.eval_seeds, args.eval_steps)

        envs.close()
        writer.close()