# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
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

from ecomodrl.algorithms.td3 import TD3
from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.utils.stopper import Stopper
from ecomodrl.utils.hyperparam_import import import_tuned_hp
from ecomodrl.utils.logger import TD3Logger

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seeds: list = field(default_factory=lambda: list(range(15, 16)))
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "CassKoopmans-v0"
    """the id of the environment"""
    max_episode_steps: str = 100
    """The number of steps run in each env instance before resetting"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 10000
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    
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

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, np.prod(env.single_action_space.shape))
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
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


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
        writer = TD3Logger(f"runs/{run_name}")
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

        # Early stopping mechanism setup
        min_iter_esm = args.total_timesteps//(10*args.max_episode_steps)
        stopper = Stopper(
            max_iter=args.total_timesteps, 
            min_iter=min_iter_esm,
            early_stopping=20,
            min_episode_length=args.max_episode_steps*(1-0.05),
            max_deviation=0.01,
            )
        
        td3 = TD3(
            envs,
            Actor,
            QNetwork,
            writer,
            stopper,
            args,
            run_name,
            device,
            seed,
            floattype,
        )
        td3.train(args.save_model)
        td3.evaluate(args.eval_seeds, args.eval_steps)
        
        envs.close()
        writer.close()
        
        