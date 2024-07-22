# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import time
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta

from ecomodrl.algorithms.ppo import PPO
from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.utils.stopper import Stopper
from ecomodrl.utils.hyperparam_import import import_tuned_hp
from ecomodrl.utils.logger import PPOLogger

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] 
    """the name of this experiment"""
    seeds: list = list(range(15, 25))
    """A list of training seeds of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "CassKoopmans-v0"
    """the id of the environment"""
    max_episode_steps: str = 100
    """The number of steps run in each env instance before resetting"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 0.002375
    """the learning rate of the optimizer"""
    num_envs: int = 5
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.95
    """the discount factor gamma"""
    gae_lambda: float = 0.85
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 6
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.31
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.003
    """coefficient of the entropy"""
    vf_coef: float = 0.05
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    # Evaluation specific arguments
    eval_seeds: list = list(range(100, 200, 1))
    """A list of seeds for evaluation"""
    eval_steps: int = 100
    """The number of environment steps per eval seed"""


def make_env(env_id, max_episode_steps, gamma):
    def thunk():
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)*2), std=0.01),
            nn.Softplus(), # softplus function + 1 to transform action_params to values > 1 (for beta distribution)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_params = self.actor(x) + 1
        alphas = action_params[:,0::2]
        betas = action_params[:,1::2]
        probs = Beta(alphas, betas)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    

if __name__ == "__main__":
    # load (tuned) hyperparameters
    args = tyro.cli(Args)
    args = import_tuned_hp(args, args.exp_name)
    
    # Setup up remaining hyperparameters
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    floattype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    for seed in args.seeds:
        # logger setup
        run_name = f"{args.env_id}__{args.exp_name}__{seed}__{int(time.time())}"
        writer = PPOLogger(log_dir=f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        
        # seeding (DO NOT MOVE THIS -> Must be before env and agent setup)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.max_episode_steps, args.gamma) for _ in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
        
        # agent setup
        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        
        # Early stopping mechanism setup
        stopper = Stopper(
            max_iter=args.num_iterations, 
            min_iter=args.num_iterations//10,
            early_stopping=15,
            min_episode_length=args.max_episode_steps*(1-0.05),
            max_deviation=0.01,
            )
        
        #for s in range(10):
        ppo = PPO(
            envs,
            agent,
            optimizer,
            writer,
            stopper,
            args,
            run_name,
            device,
            seed,
            floattype)
        ppo.train()
        ppo.evaluate(args.eval_seeds, args.eval_steps)
        
        envs.close()
        writer.close()
        