## Differential Evolution Hyperband (DEHB) tuning of DDPG algorithm in Cass Koopmans environment
import os
import random
import time
import pickle
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

from ecomodrl.algorithms.ddpg import DDPG
from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.utils.stopper import Stopper
from ecomodrl.utils.logger import DDPGLogger

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from dehb import DEHB


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the configspace"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    eval_steps: int = 10000
    """The number of steps run for evaluation of tuning run"""
    min_fidelity: int = 1e4
    """Minimum number of total steps during tuning"""
    max_fidelity: int = 2e5
    """Maximum number of total steps during tuning"""
    eta: float = 3.0
    """Parameter for Hyperband controlling early stopping aggressiveness"""
    mutation_factor: float = 0.5
    """Parameter for Hyperband controlling mutation strength between sampled configurations"""
    output_path: str = f"./tune_results/{exp_name}"
    """Directory where tune results are stored"""
    n_workers: int = 20
    """Number of DEHB workers"""
    log_level: str = "INFO"
    """the verbosity of the loguru logger, see [their website](https://loguru.readthedocs.io/en/stable/api/logger.html#levels)"""
    runtime: int = 1000
    """Total budget of DEHB optimization in seconds"""

    # Algorithm specific arguments
    env_id: str = "CassKoopmans-v0"
    """the environment id of the economic model"""
    max_episode_steps: str = 100
    """The number of steps run in each env instance before resetting"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    learning_starts: int = min_fidelity*0.9
    """timestep to start learning"""

def get_configspace(seed=None):
    cs = CS.ConfigurationSpace(seed)
    
    batch_size = CSH.CategoricalHyperparameter("batch_size", choices=[64, 128, 256, 512, 1024])
    learning_rate = CSH.UniformFloatHyperparameter("learning_rate", lower=0.000001, upper=0.01, log=True)
    tau = CSH.UniformFloatHyperparameter("tau", lower=0.0001, upper=0.05, log=False)
    cs.add_hyperparameters([batch_size, learning_rate, tau])
    
    exploration_noise = CSH.UniformFloatHyperparameter("exploration_noise", lower=0.001, upper=1., log=False)
    cs.add_hyperparameters([exploration_noise])
    
    return cs

def make_env(env_id, max_episode_steps, seed):
    def thunk():
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
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
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
    
def objective_fun(config, fidelity, **kwargs):
    args=kwargs["args"]
    
    args.batch_size = config["batch_size"]
    args.learning_rate = config["learning_rate"]
    args.tau = config["tau"]
    args.exploration_noise = config["exploration_noise"]
    args.total_timesteps = fidelity

    device = kwargs["device"]
    tuning_seeds = kwargs["tuning_seeds"]
    eval_seeds = kwargs["eval_seeds"]
    tune_time = kwargs["tune_time"]
    
    # Empty lists for storing episode returns during evaluation
    eps_return = list()
    eps_length = list()
    
    start_time = time.time()
    
    for tuning_seed in tuning_seeds:
        # logger setup
        run_name = f"{args.env_id}__{args.exp_name}__{tuning_seed}__{int(tune_time)}"
        writer = DDPGLogger(f"tune_results/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        
        # seeding (DO NOT MOVE THIS -> Must be before env and agent setup)
        random.seed(tuning_seed)
        np.random.seed(tuning_seed)
        torch.manual_seed(tuning_seed)
        
        # env setup
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.max_episode_steps, tuning_seed)])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        # Early stopping mechanism setup
        min_iter_esm = (args.total_timesteps-args.learning_starts)//(10*args.max_episode_steps)
        stopper = Stopper(
            max_iter=args.total_timesteps, 
            min_iter=min_iter_esm,
            early_stopping=40,
            min_episode_length=args.max_episode_steps*(1-0.05),
            max_deviation=0.05,
            )
        
        ddpg = DDPG(envs,
                    Actor,
                    QNetwork,
                    writer, 
                    args,
                    run_name,
                    device,
                    tuning_seed,
                    stopper
                    )
        ddpg.train()
        ddpg.evaluate(eval_seeds, args.eval_steps)
        eps_buffer = writer.get_eps_buffer()
        eps_return = eps_return.append(eps_buffer["episode_returns"])
        eps_length = eps_length.append(eps_buffer["episode_lens"])
        
        envs.close()
        writer.close()
    
    cost = time.time()-start_time
    
    res = {
    "fitness": -np.mean(eps_return, dtype=np.float64),
    "cost": cost,
    "info": {"eps_length": np.mean(eps_length), "training_steps": fidelity-args.learning_starts}
    }
    return res


if __name__ == "__main__":
    args = tyro.cli(Args)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    tuning_seeds = range(5)
    eval_seeds = range(5, 15)
    
    cs = get_configspace(args.seed)
    dimensions = len(list(cs.values()))
    
    tune_time = time.time()
    
    dehb = DEHB(f=objective_fun, 
                cs=cs, 
                dimensions=dimensions, 
                min_fidelity=args.min_fidelity,
                max_fidelity=args.max_fidelity,
                eta=args.eta,
                mutation_factor=args.mutation_factor,
                output_path=args.output_path,
                client=None,
                n_workers=args.n_workers,
                log_level=args.log_level)
    traj, runtime, history = dehb.run(
        total_cost=args.runtime, 
        device=device,
        tuning_seeds=tuning_seeds,
        eval_seeds=eval_seeds,
        tune_time=tune_time,
        args=args
        )
    # Saving optimisation trace history
    name = time.strftime("%x %X %Z", time.localtime(dehb.start))
    name = name.replace("/", "-").replace(":", "-").replace(" ", "_")
    dehb.logger.info("Saving optimisation trace history...")
    with open(os.path.join(args.output_path, f"history_{name}.pkl"), "wb") as f:
        pickle.dump(history, f)