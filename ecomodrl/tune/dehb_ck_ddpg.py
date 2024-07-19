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

def make_env(env_id, max_episode_steps, floattype):
    def thunk():
        env = gym.make(env_id, max_episode_steps=max_episode_steps, floattype=floattype)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
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
    def __init__(self, env, floattype):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=floattype)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=floattype)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
    
def objective_fun(config, fidelity, **kwargs):
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    tau = config["tau"]
    exploration_noise = config["exploration_noise"]
    
    envs = kwargs["envs"]
    device = kwargs["device"]
    buffer_size = kwargs["buffer_size"]
    learning_starts = kwargs["learning_starts"]
    gamma = kwargs["gamma"]
    tuning_seeds = kwargs["tuning_seeds"]
    eval_seeds = kwargs["eval_seeds"]
    eval_steps = kwargs["eval_steps"]
    torch_floattype = kwargs["torch_floattype"]
    torch.set_default_dtype(torch_floattype)
    
    
    # Empty lists for storing episode returns during testing
    eps_return = list()
    eps_length = list()
    
    for tuning_seed in tuning_seeds:
        # seeding
        random.seed(tuning_seed)
        np.random.seed(tuning_seed)
        torch.manual_seed(tuning_seed)
    
        actor = Actor(envs, torch_floattype).to(device)
        qf1 = QNetwork(envs).to(device)
        qf1_target = QNetwork(envs).to(device)
        target_actor = Actor(envs, torch_floattype).to(device)
        target_actor.load_state_dict(actor.state_dict())
        qf1_target.load_state_dict(qf1.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()), lr=learning_rate)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=learning_rate)
        
        rb = ReplayBuffer(
            buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )

        # TRY NOT TO MODIFY: start the game
        start_time = time.time()
        obs, _ = envs.reset(seed=tuning_seed)
        for global_step in range(int(fidelity)):
            # ALGO LOGIC: put action logic here
            if global_step < learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = actor(torch.Tensor(obs).to(device))
                    actions += torch.normal(0, actor.action_scale * exploration_noise)
                    actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

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
                    next_state_actions = target_actor(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (qf1_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                q_optimizer.zero_grad()
                qf1_loss.backward()
                q_optimizer.step()

                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        cost = time.time()-start_time
        
        # evaluate algo (test runs)   
        for eval_seed in eval_seeds:
            next_obs, _ = envs.reset(seed=eval_seed)
            next_obs = torch.Tensor(next_obs).to(device)
            for _ in range(0, eval_steps//10):
                with torch.no_grad():
                    actions = actor(torch.Tensor(obs).to(device))
                    actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
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
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    tuning_seeds = range(5)
    eval_seeds = range(5, 15)
    
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.max_episode_steps, np.float64)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    cs = get_configspace(args.seed)
    dimensions = len(list(cs.values()))
    
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
        envs=envs,
        device=device,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        gamma=args.gamma,
        torch_floattype=torch.float64,
        tuning_seeds=tuning_seeds,
        eval_seeds=eval_seeds,
        eval_steps=args.eval_steps
        )
    # Saving optimisation trace history
    name = time.strftime("%x %X %Z", time.localtime(dehb.start))
    name = name.replace("/", "-").replace(":", "-").replace(" ", "_")
    dehb.logger.info("Saving optimisation trace history...")
    with open(os.path.join(args.output_path, f"history_{name}.pkl"), "wb") as f:
        pickle.dump(history, f)