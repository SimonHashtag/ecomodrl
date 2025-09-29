## Differential Evolution Hyperband (DEHB) tuning of PPO algorithm in Cass Koopmans environment

import os
import random
import time
import pickle
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.beta import Beta
import tyro
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
    max_fidelity: int = 1e6
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
    runtime: int = 500
    """Total budget of DEHB optimization in seconds"""

    # Algorithm specific arguments
    env_id: str = "CassKoopmans-v0"
    """the id of the environment"""
    max_episode_steps: int = 100
    """The number of steps run in each env instance before resetting"""
    num_envs: int = 5
    """the number of parallel game environments"""
    gamma: float = 0.95
    """the discount factor gamma"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    
def get_configspace(seed=None):
    cs = CS.ConfigurationSpace(seed)
    
    num_steps = CSH.CategoricalHyperparameter("num_steps", choices=[128, 256, 512, 1024, 2048])
    num_minibatches = CSH.CategoricalHyperparameter("num_minibatches", choices=[1, 2, 4, 8, 16, 32])
    update_epochs = CSH.UniformIntegerHyperparameter("update_epochs", lower=5, upper=20, log=False)
    cs.add_hyperparameters([num_steps, num_minibatches, update_epochs])
    
    learning_rate = CSH.UniformFloatHyperparameter("learning_rate", lower=0.000001, upper=0.01, log=True)
    anneal_lr = CSH.CategoricalHyperparameter("anneal_lr", choices=[True, False])
    cs.add_hyperparameters([anneal_lr, learning_rate])
    
    gae_lambda = CSH.UniformFloatHyperparameter("gae_lambda", lower=0.8, upper=1.0, log=False)
    norm_adv = CSH.CategoricalHyperparameter("norm_adv", choices=[True, False])
    clip_vloss = CSH.CategoricalHyperparameter("clip_vloss", choices=[True, False])
    clip_coef = CSH.UniformFloatHyperparameter("clip_coef", lower=0.0, upper=0.5, log=False)
    vf_coef = CSH.UniformFloatHyperparameter("vf_coef", lower=0.001, upper=5.0, log=True)
    ent_coef = CSH.UniformFloatHyperparameter("ent_coef", lower=0.001, upper=5.0, log=True)
    cs.add_hyperparameters([gae_lambda, norm_adv, clip_coef, clip_vloss, vf_coef, ent_coef])
    
    return cs


def make_env(env_id, max_episode_steps, gamma, floattype):
    def thunk():
        env = CassKoopmansEnv(max_episode_steps=max_episode_steps, floattype=floattype)
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


def objective_fun(config, fidelity, **kwargs):
    num_steps = int(config["num_steps"])
    batch_size = int(args.num_envs * num_steps)
    minibatch_size = int(batch_size // int(config["num_minibatches"]))
    num_iterations = int(fidelity // batch_size)
    
    envs = kwargs["envs"]
    device = kwargs["device"]
    tuning_seeds = kwargs["tuning_seeds"]
    eval_seeds = kwargs["eval_seeds"]
    eval_steps = kwargs["eval_steps"]
    floattype = kwargs["floattype"]
    torch.set_default_dtype(floattype)
    
    # Empty lists for storing episode returns during testing
    eps_return = list()
    eps_length = list()
    
    for tuning_seed in tuning_seeds:
        # seeding
        random.seed(tuning_seed)
        np.random.seed(tuning_seed)
        torch.manual_seed(tuning_seed)

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((num_steps, args.num_envs) + envs.single_observation_space.shape).to(device, dtype=floattype)
        real_obs = torch.zeros((num_steps, args.num_envs) + envs.single_observation_space.shape).to(device, dtype=floattype)
        actions = torch.zeros((num_steps, args.num_envs) + envs.single_action_space.shape).to(device, dtype=floattype)
        logprobs = torch.zeros((num_steps, args.num_envs)).to(device, dtype=floattype)
        rewards = torch.zeros((num_steps, args.num_envs)).to(device, dtype=floattype)
        dones = torch.zeros((num_steps, args.num_envs)).to(device)
        truncs = torch.zeros((num_steps, args.num_envs)).to(device)
        values = torch.zeros((num_steps, args.num_envs)).to(device, dtype=floattype)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start = time.time()
        next_obs, _ = envs.reset(seed=tuning_seed)
        next_obs = torch.Tensor(next_obs).to(device, dtype=floattype)
        real_next_obs = next_obs.detach().clone()
        next_done = torch.zeros(args.num_envs).to(device)
        next_trunc = torch.zeros(args.num_envs).to(device)

        for iteration in range(1, num_iterations+1):
            
            # Annealing the rate if instructed to do so.
            if config["anneal_lr"]:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * config["learning_rate"]
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                real_obs[step] = real_next_obs
                dones[step] = next_done
                truncs[step] = next_trunc

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                
                # Deal with truncated observations
                real_next_obs = next_obs.copy()
                for idx, trunc in enumerate(truncations):
                    if trunc:
                        real_next_obs[idx] = infos["final_observation"][idx]
                
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, real_next_obs, next_done, next_trunc = torch.Tensor(next_obs).to(device, dtype=floattype), torch.Tensor(real_next_obs).to(device, dtype=floattype), torch.Tensor(terminations).to(device), torch.Tensor(truncations).to(device)      
            
            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device, dtype=floattype)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminated = 1.0 - next_done
                        nextnontruncated = 1.0 - next_trunc
                        nextvalues = next_value
                    else:
                        nextnonterminated = 1.0 - dones[t + 1]
                        nextnontruncated = 1.0 - truncs[t + 1]
                        nextvalues = torch.where(truncs[t + 1] == 1.0, agent.get_value(real_obs[t + 1]).flatten(), values[t + 1])
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminated - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * config["gae_lambda"] * torch.logical_and(nextnonterminated, nextnontruncated) * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []
            for _ in range(int(config["update_epochs"])):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        # old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > config["clip_coef"]).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if config["norm_adv"]:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["clip_coef"], 1 + config["clip_coef"])
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if config["clip_vloss"]:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -config["clip_coef"],
                            config["clip_coef"],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - config["ent_coef"] * entropy_loss + v_loss * config["vf_coef"]

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
        cost = time.time()-start        
        
        # evaluate algo (test runs)   
        for eval_seed in eval_seeds:
            next_obs, _ = envs.reset(seed=eval_seed)
            next_obs = torch.Tensor(next_obs).to(device, dtype=floattype)
            for step in range(0, eval_steps//10):
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(next_obs)
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                        
                next_obs = torch.Tensor(next_obs).to(device, dtype=floattype)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if "episode" in info:
                            eps_return.append(info["episode"]["r"])
                            eps_length.append(info["episode"]["l"])
        
    envs.close()
    
    res = {
        "fitness": -np.mean(eps_return, dtype=np.float64),
        "cost": cost,
        "info": {"eps_length": np.mean(eps_length), "training_steps": num_iterations}
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
        [make_env(args.env_id, args.max_episode_steps, args.gamma, np.float64) for _ in range(args.num_envs)]
    )
    
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
        floattype=torch.float64,
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
