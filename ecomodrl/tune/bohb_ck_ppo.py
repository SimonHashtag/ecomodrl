## Bayesian Optimization Hyperband (BOHB) tuning of PPO algorithm in Cass Koopmans environment

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta

from ray import air, tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

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
    env_id: str = "CassKoopmans-v0"
    """the id of the environment"""
    max_episode_steps: str = 100
    """The number of steps run in each env instance before resetting"""

    # Algorithm specific arguments
    total_timesteps: int = 200000
    """total timesteps of the experiments"""
    num_envs: int = 5
    """the number of parallel game environments"""
    gamma: float = 0.95
    """the discount factor gamma"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, max_episode_steps, gamma):
    def thunk():
        env = CassKoopmansEnv(max_episode_steps=max_episode_steps)
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


def run_ppo(config):
    args.batch_size = int(args.num_envs * int(int(config["num_steps"])))
    args.minibatch_size = int(args.batch_size // int(config["num_minibatches"]))
    args.num_iterations = args.total_timesteps // args.batch_size
    floattype = torch.float32
    
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.max_episode_steps, args.gamma) for _ in range(args.num_envs)]
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    envs.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((int(int(config["num_steps"])), args.num_envs) + envs.single_observation_space.shape).to(device, dtype=floattype)
    real_obs = torch.zeros((int(config["num_steps"]), args.num_envs) + envs.single_observation_space.shape).to(device, dtype=floattype)
    actions = torch.zeros((int(config["num_steps"]), args.num_envs) + envs.single_action_space.shape).to(device, dtype=floattype)
    logprobs = torch.zeros((int(config["num_steps"]), args.num_envs)).to(device, dtype=floattype)
    rewards = torch.zeros((int(config["num_steps"]), args.num_envs)).to(device, dtype=floattype)
    dones = torch.zeros((int(config["num_steps"]), args.num_envs)).to(device)
    truncs = torch.zeros((int(config["num_steps"]), args.num_envs)).to(device)
    values = torch.zeros((int(config["num_steps"]), args.num_envs)).to(device, dtype=floattype)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    # start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device, dtype=floattype)
    real_next_obs = next_obs.detach().clone()
    next_done = torch.zeros(args.num_envs).to(device)
    next_trunc = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        eps_return = list()
        eps_length = list()
        
        # Annealing the rate if instructed to do so.
        if config["anneal_lr"]:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * config["learning_rate"]
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, int(config["num_steps"])):
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
            
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" in info:
                        eps_return.append(info["episode"]["r"])
                        eps_length.append(info["episode"]["l"])
        
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device, dtype=floattype)
            lastgaelam = 0
            for t in reversed(range(int(config["num_steps"]))):
                if t == int(config["num_steps"]) - 1:
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
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _ in range(int(config["update_epochs"])):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
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
        
        session.report({"eps_return": np.mean(eps_return), "eps_length": np.mean(eps_length), "training_steps": iteration})

    envs.close()
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.smoke_test = False
    torch.backends.cudnn.deterministic = args.torch_deterministic

    trial_space = {
        #"env": args.env_id,
        "learning_rate": tune.choice([3e-5, 3e-4, 6e-4, 1e-3]),  # tune.loguniform(5e-5, 1e-3),
        "anneal_lr": tune.choice([True, False]),
        "num_steps": tune.choice([512, 1024, 2048, 4096]),
        "num_minibatches": tune.choice([1, 4, 8, 16, 32]),  # tune.sample_from(lambda spec: np.random.randint(1, 2 * np.sqrt(spec.config.num_ro_steps))),
        "update_epochs": tune.choice([1, 5, 10, 15]),
        "gae_lambda": tune.choice([0.85, 0.9, 0.95, 1.0]),
        "norm_adv": tune.choice([True, False]),
        "clip_vloss": tune.choice([True, False]),
        "clip_coef": tune.choice([0.0, 0.1, 0.15, 0.2, 0.3]),
        "vf_coef": tune.choice([0.1, 0.25, 0.5, 1.0]),
        "ent_coef": tune.choice([0.0, 0.1, 0.5, 1.0]),
        "target_kl": None,  # tune.choice([None, tune.quniform(0.003, 0.03, 0.001)])
    }

    algo = HyperOptSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=10)

    scheduler = AsyncHyperBandScheduler(
        time_attr="training_steps",
        max_t=2 if args.smoke_test else 250,
        grace_period=1 if args.smoke_test else 20,
    )
    stopping_criteria = {"training_steps": 2 if args.smoke_test else 250}
    
    storage_path = os.path.abspath("./tune_results")
    exp_name = "tune_ck_ppo_pb2"
    path = os.path.join(storage_path, exp_name)

    if tune.Tuner.can_restore(path):
        tuner = tune.Tuner.restore(path, trainable=run_ppo)
    else:
        tuner = tune.Tuner(
            run_ppo,
            param_space=trial_space,
            tune_config=tune.TuneConfig(
                num_samples=2 if args.smoke_test else 500, 
                metric="eps_return", 
                mode="max", 
                search_alg=algo, 
                scheduler=scheduler,
                trial_dirname_creator=lambda trial: trial.trial_id,
                trial_name_creator=lambda trial: f"{exp_name}-{trial.trial_id}"),
            run_config=air.RunConfig(
                failure_config=air.FailureConfig(),
                storage_path=storage_path,
                name=exp_name,
                stop=stopping_criteria,
                verbose=2,
            ),
        )

    results = tuner.fit()

    best_result = results.get_best_result(metric="eps_return", mode="max")  # Get best result object
    best_config = best_result.config  # Get best trial's hyperparameters
    best_logdir = best_result.log_dir  # Get best trial's logdir
    best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
    best_metrics = best_result.metrics  # Get best trial's last results
    best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe
    print(best_result.config)
    print(best_result_df)

    results_df = results.get_dataframe()
    results_df.to_csv()