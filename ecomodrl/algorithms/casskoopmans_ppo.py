# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.utils.stopper import Stopper

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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "EcoModRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CassKoopmans-v0"
    """the id of the environment"""
    max_episode_steps: str = 100
    """The number of steps run in each env instance before resetting"""
    total_timesteps: int = 200000
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
    args = tyro.cli(Args)
    for s in range(10):
        args.seed = s
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        floattype = torch.float32

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.max_episode_steps, args.gamma) for _ in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        
        # Early stopping mechanism
        stopper = Stopper(
            max_iter=args.num_iterations, 
            min_iter=args.num_iterations//10,
            early_stopping=15,
            min_episode_length=args.max_episode_steps*(1-0.05),
            max_deviation=0.01,
            )
        eps_results = defaultdict(list)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device, dtype=floattype)
        real_obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device, dtype=floattype)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device, dtype=floattype)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device, dtype=floattype)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device, dtype=floattype)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        truncs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device, dtype=floattype)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device, dtype=floattype)
        real_next_obs = next_obs.detach().clone()
        next_done = torch.zeros(args.num_envs).to(device)
        next_trunc = torch.zeros(args.num_envs).to(device)

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
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
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("eps_stats/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("eps_stats/episodic_length", info["episode"]["l"], global_step)
                            writer.add_scalar("period_stats/mean_consumption_frac", info["consumption_frac"], global_step)
                            writer.add_scalar("period_stats/mean_next_capital", info["capital"], global_step)
                            writer.add_scalar("period_stats/mean_consumption", info["consumption"], global_step)
                            writer.add_scalar("period_stats/mean_reward", info["reward"], global_step)
                            eps_results["episode_returns"].append(info["episode"]["r"])
                            eps_results["episode_lens"].append(info["episode"]["l"])
                else:       
                    writer.add_scalar("period_stats/mean_consumption_frac", infos["consumption_frac"].mean(), global_step)
                    writer.add_scalar("period_stats/mean_next_capital", infos["capital"].mean(), global_step)
                    writer.add_scalar("period_stats/mean_consumption", infos["consumption"].mean(), global_step)
                    writer.add_scalar("period_stats/mean_reward", infos["reward"].mean(), global_step)
            
            if stopper(eps_results):
                break
            eps_results.clear()
            
            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device, dtype=floattype)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminated = 1.0 - next_done
                        nextnontruncated = 1.0 - next_trunc
                        nextvalues = next_value
                    else:
                        nextnonterminated = 1.0 - dones[t + 1]
                        nextnontruncated = 1.0 - truncs[t + 1]
                        nextvalues = torch.where(truncs[t + 1] == 1.0, agent.get_value(real_obs[t + 1]).flatten(), values[t + 1])
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminated - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * torch.logical_and(nextnonterminated, nextnontruncated) * lastgaelam
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
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("performance/value_loss", v_loss.item(), global_step)
            writer.add_scalar("performance/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("performance/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("performance/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("performance/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("performance/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("performance/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("performance/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("performance/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.ecomodrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
            from cleanrl_utils.evals.ppo_eval import evaluate

            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=Agent,
                device=device,
                gamma=args.gamma,
            )
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)

            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub

                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

        envs.close()
        writer.close()