import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from gymnasium.wrappers import TimeLimit, TransformReward
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from ecomodrl.envs.casskoopmans import CassKoopmansEnv

# ─────────────────────────────────────────────────────────────
# 1. Environment setup
# ─────────────────────────────────────────────────────────────
MAX_EPISODE_STEPS = 30
MAX_EVAL_EPISODE_STEPS = 100
BETA = 0.95
SEED = 42
N_ENVS = 8

set_random_seed(SEED)

def make_env(max_episode_steps=MAX_EPISODE_STEPS, fixed_start=False, training=True):
    def _thunk():
        env = CassKoopmansEnv(
            max_episode_steps=max_episode_steps,
            fixed_start=fixed_start,
            seed=SEED
        )
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        if training:
            env = TransformReward(env, lambda r: max(r, -100))  # stability: avoid large negative rewards during training
        return env
    return _thunk

# Vectorized training env with observation & reward normalization
train_env = make_vec_env(make_env(fixed_start=False), n_envs=N_ENVS, seed=SEED)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=BETA)

# Sanity-check a single raw instance (not the normalized VecEnv)
check_env(TimeLimit(CassKoopmansEnv(seed=SEED), max_episode_steps=MAX_EPISODE_STEPS), warn=True)

# Separate eval env: same dynamics and TimeLimit, but fixed start for determinism
eval_env_raw = DummyVecEnv([make_env(max_episode_steps=MAX_EVAL_EPISODE_STEPS, fixed_start=True, training=False)])
eval_env = VecNormalize(eval_env_raw, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=1000.0, gamma=BETA)

# ─────────────────────────────────────────────────────────────
# 2. Define network & PPO hyperparameters
# ─────────────────────────────────────────────────────────────
policy_kwargs = dict(
    activation_fn=th.nn.ReLU, 
    net_arch=dict(pi=[64, 64], vf=[64, 64]), 
    log_std_init=-2.0,            # ↓ smaller initial exploration to avoid extremes
)

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    gamma=BETA,
    n_steps=MAX_EPISODE_STEPS,   # per-env rollout length; aligns with episode length
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.0,
    gae_lambda=0.95,
    clip_range=0.2,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
)

# ─────────────────────────────────────────────────────────────
# 3. Train with early stopping
# ─────────────────────────────────────────────────────────────
def evaluate(model, eval_env, beta=BETA, n_episodes=5):
    # Make sure eval env does not update running stats
    eval_env.training = False
    returns = []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        if isinstance(obs, tuple):  # compatibility with wrappers
            obs = obs[0]
        done = np.array([False])
        G = 0.0
        power = 1.0
        t = 0
        while not done.all():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            G += float(reward.mean()) * power
            power *= beta
            t += 1
        returns.append(G)
    return float(np.mean(returns))


n_eval_every = 2_000  # training steps between evaluations
patience = 5
min_delta = 1e-3
total_timesteps = 50_000

best_score = -np.inf
patience_counter = 0
reward_history = []
timesteps_so_far = 0
best_model_path = "./runs/best_model_ppo.zip"

print("────────────────────────────────────────────")
print("     PPO TRAINING LOOP WITH EARLY STOPPING")
print("────────────────────────────────────────────")

while timesteps_so_far < total_timesteps:
    # Train for a short chunk
    model.learn(total_timesteps=n_eval_every, reset_num_timesteps=False)
    timesteps_so_far += n_eval_every
    
    # Sync eval env stats with training env stats (so obs/reward normalizers match)
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    
    avg_return = evaluate(model, eval_env, n_episodes=1)    # n_episodes=1 since fixed start and deterministic policy and environment (if stochastic env, increase n_episodes and vary seeds!)
    reward_history.append((timesteps_so_far, avg_return))
    print(f"[{timesteps_so_far:7d}] avg true return: {avg_return: .6f} | best: {best_score: .6f} | patience: {patience_counter}/{patience}")

    if avg_return > best_score + min_delta:
        best_score = avg_return
        model.save(best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            model = PPO.load(best_model_path, env=train_env)  # Restores best policy
            train_env.save("./runs/best_vecnormalize.pkl")
            break

# ─────────────────────────────────────────────────────────────
# 4) Plot reward curve (true scale)
# ─────────────────────────────────────────────────────────────
benchmark_ret = -12.165960817708347  # from value function iteration (see casskoopmans_benchmark.py)
if reward_history:
    steps, returns = zip(*reward_history)
    plt.figure(figsize=(9, 4.5))
    plt.plot(steps, returns, label="Avg true return")
    plt.axhline(y=best_score, linestyle="--", label=f"Best = {best_score:.4f}")
    plt.axhline(y=benchmark_ret, linestyle="--", color="red", label=f"Benchmark (VFI) = {benchmark_ret:.4f}")
    plt.xlabel("Timesteps")
    plt.ylabel("Average discounted episode return (true scale)")
    plt.title("PPO on Cass–Koopmans (evaluation)")
    plt.legend()
    plt.tight_layout()
    plt.show()