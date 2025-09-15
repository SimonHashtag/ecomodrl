# %%

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
from benchmarks.casskoopmans_benchmark import problem, time_iterations, get_vfrom_copt

# ─────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────
MAX_EPISODE_STEPS = 30
MAX_EVAL_EPISODE_STEPS = 100
BETA = 0.95
SEED = 42
N_ENVS = 8
EVAL_EVERY = 8_000 # training steps between evaluations
PATIENCE = 10
MIN_DELTA = 1e-3
TOTAL_TIMESTEPS = 100_000
BEST_PATH = "./runs/best_model_ppo.zip"
VECNORM_PATH = "./runs/best_vecnormalize.pkl"
BENCHMARK_RET = -12.165960817708347  # from value function iteration (see casskoopmans_benchmark.py)
STATE_MIN = 1.0

set_random_seed(SEED)

# ────────────────────────────────────────────────────────────
# 2. Environment
# ────────────────────────────────────────────────────────────

def make_env(max_episode_steps=MAX_EPISODE_STEPS, fixed_start=False, k_min=STATE_MIN, training=True):
    def _thunk():
        env = CassKoopmansEnv(
            max_episode_steps=max_episode_steps,
            fixed_start=fixed_start,
            k_min=k_min,
            seed=SEED
        )
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        # stabilize: avoid large negative rewards during training
        if training:
            env = TransformReward(env, lambda r: max(r, -100))  
        return env
    return _thunk

# Vectorized training env with observation & reward normalization
train_env = make_vec_env(make_env(fixed_start=False), n_envs=N_ENVS, seed=SEED)
train_env = VecNormalize(
    train_env, norm_obs=True, norm_reward=True, 
    clip_obs=10.0, clip_reward=10.0, gamma=BETA
)

# Sanity-check a single raw instance (not the normalized VecEnv)
check_env(TimeLimit(CassKoopmansEnv(seed=SEED), max_episode_steps=MAX_EPISODE_STEPS), warn=True)

# Separate eval env: same dynamics and TimeLimit, but fixed start for determinism
eval_env_raw = DummyVecEnv([make_env(max_episode_steps=MAX_EVAL_EPISODE_STEPS, fixed_start=True, training=False)])
eval_env = VecNormalize(
    eval_env_raw, training=False, norm_obs=True, norm_reward=False, 
    clip_obs=10.0, clip_reward=1000.0, gamma=BETA
)

# ─────────────────────────────────────────────────────────────
# 3. PPO model & hyperparameters
# ─────────────────────────────────────────────────────────────
policy_kwargs = dict(
    activation_fn=th.nn.ReLU, 
    net_arch=dict(pi=[64, 64], vf=[64, 64]), 
    log_std_init=-2.0,            # smaller initial exploration to avoid extremes
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
# 4. Utilities
# ─────────────────────────────────────────────────────────────
def sync_vecnormalize(dst: VecNormalize, src: VecNormalize):
    """keep eval stats in lockstep with training stats"""
    dst.obs_rms = src.obs_rms
    dst.ret_rms = src.ret_rms

def evaluate(model, eval_env, beta=BETA, n_episodes=1):
    """fixed start + deterministic => 1 episode is fine."""
    eval_env.training = False
    returns = []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = np.array([False])
        G, power = 0.0, 1.0
        while not done.all():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            G += float(reward.mean()) * power
            power *= beta
        returns.append(G)
    return float(np.mean(returns))

# ─────────────────────────────────────────────────────────────
# 5. Training loop with early stopping based on evaluation
# ─────────────────────────────────────────────────────────────
print("────────────────────────────────────────────")
print("     PPO TRAINING LOOP WITH EARLY STOPPING")
print("────────────────────────────────────────────")

best_score = -np.inf
patience_counter = 0
return_history = []
timesteps = 0

while timesteps < TOTAL_TIMESTEPS:
    # Train for a short chunk
    model.learn(total_timesteps=EVAL_EVERY, reset_num_timesteps=False)
    timesteps += EVAL_EVERY
    
    sync_vecnormalize(eval_env, train_env)
    
    avg_return = evaluate(model, eval_env, n_episodes=1)    # n_episodes=1 since fixed start and deterministic policy and environment (if stochastic env, increase n_episodes and vary seeds!)
    return_history.append((timesteps, avg_return))
    print(f"[{timesteps:7d}] avg true return: {avg_return: .6f} | best: {best_score: .6f} | PATIENCE: {patience_counter}/{PATIENCE}")

    if avg_return > best_score + MIN_DELTA:
        best_score = avg_return
        model.save(BEST_PATH)
        train_env.save(VECNORM_PATH)
        PATIENCE_counter = 0
    else:
        PATIENCE_counter += 1
        if PATIENCE_counter >= PATIENCE:
            print("Early stopping.")
            model = PPO.load(BEST_PATH)  # Restores best policy
            #train_env.save(VECNORM_PATH)
            break

# ─────────────────────────────────────────────────────────────
# 6. Plot return curve (true scale)
# ─────────────────────────────────────────────────────────────
if return_history:
    steps, returns = zip(*return_history)
    plt.figure(figsize=(9, 4.5))
    plt.plot(steps, returns, label="Avg true return")
    plt.axhline(y=best_score, linestyle="--", label=f"Best = {best_score:.4f}")
    plt.axhline(y=BENCHMARK_RET, linestyle="--", color="red", label=f"Benchmark (VFI) = {BENCHMARK_RET:.4f}")
    plt.xlabel("Timesteps")
    plt.ylabel("Average discounted episode return (true scale)")
    plt.title("PPO on Cass–Koopmans (evaluation)")
    plt.legend()
    plt.tight_layout()
    plt.show()
# %%
# ─────────────────────────────────────────────────────────────
# 7. Analyze performance (Euler equation errors & Bellman residuals)
# ─────────────────────────────────────────────────────────────
#vecnorm = VecNormalize.load(VECNORM_PATH, DummyVecEnv([lambda: None]))
model = PPO.load(BEST_PATH)  # Restores best policy
vecnorm = VecNormalize.load("./runs/best_vecnormalize.pkl", train_env)
#vecnorm.norm_reward = False
vecnorm.training = False

k_min, k_max, n_pts = STATE_MIN, 15., 100
obs = np.linspace(k_min, k_max, n_pts, dtype=np.float32).reshape(-1, 1)
obs_norm = vecnorm.normalize_obs(obs)

# deterministic policy: actions and value function for normalized observations
with th.no_grad():
    #policy_det, _ = model.policy.predict(obs, deterministic=True)
    obs_tensor = th.as_tensor(obs_norm)
    policy_det = model.policy._predict(obs_tensor, deterministic=True).cpu().numpy().squeeze()
    Vs = model.policy.predict_values(obs_tensor).cpu().numpy().squeeze()
    
def c_from_next_k(k, a, alpha=0.3, delta=0.1):
    f = k**alpha
    y_res = f + (1.0 - delta) * k
    return np.clip(a, 0.0, 1.0) * y_res

c_rl = c_from_next_k(obs.squeeze(), policy_det, vecnorm.get_attr("ALPHA")[0], vecnorm.get_attr("DELTA")[0])

# VecNormalize: undo reward normalization to recover true-scale values (count=epsilon to avoid division by zero)
Vs_true = Vs * np.sqrt(vecnorm.ret_rms.var + vecnorm.epsilon)

# Benchmark value and policy functions via time iterations
p = problem()
grid, policy_benchmark = time_iterations(
    p, grid_points=30, lower_limit=k_min, upper_limit=k_max, 
    tol=1e-5, conv=0.0001, verbose=False)
V = get_vfrom_copt(p, grid, policy_benchmark, conv=0.001)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(obs.squeeze(), c_rl, label="RL Policy")
ax[0].plot(grid, policy_benchmark(grid), linestyle="--", color="orange", label="Benchmark Policy")
ax[0].set_title("Deterministic policy")
ax[0].set_xlabel("Capital k")
ax[0].set_ylabel("Consumption")

ax[1].plot(obs.squeeze(), Vs_true, label="RL Value")
ax[1].plot(grid, V(grid), linestyle="--", color="orange", label="Benchmark Value")
ax[1].set_title("Value function V(k) (true scale)")
ax[1].set_xlabel("Capital k")
ax[1].set_ylabel("Value")

plt.tight_layout()
plt.legend()
plt.show()