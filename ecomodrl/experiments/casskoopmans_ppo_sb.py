# %%
#%cd C:\Users\s_haas09\VSC_Python\EcoModRL

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from gymnasium.wrappers import TimeLimit, TransformReward
from scipy.interpolate import CubicSpline
from scipy.stats import bootstrap
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from typing import Optional

from ecomodrl.envs.casskoopmans import CassKoopmansEnv, c_from_next_k
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem
from ecomodrl.benchmarks.casskoopmans.dynopt import time_iterations, get_vfrom_copt, simulation, get_euler_errors
from ecomodrl.utils.params_loader import CassKoopmansParams
from ecomodrl.utils.metrics.accuracy import wev, cev_crra

#%%
# ─────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────
toml_path = "./ecomodrl/experiments/model_params/casskoopmans.toml" # path to TOML file with model params (optional)
model_params = CassKoopmansParams.load_params(toml_path)
MAX_EPISODE_STEPS = 30
MAX_EVAL_EPISODE_STEPS = 100
#BETA = 0.95
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
STATE_MAX = 15.0

plt.style.use('seaborn-v0_8-colorblind')
set_random_seed(SEED)

# ────────────────────────────────────────────────────────────
# 2. Environment
# ────────────────────────────────────────────────────────────

def make_env(
    params: CassKoopmansParams,
    *,
    max_episode_steps=MAX_EPISODE_STEPS, 
    fixed_start=False, 
    k_min=STATE_MIN,
    k_max=STATE_MAX,
    seed: Optional[int] = None,
    floattype=np.float32,
    training=True) -> "CassKoopmansEnv":
    def _thunk():
        env = CassKoopmansEnv(
            max_episode_steps=max_episode_steps,
            A=params.A,
            GAMMA=params.crra,          
            DELTA=params.delta,         
            ALPHA=params.alpha,         
            RESETSTATE=params.reset_k,
            fixed_start=fixed_start,
            k_min=k_min,
            k_max=k_max,
            seed=seed,
            floattype=floattype,
        )
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        # stabilize: avoid large negative rewards during training
        if training:
            env = TransformReward(env, lambda r: max(r, -100))  
        return env
    return _thunk

# Vectorized training env with observation & reward normalization
train_env = make_vec_env(make_env(model_params, fixed_start=False), n_envs=N_ENVS, seed=SEED)
train_env = VecNormalize(
    train_env, norm_obs=True, norm_reward=True, 
    clip_obs=10.0, clip_reward=10.0, gamma=model_params.beta,
)

# Sanity-check a single raw instance (not the normalized VecEnv)
check_env(TimeLimit(CassKoopmansEnv(seed=SEED), max_episode_steps=MAX_EPISODE_STEPS), warn=True)

# Separate eval env: same dynamics and TimeLimit, but fixed start for determinism
eval_env_raw = DummyVecEnv([make_env(model_params, max_episode_steps=MAX_EVAL_EPISODE_STEPS, fixed_start=True, training=False)])
eval_env = VecNormalize(
    eval_env_raw, training=False, norm_obs=True, norm_reward=False, 
    clip_obs=10.0, clip_reward=1000.0, gamma=model_params.beta,
)

# %%
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
    gamma=model_params.beta,
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

def evaluate(model, eval_env, beta, n_episodes=1):
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
    model.learn(total_timesteps=EVAL_EVERY, reset_num_timesteps=False)
    timesteps += EVAL_EVERY
    
    sync_vecnormalize(eval_env, train_env)
    
    avg_return = evaluate(model, eval_env, beta=model_params.beta, n_episodes=1)    # n_episodes=1 since fixed start and deterministic policy and environment (if stochastic env, increase n_episodes and vary seeds!)
    return_history.append((timesteps, avg_return))
    print(f"[{timesteps:7d}] avg true return: {avg_return: .6f} | best: {best_score: .6f} | PATIENCE: {patience_counter}/{PATIENCE}")

    if avg_return > best_score + MIN_DELTA:
        best_score = avg_return
        model.save(BEST_PATH)
        train_env.save(VECNORM_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            model = PPO.load(BEST_PATH)  # Restores best policy
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

model = PPO.load(BEST_PATH)  # Restores best policy
vecnorm = VecNormalize.load(VECNORM_PATH, train_env)
vecnorm.training = False

k_min, k_max, n_pts = STATE_MIN, 15., 100
grid = np.linspace(k_min, k_max, n_pts, dtype=np.float32)
obs = grid.reshape(-1, 1)
obs_norm = vecnorm.normalize_obs(obs)
obs_tensor = th.as_tensor(obs_norm)

# deterministic policy: actions and value function for normalized observations
with th.no_grad():
    policy_det = model.policy._predict(obs_tensor, deterministic=True).cpu().numpy().squeeze()
    V_norm = model.policy.predict_values(obs_tensor).cpu().numpy().squeeze()

c_rl = c_from_next_k(obs.squeeze(), policy_det, vecnorm.get_attr("ALPHA")[0], vecnorm.get_attr("DELTA")[0])
policy_rl = CubicSpline(obs.squeeze(), c_rl, bc_type="natural", extrapolate=True)

# VecNormalize: undo reward normalization to recover true-scale values (count=epsilon to avoid division by zero)
V_nn = V_norm * np.sqrt(vecnorm.ret_rms.var + vecnorm.epsilon)

# Benchmark value and policy functions via time iterations
p = PlanningProblem(
    σ=model_params.crra,
    β=model_params.beta,
    δ=model_params.delta,
    α=model_params.alpha,
    A=model_params.A,
)
grid_bm, policy_bm = time_iterations(
    p, grid_points=30, lower_limit=k_min, upper_limit=k_max, 
    tol=1e-5, conv=0.0001, verbose=False)
V_bm = get_vfrom_copt(p, grid_bm, policy_bm, conv=0.001)
V_rl = get_vfrom_copt(p, grid, policy_rl, conv=0.001)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(grid, policy_rl(grid), label="RL Policy")
ax[0].plot(grid, policy_bm(grid), linestyle="--", label="Benchmark Policy")
ax[0].set_title("Policy Function")
ax[0].set_xlabel("Capital k")
ax[0].set_ylabel("Consumption")
ax[0].legend()

ax[1].plot(obs.squeeze(), V_nn, label="RL Value (directly from NN)")
ax[1].plot(grid, V_bm(grid), linestyle="--", label="Benchmark Value")
ax[1].plot(grid, V_rl(grid), label="RL Value (recomputed from policy)")
ax[1].set_title("Value function V(k) ")
ax[1].set_xlabel("Capital k")
ax[1].set_ylabel("Value")

plt.tight_layout()
plt.legend()
plt.show()

# %%
# Euler equation errors under RL policy
euler_errors_rl, states = get_euler_errors(p, grid, policy_rl, n_eps=100)
euler_errors_bm, _ = get_euler_errors(p, grid, policy_bm, n_eps=100)
plt.hist(np.log10(abs(euler_errors_rl)), bins=50, density=True, alpha=0.7, label='RL policy')
plt.hist(np.log10(abs(euler_errors_bm)), bins=50, density=True, color="grey", alpha=0.7, label='Time iterations')
plt.axvline(x=np.log10(abs(euler_errors_rl)).mean(), linestyle='--', linewidth=1)
plt.axvline(x=np.log10(abs(euler_errors_bm)).mean(), color='grey', linestyle='--', linewidth=1)
plt.title("Log10 Euler equation errors")
plt.xlabel("Euler equation error")
plt.ylabel("Density")
plt.tight_layout()
plt.legend()
plt.show()
print(f'Max absolute Euler equation error: \r\n  RL: {max(abs(euler_errors_rl))}      Benchmark (TI): {max(abs(euler_errors_bm))}')
_, _, U_rl = simulation(p, policy_rl, 100, start_value=5.0)
_, _, U_bm = simulation(p, policy_bm, 100, start_value=5.0)
print(f'The optimal discounted episode return is: \r\n  RL: {U_rl}      Benchmark (TI): {U_bm}')
# %%
# Wealth equivalent (cash-on-hand) variation


CEV = cev_crra(grid, V_bm, V_rl, σ=model_params.crra)
WEV = wev(grid, V_bm, V_rl)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(grid, WEV)
ax[0].set_ylim(bottom=0.0)
ax[0].set_xlabel("Capital k")
ax[0].set_ylabel("WEV", color="blue")
ax[0].tick_params(axis='y', labelcolor="blue")
ax[0].set_title("Wealth-equivalent variation (WEV)")

ax0_prime = ax[0].twinx()
ax0_prime.hist(states, bins=30, density=True, alpha=0.3, color="gray", label="States visited (simulations)")
ax0_prime.set_ylabel("Density", color="gray")
ax0_prime.tick_params(axis='y', labelcolor="gray")

ax[1].plot(grid, CEV)
ax[1].set_ylim(bottom=0.0)
ax[1].set_xlabel("Capital k")
ax[1].set_ylabel("CEV", color="blue")
ax[1].tick_params(axis='y', labelcolor="blue")
ax[1].set_title("Consumption-equivalent variation (CEV)")

ax1_prime = ax[1].twinx()
ax1_prime.hist(states, bins=30, density=True, alpha=0.3, color="gray", label="States visited (simulations)")
ax1_prime.set_ylabel("Density", color="gray")
ax1_prime.tick_params(axis='y', labelcolor="gray")

fig.tight_layout()
plt.show()

avg_wev_relevant = wev(states, V_bm, V_rl).mean()
print(f"Average WEV: {np.mean(WEV):.4f}")
print(f"Average relevant WEV: {np.mean(avg_wev_relevant):.4f}")

avg_cev_relevant = cev_crra(states, V_bm, V_rl, σ=model_params.crra).mean()
print(f"Average CEV: {np.mean(CEV):.4f}")
print(f"Average relevant CEV: {np.mean(avg_cev_relevant):.8f}")

# # CI do not make sense for a single deterministic policy and envs!!!
# # Bias-corrected and accelerated bootstrap CI
# rng = np.random.default_rng(SEED)
# n_bootstraps = 1_000
# cev_crra_mean = lambda data: np.mean(cev_crra(data, V_bm, V_rl, σ=model_params.crra))
# bs_res = bootstrap(
#     (states,), cev_crra_mean, vectorized=False, 
#     n_resamples=n_bootstraps, method="BCa", 
#     confidence_level=0.95, random_state=rng)
# print(f"95% BCa bootstrap CI for average relevant CEV: [{bs_res.confidence_interval[0]:.8f}, {bs_res.confidence_interval[1]:.8f}]")
# %%