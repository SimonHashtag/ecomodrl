#%cd C:\Users\s_haas09\VSC_Python\EcoModRL
import csv
import json
import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from gymnasium.wrappers import TimeLimit, TransformReward
from scipy.interpolate import CubicSpline
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from ecomodrl.envs.casskoopmans import CassKoopmansEnv, c_from_next_k
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem
from ecomodrl.benchmarks.casskoopmans.dynopt import egm, get_vfrom_copt, get_euler_errors
from ecomodrl.utils.params_loader import CassKoopmansParams
from ecomodrl.utils.metrics.accuracy import wev, cev_crra
from ecomodrl.utils.stopper_sb import EarlyStoppingEvalCallback

# ─────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────
toml_path = "./ecomodrl/experiments/model_params/casskoopmans.toml" # path to TOML file with model params (optional)
model_params = CassKoopmansParams.load_params(toml_path)
out_path = "./runs/CassKoopmans/PPO"
MAX_EPISODE_STEPS = 30
MAX_EVAL_EPISODE_STEPS = 100
N_ENVS = 8
TOTAL_TIMESTEPS = 200_000
EVAL_EVERY = 5_000 # training steps between evaluations
PATIENCE = 5
MIN_DELTA = 1e-3
BENCHMARK_RET = -11.434657802305646  # from EGM (see benchmarks/casskoopmans/dynopt.py)
STATE_MIN = 1.0
STATE_MAX = 15.0
BASE_SEED = 42
N_SEEDS = 1
SEEDS = list(range(BASE_SEED, BASE_SEED + N_SEEDS))

plt.style.use('seaborn-v0_8-colorblind')

def make_env(
    params: CassKoopmansParams,
    *,
    max_episode_steps=MAX_EPISODE_STEPS, 
    fixed_start=False, 
    RESETSTATE=5.0,
    k_min=STATE_MIN,
    k_max=STATE_MAX,
    floattype=np.float32,
    training=True) -> "CassKoopmansEnv":
    def _thunk():
        env = CassKoopmansEnv(
            max_episode_steps=max_episode_steps,
            A=params.A,
            GAMMA=params.crra,          
            DELTA=params.delta,         
            ALPHA=params.alpha,         
            RESETSTATE=RESETSTATE,
            fixed_start=fixed_start,
            k_min=k_min,
            k_max=k_max,
            floattype=floattype,
        )
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        # stabilize: avoid large negative rewards during training
        if training:
            env = TransformReward(env, lambda r: max(r, -100))  
        return env
    return _thunk

def build_envs(
    model_params,
    n_envs=N_ENVS,
    max_eval_episode_steps=MAX_EVAL_EPISODE_STEPS,
    eval_state_min=STATE_MIN,
    eval_state_max=STATE_MAX,
    eval_num_states=20,
    seed=BASE_SEED,
    ):
    # Vectorized training env with observation & reward normalization
    # Seeding: VecEnv seeds each sub-env with seed + env_number
    train_env = make_vec_env(make_env(model_params, fixed_start=False), n_envs=n_envs, seed=seed)
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, 
        clip_obs=10.0, clip_reward=10.0, gamma=model_params.beta,
    )

    # Eval env: fixed-start grid, deterministic eval
    eval_grid = np.linspace(eval_state_min, 
                            eval_state_max, 
                            num=eval_num_states, 
                            dtype=np.float32)
    eval_env_thunks = [
        make_env(
            model_params, max_episode_steps=max_eval_episode_steps, 
            fixed_start=True, RESETSTATE=k0, training=False) 
        for k0 in eval_grid
        ]
    eval_env_raw = DummyVecEnv(
        eval_env_thunks,
    )
    _ = eval_env_raw.seed(seed)
    try:
        eval_env_raw.action_space.seed(seed)
        eval_env_raw.observation_space.seed(seed)
    except Exception:
        pass
    eval_env = VecNormalize(
        eval_env_raw, training=False, norm_obs=True, norm_reward=False, 
        clip_obs=10.0, clip_reward=1000.0, gamma=model_params.beta,
    )
    return train_env, eval_env

def get_policy_value_functions(model_path, vecnorm_path, train_env, grid):
    model = PPO.load(model_path)  # Restores best policy
    vecnorm = VecNormalize.load(vecnorm_path, train_env)
    vecnorm.training = False

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
    value_rl = CubicSpline(obs.squeeze(), V_nn, bc_type="natural", extrapolate=True)
    return policy_rl, value_rl

def single_training_run(seed: int):
    out_dir = f"{out_path}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, "best_model_ppo.zip")
    vecnorm_path = os.path.join(out_dir, "best_vecnormalize.pkl")
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # global seeding (numpy/torch/python)
    set_random_seed(seed)
    
    # ────────────────────────────────────────────────────────────
    # 1. Environment
    # ────────────────────────────────────────────────────────────
    train_env, eval_env = build_envs(
        model_params, 
        seed=seed,
        )
    
    # ─────────────────────────────────────────────────────────────
    # 2. PPO training
    # ─────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU, 
        net_arch=dict(pi=[32, 32], vf=[32, 32]), 
        log_std_init=-2.0,            # smaller initial exploration to avoid extremes
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        gamma=model_params.beta,
        n_steps=MAX_EPISODE_STEPS,   # per-env rollout length; aligns with episode length
        batch_size=N_ENVS * MAX_EPISODE_STEPS // 4,  # minibatch size; 4 minibatches per update
        learning_rate=3e-4,
        ent_coef=0.0,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=out_dir,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
    )

    callback = EarlyStoppingEvalCallback(
        eval_env=eval_env,
        beta=model_params.beta,
        eval_freq=EVAL_EVERY,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        n_eval_episodes=1,            # increase for stochastic envs
        deterministic_eval=True,
        best_model_path=best_model_path,
        vecnorm_path=vecnorm_path,
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    # ─────────────────────────────────────────────────────────────
    # 3. Analyze performance
    # ─────────────────────────────────────────────────────────────

    # Return curve
    steps, returns = zip(*callback.return_history)
    plt.figure(figsize=(9, 4.5))
    plt.plot(steps, returns, label="Avg true return")
    plt.axhline(y=callback.best_score, linestyle="--", 
                label=f"Best = {callback.best_score:.4f}")
    plt.axhline(y=BENCHMARK_RET, linestyle="--", color="red", 
                label=f"Benchmark (EGM) = {BENCHMARK_RET:.4f}")
    plt.xlabel("Timesteps")
    plt.ylabel("Average discounted episode return (true scale)")
    plt.title("PPO on Cass–Koopmans (evaluation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "return_curve.png"), dpi=150)
    plt.close()
    
    # Store learning curve (steps/returns) for later aggregation
    np.savez(os.path.join(out_dir, "learning_curve.npz"),
             steps=np.array(steps), returns=np.array(returns))

    k_min, k_max, n_pts = STATE_MIN, STATE_MAX, 100
    grid = np.linspace(k_min, k_max, n_pts, dtype=np.float32)

    policy_rl, value_rl = get_policy_value_functions(
        best_model_path, 
        vecnorm_path, 
        train_env, 
        grid
        )

    # Benchmark value and policy functions via egm
    p = PlanningProblem(
        σ=model_params.crra,
        β=model_params.beta,
        δ=model_params.delta,
        α=model_params.alpha,
        A=model_params.A,
    )
    grid_bm, policy_bm = egm(
        p, grid_points=100, lower_limit=k_min, upper_limit=k_max, 
        conv=0.0001, verbose=False)
    V_bm = get_vfrom_copt(p, grid_bm, policy_bm, conv=0.001)
    V_rl = get_vfrom_copt(p, grid, policy_rl, conv=0.001)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(grid, policy_rl(grid), label="RL Policy")
    ax[0].plot(grid, policy_bm(grid), linestyle="--", label="Benchmark Policy")
    ax[0].set_title("Policy Function")
    ax[0].set_xlabel("Capital k")
    ax[0].set_ylabel("Consumption")
    ax[0].legend()

    ax[1].plot(grid, value_rl(grid), label="RL Value (directly from NN)")
    ax[1].plot(grid, V_bm(grid), linestyle="--", label="Benchmark Value")
    ax[1].plot(grid, V_rl(grid), label="RL Value (recomputed from policy)")
    ax[1].set_title("Value function V(k) ")
    ax[1].set_xlabel("Capital k")
    ax[1].set_ylabel("Value")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "policy_value.png"), dpi=150)
    plt.close(fig)
    
    # Euler equation errors under RL policy
    euler_errors_rl, states, U_rl = get_euler_errors(p, grid, policy_rl, n_eps=100)
    euler_errors_bm, _, U_bm = get_euler_errors(p, grid, policy_bm, n_eps=100)
    plt.hist(np.log10(abs(euler_errors_rl)), bins=50, density=True, alpha=0.7, label='RL policy')
    plt.hist(np.log10(abs(euler_errors_bm)), bins=50, density=True, color="grey", alpha=0.7, label='EGM')
    plt.axvline(x=np.log10(abs(euler_errors_rl)).mean(), linestyle='--', linewidth=1)
    plt.axvline(x=np.log10(abs(euler_errors_bm)).mean(), color='grey', linestyle='--', linewidth=1)
    plt.title("Log10 Euler equation errors")
    plt.xlabel("Euler equation error")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "euler_errors.png"), dpi=150)
    plt.close()
    
    # Welfare metrics (Wealth-equivalent (cash-on-hand) and consumption-equivalent variations)
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
    fig.savefig(os.path.join(figures_dir, "wev_cev.png"), dpi=150)
    plt.close(fig)

    avg_wev = float(np.mean(WEV))
    avg_wev_relevant = float(wev(states, V_bm, V_rl).mean())
    avg_cev = float(np.mean(CEV))
    avg_cev_relevant = float(cev_crra(states, V_bm, V_rl, σ=model_params.crra).mean())
    
    metrics = {
        "seed": seed,
        "best_eval_return": float(callback.best_score),
        "benchmark_return": float(BENCHMARK_RET),
        "U_rl": float(U_rl),
        "U_bm": float(U_bm),
        "max_abs_euler_error_rl": float(np.max(np.abs(euler_errors_rl))),
        "max_abs_euler_error_bm": float(np.max(np.abs(euler_errors_bm))),
        "avg_wev": avg_wev,
        "avg_wev_relevant": avg_wev_relevant,
        "avg_cev": avg_cev,
        "avg_cev_relevant": avg_cev_relevant,
        "best_model_path": best_model_path,
        "vecnorm_path": vecnorm_path,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def run_multiple_seeds(seeds):
    all_metrics = []
    for s in seeds:
        print(f"\n===== Running seed {s} =====")
        m = single_training_run(s)
        all_metrics.append(m)
        
    csv_path = f"{out_path}/seed_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "best_eval_return",
                "benchmark_return",
                "U_rl",
                "U_bm",
                "max_abs_euler_error_rl",
                "max_abs_euler_error_bm",
                "avg_wev",
                "avg_wev_relevant",
                "avg_cev",
                "avg_cev_relevant",
                "best_model_path",
                "vecnorm_path",
            ],
        )
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nWrote summary to {csv_path}")
    return all_metrics


if __name__ == "__main__":
    run_multiple_seeds(SEEDS)
    curves = []
    global_steps = np.linspace(0, TOTAL_TIMESTEPS, num=100)
    for s in SEEDS:
        d = np.load(f"{out_path}/seed_{s}/learning_curve.npz")
        curves.append(np.interp(global_steps, d["steps"], d["returns"]))
    curves = np.stack(curves)  # shape [n_seeds, n_eval_points]

    mean = curves.mean(0)
    # percentiles are calculated with linear interpolation!
    q25, q75 = np.percentile(curves, [25, 75], axis=0)

    plt.fill_between(global_steps, q25, q75, alpha=0.3, label="IQR")
    plt.plot(global_steps, mean, label="Mean")
    plt.legend()
    plt.show()

# # CI do not make sense for a single deterministic policy and envs!!!
# # Bias-corrected and accelerated bootstrap CI
# from scipy.stats import bootstrap
# rng = np.random.default_rng(SEED)
# n_bootstraps = 1_000
# cev_crra_mean = lambda data: np.mean(cev_crra(data, V_bm, V_rl, σ=model_params.crra))
# bs_res = bootstrap(
#     (states,), cev_crra_mean, vectorized=False, 
#     n_resamples=n_bootstraps, method="BCa", 
#     confidence_level=0.95, random_state=rng)
# print(f"95% BCa bootstrap CI for average relevant CEV: [{bs_res.confidence_interval[0]:.8f}, {bs_res.confidence_interval[1]:.8f}]")
# %%