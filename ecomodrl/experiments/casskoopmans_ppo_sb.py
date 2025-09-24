import json
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from gymnasium.wrappers import TimeLimit, TransformReward
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
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
from ecomodrl.utils.tb_reader import tb_read_scalar

@dataclass
class RunConfig:
    toml_path: str = "./ecomodrl/experiments/model_params/casskoopmans.toml"
    out_dir: Path = Path("D:/ECOMODRL/runs/CassKoopmans/PPO")
    max_episode_steps: int = 30
    max_eval_episode_steps: int = 100
    n_envs: int = 8
    total_timesteps: int = 200_000
    eval_every: int = 5_000
    patience: int = 5
    min_delta: float = 1e-3
    state_min: float = 1.0
    state_max: float = 15.0
    base_seed: int = 42
    n_seeds: int = 10
    @property
    def seeds(self) -> list[int]:
        return list(range(self.base_seed, self.base_seed + self.n_seeds))

def make_env(
    params: CassKoopmansParams,
    *,
    max_episode_steps: int, 
    fixed_start: bool, 
    fixed_state: float,
    k_min: float,
    k_max: float,
    floattype=np.float32,
    training: bool
    ) -> "CassKoopmansEnv":
    """
    Factory that returns a thunk creating a configured CassKoopmansEnv.
    """
    def _thunk():
        env = CassKoopmansEnv(
            max_episode_steps=max_episode_steps,
            A=params.A,
            GAMMA=params.crra,          
            DELTA=params.delta,         
            ALPHA=params.alpha,         
            RESETSTATE=fixed_state,
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
    params: CassKoopmansParams,
    cfg: RunConfig,
    seed: int,
    ):
    """ 
    Creates vectorized training env and eval env with normalization and explicit seeding.
    """
    train_env = make_vec_env(
        make_env(
            params,
            max_episode_steps=cfg.max_episode_steps,
            fixed_start=False,
            fixed_state=float(np.nan),  # ignored when fixed_start=False
            k_min=cfg.state_min,
            k_max=cfg.state_max,
            training=True,
        ), 
        n_envs=cfg.n_envs, 
        seed=seed
        ) # Seeding: VecEnv seeds each sub-env with seed + env_number
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, 
        clip_obs=10.0, clip_reward=10.0, gamma=params.beta,
    )

    # Eval env: fixed-start grid, deterministic eval
    eval_grid = np.linspace(cfg.state_min, 
                            cfg.state_max, 
                            num=20, 
                            dtype=np.float32)
    eval_env_thunks = [
        make_env(
            params,
            max_episode_steps=cfg.max_eval_episode_steps,
            fixed_start=True,
            fixed_state=float(k0),
            k_min=cfg.state_min,
            k_max=cfg.state_max,
            training=False,
        ) 
        for k0 in eval_grid
        ]
    eval_env_raw = DummyVecEnv(
        eval_env_thunks,
    )
    try:
        eval_env_raw.seed(seed)
        eval_env_raw.action_space.seed(seed)
        eval_env_raw.observation_space.seed(seed)
    except Exception:
        pass
    eval_env = VecNormalize(
        eval_env_raw, training=False, norm_obs=True, norm_reward=False, 
        clip_obs=10.0, clip_reward=1000.0, gamma=params.beta,
    )
    return train_env, eval_env

def train_single_seed(cfg: RunConfig, params:CassKoopmansParams, seed: int) -> dict:
    """
    Train PPO for one seed and save artifacts (model, vecnorm, learning_curve, meta).
    """
    # global seeding (numpy/torch/python)
    set_random_seed(seed)
    
    # Output dirs
    out_dir = cfg.out_dir / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "best_model_ppo.zip"
    vecnorm_path = out_dir / "best_vecnormalize.pkl"

    train_env, eval_env = build_envs(
        params, 
        cfg,
        seed=seed,
        )
    
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU, 
        net_arch=dict(pi=[32, 32], vf=[32, 32]), 
        log_std_init=-2.0,            # smaller initial exploration to avoid extremes
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        gamma=params.beta,
        n_steps=cfg.max_episode_steps,   # per-env rollout length; aligns with episode length
        batch_size=cfg.n_envs * cfg.max_episode_steps // 4,  # minibatch size; 4 minibatches per update
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
        beta=params.beta,
        eval_freq=cfg.eval_every,
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        n_eval_episodes=1,            # increase for stochastic envs
        deterministic_eval=True,
        best_model_path=best_model_path,
        vecnorm_path=vecnorm_path,
        verbose=1,
    )

    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)
    
    train_cfg = {
        "seed": seed,
        "best_eval_return": float(callback.best_score),
        "num_timesteps": int(callback.num_timesteps),
        "best_model_path": str(best_model_path),
        "vecnorm_path": str(vecnorm_path),
        "cfg": asdict(cfg),
        "params": vars(params)
    }
    return train_cfg

def train_multiple_seeds(cfg: RunConfig, params: CassKoopmansParams):
    """ 
    Train multiple seeds and save summary JSON.
    """
    train_cfgs = [train_single_seed(cfg, params, s) for s in cfg.seeds]
    with open(cfg.out_dir / "train_cfgs.json", "w") as f:
        json.dump(train_cfgs, f, indent=2, default=str)

def get_policy_value_functions(
    model_path: Path, 
    vecnorm_path: Path, 
    params,
    grid):
    """ 
    Given a trained model and VecNormalize, return policy and value functions
    as CubicSpline interpolators on the specified grid.
    """
    model = PPO.load(model_path)  # Restores best policy
    # create a dummy VecNormalize with same stats for obs normalization
    dummy_env = make_vec_env(
        make_env(
            params,
            max_episode_steps=100,
            fixed_start=False,
            fixed_state=float(np.nan),  # ignored when fixed_start=False
            k_min=1.0,
            k_max=15.0,
            training=False,
        ), 
        n_envs=1, 
        seed=1234
        )  # placeholder, not used
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
    vecnorm.training = False

    obs = grid.reshape(-1, 1)
    obs_norm = vecnorm.normalize_obs(obs)
    
    # Get policy function for normalized observations
    actions, _ = model.predict(obs_norm, deterministic=True)
    c_rl = c_from_next_k(obs.squeeze(), np.asarray(actions).squeeze(), params.alpha, params.delta)
    policy_rl = CubicSpline(obs.squeeze(), c_rl, bc_type="natural", extrapolate=True)

    # Get value function for normalized observations
    with th.no_grad():
        V_norm = model.policy.predict_values(th.as_tensor(obs_norm)).cpu().numpy().squeeze()
    ret_rms = getattr(vecnorm, "ret_rms", None)
    eps = getattr(vecnorm, "epsilon", 1e-8)
    scale = np.sqrt(getattr(ret_rms, "var", 1.0) + eps) if ret_rms is not None else 1.0
    V_nn = V_norm * scale
    value_rl = CubicSpline(obs.squeeze(), V_nn, bc_type="natural", extrapolate=True)
    return policy_rl, value_rl

def compute_benchmark(params: CassKoopmansParams, k_min: float, k_max: float):
    p = PlanningProblem(σ=params.crra, β=params.beta, δ=params.delta, α=params.alpha, A=params.A)
    grid_bm, policy_bm = egm(p, grid_points=200, lower_limit=k_min, upper_limit=k_max, conv=1e-4, verbose=False)
    V_bm = get_vfrom_copt(p, grid_bm, policy_bm, conv=1e-3)
    return p, grid_bm, policy_bm, V_bm

def bootstrap_mean_ci(Y: np.ndarray, *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 123):
    """
    Y: shape [n_seeds, n_points] -> returns (mean, lower, upper) each [n_points]
    Two-sided bootstrap over seeds with replacement; reproducible via 'seed'.
    """
    rng = np.random.default_rng(seed)
    n_seeds, _ = Y.shape
    idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    samples = Y[idx, :]
    boot_means = samples.mean(axis=1)
    mean = Y.mean(axis=0)
    lower = np.percentile(boot_means, 100 * (alpha / 2), axis=0)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return mean, lower, upper

def analyze_all_seeds(
    cfg: RunConfig, 
    params: CassKoopmansParams,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    bootstrap_seed: int = 123
    ):
    """
    Aggregate analysis and plotting across seeds:
      1) Policy + Value plots with mean and 95% bootstrap CI (CI only for NN value).
      2) Euler error histogram (pooled) + per-seed KDE overlays; EGM histogram as reference.
      3) WEV/CEV plots with mean and 95% bootstrap CI; pooled state histogram on twin axes.
    Saves: policy_value_agg.png, euler_errors_agg.png, wev_cev_agg.png in cfg.out_dir.
    """
    plt.style.use('seaborn-v0_8-colorblind')
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # common grid for RL & Benchmark evals
    grid = np.linspace(cfg.state_min, cfg.state_max, 200, dtype=np.float32)
    
    # Grid for average return plot
    global_steps = np.linspace(0, cfg.total_timesteps, num=100)
    
    # Benchmark once
    p, _, policy_bm, V_bm = compute_benchmark(params, cfg.state_min, cfg.state_max)
    euler_bm, _, U_bm = get_euler_errors(p, grid, policy_bm, n_eps=100)
    euler_bm = np.log10(np.abs(euler_bm))
    
    # storage across seeds
    return_curves = []
    rl_policy_vals = []       
    rl_value_nn_vals = []     
    rl_value_from_policy = [] 
    euler_logabs_list = []    
    pooled_states = []        
    Us_rl = []               
    WEV_mat = []
    CEV_mat = []
    WEV_rel_mat = []
    CEV_rel_mat = []
    seed_ids = list(cfg.seeds)
    
    train_cfgs_path = out_dir / "train_cfgs.json"
    with open(train_cfgs_path, "r") as f:
        train_cfgs = json.load(f)   # list of dicts

    # Policy/value/euler/wev/cev per seed
    for seed_entry in train_cfgs:
        best_model_path = Path(seed_entry["best_model_path"])
        vecnorm_path = Path(seed_entry["vecnorm_path"])
        
        # average evaluation return/learning curves
        s = seed_entry["seed"]
        tb_path = out_dir / f"seed_{s}"
        steps, returns = tb_read_scalar(tb_path, "eval/avg_discounted_return")
        
        # RL policy + value (NN spline)
        policy_rl, value_rl = get_policy_value_functions(best_model_path, vecnorm_path, params, grid)

        # Value recomputed from policy (for plotting mean only)
        V_rl = get_vfrom_copt(p, grid, policy_rl, conv=1e-3)

        # Euler errors (RL), states, and U metrics
        euler_rl, states, U_rl = get_euler_errors(p, grid, policy_rl, n_eps=100)
        
        wev_seed = wev(grid, V_bm, V_rl)
        if wev_seed is not None:
            wev_on_grid = wev_seed
            avg_wev_relevant = wev_seed.mean()
            WEV_mat.append(wev_on_grid)
            WEV_rel_mat.append(avg_wev_relevant)
        cev_on_grid = cev_crra(grid, V_bm, V_rl, σ=params.crra)
        
        avg_cev_relevant = cev_crra(states, V_bm, V_rl, σ=params.crra).mean()
        
        # stash
        return_curves.append(np.interp(global_steps, steps, returns))
        rl_policy_vals.append(policy_rl(grid))
        rl_value_nn_vals.append(value_rl(grid))
        rl_value_from_policy.append(V_rl(grid))
        euler_logabs_list.append(np.log10(np.abs(euler_rl)))
        pooled_states.append(states)
        Us_rl.append(U_rl)
        CEV_mat.append(cev_on_grid)
        CEV_rel_mat.append(avg_cev_relevant)

    # stack for bootstrap shapes [n_seeds, n_grid]
    return_curves = np.vstack(return_curves)
    rl_policy_vals = np.vstack(rl_policy_vals)
    rl_value_nn_vals = np.vstack(rl_value_nn_vals)
    rl_value_from_policy = np.vstack(rl_value_from_policy)
    WEV_mat = np.vstack(WEV_mat)  
    CEV_mat = np.vstack(CEV_mat) 
    
    # ----------- Returns ---------------------------
    # Average Returns for RL policies on k0-grid + CI
    mean_Us, lo_Us, hi_Us = bootstrap_mean_ci(
        np.vstack(Us_rl), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    
    # Average eval return over training run with IQR
    mean_return = return_curves.mean(axis=0)
    q25_return, q75_return = np.percentile(return_curves, [25, 75], axis=0)

    fig = plt.figure(figsize=(9,4.5))
    plt.fill_between(global_steps, q25_return, q75_return, alpha=0.3, label="IQR")
    plt.plot(global_steps, mean_return, label="Mean")
    plt.xlabel("Timesteps"); plt.ylabel("Avg discounted episode return (true scale)")
    plt.title("PPO on Cass–Koopmans (evaluation)")
    plt.legend(); plt.tight_layout()
    fig.savefig(cfg.out_dir / "learning_curve_agg.png", dpi=150)
    plt.close(fig)
    
    # ---------- 1) Policy + Value aggregate ----------
    mean_pol, lo_pol, hi_pol = bootstrap_mean_ci(
        rl_policy_vals, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    mean_Vnn, lo_Vnn, hi_Vnn = bootstrap_mean_ci(
        rl_value_nn_vals, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed + 1
    )
    mean_Vfrom = rl_value_from_policy.mean(axis=0)  # mean only (no CI)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    # Policy
    ax[0].plot(grid, mean_pol, label="RL Policy (mean)")
    ax[0].fill_between(grid, lo_pol, hi_pol, alpha=0.25, label="95% CI (bootstrap)")
    ax[0].plot(grid, policy_bm(grid), linestyle="--", label="Benchmark Policy (EGM)")
    ax[0].set_title("Policy Function")
    ax[0].set_xlabel("Capital k")
    ax[0].set_ylabel("Consumption")
    ax[0].legend()

    # Value
    ax[1].plot(grid, mean_Vnn, label="RL Value (NN mean)")
    ax[1].fill_between(grid, lo_Vnn, hi_Vnn, alpha=0.25, label="95% CI (bootstrap)")
    ax[1].plot(grid, V_bm(grid), linestyle="--", label="Benchmark Value (EGM)")
    ax[1].plot(grid, mean_Vfrom, label="RL Value (from policy mean)")
    ax[1].set_title("Value Function V(k)")
    ax[1].set_xlabel("Capital k")
    ax[1].set_ylabel("Value")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(out_dir / "policy_value_agg.png", dpi=150)
    plt.close(fig)

    # ---------- 2) Euler histogram (pooled) + per-seed KDEs ----------
    # pool RL errors and EGM errors across seeds
    mean_max_euler, lo_max_euler, hi_max_euler = bootstrap_mean_ci(
        np.max(euler_logabs_list, axis=1).reshape(-1, 1), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    
    rl_all = np.concatenate(euler_logabs_list, axis=0)

    fig = plt.figure(figsize=(7.2, 4.5))
    # pooled histograms
    plt.hist(rl_all, bins=50, density=True, alpha=0.7, label="RL policy (pooled)")
    plt.hist(euler_bm, bins=50, density=True, alpha=0.7, label="EGM (pooled)", color="gray")

    # per-seed KDE overlays (subtle)
    for i, arr in enumerate(euler_logabs_list):
        try:
            kde = gaussian_kde(arr)
            xs = np.linspace(
                min(np.min(rl_all), np.min(euler_bm)),
                max(np.max(rl_all), np.max(euler_bm)), 
                256
                )
            plt.plot(xs, kde(xs), alpha=0.2, linewidth=1.0)
        except Exception:
            # KDE can fail if variance ~0; skip quietly
            logging.warning(f"Skipping KDE for seed {seed_ids[i]} due to error.")
            pass

    plt.title("Log10 Euler Equation Errors")
    plt.xlabel("log10(|error|)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "euler_errors_agg.png", dpi=150)
    plt.close(fig)

    # ---------- 3) WEV / CEV with CIs; pooled state hist ----------
    # Average wev on grid + CI
    mean_wev, lo_wev, hi_wev = bootstrap_mean_ci(
        WEV_mat.mean(axis=1).reshape(-1, 1), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    # Average wev for relevant states only + CI
    mean_rel_wev, lo_rel_wev, hi_rel_wev = bootstrap_mean_ci(
        np.vstack(WEV_rel_mat), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    # Full wev on grid + CI
    mean_WEV, lo_WEV, hi_WEV = bootstrap_mean_ci(
        WEV_mat, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed + 2
    )
    # Average cev on grid + CI
    mean_cev, lo_cev, hi_cev = bootstrap_mean_ci(
        CEV_mat.mean(axis=1).reshape(-1, 1), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    # Average cev for relevant states only + CI
    mean_rel_cev, lo_rel_cev, hi_rel_cev = bootstrap_mean_ci(
        np.vstack(CEV_rel_mat), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    # Full wev on grid + CI
    mean_CEV, lo_CEV, hi_CEV = bootstrap_mean_ci(
        CEV_mat, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed + 3
    )

    # pooled states across seeds
    states_all = np.concatenate(pooled_states, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # WEV
    ax[0].plot(grid, mean_WEV, label="WEV (mean)")
    ax[0].fill_between(grid, lo_WEV, hi_WEV, alpha=0.25, label="95% CI (bootstrap)")
    ax[0].set_ylim(bottom=0.0)
    ax[0].set_xlabel("Capital k")
    ax[0].set_ylabel("WEV")
    ax[0].set_title("Wealth-Equivalent Variation")
    ax0p = ax[0].twinx()
    ax0p.hist(states_all, bins=30, density=True, alpha=0.3, color="gray")
    ax0p.set_ylabel("State density")

    # CEV
    ax[1].plot(grid, mean_CEV, label="CEV (mean)")
    ax[1].fill_between(grid, lo_CEV, hi_CEV, alpha=0.25, label="95% CI (bootstrap)")
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_xlabel("Capital k")
    ax[1].set_ylabel("CEV")
    ax[1].set_title("Consumption-Equivalent Variation")
    ax1p = ax[1].twinx()
    ax1p.hist(states_all, bins=30, density=True, alpha=0.3, color="gray")
    ax1p.set_ylabel("State density")

    for a in ax:
        a.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "wev_cev_agg.png", dpi=150)
    plt.close(fig)
    
    
    metrics = {
        "U_bm": float(U_bm),
        "mean_U_rl": float(mean_Us[0]),
        "lo_U_rl": float(lo_Us[0]),
        "hi_U_rl": float(hi_Us[0]),
        "mean_wev": float(mean_wev[0]),
        "lo_wev": float(lo_wev[0]), 
        "hi_wev": float(hi_wev[0]),
        "mean_wev_relevant": float(mean_rel_wev[0]),
        "lo_wev_relevant": float(lo_rel_wev[0]),
        "hi_wev_relevant": float(hi_rel_wev[0]),
        "mean_cev": float(mean_cev[0]),
        "lo_cev": float(lo_cev[0]),
        "hi_cev": float(hi_cev[0]),
        "mean_cev_relevant": float(mean_rel_cev[0]),
        "lo_cev_relevant": float(lo_rel_cev[0]),
        "hi_cev_relevant": float(hi_rel_cev[0]),
        "max_abs_euler_error_bm": float(np.max(euler_bm)),
        "mean_max_abs_euler_error_rl": float(mean_max_euler[0]),
        "lo_max_abs_euler_error_rl": float(lo_max_euler[0]),
        "hi_max_abs_euler_error_rl": float(hi_max_euler[0]),
    }
    with open(out_dir / "metrics_analysis.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    cfg = RunConfig()
    params = CassKoopmansParams.load_params(cfg.toml_path)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    train_multiple_seeds(cfg, params)
    analyze_all_seeds(cfg, params, n_boot=1000, alpha=0.05, bootstrap_seed=12345)

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