# =============================================================================
# Computational-complexity metrics (FLOPs).
#
# Two hardware-agnostic measures:
#   1. FLOPs per update  -- deterministic "design intensity" of one minibatch
#      optimizer step (twin critics, batch size, policy delay, etc.).
#   2. Cumulative FLOPs to convergence -- headline cost of coming close to the
#      best training performance, aggregated over training seeds.
#
# Also exposes the number of sampled environment states to convergence.
#
# Measurement strategy
# --------------------
# Network forward+backward FLOPs (~99% of the total) are measured directly with
# PyTorch's FlopCounterMode around a short, offline `learn()` burst on a
# model loaded from a checkpoint (so the actual, possibly tuned,
# hyperparameters and architecture are used). Element-wise Adam and Polyak
# soft-update FLOPs are added analytically; they are < 1% of the matmul cost.
# Replay-buffer indexing (pure memory) and raw environment transition
# arithmetic are omitted as negligible; sample consumption enters the cumulative
# metric through inference forward passes.
#
# Unit of "one update" = one minibatch optimizer step (consistent across algos).
# =============================================================================

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from ecomodrl.analysis.efficiency_metrics import normalized_regret


# -----------------------------------------------------------------------------
# Analytic per-parameter constants for the non-matmul optimizer arithmetic.
#
# Adam (no amsgrad), per trainable parameter per optimizer step:
#   m     = b1*m + (1-b1)*g            -> mul, mul, add            = 3
#   v     = b2*v + (1-b2)*g^2          -> mul(g^2), mul, mul, add  = 4
#   denom = sqrt(v/bc2) + eps    -> sqrt, div, add           = 3
#   p    -= (lr/bc1) * m / denom       -> div, mul, sub            = 3
#   (bias-correction denominators bc1, bc2 are per-step scalars)
#   total ~= 13 FLOPs / param.
ADAM_FLOPS_PER_PARAM = 13.0

# Polyak soft update, per target parameter per update:
#   target = (1-tau)*target + tau*online -> mul, mul, add = 3.
POLYAK_FLOPS_PER_PARAM = 3.0

# -----------------------------------------------------------------------------
# Parameter counts and minibatch bookkeeping
# -----------------------------------------------------------------------------

def _count(params) -> int:
    return int(sum(int(p.numel()) for p in params))


def param_counts(model, algo: str) -> dict:
    """Trainable (optimized) and target parameter counts for one SB3 model."""
    algo = algo.lower()

    if algo == "ppo":
        return {"n_optimized": _count(model.policy.parameters()), "n_target": 0}

    n_opt = _count(model.actor.parameters()) + _count(model.critic.parameters())

    n_tgt = 0
    if getattr(model, "critic_target", None) is not None:
        n_tgt += _count(model.critic_target.parameters())
    if getattr(model, "actor_target", None) is not None:
        n_tgt += _count(model.actor_target.parameters())

    if algo == "sac" and getattr(model, "log_ent_coef", None) is not None:
        n_opt += 1  # entropy temperature

    return {"n_optimized": n_opt, "n_target": n_tgt}


def n_minibatches(model, n_envs: int) -> int:
    """Minibatch optimizer steps per epoch (PPO). 1 for off-policy algorithms."""
    n_steps = int(getattr(model, "n_steps"))
    batch = int(getattr(model, "batch_size"))
    total = n_steps * int(n_envs)
    return max(1, int(math.ceil(total / max(1, batch))))


# -----------------------------------------------------------------------------
# Measurement
# -----------------------------------------------------------------------------

def measure_policy_flops(model, algo: str, batch: int = 256) -> float:
    """Per-sample policy (action-selection) forward FLOPs.

    Uses the same policy forward call that SB3 executes during rollout:
    off-policy -> actor forward; PPO -> actor + value + log-prob.
    """

    algo = algo.lower()
    obs_space = model.observation_space

    samples = np.stack([obs_space.sample() for _ in range(int(batch))]).astype(np.float32)
    device = getattr(model.policy, "device", "cpu")
    obs = torch.as_tensor(samples, device=device)

    model.policy.set_training_mode(False)

    fcm = FlopCounterMode(display=False)
    with torch.no_grad():
        with fcm:
            model.policy(obs)

    return float(fcm.get_total_flops()) / float(batch)


def _auto_forward_steps(model, algo: str, *, cycles: int = 4, floor: int = 256) -> int:
    """Smallest env-step budget that reliably triggers several update phases.
    
    model : SB3 RL model (loaded from a checkpoint, so the actual hyperparameters are used)
    algo  : algorithm name (lowercase)
    cycles: number of update phases to trigger (default 4)
    floor : minimum number of forward env steps to use (default 256)

    FLOPs per optimizer step is deterministic (fixed matmuls on a `batch_size`
    minibatch), so only a few update phases are needed to measure it exactly.
    This scales the forward pass to the model's own update cadence instead of using a
    large fixed budget.
    """
    algo = algo.lower()

    if algo == "ppo":
        n_steps = int(getattr(model, "n_steps", 2048))
        n_envs = int(getattr(model, "n_envs", 1))
        return max(floor, cycles * n_steps * max(1, n_envs))

    learning_starts = int(getattr(model, "learning_starts", 0))

    freq = 100
    tf = getattr(model, "train_freq", None)
    if tf is not None:
        f = getattr(tf, "frequency", None)
        unit = getattr(getattr(tf, "unit", None), "value", "step")
        if f is not None and unit == "step":
            freq = int(f)

    return max(floor, learning_starts + cycles * freq)


def measure_algo_flops(
    model,
    algo: str,
    *,
    n_envs_train: int,
    forward_env_steps: int | None = None,
    adam_per_param: float = ADAM_FLOPS_PER_PARAM,
    polyak_per_param: float = POLYAK_FLOPS_PER_PARAM,
) -> dict:
    """Measure FLOPs on a per-gradient-sample basis (robust to the forward n_envs).

    Matmul FLOPs scale with the number of gradient samples processed; Adam and
    Polyak (element-wise, not seen by FlopCounter) scale with optimizer steps.

    Returns
    -------
    dict with:
        f_sample     : matmul FLOPs per gradient sample (measured).
        f_opt_fixed  : Adam + Polyak FLOPs per optimizer step (analytic).
        f_policy        : per-sample policy forward FLOPs (measured).
        f_per_update : FLOPs of one real minibatch step (batch_size samples).
        batch_size   : real minibatch size.
        n_steps      : PPO rollout length per env (None for off-policy).
        n_mb_real    : PPO minibatches per epoch at the real n_envs (1 off-policy).
        is_ppo       : bool.
    """

    algo = algo.lower()

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        model.lr_schedule = (lambda _progress_remaining: 1e-3)
    except Exception:
        pass
    if hasattr(model, "learning_starts"):
        model.learning_starts = 0
    if hasattr(model, "verbose"):
        model.verbose = 0
    if hasattr(model, "action_noise"):
        model.action_noise = None

    n_envs_forward = int(getattr(model, "n_envs", 1))

    auto = _auto_forward_steps(model, algo)
    steps = auto if forward_env_steps is None else max(int(forward_env_steps), auto)

    f_policy = measure_policy_flops(model, algo)

    updates0 = float(getattr(model, "_n_updates", 0.0))

    fcm = FlopCounterMode(display=False)
    with fcm:
        model.learn(
            total_timesteps=int(steps),
            reset_num_timesteps=True,
            progress_bar=False,
        )
    f_total = float(fcm.get_total_flops())

    d_steps = float(getattr(model, "num_timesteps", 0.0))
    d_updates = float(getattr(model, "_n_updates", 0.0)) - updates0

    if d_updates <= 0:
        raise RuntimeError(
            f"no gradient updates during a {int(steps)}-step forward pass for {algo}"
        )

    batch = int(getattr(model, "batch_size"))
    is_ppo = (algo == "ppo")

    if is_ppo:
        # SB3 PPO increments _n_updates by n_epochs per train() call, so
        # d_updates counts epochs; each epoch processes one full forward rollout.
        n_steps = int(getattr(model, "n_steps"))
        grad_samples_forward = d_updates * (n_steps * n_envs_forward)
        n_mb_real = n_minibatches(model, int(n_envs_train))
    else:
        # Off-policy: d_updates counts optimizer steps, each of batch samples.
        n_steps = None
        grad_samples_forward = d_updates * batch
        n_mb_real = 1

    f_updates_matmul = max(f_total - f_policy * d_steps, 0.0)
    f_sample = f_updates_matmul / grad_samples_forward

    pc = param_counts(model, algo)
    f_opt_fixed = adam_per_param * pc["n_optimized"] + polyak_per_param * pc["n_target"]

    return {
        "f_sample": float(f_sample),
        "f_opt_fixed": float(f_opt_fixed),
        "f_policy": float(f_policy),
        "f_per_update": float(f_sample * batch + f_opt_fixed),
        "batch_size": int(batch),
        "n_steps": n_steps,
        "n_mb_real": int(n_mb_real),
        "is_ppo": bool(is_ppo),
    }


# -----------------------------------------------------------------------------
# Convergence anchoring (mirrors efficiency_metrics.time_to_converged)
# -----------------------------------------------------------------------------

def _converged_index(t, u, egm, tol) -> int | None:
    """Index (in time-sorted order) of practical convergence, or None.

    Convergence = earliest evaluation whose normalized regret is within `tol`
    of the best (smallest) normalized regret. Matches the efficiency metric.
    """
    t = np.asarray(t, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    if t.size == 0 or not np.isfinite(egm) or egm == 0.0:
        return None

    order = np.argsort(t)
    uu = u[order]

    r = normalized_regret(egm, uu)
    if not np.any(np.isfinite(r)):
        return None

    r_best = float(np.nanmin(r))
    hit = np.where(r <= r_best + float(tol))[0]

    return int(hit[0]) if hit.size else None


# -----------------------------------------------------------------------------
# Cumulative FLOPs and sampled states, per seed
# -----------------------------------------------------------------------------

def cumulative_flops_per_seed(
    histories,
    egm_returns,
    *,
    flops: dict,
    n_envs_train: int,
    tol: float,
):
    """Per-seed cumulative training FLOPs and sampled states at convergence.

    Matmul FLOPs scale with gradient samples; Adam/Polyak with optimizer steps;
    inference with environment steps:

        cumulative = f_sample * N_samples + f_opt_fixed * N_optsteps
                   + f_policy   * steps_at_conv

    Off-policy:  N_optsteps = updates_c,          N_samples = updates_c * batch.
    PPO:         N_optsteps = epochs_c * n_mb_real, N_samples = epochs_c * n_steps * n_envs_train.
    """
    cumflops_out, states_out = [], []

    for hist, egm in zip(histories, egm_returns):
        if hist is None or hist.get("steps") is None or hist.get("updates") is None:
            cumflops_out.append(np.nan)
            states_out.append(np.nan)
            continue

        t = np.asarray(hist["t"], dtype=np.float64)
        u = np.asarray(hist["u"], dtype=np.float64)
        steps = np.asarray(hist["steps"], dtype=np.float64)
        updates = np.asarray(hist["updates"], dtype=np.float64)

        idx = _converged_index(t, u, egm, tol)
        if idx is None:
            cumflops_out.append(np.nan)
            states_out.append(np.nan)
            continue

        order = np.argsort(t)
        s_c = steps[order][idx]
        u_c = updates[order][idx]

        if not (np.isfinite(s_c) and np.isfinite(u_c)):
            cumflops_out.append(np.nan)
            states_out.append(np.nan)
            continue

        if flops["is_ppo"]:
            n_optsteps = u_c * float(flops["n_mb_real"])
            n_samples = u_c * float(flops["n_steps"] * int(n_envs_train))
        else:
            n_optsteps = u_c
            n_samples = u_c * float(flops["batch_size"])

        cumflops = (
            flops["f_sample"] * n_samples
            + flops["f_opt_fixed"] * n_optsteps
            + flops["f_policy"] * s_c
        )
        cumflops_out.append(float(cumflops))
        states_out.append(float(s_c))

    return cumflops_out, states_out