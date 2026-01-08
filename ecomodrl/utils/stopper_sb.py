import os
import time
import numpy as np
from typing import Optional, List, Tuple
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize


def unwrap_vecnormalize(env) -> Optional[VecNormalize]:
    """Return the inner VecNormalize wrapper if present, else None."""
    e = env
    while isinstance(e, VecEnvWrapper):
        if isinstance(e, VecNormalize):
            return e
        e = e.venv
    return None


def sync_vecnormalize(dst: VecNormalize, src: VecNormalize) -> None:
    """Sync eval stats with training stats."""
    dst.obs_rms = src.obs_rms
    dst.ret_rms = src.ret_rms


def evaluate_policy_discounted(
    model,
    eval_env,
    beta: float,
    n_episodes: int = 1,
    deterministic: bool = True,
) -> float:
    """
    Discounted-return evaluation on a VecEnv(-like) interface.
    Uses mean reward across vectorized envs as a proxy for aggregate return.
    """
    if hasattr(eval_env, "training"):
        eval_env.training = False

    avg_return = 0.0
    n_envs = int(getattr(eval_env, "num_envs", 1))

    for _ in range(int(n_episodes)):
        obs = eval_env.reset()
        dones = np.zeros(n_envs, dtype=bool)
        G, power = 0.0, 1.0

        while not dones.all():
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, done_step, _ = eval_env.step(action)
            G += float(np.mean(rewards)) * power
            power *= float(beta)
            dones |= done_step

        avg_return += G

    return avg_return / float(n_episodes)


class EarlyStoppingCallback(BaseCallback):
    """
    SB3 callback for PPO, SAC, TD3, DDPG that:
      1) Tracks and limits a processed-samples budget:
         - PPO:       planned_samples = (n_envs * n_steps) * n_epochs
         - Off-policy: planned_samples = gradient_steps_eff * batch_size
           where gradient_steps_eff matches SB3 learn() semantics:
             * if gradient_steps >= 0: use it directly
             * if gradient_steps < 0 : use no. of transitions collected in the rollout
      2) Runs discounted evaluation at rollout boundaries, scheduled by processed-sample frequency.
      3) Syncs VecNormalize statistics (train -> eval) and evaluates with raw rewards.
      4) Saves best model and optional VecNormalize stats.
      5) Implements early stopping (patience/min_delta).
      6) Reloads best model at the end.

    Budget stopping is "non-strict": training stops at the *next* environment step after the
    final update chunk has executed (overshoot bounded by one update chunk).
    """

    def __init__(
        self,
        eval_env,
        beta: float,
        eval_freq: int,
        patience: int,
        min_delta: float,
        max_processed_samples: Optional[int] = None,
        n_eval_episodes: int = 1,
        deterministic_eval: bool = True,
        best_model_path: str = "best_model.zip",
        vecnorm_path: Optional[str] = "vecnormalize.pkl",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.beta = float(beta)
        self.eval_freq = int(eval_freq)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.max_processed_samples = None if max_processed_samples is None else int(max_processed_samples)

        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic_eval = bool(deterministic_eval)

        self.best_model_path = str(best_model_path)
        self.vecnorm_path = None if vecnorm_path is None else str(vecnorm_path)

        # runtime state
        self.best_score = -np.inf
        self.last_eval_score = -np.inf
        self.patience_counter = 0

        self.processed_samples = 0
        self._stop_reason = "none"

        # evaluation scheduling (processed-samples axis)
        self._last_eval_proc = 0

        # timing
        self._t0 = 0.0

        # histories (three axes)
        self.return_history_env: List[Tuple[int, float]] = []
        self.return_history_proc: List[Tuple[int, float]] = []
        self.return_history_time: List[Tuple[float, float]] = []

        # cached algo params
        self._algo_family = "unknown"  # "ppo" or "off_policy"
        self._n_envs = 1
        self._ppo_n_steps = None
        self._ppo_n_epochs = None
        self._off_batch_size = None
        self._off_gradient_steps = None
        self._off_learning_starts = None

    # -------- SB3 hooks --------

    def _init_callback(self) -> None:
        # Cache env info
        if self.training_env is None:
            self._n_envs = 1
        else:
            self._n_envs = int(getattr(self.training_env, "num_envs", 1))

        # Detect algorithm family + cache constants
        algo_name = self.model.__class__.__name__.lower()
        if "ppo" in algo_name:
            self._algo_family = "ppo"
            self._ppo_n_steps = int(getattr(self.model, "n_steps", 0)) or None
            self._ppo_n_epochs = int(getattr(self.model, "n_epochs", 0)) or None
        elif any(k in algo_name for k in ("sac", "td3", "ddpg")):
            self._algo_family = "off_policy"
            self._off_batch_size = int(getattr(self.model, "batch_size", 0)) or None
            self._off_gradient_steps = int(getattr(self.model, "gradient_steps", 0))
            self._off_learning_starts = int(getattr(self.model, "learning_starts", 0))
        else:
            self._algo_family = "unknown"

        # Reset counters/state
        self.best_score = -np.inf
        self.patience_counter = 0
        self.processed_samples = 0
        self._stop_reason = "none"

        self._last_eval_proc = 0

        self.return_history_env = []
        self.return_history_proc = []
        self.return_history_time = []

        self._t0 = time.perf_counter()

        if self.verbose:
            print("────────────────────────────────────────────", flush=True)
            print("  TRAINING WITH EARLY STOPPING", flush=True)
            print("────────────────────────────────────────────", flush=True)
            
    def _on_rollout_start(self) -> None:
        if self._stop_reason != "none":
            return

        if self.eval_freq <= 0:
            return

        if (self.processed_samples - self._last_eval_proc) >= self.eval_freq:
            self._last_eval_proc = self.processed_samples
            self._run_eval_and_maybe_stop()

    def _on_rollout_end(self) -> None:
        """
        Budget accounting happens here: end of rollout collection, right before the update chunk runs.
        (Non-strict stopping means we stop at the next env step after the final chunk.)
        """
        if self.max_processed_samples is None or self._stop_reason != "none":
            return

        planned = self._planned_processed_samples_for_next_update()
        if planned <= 0:
            return

        self.processed_samples += planned

        # Budget logging
        self.logger.record("budget/processed_samples", float(self.processed_samples))
        self.logger.record("budget/max_processed_samples", float(self.max_processed_samples))
        self.logger.record("budget/usage_fraction", float(self.processed_samples) / float(self.max_processed_samples))

        if self.processed_samples >= self.max_processed_samples:
            self._stop_reason = "budget"
            if self.verbose:
                print(
                    f"[Budget] Reached: +{planned} -> {self.processed_samples}/{self.max_processed_samples}. "
                    f"Stopping at next env step.",
                    flush=True,
                )

    def _on_step(self) -> bool:
        return self._stop_reason == "none"

    def _on_training_end(self) -> None:
        # Reload best model parameters (so model after learn() corresponds to best checkpoint)
        try:
            if self.best_model_path and os.path.exists(self.best_model_path):
                self.model.set_parameters(self.best_model_path)
        except Exception as e:
            if self.verbose:
                print(f"Warning: could not reload best model ({e}).", flush=True)

        if self.verbose:
            print(f"[Stop] reason={self._stop_reason}", flush=True)

    # -------- internal helpers --------

    def _planned_processed_samples_for_next_update(self) -> int:
        """Planned processed samples for the next training update chunk."""
        if self._algo_family == "ppo":
            if not self._ppo_n_steps or not self._ppo_n_epochs:
                return 0
            return int(self._n_envs * self._ppo_n_steps * self._ppo_n_epochs)

        if self._algo_family == "off_policy":
            if self._off_batch_size is None:
                return 0

            # learning_starts gating
            if self._off_learning_starts is not None and int(self.model.num_timesteps) <= int(self._off_learning_starts):
                return 0

            gs = int(self._off_gradient_steps if self._off_gradient_steps is not None else 0)
            if gs >= 0:
                gradient_steps_eff = gs
            else:
                gradient_steps_eff = self._offpolicy_rollout_transitions()

            if gradient_steps_eff <= 0:
                return 0

            return int(gradient_steps_eff * int(self._off_batch_size))

        return 0

    def _offpolicy_rollout_transitions(self) -> int:
        """
        Reconstruct number of transitions collected during the rollout.
        Prefer callback locals when present; fall back to train_freq if step-based.
        """
        num_collected_steps = self.locals.get("num_collected_steps", None)
        env = self.locals.get("env", None)

        n_envs = self._n_envs
        if env is not None and hasattr(env, "num_envs"):
            n_envs = int(env.num_envs)

        if num_collected_steps is not None:
            return int(num_collected_steps) * n_envs

        tf = getattr(self.model, "train_freq", None)
        freq = getattr(tf, "frequency", None)
        unit = getattr(tf, "unit", None)
        unit_name = getattr(unit, "name", str(unit))

        if freq is not None and "STEP" in str(unit_name).upper():
            return int(freq) * n_envs

        return 0

    def _run_eval_and_maybe_stop(self) -> None:
        # Sync VecNormalize (if present)
        train_norm = None
        try:
            train_norm = self.model.get_vec_normalize_env()
        except Exception:
            train_norm = unwrap_vecnormalize(self.model.get_env())

        eval_norm = unwrap_vecnormalize(self.eval_env)

        if train_norm is not None and eval_norm is not None:
            sync_vecnormalize(eval_norm, train_norm)
            eval_norm.training = False
            eval_norm.norm_reward = False
        elif hasattr(self.eval_env, "training"):
            self.eval_env.training = False

        # Evaluate
        avg_return = float(
            evaluate_policy_discounted(
                self.model,
                self.eval_env,
                beta=self.beta,
                n_episodes=self.n_eval_episodes,
                deterministic=self.deterministic_eval,
            )
        )
        self.last_eval_score = avg_return

        # Record histories on three axes
        t_env = int(self.num_timesteps)
        t_proc = int(self.processed_samples)
        t_wall = float(time.perf_counter() - self._t0)

        self.return_history_env.append((t_env, avg_return))
        self.return_history_proc.append((t_proc, avg_return))
        self.return_history_time.append((t_wall, avg_return))

        # Early stopping logic + saving best
        improved = avg_return > (self.best_score + self.min_delta)
        if improved:
            self.best_score = avg_return
            self.patience_counter = 0

            if self.best_model_path:
                self.model.save(self.best_model_path)

            if self.vecnorm_path and train_norm is not None and hasattr(train_norm, "save"):
                train_norm.save(self.vecnorm_path)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self._stop_reason = "early_stop"

        # Logging (dump at eval cadence)
        self.logger.record("eval/avg_discounted_return", avg_return)
        self.logger.record("eval/best_discounted_return", float(self.best_score))
        self.logger.record("eval/patience_counter", float(self.patience_counter))
        self.logger.record("eval/x_env_steps", float(t_env))
        self.logger.record("eval/x_processed_samples", float(t_proc))
        self.logger.record("eval/x_wallclock_seconds", float(t_wall))
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            budget_msg = ""
            if self.max_processed_samples is not None:
                budget_msg = f" | Budget: {self.processed_samples}/{self.max_processed_samples}"
            print(
                f"[{t_env:8d}] eval_return={avg_return: .6f} | best={self.best_score: .6f} | "
                f"patience={self.patience_counter}/{self.patience} | proc={t_proc} | time={t_wall:.1f}s{budget_msg}",
                flush=True,
            )
