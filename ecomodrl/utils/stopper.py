import os
import time
import numpy as np
from typing import Optional, List, Tuple

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize

from ecomodrl.utils.learning_schedule import LinearScheduleByProcessedSamples, LinearScheduleByWallClockTime


# -----------------------
# Helper functions
# -----------------------

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


# -----------------------
# Shared base class
# -----------------------

class _BaseEarlyStoppingEvalCallback(BaseCallback):
    """
    Shared functionality:
      - rollout-boundary evaluation (executed in _on_rollout_start)
      - env-step scheduling of evaluation (set in _on_step)
      - VecNormalize sync + raw reward evaluation
      - early stopping, best checkpoint save + reload
      - histories for env/proc/time axes
    Subclasses implement the budget mechanism.
    """

    def __init__(
        self,
        eval_env,
        beta: float,
        eval_freq_steps: int,
        patience: int,
        min_delta: float,
        n_eval_episodes: int = 1,
        deterministic_eval: bool = True,
        best_model_path: str = "best_model.zip",
        vecnorm_path: Optional[str] = "vecnormalize.pkl",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.beta = float(beta)
        self.eval_freq_steps = int(eval_freq_steps)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic_eval = bool(deterministic_eval)

        self.best_model_path = str(best_model_path)
        self.vecnorm_path = None if vecnorm_path is None else str(vecnorm_path)

        # runtime state
        self.best_score = -np.inf
        self.last_eval_score = -np.inf
        self.patience_counter = 0
        self._stop_reason = "none"

        # scheduling flags
        self._last_eval_env = 0
        self._eval_due = False
        self._final_eval_pending = False  # ensures evaluation after last update before stopping

        # timing
        self._t0 = 0.0

        # histories (three axes)
        self.return_history_env: List[Tuple[int, float]] = []
        self.return_history_proc: List[Tuple[int, float]] = []
        self.return_history_time: List[Tuple[float, float]] = []

        self.processed_samples = 0

    def _init_callback(self) -> None:
        self.best_score = -np.inf
        self.last_eval_score = -np.inf
        self.patience_counter = 0
        self._stop_reason = "none"

        self._last_eval_env = 0
        self._eval_due = False
        self._final_eval_pending = False

        self.return_history_env = []
        self.return_history_proc = []
        self.return_history_time = []

        self._t0 = time.perf_counter()

        if self.verbose:
            print("────────────────────────────────────────────", flush=True)
            print(f"  TRAINING WITH EARLY STOPPING ({self.__class__.__name__})", flush=True)
            print("────────────────────────────────────────────", flush=True)

    def _on_rollout_start(self) -> None:
        # Evaluate at rollout boundaries when due OR if a final eval was requested.
        if self._eval_due or self._final_eval_pending:
            self._eval_due = False
            self._final_eval_pending = False
            self._run_eval_and_maybe_stop()

    def _on_step(self) -> bool:
        # Subclass may update budgets / stop reason here (e.g. wall-clock budget).
        self._budget_hook_on_step()

        if self._stop_reason != "none":
            return False

        # Schedule eval (by env steps), execute at rollout boundary.
        if self.eval_freq_steps > 0 and (self.num_timesteps - self._last_eval_env) >= self.eval_freq_steps:
            self._last_eval_env = int(self.num_timesteps)
            self._eval_due = True

        return True

    def _on_training_end(self) -> None:
        # Reload best parameters
        try:
            if self.best_model_path and os.path.exists(self.best_model_path):
                self.model.set_parameters(self.best_model_path)
        except Exception as e:
            if self.verbose:
                print(f"Warning: could not reload best model ({e}).", flush=True)

        if self.verbose:
            print(f"[Stop] reason={self._stop_reason}", flush=True)

    # ---- hooks for subclasses ----

    def _budget_hook_on_step(self) -> None:
        """Subclass hook: wall-clock budget enforcement belongs here. Default: no-op."""
        return

    # ---- evaluation core ----

    def _run_eval_and_maybe_stop(self) -> None:
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

        # Early stopping + saving best
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
            print(
                f"[{t_env:8d}] eval_return={avg_return: .6f} | best={self.best_score: .6f} | "
                f"patience={self.patience_counter}/{self.patience} | proc={t_proc} | time={t_wall:.1f}s",
                flush=True,
            )


# -----------------------
# 1) Processed-samples budget callback (refined)
# -----------------------

class ProcessedSamplesEarlyStoppingCallback(_BaseEarlyStoppingEvalCallback):
    """
    Your processed-samples budget callback, with evaluation scheduling fixed:
      - eval scheduled by env steps (eval_freq_steps) but executed at rollout start
      - final evaluation guaranteed after last update when budget is reached
    """

    def __init__(
        self,
        eval_env,
        beta: float,
        eval_freq_steps: int,
        patience: int,
        min_delta: float,
        max_processed_samples: Optional[int] = None,
        n_eval_episodes: int = 1,
        deterministic_eval: bool = True,
        best_model_path: str = "best_model.zip",
        vecnorm_path: Optional[str] = "vecnormalize.pkl",
        lr_schedule_obj: Optional[LinearScheduleByProcessedSamples] = None,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            beta=beta,
            eval_freq_steps=eval_freq_steps,
            patience=patience,
            min_delta=min_delta,
            n_eval_episodes=n_eval_episodes,
            deterministic_eval=deterministic_eval,
            best_model_path=best_model_path,
            vecnorm_path=vecnorm_path,
            verbose=verbose,
        )
        self.max_processed_samples = None if max_processed_samples is None else int(max_processed_samples)
        self.lr_schedule_obj = lr_schedule_obj

        # cached algo params
        self._algo_family = "unknown"  # "ppo" or "off_policy"
        self._n_envs = 1
        self._ppo_n_steps = None
        self._ppo_n_epochs = None
        self._off_batch_size = None
        self._off_gradient_steps = None
        self._off_learning_starts = None

    def _init_callback(self) -> None:
        super()._init_callback()

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

        self.processed_samples = 0

        if self.verbose and self.max_processed_samples is not None:
            print(f"[Budget] max_processed_samples={self.max_processed_samples}", flush=True)

    def _on_rollout_end(self) -> None:
        """
        Budget accounting: end of rollout collection, right before the update chunk runs.
        If budget is reached, we request a final eval on the next rollout start (post-update),
        then training stops via _on_step().
        """
        if self.max_processed_samples is None or self._stop_reason != "none":
            return

        planned = self._planned_processed_samples_for_next_update()
        if planned <= 0:
            return

        self.processed_samples += planned

        if self.lr_schedule_obj is not None:
            self.lr_schedule_obj.update(self.processed_samples)

        # Budget logging
        self.logger.record("budget/processed_samples", float(self.processed_samples))
        self.logger.record("budget/max_processed_samples", float(self.max_processed_samples))
        self.logger.record("budget/usage_fraction", float(self.processed_samples) / float(self.max_processed_samples))

        if self.processed_samples >= self.max_processed_samples:
            self._stop_reason = "budget"
            # ensure we still evaluate once after the last update has happened
            self._final_eval_pending = True

            if self.verbose:
                print(
                    f"[Budget] Reached: +{planned} -> {self.processed_samples}/{self.max_processed_samples}. "
                    f"Will do final eval at next rollout start, then stop.",
                    flush=True,
                )

    def _planned_processed_samples_for_next_update(self) -> int:
        if self._algo_family == "ppo":
            if not self._ppo_n_steps or not self._ppo_n_epochs:
                return 0
            return int(self._n_envs * self._ppo_n_steps * self._ppo_n_epochs)

        if self._algo_family == "off_policy":
            if self._off_batch_size is None:
                return 0

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


# -----------------------
# 2) Wall-clock budget callback
# -----------------------

class WallClockEarlyStoppingCallback(_BaseEarlyStoppingEvalCallback):
    """
    Same behaviour as ProcessedSamplesEarlyStoppingCallback, but the budget is wall-clock time.

    - Budget is enforced in _on_step()
    - When budget is reached, we optionally request a final evaluation at the next rollout start.
      If you want strictly "stop immediately", set final_eval_on_budget=False.
    """

    def __init__(
        self,
        eval_env,
        beta: float,
        eval_freq_steps: int,
        patience: int,
        min_delta: float,
        max_wallclock_seconds: Optional[float] = None,
        n_eval_episodes: int = 1,
        deterministic_eval: bool = True,
        best_model_path: str = "best_model.zip",
        vecnorm_path: Optional[str] = "vecnormalize.pkl",
        final_eval_on_budget: bool = True,
        lr_schedule_obj: Optional[LinearScheduleByWallClockTime] = None,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            beta=beta,
            eval_freq_steps=eval_freq_steps,
            patience=patience,
            min_delta=min_delta,
            n_eval_episodes=n_eval_episodes,
            deterministic_eval=deterministic_eval,
            best_model_path=best_model_path,
            vecnorm_path=vecnorm_path,
            verbose=verbose,
        )
        self.max_wallclock_seconds = None if max_wallclock_seconds is None else float(max_wallclock_seconds)
        self.final_eval_on_budget = bool(final_eval_on_budget)

        self.lr_schedule_obj = lr_schedule_obj

    def _init_callback(self) -> None:
        super()._init_callback()
        if self.verbose and self.max_wallclock_seconds is not None:
            print(f"[Budget] max_wallclock_seconds={self.max_wallclock_seconds:.2f}", flush=True)

    def _budget_hook_on_step(self) -> None:
        if self.max_wallclock_seconds is None or self._stop_reason != "none":
            return

        elapsed = time.perf_counter() - self._t0
        
        if self.lr_schedule_obj is not None:
            self.lr_schedule_obj.update(elapsed)
            
        if elapsed >= self.max_wallclock_seconds:
            self._stop_reason = "budget"
            if self.final_eval_on_budget:
                self._final_eval_pending = True

            # log budget
            self.logger.record("budget/wallclock_seconds", float(elapsed))
            self.logger.record("budget/max_wallclock_seconds", float(self.max_wallclock_seconds))
            self.logger.record("budget/usage_fraction", float(elapsed) / float(self.max_wallclock_seconds))

            if self.verbose:
                print(
                    f"[Budget] Wall-clock reached: {elapsed:.2f}/{self.max_wallclock_seconds:.2f}s. "
                    f"{'Will do final eval if possible, then stop.' if self.final_eval_on_budget else 'Stopping.'}",
                    flush=True,
                )
