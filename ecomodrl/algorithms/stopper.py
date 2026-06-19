import os
import time
import numpy as np
from typing import Optional, List, Tuple

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize

from ecomodrl.algorithms.learning_schedule import LinearScheduleByWallClockTime


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


def _seed_eval_env(eval_env, seed: int) -> None:
    if seed is None:
        return
    if hasattr(eval_env, "seed"):
        try:
            eval_env.seed(seed)
            return
        except Exception:
            pass
    if hasattr(eval_env, "venv") and hasattr(eval_env.venv, "seed"):
        try:
            eval_env.venv.seed(seed)
            return
        except Exception:
            pass
    if hasattr(eval_env, "envs"):
        try:
            for i, e in enumerate(eval_env.envs):
                if hasattr(e, "seed"):
                    e.seed(int(seed) + int(i))
        except Exception:
            pass


def evaluate_policy_discounted(
    model,
    eval_env,
    beta: float,
    n_episodes: int = 1,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> float:
    """
    Discounted-return evaluation on a VecEnv(-like) interface.
    Uses mean reward across vectorized envs as a proxy for aggregate return.
    """
    if hasattr(eval_env, "training"):
        eval_env.training = False

    avg_return = 0.0
    n_envs = int(getattr(eval_env, "num_envs", 1))

    for i in range(int(n_episodes)):
        ep_seed = None if seed is None else int(seed) + int(i)
        if ep_seed is not None:
            _seed_eval_env(eval_env, ep_seed)
        obs = eval_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
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
# Wall-clock budget callback
# -----------------------

class WallClockEarlyStoppingCallback(BaseCallback):
    """
    Early-stopping evaluation callback with a wall-clock time budget.

    - Rollout-boundary evaluation (executed in _on_rollout_start).
    - Env-step scheduling of evaluation (set in _on_step).
    - VecNormalize sync + raw reward evaluation.
    - Early stopping, best checkpoint save + reload.
    - Histories for env/time axes.
    - Budget is wall-clock time, enforced in _on_step().
    - When budget is reached, optionally requests a final evaluation at the
      next rollout start. Set final_eval_on_budget=False to stop immediately.
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
        eval_seed: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.beta = float(beta)
        self.eval_freq_steps = int(eval_freq_steps)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.max_wallclock_seconds = None if max_wallclock_seconds is None else float(max_wallclock_seconds)

        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic_eval = bool(deterministic_eval)

        self.best_model_path = str(best_model_path)
        self.vecnorm_path = None if vecnorm_path is None else str(vecnorm_path)
        self.final_eval_on_budget = bool(final_eval_on_budget)

        self.lr_schedule_obj = lr_schedule_obj
        self.eval_seed = None if eval_seed is None else int(eval_seed)

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
        self.return_history_time: List[Tuple[float, float]] = []

    def _init_callback(self) -> None:
        self.best_score = -np.inf
        self.last_eval_score = -np.inf
        self.patience_counter = 0
        self._stop_reason = "none"

        self._last_eval_env = 0
        self._eval_due = False
        self._final_eval_pending = False

        self.return_history_env = []
        self.return_history_time = []

        self._t0 = time.perf_counter()

        if self.verbose:
            print("────────────────────────────────────────────", flush=True)
            print(f"  TRAINING WITH EARLY STOPPING ({self.__class__.__name__})", flush=True)
            print("────────────────────────────────────────────", flush=True)
            if self.max_wallclock_seconds is not None:
                print(f"[Budget] max_wallclock_seconds={self.max_wallclock_seconds:.2f}", flush=True)

    def _on_rollout_start(self) -> None:
        # Evaluate at rollout boundaries when due OR if a final eval was requested.
        if self._eval_due or self._final_eval_pending:
            self._eval_due = False
            self._final_eval_pending = False
            self._run_eval_and_maybe_stop()

    def _on_step(self) -> bool:
        # Wall-clock budget enforcement.
        if self.max_wallclock_seconds is not None and self._stop_reason == "none":
            elapsed = time.perf_counter() - self._t0

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

        if self._stop_reason != "none":
            return False

        # Schedule eval (by env steps), execute at rollout boundary.
        if self.eval_freq_steps > 0 and (self.num_timesteps - self._last_eval_env) >= self.eval_freq_steps:
            self._last_eval_env = int(self.num_timesteps)
            self._eval_due = True

        return True

    def _on_rollout_end(self) -> None:
        elapsed = time.perf_counter() - self._t0
        if self.lr_schedule_obj is not None:
            self.lr_schedule_obj.update(elapsed)

    def _on_training_end(self) -> None:
        if self._final_eval_pending:
            self._final_eval_pending = False
            try:
                self._run_eval_and_maybe_stop()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: could not run final eval ({e}).", flush=True)
        # Reload best parameters
        try:
            if self.best_model_path and os.path.exists(self.best_model_path):
                self.model.set_parameters(self.best_model_path)
        except Exception as e:
            if self.verbose:
                print(f"Warning: could not reload best model ({e}).", flush=True)

        if self.verbose:
            print(f"[Stop] reason={self._stop_reason}", flush=True)

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
                seed=self.eval_seed,
            )
        )
        self.last_eval_score = avg_return

        # Record histories on three axes
        t_env = int(self.num_timesteps)
        t_wall = float(time.perf_counter() - self._t0)

        self.return_history_env.append((t_env, avg_return))
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
            if self.patience_counter >= self.patience and self._stop_reason == "none":
                self._stop_reason = "early_stop"

        # Logging (dump at eval cadence)
        self.logger.record("eval/avg_discounted_return", avg_return)
        self.logger.record("eval/best_discounted_return", float(self.best_score))
        self.logger.record("eval/patience_counter", float(self.patience_counter))
        self.logger.record("eval/x_env_steps", float(t_env))
        self.logger.record("eval/x_wallclock_seconds", float(t_wall))
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            print(
                f"[{t_env:8d}] eval_return={avg_return: .6f} | best={self.best_score: .6f} | "
                f"patience={self.patience_counter}/{self.patience} | time={t_wall:.1f}s",
                flush=True,
            )