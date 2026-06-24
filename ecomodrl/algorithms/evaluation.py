from __future__ import annotations

from pathlib import Path
from typing import Union, Mapping, Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ecomodrl.algorithms.stopper import TrainingClock
from ecomodrl.analysis.policy_interface import RLPolicy
from ecomodrl.analysis.accuracy_metrics import eval_return_grid


def is_vecnormalize(obj) -> bool:
    """Return True if obj looks like an SB3 VecNormalize wrapper."""
    return (
        obj is not None
        and hasattr(obj, "normalize_obs")
        and hasattr(obj, "obs_rms")
        and hasattr(obj, "save")
    )


class BestValueEvalCallback(BaseCallback):
    """Periodically evaluate the current RL policy and save the best checkpoint.

    The callback evaluates the current policy on a fixed evaluation sample:
    common initial states and common shock paths. This gives a low-noise,
    paired comparison of policies during training.

    The policy is scored by `eval_return_grid`, which computes mean discounted
    utility over the evaluation sample.

    Notes on VecNormalize
    ---------------------
    The callback saves both:
        - model.zip
        - vecnormalize.pkl

    because an SB3 policy trained with normalized observations must later be
    evaluated using the same observation-normalization statistics.
    """

    def __init__(
        self,
        cfg: Union[Mapping[str, Any], Any],
        algo_cfg: Union[Mapping[str, Any], Any],
        clock: TrainingClock,
        eco_model,
        evaluation_sample,
        out_dir,
        verbose: bool = True,
        vecnormalize=None,
        eval_on_training_end: bool = True,
    ):
        super().__init__(verbose=0)

        self.clock = clock
        self.economic_model = eco_model
        self.evaluation_sample = evaluation_sample
        self.out_dir = Path(out_dir)

        self.algo_name = str(algo_cfg.name)
        self.model_name = str(cfg.env_name)

        self.wallclock_seconds = float(cfg.max_wallclock_seconds)
        self.eval_interval_seconds = float(cfg.eval_interval_seconds)

        self.print_progress = bool(verbose)
        self.explicit_vecnormalize = vecnormalize
        self.eval_on_training_end = bool(eval_on_training_end)

        self.eval_times = []
        self.eval_returns = []

        self.best_value = -np.inf
        self.improved_at_least_once = False

        self._last_eval_time = None
        self._n_evaluations = 0

    def _on_training_start(self) -> None:
        self._last_eval_time = self.clock.now()

        if self.print_progress:
            n_states, horizon = int(self.evaluation_sample.n_states), int(self.evaluation_sample.horizon)

            print(
                f"[{self.algo_name} | {self.model_name}] training started "
                f"(budget={self.wallclock_seconds:.0f}s, "
                f"eval every {self.eval_interval_seconds:.1f}s, "
                f"N={n_states}, T={horizon})",
                flush=True,
            )

        self._evaluate_and_maybe_save(reason="initial")

    def _on_step(self) -> bool:
        now = self.clock.now()

        if now - self._last_eval_time >= self.eval_interval_seconds:
            self._evaluate_and_maybe_save(reason="periodic")
            self._last_eval_time = self.clock.now()

        return True

    def _on_training_end(self) -> None:
        if self.eval_on_training_end:
            self._evaluate_and_maybe_save(reason="final")

        self._save_history()

        if self.print_progress:
            best = f"{self.best_value:.6f}" if np.isfinite(self.best_value) else "nan"

            print(
                f"[{self.algo_name} | {self.model_name}] training ended | "
                f"best value = {best}",
                flush=True,
            )

    def _evaluate_and_maybe_save(self, reason: str) -> None:
        """Evaluate current policy and save checkpoint if value improves."""
        self.clock.pause()

        try:
            value = self._evaluate_current_policy()
            elapsed = self.clock.now()

            self.eval_times.append(float(elapsed))
            self.eval_returns.append(float(value))
            self._n_evaluations += 1

            improved = bool(np.isfinite(value) and value > self.best_value)

            if improved:
                self.best_value = float(value)
                self.improved_at_least_once = True
                self._save_best()

            if self.print_progress:
                marker = " *" if improved else ""
                value_str = f"{value:>12.6f}" if np.isfinite(value) else "nan"

                print(
                    f"[{self.algo_name} | {self.model_name}] "
                    f"eval={self._n_evaluations:03d} ({reason}) | "
                    f"t={elapsed:>6.1f}s / {self.wallclock_seconds:.0f}s | "
                    f"value={value_str}{marker}",
                    flush=True,
                )

        except Exception as exc:
            if self.print_progress:
                elapsed = self.clock.now()

                print(
                    f"[{self.algo_name} | {self.model_name}] "
                    f"error during {reason} evaluation at t={elapsed:.1f}s: {exc}",
                    flush=True,
                )

        finally:
            self.clock.resume()

    def _evaluate_current_policy(self) -> float:
        """Evaluate the current SB3 policy on the fixed evaluation sample."""
        vecnormalize = self._get_vecnormalize()

        old_training = (
            getattr(vecnormalize, "training", None)
            if vecnormalize is not None
            else None
        )

        try:
            policy = RLPolicy(
                sb3_model=self.model,
                vecnormalize=vecnormalize,
                model=self.economic_model,
            )

            return eval_return_grid(
                self.economic_model,
                policy,
                self.evaluation_sample,
            )

        finally:
            if vecnormalize is not None and old_training is not None:
                vecnormalize.training = old_training

    def _save_best(self) -> None:
        """Save best SB3 model, VecNormalize stats, and best-evaluation info."""
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.model.save(str(self.out_dir / "model.zip"))

        vecnormalize = self._get_vecnormalize()
        if vecnormalize is not None:
            vecnormalize.save(str(self.out_dir / "vecnormalize.pkl"))

        self._save_history()

        np.savez(
            self.out_dir / "best_info.npz",
            best_U=np.asarray(self.best_value),
            history_t=np.asarray(self.eval_times, dtype=np.float64),
            history_U=np.asarray(self.eval_returns, dtype=np.float64),
            n_evaluations=np.asarray(self._n_evaluations),
            improved_at_least_once=np.asarray(self.improved_at_least_once),
        )

    def _save_history(self) -> None:
        """Save the full evaluation history."""
        self.out_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            self.out_dir / "eval_history.npz",
            best_U=np.asarray(self.best_value),
            history_t=np.asarray(self.eval_times, dtype=np.float64),
            history_U=np.asarray(self.eval_returns, dtype=np.float64),
            n_evaluations=np.asarray(self._n_evaluations),
            improved_at_least_once=np.asarray(self.improved_at_least_once),
        )