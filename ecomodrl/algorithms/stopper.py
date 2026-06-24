from __future__ import annotations

import time

from stable_baselines3.common.callbacks import BaseCallback


class TrainingClock:
    """Wall clock excluding paused intervals.

    Used to make wall-clock budgets and wall-clock learning-rate schedules
    exclude time spent in evaluation, checkpointing, plotting, etc.
    """

    def __init__(self):
        self._t0 = time.perf_counter()
        self._paused_total = 0.0
        self._pause_started = None

    def now(self) -> float:
        """Elapsed unpaused seconds."""
        if self._pause_started is not None:
            return self._pause_started - self._t0 - self._paused_total

        return time.perf_counter() - self._t0 - self._paused_total

    def pause(self):
        """Start a paused interval if not already paused."""
        if self._pause_started is None:
            self._pause_started = time.perf_counter()

    def resume(self):
        """End a paused interval if currently paused."""
        if self._pause_started is not None:
            self._paused_total += time.perf_counter() - self._pause_started
            self._pause_started = None

    @property
    def paused(self) -> bool:
        return self._pause_started is not None


class WallClockBudgetCallback(BaseCallback):
    """Stop training after `seconds` of unpaused wall-clock time."""

    def __init__(self, clock: TrainingClock, seconds: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.clock = clock
        self.seconds = float(seconds)

    def _on_step(self) -> bool:
        return self.clock.now() < self.seconds
