from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class LinearScheduleByWallClockTime:
    """
    Linear LR schedule from initial_lr to final_lr as a function of
    elapsed_seconds / max_wallclock_seconds.

    The argument `_progress_remaining` (SB3 interface) is ignored. The internal
    counter is updated externally (e.g., by a wall-clock budget callback).

    Notes:
      - Uses seconds as the budget unit.
      - If elapsed_seconds exceeds max_wallclock_seconds, the schedule is clamped at final_lr.
    """
    initial_lr: float
    final_lr: float
    max_wallclock_seconds: float
    elapsed_seconds: float = 0.0

    _slope: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_wallclock_seconds <= 0:
            raise ValueError("max_wallclock_seconds must be positive.")
        self._slope = float(self.final_lr) - float(self.initial_lr)

    def update(self, elapsed_seconds: float) -> None:
        es = float(elapsed_seconds)
        if es < 0.0:
            raise ValueError("elapsed_seconds must be non-negative.")
        self.elapsed_seconds = es

    def __call__(self, _progress_remaining: float) -> float:
        frac = self.elapsed_seconds / float(self.max_wallclock_seconds)
        if frac > 1.0:
            frac = 1.0
        return float(self.initial_lr + self._slope * frac)
