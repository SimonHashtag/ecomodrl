from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ecomodrl.algorithms.stopper import TrainingClock


class WallClockCosineLR:
    """Cosine learning-rate schedule based on training-only wall clock.

    SB3 expects:

        learning_rate(progress_remaining) -> float

    but we ignore `progress_remaining` and use the shared TrainingClock.

    Parameters
    ----------
    clock:
        TrainingClock instance.

    lr0:
        Initial learning rate.

    drop:
        Log10 drop by the end of training. For example:
            drop=1 -> lr_end = lr0 / 10
            drop=2 -> lr_end = lr0 / 100
            drop=0 -> constant lr0

    wallclock_seconds:
        Training-only wall-clock budget.
    """

    def __init__(self, clock: Optional[TrainingClock], lr0, drop, wallclock_seconds):
        self.clock = clock
        self.lr0 = float(lr0)
        self.end = float(lr0) / (10.0 ** float(drop))
        self.T = max(float(wallclock_seconds), 1e-12)

    def __call__(self, _progress_remaining):
        t = self.clock.now() if self.clock is not None else 0.0
        frac = np.clip(t / self.T, 0.0, 1.0)

        return self.end + 0.5 * (self.lr0 - self.end) * (
            1.0 + math.cos(math.pi * frac)
        )
