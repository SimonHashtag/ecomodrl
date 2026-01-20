from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LinearScheduleByProcessedSamples:
    """
    Linear LR schedule from initial_lr to final_lr as a function of
    processed_samples / max_processed_samples.

    The argument `_progress_remaining` (SB3 interface) is ignored. The internal
    counter is updated externally (e.g., by an EarlyStopping callback).
    """
    initial_lr: float
    final_lr: float
    max_processed_samples: int
    processed_samples: int = 0

    _slope: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_processed_samples <= 0:
            raise ValueError("max_processed_samples must be a positive integer.")
        self._slope = float(self.final_lr) - float(self.initial_lr)

    def update(self, processed_samples: int) -> None:
        ps = int(processed_samples)
        if ps < 0:
            raise ValueError("processed_samples must be non-negative.")
        self.processed_samples = ps

    def __call__(self, _progress_remaining: float) -> float:
        frac = self.processed_samples / float(self.max_processed_samples)
        if frac > 1.0:
            frac = 1.0
        return float(self.initial_lr + self._slope * frac)