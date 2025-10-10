from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def tb_read_scalar(log_dir: str | Path, tag: str):
    """
    Recursively search `log_dir` for TensorBoard event files,
    pick the newest one (by modification time), and load the given scalar tag.
    Returns (steps, values) as numpy arrays.
    """
    log_dir = Path(log_dir)
    # pick the newest event file in the run dir
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {log_dir}")
    ev = max(event_files, key=lambda p: p.stat().st_mtime)
    acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        raise KeyError(f"Tag {tag!r} not found. Available: {acc.Tags().get('scalars', [])}")
    scalars = acc.Scalars(tag)
    steps  = np.array([s.step for s in scalars])
    values = np.array([s.value for s in scalars], dtype=float)
    return steps, values