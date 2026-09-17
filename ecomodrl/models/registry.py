# -----------------------------------------------------------------------------
# Model registry and the post-decision spec.
#
# The post-decision critics (ecomodrl/algorithms/post_decision) carry a small,
# picklable spec in SB3's `policy_kwargs` so that a checkpoint can be reloaded
# without an environment. The spec holds the model class name, the calibration
# and the training reward floor; the economic model is rebuilt from it.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ecomodrl.models.base_model import EconomicModel
from ecomodrl.models.casskoopmans import CassKoopmans
from ecomodrl.models.brockmirman import BrockMirman
from ecomodrl.models.bufferstock import BufferStock
from ecomodrl.models.portfoliochoice import PortfolioChoice


MODEL_CLASSES: dict[str, type[EconomicModel]] = {
    cls.__name__: cls for cls in (CassKoopmans, BrockMirman, BufferStock, PortfolioChoice)
}


def make_pd_spec(model: EconomicModel, reward_floor: float) -> dict:
    """Plain-Python spec from which `model_from_pd_spec` rebuilds `model`."""
    params = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in asdict(model.params).items()
    }
    return {
        "model_cls": type(model).__name__,
        "params": params,
        "reward_floor": float(reward_floor),
    }


def model_from_pd_spec(spec: dict) -> EconomicModel:
    """Rebuild the economic model with the same constructor call the envs use."""
    if spec is None:
        raise ValueError("A post-decision spec is required (see make_pd_spec).")
    try:
        cls = MODEL_CLASSES[str(spec["model_cls"])]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model class {spec.get('model_cls')!r}; known: {sorted(MODEL_CLASSES)}"
        ) from exc
    return cls(params=cls.PARAMS_CLS(**spec["params"]))
