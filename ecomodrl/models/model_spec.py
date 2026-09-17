"""Stable, lightweight serialization contract for economic models.

SB3 reconstructs a policy before loading its parameter state. Post-decision
policies therefore need enough model information in their constructor data to
rebuild economic transitions and features without serializing the model's
large derived grids and shock arrays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ecomodrl.models.base_model import EconomicModel
from ecomodrl.models.casskoopmans import CassKoopmans
from ecomodrl.models.brockmirman import BrockMirman
from ecomodrl.models.bufferstock import BufferStock
from ecomodrl.models.portfoliochoice import PortfolioChoice


_MODEL_CLASSES: dict[str, type[EconomicModel]] = {
    cls.__name__: cls
    for cls in (CassKoopmans, BrockMirman, BufferStock, PortfolioChoice)
}


def _plain_value(value: Any) -> Any:
    """Convert NumPy/dataclass values to checkpoint-stable Python containers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(v) for v in value]
    return value


@dataclass(frozen=True)
class ModelSpec:
    """Constructor description for one EconomicModel.

    params contains the calibration dataclass fields. constructor_kwargs
    contains behavior-affecting model settings that are not calibration fields,
    such as grid sizes and PortfolioChoice RQMC settings. Derived arrays are
    rebuilt by the model constructor and are never stored in the checkpoint.
    """

    model_cls: str
    params: dict[str, Any]
    constructor_kwargs: dict[str, Any] = field(default_factory=dict)


def make_model_spec(model: EconomicModel) -> ModelSpec:
    """Capture the complete lightweight constructor contract of model."""
    if not hasattr(model.params, "__dataclass_fields__"):
        raise TypeError(
            f"{type(model).__name__}.params must be a dataclass to create ModelSpec."
        )
    return ModelSpec(
        model_cls=type(model).__name__,
        params=_plain_value(asdict(model.params)),
        constructor_kwargs=_plain_value(model.model_spec_kwargs()),
    )


def model_from_spec(spec: ModelSpec) -> EconomicModel:
    """Rebuild an economic model from its constructor contract."""
    if not isinstance(spec, ModelSpec):
        raise TypeError(f"model_spec must be a ModelSpec, got {type(spec).__name__}.")
    try:
        model_cls = _MODEL_CLASSES[spec.model_cls]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model class {spec.model_cls!r}; known: {sorted(_MODEL_CLASSES)}"
        ) from exc

    params = model_cls.PARAMS_CLS(**spec.params)
    return model_cls(params=params, **spec.constructor_kwargs)
