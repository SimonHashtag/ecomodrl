from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    raise RuntimeError(
        "Python 3.11+ is required for tomllib (stdlib TOML parser). "
        "On older versions, install 'tomli' and switch the import."
    )

class BaseParams:
    """Base class to load parameters and override defaults from TOML or dict.
    NOTE: This class is intentionally NOT a dataclass. All methods below
    must operate on subclasses that ARE dataclasses.
    """

    @classmethod
    def _field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, strict: bool = True):
        """
        Build from a dict; optionally error on unknown keys to catch typos.
        """
        if strict:
            unknown = set(d) - cls._field_names()
            if unknown:
                raise KeyError(
                    f"Unknown parameter(s) in config: {sorted(unknown)}. "
                    f"Valid keys: {sorted(cls._field_names())}"
                )
        # keep only known keys (in case strict=False)
        clean = {k: v for k, v in d.items() if k in cls._field_names()}
        return cls(**{**asdict(cls()), **clean})

    @classmethod
    def load_from_toml(
        cls,
        toml_path: str | Path,
        *,
        base: BaseParams | None = None,
        strict: bool = True,
    ):
        """
        Load parameters from TOML and merge over either:
          - provided base params, or
          - the class defaults if base=None.

        The result is a NEW frozen ModelParams instance.
        """
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        if strict:
            unknown = set(data) - cls._field_names()
            if unknown:
                raise KeyError(
                    f"Unknown parameter(s) in TOML: {sorted(unknown)}. "
                    f"Valid keys: {sorted(cls._field_names())}"
                )

        base_params = base if base is not None else cls()
        # dataclass is frozen; build a new instance with overrides
        merged = {
            **asdict(base_params),
            **{k: v for k, v in data.items() if k in cls._field_names()}
        }
        return cls(**merged)

    @classmethod
    def load_params(cls, toml_path: str | Path | None = None, params_dict: dict | None = None) -> BaseParams:
        """
        Returns a frozen ModelParams object.
        - If toml_path is None: return defaults from the dataclass.
        - If provided: overrides defaults with values from TOML.
        """
        if toml_path is None and params_dict is None:
            return cls()
        elif params_dict is None:
            return cls.load_from_toml(toml_path, base=None, strict=True)
        else:
            return cls.from_dict(params_dict, strict=True)
        
    
@dataclass(frozen=True)
class CassKoopmansParams(BaseParams):
    """
    Canonical, frozen configuration for the Cass–Koopmans model and related planners.

    Attributes
    ----------
    gamma : float
        Coefficient of relative risk aversion σ.
        Must be > 0. The σ=1 case corresponds to log utility as a limit.
    beta : float
        Subjective discount factor β. Must lie in (0, 1).
    delta : float
        Capital depreciation rate δ per period in [0, 1].
    alpha : float
        Capital share α (output elasticity of capital) in (0, 1).
    A : float
        Productivity / technology level A > 0 in y = A * k^α.
    """
    gamma: float = 2.0
    beta: float = 0.95
    delta: float = 0.02
    alpha: float = 0.33
    A: float = 1.0

    # -------- Validation --------
    def __post_init__(self):
        if not (self.gamma > 0):
            raise ValueError("crra (σ) must be > 0")
        if not (0 < self.beta < 1):
            raise ValueError("beta (β) must be in (0, 1)")
        if not (0 <= self.delta <= 1):
            raise ValueError("delta (δ) must be in [0, 1]")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha (α) must be in (0, 1)")
        if not (self.A > 0):
            raise ValueError("A must be > 0")
