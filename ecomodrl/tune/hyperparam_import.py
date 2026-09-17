from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from ecomodrl.algorithms.hyperparams import DEFAULT_HP_BY_ALGO
from ecomodrl.configs.variants import DEFAULT_VARIANT, algo_dir_name


def _candidate_incumbents(
    env_name: str,
    algo_name: str,
    tune_results_dir: Optional[Path],
    variant: str = DEFAULT_VARIANT,
) -> Iterable[Path]:
    """Generate candidate paths for incumbent.json files to load tuned hyperparameters from.
    First, a custom tune_results_dir path is checked, then the TUNERESULTS_PATH environment variable, and finally a default relative path.

    Args:
        env_name (str): _environment name, e.g. "CassKoopmans". Used to construct candidate paths.
        algo_name (str): _algorithm name, e.g. "ddpg". Used to construct candidate paths.
        tune_results_dir (Optional[Path]): If provided, this path is checked first for incumbent.json files.
        variant (str): Algorithm variant ("mf" or "pd"); post-decision incumbents live in "<ALGO>_PD".

    Returns:
        Iterable[Path]: An iterable of candidate paths to incumbent.json files, in order of preference.

    Yields:
        Iterator[Iterable[Path]]: Paths to check for incumbent.json files.
    """
    bases: list[Path] = []
    if tune_results_dir:
        bases.append(Path(tune_results_dir))

    env_path = os.environ.get("TUNERESULTS_PATH")
    if env_path and env_path.strip():
        bases.append(Path(env_path.strip()) / env_name)

    bases.append(Path("./tune_results") / env_name)

    seen: set[str] = set()
    for base in bases:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        yield base / algo_dir_name(algo_name, variant) / "incumbent.json"
        yield base / "incumbent.json"


def load_tuned_hp(
    *,
    env_name: str,
    algo_name: str,
    tune_results_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    variant: str = DEFAULT_VARIANT,
) -> Dict[str, Any]:
    """Load a set of tuned hyperparameters.

    Args:
        env_name (str): _environment name, e.g. "CassKoopmans". Used to construct candidate paths.
        algo_name (str): _algorithm name, e.g. "ddpg". Used to construct candidate paths.
        tune_results_dir (Optional[Path], optional): If provided, this path is checked first for incumbent.json files. Defaults to None.
        logger (Optional[logging.Logger], optional): Logger instance for logging messages. Defaults to None.
        variant (str): Algorithm variant ("mf" or "pd"); selects the "<ALGO>" or "<ALGO>_PD" incumbent.

    Raises:
        ValueError: If algo_name is not supported.

    Returns:
        Dict[str, Any]: A dictionary of hyperparameters loaded from the incumbent.json file, or default hyperparameters if no file is found or an error occurs.
        A loaded config is merged over the defaults, so hyperparameters added after the tuning run fall back to their default value.
    """
    logger = logger or logging.getLogger(__name__)
    algo_key = algo_name.strip().lower()
    defaults = DEFAULT_HP_BY_ALGO.get(algo_key)
    if defaults is None:
        raise ValueError(f"Unsupported algo_name: {algo_name}")

    path = next(
        (p for p in _candidate_incumbents(env_name, algo_key, tune_results_dir, variant) if p.exists()),
        None,
    )
    if path is None:
        logger.warning(
            "No tuned incumbent found for %s/%s. Using defaults.",
            env_name,
            algo_key,
        )
        return dict(defaults)

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read incumbent.json at %s: %s", path, exc)
        logger.warning(
            "No tuned incumbent found for %s/%s. Using defaults.",
            env_name,
            algo_key,
        )
        return dict(defaults)

    if isinstance(data, Mapping):
        config = data.get("config", data)
        if isinstance(config, Mapping):
            logger.info(
                "Loaded tuned hyperparameters for %s/%s from %s.",
                env_name,
                algo_dir_name(algo_key, variant),
                path,
            )
            return {**defaults, **dict(config)}

    logger.warning(
        "incumbent.json at %s does not contain a config mapping. Using defaults.",
        path,
    )
    return dict(defaults)