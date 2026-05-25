from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from ecomodrl.algorithms.hyperparams import DEFAULT_HP_BY_ALGO


def _normalize_algo_name(algo_name: str) -> str:
    return algo_name.strip().lower()


def _candidate_incumbents(
    env_name: str,
    algo_name: str,
    tune_results_dir: Optional[Path],
) -> Iterable[Path]:
    bases: list[Path] = []
    if tune_results_dir:
        bases.append(Path(tune_results_dir))

    env_path = os.environ.get("TUNERESULTS_PATH")
    if env_path and env_path.strip():
        bases.append(Path(env_path.strip()))

    bases.append(Path("./tune_results") / env_name)

    seen: set[str] = set()
    for base in bases:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        yield base / algo_name.upper() / "incumbent.json"
        yield base / "incumbent.json"


def load_tuned_config(
    *,
    env_name: str,
    algo_name: str,
    tune_results_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    algo_key = _normalize_algo_name(algo_name)
    defaults = DEFAULT_HP_BY_ALGO.get(algo_key)
    if defaults is None:
        raise ValueError(f"Unsupported algo_name: {algo_name}")

    path = next(
        (p for p in _candidate_incumbents(env_name, algo_key, tune_results_dir) if p.exists()),
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
                algo_key,
                path,
            )
            return dict(config)

    logger.warning(
        "incumbent.json at %s does not contain a config mapping. Using defaults.",
        path,
    )
    return dict(defaults)