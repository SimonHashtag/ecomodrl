# -----------------------------------------------------------------------------
# Common policy interface for EGM and RL.
#
# A Policy exposes:
#   - policy.share(state)        -> consumption share s
#   - policy.consumption(state)  -> consumption c = s * m
#   - policy(state)              -> alias for policy.consumption(state)
#
# State convention:
#   state = (m, x1, ..., xd)
#
# Both single-state inputs with shape (state_dim,) and batched inputs with shape
# (..., state_dim) are supported.
#
# For RL policies trained behind VecNormalize:
#   - obs_from_state(state, model) creates the raw environment observation;
#   - VecNormalize.normalize_obs is then applied before model.predict().
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from ecomodrl.envs.envs import obs_from_state


class EGMPolicy:
    """Wrap a solved EGM model in the common policy interface."""

    def __init__(self, model):
        self.model = model

    def consumption(self, state):
        return self.model.consumption(state)

    def share(self, state):
        state = np.asarray(state, dtype=np.float64)
        c = self.consumption(state)
        m = state[..., 0]
        return c / m

    def __call__(self, state):
        return self.consumption(state)


class RLPolicy:
    """Wrap a trained SB3 model in the common policy interface.

    Parameters
    ----------
    sb3_model:
        Trained Stable-Baselines3 model.

    vecnormalize:
        Optional VecNormalize object with saved normalization statistics.
        If supplied, observations are normalized before calling
        `sb3_model.predict`.

    model:
        Economic model instance. Required for `obs_from_state` and for mapping
        actions to economic controls. If not supplied, this class tries to infer
        it from the environment attached to the SB3 model.

    Notes
    -----
    SB3 policies trained with VecNormalize see normalized observations.
    Therefore, when evaluating outside the environment, you must normalize the
    deterministic observation transform manually:

        raw_obs = obs_from_state(state, model)
        obs = vecnormalize.normalize_obs(raw_obs)

    Passing unnormalized observations to a policy trained with VecNormalize can
    produce completely wrong actions.
    """

    def __init__(
        self,
        sb3_model,
        vecnormalize=None,
        model=None,
    ):
        self.sb3_model = sb3_model
        self.vecnormalize = vecnormalize

        if self.vecnormalize is None:
            self.vecnormalize = self._infer_vecnormalize_from_sb3(sb3_model)

        if model is None:
            model = self._infer_model_from_sb3(sb3_model)

        if model is None:
            raise ValueError(
                "RLPolicy needs an economic model. Pass model=..., or attach "
                "an env with `.model` to the SB3 algorithm."
            )

        self.model = model

    # ------------------------------------------------------------------
    # Public policy API
    # ------------------------------------------------------------------

    def share(self, state):
        state_arr, single = self._prepare_state(state)

        flat_shape = state_arr.shape[:-1]
        state_flat = state_arr.reshape(-1, state_arr.shape[-1])

        obs_flat = obs_from_state(state_flat, self.model)
        obs_flat = self._normalize_obs_if_needed(obs_flat)

        actions, _ = self.sb3_model.predict(obs_flat, deterministic=True)
        actions = np.asarray(actions, dtype=np.float64)
        actions = actions.reshape(flat_shape + (self.model.action_dim,))

        controls = self.model.controls_from_action(actions, state=state_arr)

        if "share" not in controls:
            raise KeyError(
                "model.controls_from_action(...) must return a 'share' entry "
                "for RLPolicy.share()."
            )

        s = np.asarray(controls["share"], dtype=np.float64)

        if single:
            return float(s.reshape(-1)[0])

        return s

    def consumption(self, state):
        state_arr, single = self._prepare_state(state)

        s = self.share(state_arr)
        m = state_arr[..., 0]
        c = np.asarray(s) * m

        if single:
            return float(np.asarray(c).reshape(-1)[0])

        return c

    def __call__(self, state):
        return self.consumption(state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_state(state):
        state_arr = np.asarray(state, dtype=np.float64)
        single = state_arr.ndim == 1

        if single:
            state_arr = state_arr[None, :]

        return state_arr, single

    def _normalize_obs_if_needed(self, obs):
        obs = np.asarray(obs, dtype=np.float32)

        if self.vecnormalize is None:
            return obs

        # VecNormalize.normalize_obs expects observations in env observation
        # units and returns normalized observations in policy-network units.
        obs_norm = self.vecnormalize.normalize_obs(obs.copy())
        return np.asarray(obs_norm, dtype=np.float32)

    @classmethod
    def _infer_vecnormalize_from_sb3(cls, sb3_model):
        """Try to find a VecNormalize wrapper attached to an SB3 model."""
        try:
            env = sb3_model.get_env()
        except Exception:
            return None

        return cls._find_vecnormalize(env)

    @classmethod
    def _find_vecnormalize(cls, obj, seen=None):
        """Duck-typed recursive VecNormalize search."""
        if obj is None:
            return None

        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)

        # Duck typing instead of importing SB3 VecNormalize.
        if hasattr(obj, "normalize_obs") and hasattr(obj, "obs_rms"):
            return obj

        # VecNormalize.venv, Monitor.env, wrappers, etc.
        for attr in ("venv", "env"):
            child = getattr(obj, attr, None)
            found = cls._find_vecnormalize(child, seen)
            if found is not None:
                return found

        # DummyVecEnv/SubprocVecEnv style.
        envs = getattr(obj, "envs", None)
        if envs is not None:
            for child in envs:
                found = cls._find_vecnormalize(child, seen)
                if found is not None:
                    return found

        return None

    @classmethod
    def _infer_model_from_sb3(cls, sb3_model):
        """Try to recover the economic model from an SB3 model's env."""
        try:
            env = sb3_model.get_env()
        except Exception:
            return None

        return cls._find_model(env)

    @classmethod
    def _find_model(cls, obj, seen=None):
        """Recursively search env/wrapper structure for `.model`."""
        if obj is None:
            return None

        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)

        model = getattr(obj, "model", None)
        if model is not None:
            return model

        for attr in ("venv", "env"):
            child = getattr(obj, attr, None)
            found = cls._find_model(child, seen)
            if found is not None:
                return found

        envs = getattr(obj, "envs", None)
        if envs is not None:
            for child in envs:
                found = cls._find_model(child, seen)
                if found is not None:
                    return found

        return None