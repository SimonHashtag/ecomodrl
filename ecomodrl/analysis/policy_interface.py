# -----------------------------------------------------------------------------
# Common policy interface for EGM and RL.
#
# A Policy exposes:
#   - policy.share(state)        -> consumption share s
#   - policy.consumption(state)  -> consumption c
#   - policy.controls(state)     -> dict of controls (share, consumption, portfolio_weights)
#   - policy(state)              -> alias for policy.consumption(state)
#
# State convention:
#   state = (m, x1, ..., xd)
#
# Both single-state inputs with shape (state_dim,) and batched inputs with shape
# (..., state_dim) are supported.
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from ecomodrl.envs.envs import obs_from_state

# ==============================================================================
# Batch state helper
# ==============================================================================

def as_state_batch(state):
    """Convert a state array to batched form.

    Returns
    -------
    state_batch: Array with shape (N, state_dim) or more generally (..., state_dim).
    single: True if the input was a single state with shape (state_dim,).
    """
    state_arr = np.asarray(state, dtype=np.float64)
    single = state_arr.ndim == 1

    if single:
        state_arr = state_arr[None, :]

    return state_arr, single

# ==============================================================================
# Policy interface classes
# ==============================================================================

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
    
    def controls(self, state):
        state = np.asarray(state, dtype=np.float64)

        c = self.consumption(state)
        m = state[..., 0]
        share = c / m

        out = {
            "share": share,
            "consumption": c,
        }

        if hasattr(self.model, "portfolio_weights"):
            out["portfolio_weights"] = self.model.portfolio_weights(state)

        return out

    def __call__(self, state):
        return self.consumption(state)


class RLPolicy:
    """Wrap a trained SB3 model in the common policy interface.

    Parameters
    ----------
    sb3_model: Trained Stable-Baselines3 model.

    model: Economic model instance.

    Observations are fed to the SB3 model exactly as `obs_from_state` produces
    them; the training environments apply no observation normalization.
    """

    def __init__(
        self,
        sb3_model,
        model,
    ):
        self.sb3_model = sb3_model
        self.model = model

    def share(self, state):
        controls, single = self._controls(state)

        if "share" not in controls:
            raise KeyError(
                "model.controls_from_action(...) must return a 'share' entry "
                "for RLPolicy.share()."
            )

        share = np.asarray(controls["share"], dtype=np.float64)

        if single:
            return float(share.reshape(-1)[0])

        return share
    
    def controls(self, state):
        controls, single = self._controls(state)

        if single:
            out = {}
            for k, v in controls.items():
                arr = np.asarray(v)
                if arr.ndim == 0:
                    out[k] = float(arr)
                elif arr.shape[0] == 1:
                    out[k] = arr[0]
                else:
                    out[k] = arr
            return out

        return controls

    def consumption(self, state):
        controls = self.controls(state)

        if "consumption" in controls:
            return controls["consumption"]

        state_arr, single = as_state_batch(state)
        c = np.asarray(controls["share"], dtype=np.float64) * state_arr[..., 0]

        if single:
            return float(c.reshape(-1)[0])

        return c

    def __call__(self, state):
        return self.consumption(state)

    def _predict_actions(self, state_arr):
        """Predict actions from raw economic states."""
        flat_shape = state_arr.shape[:-1]
        state_flat = state_arr.reshape(-1, state_arr.shape[-1])

        obs = obs_from_state(state_flat, self.model)

        actions, _ = self.sb3_model.predict(obs, deterministic=True)
        actions = np.asarray(actions, dtype=np.float64)

        return actions.reshape(flat_shape + (self.model.action_dim,))
    
    def _controls(self, state):
        """Predict SB3 action and map it to economic controls."""
        state_arr, single = as_state_batch(state)
        actions = self._predict_actions(state_arr)

        controls = self.model.controls_from_action(
            actions,
            state=state_arr,
        )

        return controls, single