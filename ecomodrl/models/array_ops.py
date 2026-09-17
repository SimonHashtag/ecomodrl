# -----------------------------------------------------------------------------
# Minimal NumPy / Torch operation adapter for the economic models.
#
# The post-decision critics (ecomodrl/algorithms/post_decision) need a few model
# methods (utility, controls_from_action, state transforms, post-decision features)
# to be differentiable in torch. Instead of re-implementing the economics in
# torch, those methods are written once against the small op table below and
# dispatch on the input type:
#
#     xp = ops(x)          -> _NumpyOps if x is a numpy array / scalar
#                          -> _TorchOps if x is a torch.Tensor
#
# The numpy side delegates 1:1 to the numpy calls the models used before, so the
# environment path is unchanged. The torch side works in float64 and only uses
# differentiable ops (sort/gather/where/clamp), so autograd flows through.
#
# torch is never imported eagerly here: the models stay importable without it.
# -----------------------------------------------------------------------------

from __future__ import annotations

import sys

import numpy as np


def is_tensor(x) -> bool:
    """True if `x` is a torch.Tensor (without importing torch)."""
    th = sys.modules.get("torch")
    return th is not None and isinstance(x, th.Tensor)


class _NumpyOps:
    """NumPy implementation of the shared operation table."""

    @staticmethod
    def asarray(x, dtype=None):
        return np.asarray(x) if dtype is None else np.asarray(x, dtype=dtype)

    clip = staticmethod(np.clip)
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    log1p = staticmethod(np.log1p)
    maximum = staticmethod(np.maximum)
    where = staticmethod(np.where)
    atleast_1d = staticmethod(np.atleast_1d)

    @staticmethod
    def concatenate(seq, axis=-1):
        return np.concatenate(seq, axis=axis)

    @staticmethod
    def zeros(shape, like):
        return np.zeros(shape, dtype=np.float64)

    @staticmethod
    def sort_descending(x):
        return np.sort(x, axis=-1)[..., ::-1]

    @staticmethod
    def cumsum(x):
        return np.cumsum(x, axis=-1)

    @staticmethod
    def arange(start, stop, like):
        return np.arange(start, stop)

    @staticmethod
    def amax(x):
        return np.asarray(np.max(x, axis=-1))

    @staticmethod
    def take_along_axis(a, idx):
        return np.take_along_axis(a, idx, axis=-1)

    @staticmethod
    def sum_keepdims(x):
        return x.sum(axis=-1, keepdims=True)


class _TorchOps:
    """Torch implementation of the shared operation table, using float64."""

    @staticmethod
    def asarray(x, dtype=None):
        import torch as th
        if is_tensor(x):
            return x.to(th.float64)
        return th.as_tensor(np.asarray(x, dtype=np.float64))

    @staticmethod
    def clip(x, lo, hi):
        import torch as th
        return th.clamp(x, lo, hi)

    @staticmethod
    def exp(x):
        import torch as th
        return th.exp(x)

    @staticmethod
    def log(x):
        import torch as th
        return th.log(x)

    @staticmethod
    def log1p(x):
        import torch as th
        return th.log1p(x)

    @staticmethod
    def maximum(x, y):
        import torch as th
        if not is_tensor(y):
            y = th.as_tensor(y, dtype=x.dtype, device=x.device)
        return th.maximum(x, y)

    @staticmethod
    def where(cond, a, b):
        import torch as th
        if not is_tensor(a):
            a = th.as_tensor(a, device=cond.device)
        if not is_tensor(b):
            b = th.as_tensor(b, dtype=a.dtype, device=cond.device)
        return th.where(cond, a, b)

    @staticmethod
    def atleast_1d(x):
        return x.reshape(1) if x.ndim == 0 else x

    @staticmethod
    def concatenate(seq, axis=-1):
        import torch as th
        return th.cat(list(seq), dim=axis)

    @staticmethod
    def zeros(shape, like):
        import torch as th
        return th.zeros(shape, dtype=like.dtype, device=like.device)

    @staticmethod
    def sort_descending(x):
        import torch as th
        return th.sort(x, dim=-1, descending=True).values

    @staticmethod
    def cumsum(x):
        import torch as th
        return th.cumsum(x, dim=-1)

    @staticmethod
    def arange(start, stop, like):
        import torch as th
        return th.arange(start, stop, device=like.device)

    @staticmethod
    def amax(x):
        return x.max(dim=-1).values

    @staticmethod
    def take_along_axis(a, idx):
        import torch as th
        return th.take_along_dim(a, idx, dim=-1)

    @staticmethod
    def sum_keepdims(x):
        return x.sum(dim=-1, keepdim=True)


def ops(*xs):
    """Backend namespace chosen by the first torch tensor among `xs`; numpy otherwise."""
    return _TorchOps if any(is_tensor(x) for x in xs) else _NumpyOps
