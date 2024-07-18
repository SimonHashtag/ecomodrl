# Single-file implementation of single-task TD-MPC2, see https://github.com/nicklashansen/tdmpc2

# Packages
from copy import deepcopy
import os
import random
from dataclasses import dataclass
import tyro

import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble

import numpy as np

# Args
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

# Make environment


# Architecture (World model)
class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		modules = nn.ModuleList(modules)
		fn, params, _ = combine_state_for_ensemble(modules)
		self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness='different', **kwargs)
		self.params = nn.ParameterList([nn.Parameter(p) for p in params])
		self._repr = str(modules)

	def forward(self, *args, **kwargs):
		return self.vmap([p for p in self.params], (), *args, **kwargs)

class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""
	
	def __init__(self, simnorm_dim):
		super().__init__()
		self.dim = simnorm_dim
	
	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=nn.Mish(inplace=True), **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
    """MLP with LayerNorm, Mish activations, and optionally dropout"""
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)
    

class WorldModel(nn.Module):
    """
    World Model architecture including state encoder, forward dynamics model, 
    reward model, policy prior, and Q-Function
    """
    def __init__(
        self, 
        env, 
        latent_dim, 
        enc_dim, 
        num_enc_layers, 
        mlp_dim, 
        simnorm_dim, 
        num_bins,
        dropout,
        num_q, 
        log_std_min, 
        log_std_max
        ):
        super().__init__()
        self._encoder = mlp(np.array(env.single_observation_space.shape).prod(),  max(num_enc_layers-1, 1)*[enc_dim], latent_dim, act=SimNorm(simnorm_dim))
        self._dynamics = mlp(latent_dim + np.prod(env.single_action_space.shape), 2*[mlp_dim], latent_dim, act=SimNorm(simnorm_dim))
        self._rewards = mlp(latent_dim + np.prod(env.single_action_space.shape), 2*[mlp_dim], max(num_bins, 1))  
        self._pi = mlp(latent_dim, 2*[mlp_dim], 2*np.prod(env.single_action_space.shape)) # Output: mean and var of multivariate Gaussian to predict rewards
        self._Qs = Ensemble([mlp(latent_dim + np.prod(env.single_action_space.shape), 2*[mlp_dim], max(num_bins, 1), dropout=dropout) for _ in range(num_q)])
        # weight initialization: self.apply(init.weight_init)
        # initialize some parameters to zero: init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(log_std_min)
        self.log_std_dif = torch.tensor(log_std_max) - self.log_std_min

## encoder

## dynamics

## reward

## policy prior

## state-action value (critic)


# Training

if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")