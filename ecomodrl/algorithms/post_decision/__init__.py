"""Post-decision ("structure-exploiting") critics with quadrature targets.

For DDPG, TD3 and SAC the critic knows the utility function and the
deterministic part of the budget identity and learns only the continuation
value over the post-decision state; its target integrates over the model's own
shock nodes. PPO keeps the standard pre-decision value net and only replaces
the one-step bootstrap in its GAE TD error by the same quadrature expectation.
See `economics.py` (torch model pieces), `critic.py` (critic + target) and
`algorithms.py` (PDDDPG, PDTD3, PDSAC, PDPPO).
"""
from ecomodrl.algorithms.post_decision.algorithms import (
    PDDDPG,
    PDPPO,
    PDPPOPolicy,
    PDSAC,
    PDSACPolicy,
    PDTD3,
    PDTD3Policy,
)
from ecomodrl.algorithms.post_decision.critic import PostDecisionCritic, quadrature_target
from ecomodrl.algorithms.post_decision.economics import PDEconomics, make_economics

__all__ = [
    "PDDDPG",
    "PDPPO",
    "PDPPOPolicy",
    "PDSAC",
    "PDSACPolicy",
    "PDTD3",
    "PDTD3Policy",
    "PostDecisionCritic",
    "quadrature_target",
    "PDEconomics",
    "make_economics",
]
