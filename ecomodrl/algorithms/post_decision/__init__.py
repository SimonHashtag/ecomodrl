"""Post-decision ("structure-exploiting") critics with quadrature targets.

Selected with `variant="pd"` (see ecomodrl/configs/variants.py). For DDPG, TD3
and SAC the critic knows the utility function and the deterministic part of the
budget identity and learns only the continuation value over the post-decision
state; its target integrates over the model's own shock nodes. PPO keeps the
standard pre-decision value net and only replaces the one-step bootstrap in its
GAE TD error by the same quadrature expectation. All economics come from the
EconomicModel (`critic.py`: adapter, critic and target; `algorithms.py`:
PDDDPG, PDTD3, PDSAC, PDPPO).
"""
from ecomodrl.algorithms.post_decision.algorithms import (
    PDDDPG,
    PDHybridPPOPolicy,
    PDPPO,
    PDPPOPolicy,
    PDSAC,
    PDSACPolicy,
    PDTD3,
    PDTD3Policy,
)
from ecomodrl.algorithms.post_decision.critic import (
    PostDecisionAdapter,
    PostDecisionCritic,
    make_post_decision_adapter,
    quadrature_target,
)

__all__ = [
    "PDDDPG",
    "PDHybridPPOPolicy",
    "PDPPO",
    "PDPPOPolicy",
    "PDSAC",
    "PDSACPolicy",
    "PDTD3",
    "PDTD3Policy",
    "PostDecisionAdapter",
    "PostDecisionCritic",
    "make_post_decision_adapter",
    "quadrature_target",
]
