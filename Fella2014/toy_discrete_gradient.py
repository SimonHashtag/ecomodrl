"""Toy experiment: can an actor that outputs a continuous number learn a discrete choice?

Companion script to RL_discrete.tex. A one-step problem (a contextual bandit)
keeps everything except the gradient question out of the way:

    context  s = (x, d):  x ~ U[0, 1] (where the household would like to be),
                          d ~ uniform level in {0, ..., 6} (where it is now)
    choice   k in {0, ..., 6} (the new level)
    reward   r(k | s) = -0.05 (k - 6x)^2 - PHI * 1{k != d}  (+ small noise)

The fixed cost PHI makes "stay at d" attractive, exactly like Fella's durable
adjustment cost. The actor outputs u in [-1, 1]; the environment executes the
bin k = floor((u + 1) / 2 * 7), i.e. 7 equal-width ordinal bins.

Methods (all train a deterministic tanh actor for the same number of steps):

  model-free ("mf"):   a neural critic Q(s, u) is fitted to the rewards of noisy
                       actions u = actor(s) + sigma * noise; the actor ascends Q.
                       This is what DDPG/TD3/SAC do in the model-free variant.
  post-decision ("pd"): the value of every level is known from the model
                       (here exactly: r(k | s)); only the action-to-level map is
                       a step function.
      hard:      Q(u) = r(bin(u))                          -> gradient exactly 0
      hat:       linear interpolation of r between the two neighbouring bins
      expected:  Q(u) = sum_k P_k(u) r(k), P_k = probability that
                 u + N(0, (1/7)^2) falls into bin k        -> used in the code

Output: accuracy (share of contexts where the chosen level is the best one) and
regret (average reward loss) for 4 seeds per method. Runs in ~3 minutes on a
laptop CPU. Usage: python Fella2014/toy_discrete_gradient.py
"""

import time

import torch as th
from torch import nn

th.set_num_threads(4)

K = 7                                   # number of levels / bins
EDGES = th.linspace(-1.0, 1.0, K + 1)   # bin edges on [-1, 1]
PHI = 0.1                               # fixed adjustment cost
NOISE = 0.05                            # reward noise (the critic sees noisy data)
STEPS, BATCH, SEEDS = 3000, 256, range(4)
NORMAL = th.distributions.Normal(0.0, 1.0)


def rewards(x, d):
    """r(k | x, d) for all levels k, shape (B, K)."""
    k = th.arange(K, dtype=th.float32)
    return -0.05 * (k[None, :] - 6 * x[:, None]) ** 2 - PHI * (k[None, :] != d[:, None]).float()


def bin_of(u):
    """Level executed for action u: equal-width bins on [-1, 1]."""
    return th.clamp(((u + 1) / 2 * K).floor().long(), 0, K - 1)


def context(x, d):
    return th.stack([x, d.float() / (K - 1)], dim=-1)


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_out))


def evaluate(actor):
    """Accuracy and regret of the deterministic actor on fresh contexts."""
    g = th.Generator().manual_seed(123)
    x, d = th.rand(20000, generator=g), th.randint(0, K, (20000,), generator=g)
    with th.no_grad():
        k = bin_of(th.tanh(actor(context(x, d))).squeeze(-1))
    r = rewards(x, d)
    accuracy = (k == r.argmax(1)).float().mean().item()
    regret = (r.max(1).values - r.gather(1, k[:, None]).squeeze(1)).mean().item()
    return accuracy, regret


def run_model_free(seed, sigma):
    """DDPG-style: learn a critic Q(s, u) from noisy actions, ascend it with the actor."""
    th.manual_seed(seed)
    actor, critic = mlp(2, 1), mlp(3, 1)
    opt_actor = th.optim.Adam(actor.parameters(), 1e-3)
    opt_critic = th.optim.Adam(critic.parameters(), 1e-3)
    for _ in range(STEPS):
        x, d = th.rand(BATCH), th.randint(0, K, (BATCH,))
        s = context(x, d)
        with th.no_grad():  # exploration: Gaussian noise on the actor's output
            u = th.clamp(th.tanh(actor(s)).squeeze(-1) + sigma * th.randn(BATCH), -1, 1)
        r = rewards(x, d).gather(1, bin_of(u)[:, None]).squeeze(1) + NOISE * th.randn(BATCH)
        critic_loss = ((critic(th.cat([s, u[:, None]], -1)).squeeze(-1) - r) ** 2).mean()
        opt_critic.zero_grad(); critic_loss.backward(); opt_critic.step()
        actor_loss = -critic(th.cat([s, th.tanh(actor(s))], -1)).mean()
        opt_actor.zero_grad(); actor_loss.backward(); opt_actor.step()
    return evaluate(actor)


def run_post_decision(seed, method, sigma=1.0 / K):
    """The level values are known; only the map from u to the level is a step function."""
    th.manual_seed(seed)
    actor = mlp(2, 1)
    opt_actor = th.optim.Adam(actor.parameters(), 1e-3)
    grad_norms = []
    for _ in range(STEPS):
        x, d = th.rand(BATCH), th.randint(0, K, (BATCH,))
        r = rewards(x, d)
        u = th.tanh(actor(context(x, d))).squeeze(-1)
        if method == "hard":       # value of the executed bin: piecewise constant in u
            q = r.gather(1, bin_of(u)[:, None]).squeeze(1)
        elif method == "hat":      # linear interpolation between neighbouring bin centres
            pos = (u + 1) / 2 * K - 0.5                 # bin centres sit at 0, 1, ..., 6
            k = th.clamp(pos.detach().floor().long(), 0, K - 2)
            w = pos - k.float()
            q = (1 - w) * r.gather(1, k[:, None]).squeeze(1) + w * r.gather(1, (k + 1)[:, None]).squeeze(1)
        else:                      # expectation over the bins under N(u, sigma^2)
            cdf = NORMAL.cdf((EDGES[None, 1:-1] - u[:, None]) / sigma)       # interior edges
            cdf = th.cat([th.zeros(BATCH, 1), cdf, th.ones(BATCH, 1)], dim=1)  # open outer bins
            q = ((cdf[:, 1:] - cdf[:, :-1]) * r).sum(1)
        loss = -q.mean()
        opt_actor.zero_grad()
        if loss.requires_grad:
            loss.backward()
        grad_norms.append(sum(float(p.grad.norm()) ** 2 for p in actor.parameters() if p.grad is not None) ** 0.5)
        opt_actor.step()
    return evaluate(actor), sum(grad_norms[:100]) / 100


def main():
    g = th.Generator().manual_seed(123)
    x, d = th.rand(20000, generator=g), th.randint(0, K, (20000,), generator=g)
    r = rewards(x, d)
    stay_acc = (d == r.argmax(1)).float().mean().item()
    stay_regret = (r.max(1).values - r.gather(1, d[:, None]).squeeze(1)).mean().item()
    print(f"baseline 'always stay':  accuracy {stay_acc:.3f}  regret {stay_regret:.4f}")
    print(f"baseline 'random level': accuracy {1 / K:.3f}")

    print("\nmodel-free (critic over the continuous action, exploration noise sigma):")
    for sigma in (0.02, 0.1, 0.3):
        t0 = time.time()
        res = [run_model_free(seed, sigma) for seed in SEEDS]
        acc = " ".join(f"{a:.3f}" for a, _ in res)
        reg = " ".join(f"{r_:.4f}" for _, r_ in res)
        print(f"  sigma={sigma:<4}  accuracy [{acc}]  regret [{reg}]  ({time.time() - t0:.0f} s)", flush=True)

    print("\npost-decision (values known per level; how to differentiate through the bins):")
    for method in ("hard", "hat", "expected"):
        res = [run_post_decision(seed, method) for seed in SEEDS]
        acc = " ".join(f"{a:.3f}" for (a, _), _ in res)
        reg = " ".join(f"{r_:.4f}" for (_, r_), _ in res)
        grad = sum(g_ for _, g_ in res) / len(res)
        print(f"  {method:<9} accuracy [{acc}]  regret [{reg}]  mean |grad| (first 100 steps) {grad:.4f}", flush=True)


if __name__ == "__main__":
    main()
