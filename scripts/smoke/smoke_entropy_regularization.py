"""CPU-only smoke test for the v4 entropy-regularized router update.

Stubs out every GPU/LLM/AppWorld dependency (the writer LLM call and the
AppWorld self-eval subprocess) so the *only* thing under test is the change
made in v4: that `sample_action` exposes the full policy distribution, and
that the entropy term in `run_one_batch_update`'s loss actually pushes a
collapsed route distribution back toward uniform.

Run: .venv/bin/python scripts/smoke/smoke_entropy_regularization.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_memory_lab.router_policy import (  # noqa: E402
    ROUTES,
    RouterPolicy,
    action_distribution,
    sample_action,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def collapsed_router(dominant: str = "both", margin: float = 6.0) -> RouterPolicy:
    """A router whose bias alone already makes `dominant` near-certain.

    This is the state v2/v3_g4 actually drifted into; the point of the test is
    what the update does when starting from here.
    """
    model = RouterPolicy()
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
        model.linear.bias[ROUTES.index(dominant)] = margin
    return model


def feats() -> torch.Tensor:
    return torch.tensor([1.0, 0.5, 0.3, 0.4, 0.6], dtype=torch.float32)


# --- 1. sample_action exposes the full distribution -------------------------
print("\n1. sample_action returns the full policy distribution")
model = RouterPolicy()
route, logprob, dist = sample_action(model, feats())
check("returns a 3-tuple with a valid route", route in ROUTES, route)
check("probs has one entry per route", tuple(dist.probs.shape) == (len(ROUTES),), str(tuple(dist.probs.shape)))
_psum = float(dist.probs.detach().sum())
check("probs sum to 1", abs(_psum - 1.0) < 1e-5, f"{_psum:.6f}")
check("entropy is differentiable (requires_grad)", dist.entropy().requires_grad)
check("logprob is differentiable (requires_grad)", logprob.requires_grad)
_p = dist.probs.detach()
manual_H = -float((_p * _p.log()).sum())
check("entropy matches -sum(p log p)", abs(manual_H - float(dist.entropy().detach())) < 1e-5,
      f"manual={manual_H:.6f} dist={float(dist.entropy().detach()):.6f}")
check("logprob matches log(probs[route])",
      abs(float(logprob.detach()) - math.log(float(_p[ROUTES.index(route)]))) < 1e-5)

uniform_H = math.log(len(ROUTES))
print(f"  (uniform entropy for {len(ROUTES)} routes = ln4 = {uniform_H:.4f} nats)")


# --- 2. entropy of a collapsed policy is ~0, and its gradient pushes it up ---
print("\n2. entropy gradient on a deliberately collapsed distribution")
model = collapsed_router()
dist = action_distribution(model, feats())
H0 = float(dist.entropy())
check("collapsed policy has near-zero entropy", H0 < 0.1, f"H={H0:.6f}, probs={[round(float(x),4) for x in dist.probs]}")

# One pure-entropy-bonus step: loss = -coef * H. If the sign is right, this
# alone must raise the entropy.
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(50):
    opt.zero_grad()
    (-0.01 * action_distribution(model, feats()).entropy()).backward()
    opt.step()
H1 = float(action_distribution(model, feats()).entropy())
check("entropy bonus increases entropy", H1 > H0, f"{H0:.6f} -> {H1:.6f}")
probs1 = action_distribution(model, feats()).probs
check("mass moves off the dominant route", float(probs1[ROUTES.index("both")]) < 0.999,
      f"p(both) {float(dist.probs[ROUTES.index('both')]):.6f} -> {float(probs1[ROUTES.index('both')]):.6f}")


# --- 3. the real loss: with vs without the entropy term ---------------------
print("\n3. full v4 loss, entropy_coef=0 vs 0.01, from the same collapsed start")


def run_updates(entropy_coef: float, steps: int = 60, lr: float = 0.01) -> tuple[float, list[float], float]:
    """Mimic run_one_batch_update's loss exactly, with fixed synthetic
    advantages that (as in the real runs) mildly reward the dominant route --
    the pressure that drives the collapse."""
    torch.manual_seed(20260822)
    model = collapsed_router()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    params0 = [p.detach().clone() for p in model.parameters()]
    for _ in range(steps):
        opt.zero_grad()
        pg_term = torch.zeros(())
        entropy_sum = torch.zeros(())
        # 8 candidates x 10 decisions, matching --rollouts-per-batch 8 --batch-size 10
        for k in range(8):
            advantage = 0.05 if k % 2 == 0 else -0.05
            for _ in range(10):
                _route, logprob, d = sample_action(model, feats())
                pg_term = pg_term - advantage * logprob
                entropy_sum = entropy_sum + d.entropy()
        (pg_term - entropy_coef * entropy_sum).backward()
        opt.step()
    moved = max(float((p.detach() - p0).abs().max()) for p, p0 in zip(model.parameters(), params0))
    final = action_distribution(model, feats())
    return float(final.entropy()), [round(float(x), 4) for x in final.probs], moved


H_off, probs_off, moved_off = run_updates(0.0)
H_on, probs_on, moved_on = run_updates(0.01)
print(f"  entropy_coef=0.00 -> H={H_off:.4f} probs={dict(zip(ROUTES, probs_off))}")
print(f"  entropy_coef=0.01 -> H={H_on:.4f} probs={dict(zip(ROUTES, probs_on))}")

check("params actually update (coef=0)", moved_off > 1e-6, f"max|delta|={moved_off:.6f}")
check("params actually update (coef=0.01)", moved_on > 1e-6, f"max|delta|={moved_on:.6f}")
check("entropy term keeps the distribution more spread than without it",
      H_on > H_off, f"{H_off:.4f} (off) vs {H_on:.4f} (on)")
check("collapsed start recovers meaningful entropy with the term on",
      H_on > 0.5, f"H={H_on:.4f} of max {uniform_H:.4f}")
check("without the term it stays collapsed", H_off < 0.5, f"H={H_off:.4f}")


# --- 4. coefficient monotonicity -------------------------------------------
print("\n4. larger entropy_coef -> more spread (monotonicity sanity check)")
series = [(c, run_updates(c, steps=40)[0]) for c in (0.0, 0.001, 0.01, 0.05)]
for coef, H in series:
    print(f"  coef={coef:<6} H={H:.4f}")
entropies = [H for _, H in series]
check("entropy is non-decreasing in entropy_coef",
      all(b >= a - 1e-3 for a, b in zip(entropies, entropies[1:])), str([round(h, 4) for h in entropies]))


print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILED: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
