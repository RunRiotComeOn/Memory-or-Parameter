"""Learnable route router: replaces the hand-written allocation rubrics.

See router_reward_v1/DESIGN.md sections 3 and 7. v1 scope, deliberately
narrow: the router is a single 4-way categorical over `route` only
(memory / sft / both / neither). It does not choose `memory_operation`
(refine / replace) -- whenever the sampled route requires memory, the
operation is always `add`. Choosing among existing bank entries to refine
is a variable-size action space; punting on it keeps v1 a fixed 4-way
softmax over a handful of linear features, trainable with GRPO's usual
one-gradient-step-per-group pattern.

Content generation (the actual memory text / sft_plan) still goes through
the existing LLM writer machinery in `writer_rubrics.py` -- a linear model
cannot generate free text. Only the discrete route choice is learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

ROUTES: tuple[str, ...] = ("memory", "sft", "both", "neither")

# Feature order (5 features -> 4 logits, ~24 params total):
#   0: base_agent success (0/1)
#   1: base_agent reward (float, AppWorld's raw scalar reward)
#   2: active_memory_count / 20  (bank size so far, soft-normalized)
#   3: position / task_ids_len   (fraction through the domain chain)
#   4: trajectories_remaining / task_ids_len
FEATURE_DIM = 5


def features_of(
    trajectory: dict[str, Any],
    bank: list[dict[str, Any]],
    position: int,
    task_ids_len: int,
) -> torch.Tensor:
    from .alloc_writer_harness import active_entries

    success = 1.0 if trajectory.get("success") else 0.0
    reward = float(trajectory.get("reward") or 0.0)
    active_count = len(active_entries(bank)) / 20.0
    frac_position = position / max(task_ids_len, 1)
    frac_remaining = (task_ids_len - position) / max(task_ids_len, 1)
    return torch.tensor(
        [success, reward, active_count, frac_position, frac_remaining],
        dtype=torch.float32,
    )


class RouterPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(FEATURE_DIM, len(ROUTES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def action_distribution(model: RouterPolicy, features: torch.Tensor) -> Categorical:
    """The full 4-way policy distribution at this state.

    Exposed separately from `sample_action` so callers can inspect the whole
    distribution -- specifically `.entropy()` for the entropy-regularization
    term and the collapse diagnostics. Built from logits (not probs) so the
    entropy/log_prob computations stay in log-space and numerically stable.
    """
    return Categorical(logits=model(features))


def greedy_action(model: RouterPolicy, features: torch.Tensor) -> str:
    """Argmax route -- what deployment/validation actually runs, no exploration."""
    with torch.no_grad():
        logits = model(features)
        index = int(torch.argmax(logits).item())
    return ROUTES[index]


def sample_action(
    model: RouterPolicy, features: torch.Tensor
) -> tuple[str, torch.Tensor, Categorical]:
    """Stochastic sample (not argmax) -- exploration is required for GRPO.

    Returns the sampled route, its logprob, and the full distribution it was
    drawn from. The distribution is returned (rather than just the entropy)
    because it carries the whole policy at this state: `.entropy()` feeds the
    entropy bonus in the training loss, `.probs` feeds the per-route
    diagnostics that tell us whether the route distribution is collapsing.
    Both stay attached to the graph, so the entropy term is differentiable.
    """
    dist = action_distribution(model, features)
    index = dist.sample()
    logprob = dist.log_prob(index)
    return ROUTES[int(index.item())], logprob, dist


@dataclass
class Checkpoint:
    iteration: int
    path: Path


def save_checkpoint(model: RouterPolicy, iteration: int, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"router_iter{iteration}.pt"
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model: RouterPolicy, path: Path) -> None:
    model.load_state_dict(torch.load(path, map_location="cpu"))
