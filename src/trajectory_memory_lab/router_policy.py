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

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

ROUTES: tuple[str, ...] = ("memory", "sft", "both", "neither")

# Text-content visibility (DESIGN.md section 13): the router previously saw
# only scalar proxies of "how much has already been written" (bank size,
# fraction through the chain). Position-in-chain features were dropped
# entirely -- `frac_remaining` is an exact linear complement of `frac_position`
# given a fixed 90-task domain, so it added zero expressive power to a linear
# model, and `frac_position` itself never had a demonstrated reason to matter
# more than the two features below. In its place the router now sees a hashed
# bag-of-words of two pieces of ACTUAL text: what changed in the bank recently,
# and what this task's own candidate write would say. Hashing (not an
# embedding model) keeps the whole policy a linear layer over a fixed-size
# vector -- no new dependency, still one GRPO-trainable nn.Linear.
TEXT_HASH_DIM = 16

# Feature order (3 + 2*TEXT_HASH_DIM -> 4 logits):
#   0: base_agent success (0/1)
#   1: base_agent reward (float, AppWorld's raw scalar reward)
#   2: active_memory_count / 20  (bank size so far, soft-normalized)
#   3..3+H-1:   hashed bag-of-words of entries added/changed in the last two
#               batches (router_bank_builder._recent_changes_text)
#   3+H..3+2H-1: hashed bag-of-words of THIS task's already-drafted candidate
#               memory + sft content (router_bank_builder._draft_content_text)
FEATURE_DIM = 3 + 2 * TEXT_HASH_DIM


def _hash_bag_of_words(text: str, dim: int) -> torch.Tensor:
    """Deterministic term-frequency vector over `dim` hashed buckets.

    Not learned, not an embedding model -- md5(token) % dim, counts normalized
    to sum to 1 so the vector reflects vocabulary *composition* rather than
    text length. Empty text is the zero vector (a real, distinct state from
    "wrote something forgettable"), which the linear layer can key off of.
    """
    from .memory_writer_harness import _tokens

    vec = torch.zeros(dim, dtype=torch.float32)
    tokens = _tokens(text) if text else []
    if not tokens:
        return vec
    for token in tokens:
        bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    return vec / vec.sum()


def features_of(
    trajectory: dict[str, Any],
    bank: list[dict[str, Any]],
    recent_changes_text: str,
    draft_text: str,
) -> torch.Tensor:
    from .alloc_writer_harness import active_entries

    success = 1.0 if trajectory.get("success") else 0.0
    reward = float(trajectory.get("reward") or 0.0)
    active_count = len(active_entries(bank)) / 20.0
    numeric = torch.tensor([success, reward, active_count], dtype=torch.float32)
    recent_feat = _hash_bag_of_words(recent_changes_text, TEXT_HASH_DIM)
    draft_feat = _hash_bag_of_words(draft_text, TEXT_HASH_DIM)
    return torch.cat([numeric, recent_feat, draft_feat])


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
