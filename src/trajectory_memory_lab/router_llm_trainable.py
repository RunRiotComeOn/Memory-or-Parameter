"""GRPO-trainable LLM router: the same prompted judgment as
`router_llm_policy.decide_route`, but as a differentiable 4-way categorical
over LoRA-adapted logits instead of an untrained, temperature-0 API call.

Why this is NOT the usual "sample with vLLM, recompute the logprob locally,
hot-swap the adapter back" RLHF pipeline (which this repo's vLLM 0.17.1 does
support -- `POST /v1/load_lora_adapter` with `load_inplace: true` behind
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`, plus chat-completions `logprobs`):

The router's action is ONE categorical draw over four routes, not a free-form
generation. Force the assistant turn to begin with `{"route": "` and the very
next token already identifies the route -- `memory`, `sft`, `both`, `neither`
have DISTINCT first tokens under the Qwen tokenizer (`memory` and `both` are
single tokens; `sft` -> `s`+`ft` and `neither` -> `ne`+`ither`, but `s` and
`ne` are still unambiguous among these four). So `log_softmax` over exactly
those four logits at one position is the whole policy: differentiable, with an
exact entropy, from a single forward pass. `_assert_route_tokens` verifies
that distinctness at construction time rather than trusting it.

That collapses the entire sample/recompute/sync problem:

- No vLLM for the router. 80 decisions/batch at ~0.2s each is nothing beside
  the ~70min-per-candidate AppWorld self-eval that produces the reward.
- Sampling and the gradient use the SAME weights, so the recomputed logprob
  is not merely "close" to the sampled one, it is the same number up to
  float nondeterminism -- `verify_recompute_matches_sample` asserts this, and
  it is a far stronger invariant than any cross-engine comparison could be.
- Nothing to sync after `optimizer.step()`, so no hot-swap, no server
  restart, no window in which the sampler and the learner disagree.

The cost, stated plainly: the router cannot reason before deciding. That
costs nothing *today* -- `router_llm_policy.ROUTER_LLM_SYSTEM` asks for
`{"route": ..., "rationale": ...}` in that order, so the rationale is already
generated AFTER the route and never conditions it. A reason-then-decide
variant would need the vLLM path back, and no measurement yet says
chain-of-thought helps this particular 4-way judgment.

The system prompt and payload are imported unchanged from
`router_llm_policy`, so an untrained forward pass through this class and
`decide_route` are looking at byte-identical text -- which is what makes
`run_router_llm_probe.py`'s untrained route distribution a true step-0
baseline for a trained run, rather than a differently-prompted cousin.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from .router_llm_policy import ROUTER_LLM_SYSTEM, ROUTES, build_router_payload

# The assistant turn is forced to start here, so the next token is the route.
# Keep this byte-identical to the opening of the JSON object ROUTER_LLM_SYSTEM
# asks for -- the model is being held to its own stated output format, not
# steered into a different one.
ROUTE_PREFIX = '{"route": "'


@dataclass
class RouterLLMConfig:
    model_path: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0  # deterministic scoring; see sample()/logprob()
    target_modules: str | list[str] = "all-linear"
    # Refuses rather than truncates (see build_prompt_ids). Raised from 8192
    # to the measured ceiling of what the 35B MoE router can BACKWARD through
    # on two 49GB cards with gradient checkpointing on: 16,384 tokens peaks at
    # 42.8GiB of 47.4GiB usable, and 24,576 OOMs. The payload itself is kept
    # well under this by router_llm_policy.ROUTER_MAX_STEP_CHARS, so reaching
    # this guard means something upstream grew unexpectedly -- which is
    # exactly when it should fire instead of silently dropping content.
    max_prompt_tokens: int = 16384
    enable_thinking: bool = False
    # Divides the four route logits before the softmax. This is part of the
    # POLICY, not a decoding detail: sampling, the logprob in the gradient and
    # the entropy bonus all use the same tempered distribution, so what gets
    # trained is exactly what gets sampled.
    #
    # Left at 1.0 because it measured fine there, not by assumption. A
    # pretrained LM's route prior is confident enough that this needed
    # checking: with all four routes live, Qwen3-8B sits at mean H=0.154 nats
    # and p(argmax)=0.931, which puts the probability that all K=8 candidates
    # in a batch make identical decisions -- every advantage exactly 0, the
    # whole batch a 6-hour no-op -- at 0.003. Acceptable. Raise this if the
    # route distribution collapses mid-run; it is the one lever that widens
    # exploration without touching the weights.
    policy_temperature: float = 1.0
    # Shards the base model across several GPUs when set (`"auto"`, or an
    # explicit {submodule: device} map). `device` is then ignored for
    # placement: `input_device` and `device` are read back off the loaded
    # model instead, because with a shard map the embeddings and the lm_head
    # are on DIFFERENT cards and a tensor built for one is invalid on the
    # other. Needed for Qwen3.5-35B-A3B, whose language model is ~68GB in
    # bf16 and does not fit on one 49GB card; left None for a model that does
    # fit, which keeps the single-card path byte-identical to before.
    device_map: str | dict[str, Any] | None = None
    # Recompute activations in the backward pass instead of storing them.
    # Not an optimization here -- it is the difference between running and
    # not. Measured on the 35B MoE across GPU 6,7: without it a 1,024-token
    # backward already peaks at 40.6GiB (8.3GiB of activations for ONE
    # decision) and 4,096 tokens OOMs, which is far below the 11.5k-token
    # median router prompt. With it, the same 1,024-token backward peaks at
    # 33.0GiB and 16,384 tokens fits. The cost is wall-clock in the update
    # (19.2s at 8k, 64.4s at 16k), which is still small against the ~70min
    # per-candidate AppWorld self-eval that produces the reward.
    gradient_checkpointing: bool = False


class TrainableLLMRouter:
    """4-way route policy over a LoRA-adapted causal LM.

    Deliberately mirrors `router_policy.RouterPolicy`'s surface
    (`sample_action` / `greedy_action` / `action_distribution`) so
    `router_bank_builder.run_router_chain` and the GRPO loop treat the two
    interchangeably -- the thing being updated changes, the loop does not.
    """

    def __init__(self, config: RouterLLMConfig) -> None:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        base = AutoModelForCausalLM.from_pretrained(
            config.model_path, dtype=config.dtype, device_map=config.device_map,
        )
        if config.device_map is None:
            base = base.to(self.device)
        base.config.use_cache = False
        # get_peft_model freezes every base weight; only the adapter is
        # returned by parameters() with requires_grad=True, so the optimizer
        # in the training loop cannot touch the base model even by accident.
        lora = LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            target_modules=config.target_modules, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora)
        self.model.eval()  # no dropout/no batchnorm drift; LoRA still trains
        if config.gradient_checkpointing:
            # Must be enabled on the base model and paired with
            # enable_input_require_grads: every base weight is frozen, so
            # without a grad-requiring input the checkpointed segment has no
            # graph to rebuild and the LoRA gradient comes back empty.
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        # Where input_ids must be built and where the logits come back. On the
        # single-card path both are `config.device` and nothing changes; under
        # a device_map they are genuinely different cards, and putting the
        # route-token index tensor on the wrong one is a hard error rather
        # than a silently wrong policy -- read them off the model rather than
        # assuming either.
        self.input_device = self.model.get_input_embeddings().weight.device
        output_embeddings = self.model.get_output_embeddings()
        self.device = (
            output_embeddings.weight.device if output_embeddings is not None else self.input_device
        )
        self.route_token_ids = self._assert_route_tokens()

    # ---- prompt construction -------------------------------------------------

    def _assert_route_tokens(self) -> list[int]:
        """The four routes must be distinguishable by their FIRST token after
        ROUTE_PREFIX, or the single-position categorical is not well defined.

        Checked here, at construction, against the real tokenizer rather than
        assumed from the Qwen vocabulary -- swapping in a model whose
        tokenizer merges (say) `both` and `b` would otherwise produce a
        silently wrong policy that still trains and still logs plausible
        numbers.
        """
        prefix_ids = self.tokenizer.encode(ROUTE_PREFIX, add_special_tokens=False)
        first_ids: list[int] = []
        for route in ROUTES:
            ids = self.tokenizer.encode(ROUTE_PREFIX + route, add_special_tokens=False)
            if ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError(
                    f"tokenizer does not preserve {ROUTE_PREFIX!r} when {route!r} follows it "
                    f"({ids[:len(prefix_ids)]} != {prefix_ids}); the forced-prefix scheme "
                    "cannot address a single route position for this tokenizer"
                )
            first_ids.append(ids[len(prefix_ids)])
        if len(set(first_ids)) != len(ROUTES):
            raise ValueError(
                f"routes {ROUTES} do not have distinct first tokens after {ROUTE_PREFIX!r}: "
                f"{dict(zip(ROUTES, first_ids))}"
            )
        return first_ids

    def build_prompt_ids(
        self,
        trajectory: dict[str, Any],
        active_memory_count: int,
        recent_changes_text: str,
        draft_memory: dict[str, Any] | None,
        draft_sft_plan: dict[str, Any] | None,
    ) -> list[int]:
        """Same system prompt and same payload JSON as
        `router_llm_policy.decide_route`, rendered through the model's own
        chat template, then forced into the assistant turn up to ROUTE_PREFIX.
        """
        import json

        payload = build_router_payload(
            trajectory, active_memory_count, recent_changes_text, draft_memory, draft_sft_plan,
        )
        messages = [
            {"role": "system", "content": ROUTER_LLM_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )
        # Tokenize prompt and forced prefix as ONE string: tokenizing them
        # separately and concatenating can merge differently at the seam, and
        # the route position must line up with what _assert_route_tokens
        # measured.
        ids = self.tokenizer.encode(rendered + ROUTE_PREFIX, add_special_tokens=False)
        if len(ids) > self.config.max_prompt_tokens:
            raise ValueError(
                f"router prompt is {len(ids)} tokens, over max_prompt_tokens="
                f"{self.config.max_prompt_tokens}; truncating would silently drop the "
                "drafted content the router is supposed to be judging"
            )
        return ids

    # ---- the policy ----------------------------------------------------------

    def route_logits(self, prompt_ids: list[int]) -> torch.Tensor:
        """The 4 route logits at the forced-prefix position. Differentiable.

        One forward pass; the last position's logits are the distribution over
        the token that would follow ROUTE_PREFIX, from which only the four
        route-identifying entries are kept. Renormalizing over just those four
        (rather than the full vocabulary) is what makes this exactly the
        4-way `Categorical` the linear router used -- the model cannot spend
        probability mass on a fifth, unparseable "route".
        """
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.input_device)
        # Only the last position is ever read, so ask for only that one: the
        # full [1, seq, vocab] tensor is ~300MB in bf16 at a 1k-token prompt
        # and every byte of it but the last row is discarded. `logits_to_keep`
        # is honored by recent transformers causal LMs; older ones ignore the
        # kwarg or reject it, hence the fallback rather than a version pin.
        try:
            out = self.model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = self.model(input_ids=input_ids, use_cache=False)
        # float32 for the softmax: bf16 log_softmax over 4 logits loses enough
        # precision to matter when advantages are already O(0.05).
        last = out.logits[0, -1, :].float()
        selected = last[torch.tensor(self.route_token_ids, device=last.device)]
        return selected / self.config.policy_temperature

    def action_distribution(self, prompt_ids: list[int]) -> Categorical:
        return Categorical(logits=self.route_logits(prompt_ids))

    @torch.no_grad()
    def sample_action(self, prompt_ids: list[int]) -> tuple[str, int, torch.Tensor, torch.Tensor]:
        """Stochastic draw, NO graph retained.

        Returns (route, index, probs, entropy). The graph is deliberately
        dropped: retaining 80 full forward graphs of an 8B model to build one
        summed loss would not fit, so the training loop re-runs each forward
        with grad during the update instead (see `logprob_and_entropy`). The
        weights do not change between the two, so this costs exactness
        nothing -- `verify_recompute_matches_sample` checks precisely that.
        """
        dist = Categorical(logits=self.route_logits(prompt_ids))
        index = int(dist.sample().item())
        return ROUTES[index], index, dist.probs.detach().cpu(), dist.entropy().detach().cpu()

    @torch.no_grad()
    def greedy_action(self, prompt_ids: list[int]) -> tuple[str, int, torch.Tensor, torch.Tensor]:
        """Argmax route -- what validation/deployment runs, no exploration."""
        dist = Categorical(logits=self.route_logits(prompt_ids))
        index = int(torch.argmax(dist.logits).item())
        return ROUTES[index], index, dist.probs.detach().cpu(), dist.entropy().detach().cpu()

    def logprob_and_entropy(self, prompt_ids: list[int], index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable recompute of a previously-sampled decision.

        This is the `logprob` that goes into the GRPO loss -- the exact
        analogue of `router_policy.sample_action`'s `Categorical.log_prob`,
        just reached through a full LLM forward pass instead of one
        `nn.Linear` call. Entropy comes from the same distribution so the v4
        entropy bonus carries over unchanged.
        """
        with self._checkpointing():
            dist = Categorical(logits=self.route_logits(prompt_ids))
        return dist.log_prob(torch.tensor(index, device=dist.logits.device)), dist.entropy()

    @contextmanager
    def _checkpointing(self):
        """Run the with-grad recompute in train() so checkpointing engages.

        transformers only takes the checkpointed path under
        `if self.gradient_checkpointing and self.training`, so eval() silently
        disables it -- the memory probe that first tried this reported
        `is_gradient_checkpointing: True` and an unchanged 40.6GiB peak.

        Switching modes is safe here, and measured rather than assumed: this
        model has `attention_dropout=0.0` and the adapter `lora_dropout=0.0`,
        and the four route logits come back BIT-IDENTICAL in the two modes
        (max |diff| = 0.0). So train() buys memory without moving the policy,
        which is what keeps `verify_recompute_matches_sample` -- sampling in
        eval() under no_grad, recompute here in train() -- a real assertion
        about the same distribution rather than a comparison of two.
        """
        if not self.config.gradient_checkpointing:
            yield
            return
        was_training = self.model.training
        self.model.train()
        try:
            yield
        finally:
            self.model.train(was_training)

    # ---- parameters / checkpoints -------------------------------------------

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def save_adapter(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        return path

    def load_adapter(self, path: Path) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = load_file(str(Path(path) / "adapter_model.safetensors"))
        set_peft_model_state_dict(self.model, state)


@dataclass
class SampledDecision:
    """One routed decision, recorded during sampling for replay at update time.

    Deliberately holds NO tensors with a graph: `prompt_ids` plus `index` is
    everything needed to reconstruct the logprob exactly, and storing 80 live
    forward graphs of an 8B model per batch would not fit in memory. `probs`
    and `entropy` are detached CPU copies kept only for the collapse
    diagnostics the v4 work added.
    """

    prompt_ids: list[int]
    index: int
    route: str
    task_id: str
    group: str
    probs: torch.Tensor
    entropy: torch.Tensor


def accumulate_policy_gradient(
    router: TrainableLLMRouter,
    decisions: list[SampledDecision],
    advantages: list[float],
    entropy_coef: float,
    max_grad_norm: float | None = 1.0,
) -> dict[str, float]:
    """The GRPO update, in the exact shape `train_router_selfreward.py` uses:

        pg_term = sum over every decision of -(advantage_k * logprob)
        loss    = pg_term - entropy_coef * entropy_sum

    `advantages` is per-decision (the candidate's advantage, repeated for each
    decision it made), so this stays a plain sum over all K x batch_size
    decisions and the entropy term keeps sharing its scale -- both grow with
    K x batch_size, which is what made `entropy_coef=0.01` calibrated in v4.

    Backward is called per decision and the grads ACCUMULATE, rather than
    building one summed loss and calling backward once. That is
    mathematically identical (d/dw of a sum is the sum of the d/dw) but needs
    only one forward graph resident at a time instead of all 80 -- the whole
    reason sampling drops its graph in the first place.

    `max_grad_norm` clips the accumulated gradient before the caller steps.
    It is on by default because the measured failure mode here is a step that
    is too LARGE, not too small: at lr=1e-4 a single update moved a route's
    probability from 0.00000 to 0.949, and the same rate drove the entropy
    bonus into a fully collapsed distribution instead of a uniform one
    (smoke_router_llm_policy_gradient.py). With only ~9 updates per iteration
    there is no opportunity to recover from one bad step, and DESIGN.md
    section 12 already established that a collapsed router has to be
    reinitialized rather than trained back.

    The caller owns `optimizer.zero_grad()` before and `optimizer.step()`
    after; this function only fills `.grad`.
    """
    if len(decisions) != len(advantages):
        raise ValueError(f"{len(decisions)} decisions but {len(advantages)} advantages")

    pg_total, entropy_total = 0.0, 0.0
    for decision, advantage in zip(decisions, advantages):
        logprob, entropy = router.logprob_and_entropy(decision.prompt_ids, decision.index)
        term = -(advantage * logprob) - entropy_coef * entropy
        term.backward()
        pg_total += float(-(advantage * logprob.detach()))
        entropy_total += float(entropy.detach())

    grad_norm = None
    if max_grad_norm is not None:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            router.trainable_parameters(), max_grad_norm,
        ))

    count = len(decisions)
    return {
        "loss": pg_total - entropy_coef * entropy_total,
        "pg_term": pg_total,
        "entropy_sum": entropy_total,
        "mean_entropy": entropy_total / count if count else 0.0,
        "decision_count": count,
        "grad_norm": grad_norm,
    }


def verify_recompute_matches_sample(
    router: TrainableLLMRouter, prompt_ids: list[int], index: int, tolerance: float = 1e-3,
) -> tuple[float, float, float]:
    """Assert the no-grad sampling pass and the with-grad update pass agree.

    The invariant that replaces cross-engine logprob comparison: because both
    passes run the same weights in the same process, any real disagreement
    means a genuine bug (wrong prompt replayed, wrong route index stored,
    adapter mutated between phases), not kernel or batching differences. A
    vLLM-vs-local comparison could never be this sharp -- it has to tolerate
    the very slop that would hide those bugs.

    Returns (sampled_logprob, recomputed_logprob, abs_difference).
    """
    with torch.no_grad():
        logits = router.route_logits(prompt_ids)
        sampled = float(Categorical(logits=logits).log_prob(
            torch.tensor(index, device=logits.device)
        ))
    recomputed_t, _ = router.logprob_and_entropy(prompt_ids, index)
    recomputed = float(recomputed_t)
    difference = abs(sampled - recomputed)
    if difference > tolerance:
        raise AssertionError(
            f"sampling and recompute disagree by {difference:.3e} "
            f"(sampled={sampled:.6f}, recomputed={recomputed:.6f}) -- the update is not "
            "scoring the decision that was actually sampled"
        )
    return sampled, recomputed, difference
