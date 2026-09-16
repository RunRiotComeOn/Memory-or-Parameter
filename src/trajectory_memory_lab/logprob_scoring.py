"""Teacher-forced logprob scoring: the dense proxy reward for memory writes.

See router_reward_v1/DESIGN.md section 2.2 for the full design rationale. In
one sentence: given a candidate memory M and an already-verified-correct probe
trajectory p in the same domain group, this computes

    reward(M, p) = sum_t logP(target_t | prompt_p (+) M) - sum_t logP(target_t | prompt_p)

where the target tokens are p's own recorded assistant turns (never
generated -- teacher forcing) and the two prompts are identical except for
whether M is injected. No sampling happens anywhere in this module, so the
only source of noise is the server's numerical determinism, which the
deterministic-serving investigation in appworld_experiment/noise_serial_v1
already established is negligible under this project's serve script.

Qwen's chat template special-cases the *last* message in a conversation when
it is an assistant turn (it inserts an empty <think></think> scaffold there,
since that is the turn a real generation call would be completing). Every
earlier message renders identically regardless of what follows it. That means
naive prefix-diffing across `apply_chat_template(messages[:k])` for
increasing k is wrong: the k-th message's rendering when it happens to be
last differs from its rendering mid-conversation. `turn_spans` works around
this by appending a synthetic trailing sentinel message to every prefix
except the true final one, so no real message is ever mistaken for the last
turn unless it actually is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI


_SENTINEL_CONTENT = "PROBE_SENTINEL"
_SENTINEL_MESSAGE = {"role": "user", "content": _SENTINEL_CONTENT}


@dataclass
class TurnSpan:
    index: int
    role: str
    start_char: int
    end_char: int


def turn_spans(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, list[TurnSpan]]:
    """Character spans of each message within the full chat-template rendering.

    Returns (full_text, spans) where full_text is exactly what a real
    inference call over `messages` would render (add_generation_prompt=False,
    enable_thinking=False, matching AppWorld's agent loop), and spans[i]
    covers messages[i]'s contribution to full_text.
    """
    n = len(messages)
    if n < 2:
        raise ValueError("need at least a system + one user message")

    def render(msgs: list[dict[str, str]]) -> str:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

    full_text = render(messages)

    def common_prefix(k: int) -> str:
        """Rendering of messages[:k] as it appears when something follows it.

        k must be < n so the sentinel trick applies; k == n uses full_text
        directly since the true final message legitimately is last.
        """
        if k == 0:
            return ""
        rendered = render(messages[:k] + [_SENTINEL_MESSAGE])
        marker = "<|im_start|>user\n" + _SENTINEL_CONTENT
        idx = rendered.index(marker)
        return rendered[:idx]

    boundaries = [0]
    for k in range(1, n):
        prefix = common_prefix(k)
        if not full_text.startswith(prefix):
            raise ValueError(
                f"turn {k} rendering is not a true prefix of the full conversation; "
                "the chat template's position-invariance assumption broke. "
                f"prefix_len={len(prefix)} full_text[:prefix_len]={full_text[:len(prefix)]!r} "
                f"prefix={prefix!r}"
            )
        boundaries.append(len(prefix))
    boundaries.append(len(full_text))

    spans = [
        TurnSpan(index=i, role=messages[i]["role"], start_char=boundaries[i], end_char=boundaries[i + 1])
        for i in range(n)
    ]
    return full_text, spans


def _token_index_range(offsets: list[tuple[int, int]], start_char: int, end_char: int) -> list[int]:
    return [
        i
        for i, (s, e) in enumerate(offsets)
        if s < end_char and e > start_char
    ]


@dataclass
class ScoreResult:
    total_logprob: float
    scored_token_count: int
    per_turn: list[dict[str, Any]]


def score_assistant_turns(
    client: OpenAI,
    tokenizer: Any,
    model: str,
    messages: list[dict[str, str]],
    *,
    roles_to_score: tuple[str, ...] = ("assistant",),
) -> ScoreResult:
    """Teacher-force `messages` and sum logP over turns whose role matches.

    Sends the exact token ids as the completions prompt (not a text string)
    so there is no risk of the server re-tokenizing differently than the
    offsets computed locally.
    """
    full_text, spans = turn_spans(tokenizer, messages)
    encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids: list[int] = encoding["input_ids"]
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]

    response = client.completions.create(
        model=model,
        prompt=input_ids,
        max_tokens=1,
        echo=True,
        logprobs=1,
        temperature=0.0,
    )
    token_logprobs = response.choices[0].logprobs.token_logprobs
    if len(token_logprobs) < len(input_ids):
        raise ValueError(
            f"server returned fewer logprobs ({len(token_logprobs)}) than prompt "
            f"tokens sent ({len(input_ids)}); echo may not be honoured"
        )

    total = 0.0
    count = 0
    per_turn: list[dict[str, Any]] = []
    for span in spans:
        if span.role not in roles_to_score:
            continue
        token_indices = _token_index_range(offsets, span.start_char, span.end_char)
        turn_logprob = 0.0
        turn_count = 0
        for i in token_indices:
            lp = token_logprobs[i]
            if lp is None:
                # Only index 0 of the whole sequence has no context; a
                # non-first assistant span should never hit this.
                continue
            turn_logprob += lp
            turn_count += 1
        total += turn_logprob
        count += turn_count
        per_turn.append(
            {
                "message_index": span.index,
                "start_char": span.start_char,
                "end_char": span.end_char,
                "token_count": turn_count,
                "logprob": turn_logprob,
            }
        )
    return ScoreResult(total_logprob=total, scored_token_count=count, per_turn=per_turn)
