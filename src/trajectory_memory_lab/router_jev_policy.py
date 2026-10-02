"""Router backed by TypeSafe's Jev (a "System One" model) instead of a prompted LLM.

Drop-in alternative to `router_llm_policy.decide_route` with the same
signature and the same failure posture. What changes is only HOW the route
is produced:

- `router_llm_policy` sends a system prompt plus a JSON payload to the task
  agent's own model and parses `{"route": ..., "rationale": ...}` out of the
  generated text.
- Here the same payload becomes Jev's `state` and the four routes become the
  `criteria` of a single `Choice` question. Jev returns a typed answer -- the
  selected option plus a probability distribution over all four and a
  confidence score -- so there is no free text to parse and no way to emit an
  unlisted route.

## Why the two arms are comparable

`build_router_payload` is imported and used UNCHANGED, so both routers read
byte-identical content: same trajectory, same elision budget, same drafted
memory and sft plan. The four option descriptions below are likewise copied
verbatim out of `ROUTER_LLM_SYSTEM`. Anything else would make this a
comparison of prompts rather than of routers.

That verbatim copy includes one pre-existing quirk: the framing says
"AppWorld tasks" even when the trajectories are ALFWorld's. The ALFWorld v4
run recorded in `alfworld_summary.md` used it that way, so keeping it keeps
the jev arm aligned with the arm it is being compared against. Fixing it for
one side only would confound the result; fixing it for both is a separate
experiment.

## What Jev adds that the prompted router cannot

`confidence` and the full `probabilities` distribution are recorded on every
decision. The prompted router emits a single token with no calibration
signal, so "the router was unsure" has never been observable in this
pipeline. A flat distribution here means low confidence by construction.

## Limits that shape the implementation

- **32k tokens for state plus the longest question.** The existing
  `ROUTER_MAX_STEP_CHARS = 31_000` elision converts to ~11.7k tokens at the
  worst-case 2.646 chars/token measured across the 90 base_train_v2
  transcripts, so the budget already in place fits with room to spare. No
  second elision is applied here -- a different budget would break the
  byte-identical-payload property above.
- **40 requests/second.** The bank builder is sequential (one task at a
  time), so nothing throttles.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .router_llm_policy import ROUTER_LLM_SYSTEM, ROUTES, build_router_payload

DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_JEV_API_KEY_FILE = Path("/nas04/yixuh/.config/continual-memory/jev_api_key")


def _route_criteria() -> dict[str, str]:
    """The four option descriptions, parsed out of `ROUTER_LLM_SYSTEM` itself.

    Parsed rather than re-typed so the two routers cannot drift apart: an
    edit to the prompt's route definitions reaches this arm automatically,
    and a parse that stops matching raises here instead of silently sending
    Jev a stale or empty description.
    """
    criteria: dict[str, str] = {}
    for line in ROUTER_LLM_SYSTEM.splitlines():
        line = line.strip()
        if not line.startswith("- `"):
            continue
        name, _, rest = line[3:].partition("`:")
        if name in ROUTES and rest.strip():
            criteria[name] = rest.strip()
    missing = [r for r in ROUTES if r not in criteria]
    if missing:
        raise RuntimeError(
            f"could not parse route descriptions {missing} out of ROUTER_LLM_SYSTEM; "
            "the prompt's '- `route`: ...' format changed and the two router arms "
            "would no longer be reading the same definitions"
        )
    return criteria


# The framing sentences from ROUTER_LLM_SYSTEM minus the four bullet lines
# (they become `criteria`) and minus the JSON-output instruction (Jev returns
# a typed answer, so asking for JSON would be wrong).
def _instructions() -> str:
    keep: list[str] = []
    for line in ROUTER_LLM_SYSTEM.splitlines():
        stripped = line.strip()
        if stripped.startswith("- `") or stripped.startswith("Return exactly one JSON object"):
            continue
        if stripped.startswith('{"route"'):
            continue
        keep.append(line)
    return "\n".join(keep).strip()


def _read_api_key(api_key_file: Path) -> str | None:
    env = os.environ.get("TYPESAFE_API_KEY")
    if env:
        return env
    if api_key_file.exists():
        key = api_key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    return None


def decide_route(
    trajectory: dict[str, Any],
    active_memory_count: int,
    recent_changes_text: str,
    draft_memory: dict[str, Any] | None,
    draft_sft_plan: dict[str, Any] | None,
    *,
    model: str = DEFAULT_JEV_MODEL,
    api_key_file: Path = DEFAULT_JEV_API_KEY_FILE,
    timeout: float = 120.0,
    max_retries: int = 4,
    **_ignored: Any,
) -> tuple[str, str]:
    """Returns (route, rationale), falling back to ("neither", <reason>) on any
    failure -- the same "not fatal, just no artifact this task" posture every
    writer in this pipeline takes.

    `**_ignored` absorbs `base_url`/`seed`/`max_tokens`, which the caller
    passes positionally-by-keyword for the prompted router and which have no
    meaning here (Jev is a hosted typed-decision model, not a chat endpoint).
    Accepting and dropping them keeps `router_bank_builder` able to call
    either arm through one code path.

    The rationale is a compact JSON string carrying the typed answer --
    confidence and the full distribution -- because the record schema stores
    rationale as a string and this is information the prompted router never
    produced.
    """
    try:
        from typesafe_sdk import Choice, TypeSafeClient
        from typesafe_sdk._core.retry import RetryPolicy
    except ImportError as exc:
        return "neither", f"jev_import_error:{exc!r}"

    key = _read_api_key(api_key_file)
    if not key:
        return "neither", f"jev_no_api_key:{api_key_file}"

    payload = build_router_payload(
        trajectory, active_memory_count, recent_changes_text, draft_memory, draft_sft_plan
    )
    try:
        client = TypeSafeClient(api_key=key)
        response = client.system_one(
            state=payload,
            questions={
                "route": Choice(instructions=_instructions(), criteria=_route_criteria()),
            },
            model=model,
            retry=RetryPolicy(max_retries=max_retries),
            timeout=timeout,
        )
        answer = response.choices["route"]
        route = str(answer.choice or "").strip().lower()
        if route not in ROUTES:
            # Jev picks from `criteria`, so this should be unreachable; treat
            # it like the prompted router's invalid-route case rather than
            # trusting an option nothing downstream knows how to act on.
            return "neither", f"invalid_route:{route!r}"
        rationale = json.dumps(
            {
                "router": "jev",
                "model": getattr(response, "model", model),
                "choice": route,
                "confidence": getattr(answer, "confidence", None),
                "probabilities": getattr(answer, "probabilities", None),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return route, rationale
    except Exception as exc:  # noqa: BLE001
        return "neither", f"jev_error:{exc!r}"
