"""Router backed by a cheap hosted Gemini Flash model.

Third router arm for the same ALFWorld ablation, alongside:

- `router_llm_policy`  -- the system default: the SAME local model that plays
  the task (Qwen3.5-35B-A3B) is prompted and its text parsed.
- `router_jev_policy`  -- TypeSafe's Jev answers a typed `Choice`.
- this module            -- a cheap external Gemini Flash model, prompted with
  the identical system prompt and payload.

## What this arm isolates

Against `router_llm_policy` the only thing that changes is WHICH model reads
the prompt: same system prompt, same payload, same four routes, same output
shape. So a difference here is attributable to model identity (and cost),
not to a different decision interface -- which is exactly what separates it
from the Jev arm, where the interface changed too.

`build_router_payload` and `ROUTER_LLM_SYSTEM` are imported and used
UNCHANGED, as in `router_jev_policy`. That includes the pre-existing quirk
that the framing says "AppWorld tasks" while the trajectories are
ALFWorld's: the v4 arm this is compared against ran it that way, and fixing
it for one arm only would confound the comparison.

## One deliberate deviation, recorded

v4 parses the model's free text with `ModelClient.json_chat`. Here the
output shape is enforced with `response_json_schema`, the same mechanism
`appworld_sft_writer` already uses for the teacher. That is a deviation, and
it is the right one: the jev run surfaced a failure class where a model
emitted syntactically invalid JSON (single quotes escaped as `\\'`) and the
task was lost. Schema-enforced output removes that failure mode rather than
re-importing it into a new arm. The consequence for interpretation is that
this arm cannot lose tasks to parse errors, so its usable-task count should
be compared against v4's with that in mind.

Unlike the teacher's helper in `appworld_sft_writer`, this passes an
explicit request timeout. The teacher's missing timeout is a recorded defect
(it produced repeated 14-15 minute silent stalls during the jev run); new
code should not reproduce it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .router_llm_policy import ROUTER_LLM_SYSTEM, ROUTES, build_router_payload

DEFAULT_GEMINI_ROUTER_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_API_KEY_FILE = Path("/nas04/yixuh/.config/continual-memory/gemini_api_key")

# Mirrors the `{"route": ..., "rationale": ...}` object ROUTER_LLM_SYSTEM asks
# for, with `route` constrained to the four legal values so an unlisted route
# cannot come back at all.
ROUTER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": list(ROUTES)},
        "rationale": {"type": "string"},
    },
    "required": ["route", "rationale"],
}


def _read_api_key(api_key_file: Path) -> str | None:
    env = os.environ.get("GEMINI_API_KEY")
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
    model: str = DEFAULT_GEMINI_ROUTER_MODEL,
    api_key_file: Path = DEFAULT_GEMINI_API_KEY_FILE,
    temperature: float = 0.0,
    timeout: float = 120.0,
    max_retries: int = 4,
    **_ignored: Any,
) -> tuple[str, str]:
    """Returns (route, rationale), falling back to ("neither", <reason>) on any
    failure -- the same posture as the other two router arms.

    `**_ignored` absorbs `base_url`/`seed`/`max_tokens`, which the caller
    passes for the local prompted router and which have no meaning for a
    hosted model. Dropping them lets `router_bank_builder` call any arm
    through one code path.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        return "neither", f"gemini_import_error:{exc!r}"

    key = _read_api_key(api_key_file)
    if not key:
        return "neither", f"gemini_no_api_key:{api_key_file}"

    payload = build_router_payload(
        trajectory, active_memory_count, recent_changes_text, draft_memory, draft_sft_plan
    )
    client = genai.Client(api_key=key)
    error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                config=types.GenerateContentConfig(
                    system_instruction=ROUTER_LLM_SYSTEM,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_json_schema=ROUTER_RESPONSE_SCHEMA,
                    http_options=types.HttpOptions(timeout=int(timeout * 1000)),
                ),
            )
            parsed = json.loads(response.text)
            route = str(parsed.get("route") or "").strip().lower()
            if route not in ROUTES:
                return "neither", f"invalid_route:{route!r}"
            rationale = json.dumps(
                {
                    "router": "gemini",
                    "model": model,
                    "choice": route,
                    "rationale": str(parsed.get("rationale") or "")[:2000],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return route, rationale
        except Exception as exc:  # noqa: BLE001
            error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(min(30, 2 ** attempt))
    return "neither", f"gemini_error:{error!r}"
