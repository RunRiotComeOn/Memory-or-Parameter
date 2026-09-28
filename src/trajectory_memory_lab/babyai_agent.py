"""Text agent for BabyAI (via AgentGym's HTTP environment server).

Fourth text-game benchmark alongside ALFWorld and ScienceWorld, and the
first one taken from AgentGym's suite -- which matters beyond BabyAI itself:
AgentGym serves fourteen environments behind one HTTP contract
(`/create`, `/reset`, `/step`, `/observation`, `/close`), so the client here
is written against that contract rather than against BabyAI, and the next
AgentGym environment should need a new writer prompt and little else.

What BabyAI contributes that the existing benchmarks do not:

- **Spatial navigation.** Observations are egocentric descriptions ("There
  is a grey key 1 1 steps in front of you and 1 steps to your left"), so the
  transferable knowledge is about layout and search order rather than about
  APIs (AppWorld/tau2), object affordances (ALFWorld), procedures
  (ScienceWorld) or product attributes (WebShop).
- **Forty task levels** (`BabyAI-GoToRedBall-v0`, `BabyAI-Open-v0`, ...)
  behind one action vocabulary, which gives a natural held-out-level split
  if cross-task-type transfer is ever measured here.

Two properties shape the harness:

1. **Actions are natural-language and enumerated every turn**, exactly like
   ALFWorld's admissible commands -- `"go to red ball 1"`, `"pickup grey
   key 2"`, `"turn left"`, `"move forward"`. The raw MiniGrid verbs
   (`left`/`forward`/...) are NOT accepted; the server rejects them with
   "The action is not recognized". So the agent copies from the list, and
   `extract_command` matches against the full list rather than guessing.
2. **Reward is continuous** (0..1, discounted by how many primitive steps a
   high-level action consumed), like WebShop's score and unlike ALFWorld's
   binary outcome. `success` is reserved for a full-credit episode; the raw
   value is kept in `reward` so a run can be read both ways.

A `data_idx` selects the task: the server maps it to
`all_levels[data_idx % 40 + 1]` with `seed = data_idx // 40`, i.e. forty
levels cycling as the index advances and a new layout seed every forty. That
is what `split_task_ids` slices into disjoint train/test ranges.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from .model_client import ModelClient

DEFAULT_ENV_URL = "http://127.0.0.1:36001"
NUM_LEVELS = 40

AGENT_SYSTEM = """You are an autonomous assistant completing a navigation and object-manipulation task in a gridworld (BabyAI), described to you entirely in text.

Every turn you are shown your goal, an egocentric description of what you can see ("There is a red ball 1 4 steps in front of you and 3 steps to your right"), and a list of VALID ACTIONS. Reply with EXACTLY ONE action copied verbatim from that list, and nothing else: no explanation, no reasoning, just the action text on its own.

Copy the action exactly as written, including the trailing index. Object names are precise: if the list offers `pickup grey key 2`, then `pickup grey key` and `pickup key 2` are different strings and will be rejected.

The list mixes low-level moves (`turn left`, `move forward`) with high-level ones (`go to red ball 1`, `pickup blue box 1`) that walk you there in one action. Prefer the high-level action that directly serves the goal when it is offered -- it is both faster and scored better, because the reward is discounted by how many primitive steps you consume. Use `turn left`/`turn right`/`move forward` to explore when the object you need is not yet visible, and `check ...` actions to inspect what you cannot identify.

The episode ends as soon as the goal is satisfied. There is no explicit finish action."""


class BabyAIEnvClient:
    """Thin client for AgentGym's environment HTTP contract.

    Deliberately not BabyAI-specific: the same five endpoints back every
    AgentGym environment, so the next one can reuse this class with a
    different base URL.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        # None means "take $BABYAI_ENV_URL, else the default port" -- the
        # rollout runner passes the CLI flag straight through, and that flag
        # defaults to None.
        import os

        resolved = base_url or os.environ.get("BABYAI_ENV_URL") or DEFAULT_ENV_URL
        self.base_url = resolved.rstrip("/")
        self.timeout = timeout
        self.env_id: int | None = None

    def _call(self, path: str, body: dict[str, Any] | None = None, method: str = "POST") -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body or {}).encode() if method == "POST" else None
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(f"env server error on {path}: {payload['error']}")
        return payload

    def create(self) -> int:
        self.env_id = int(self._call("/create")["id"])
        return self.env_id

    def reset(self, data_idx: int) -> dict[str, Any]:
        if self.env_id is None:
            self.create()
        return self._call("/reset", {"id": self.env_id, "data_idx": data_idx})

    def step(self, action: str) -> dict[str, Any]:
        return self._call("/step", {"id": self.env_id, "action": action})

    def close(self) -> None:
        if self.env_id is not None:
            try:
                self._call("/close", {"id": self.env_id})
            except Exception:  # noqa: BLE001 - closing is best effort
                pass
            self.env_id = None


def split_task_ids(split: str, train_seeds: int = 20, test_seeds: int = 2) -> list[str]:
    """Disjoint train/test task ids, as `babyai::<data_idx>`.

    Split by SEED, not by level: `data_idx // 40` is the layout seed and
    `data_idx % 40` the level, so slicing on seed keeps all forty levels
    present on both sides while guaranteeing no layout is shared. That makes
    this a same-distribution held-out set (like ScienceWorld's and WebShop's
    test splits), not a cross-task-type generalization test -- a held-out
    LEVEL split would be the latter, and is a separate experiment.

    AgentGym reports 810 train / 90 eval for BabyAI but ships no split file,
    so the ranges are defined here and recorded with each run.
    """
    if split == "train":
        lo, hi = 0, train_seeds * NUM_LEVELS
    elif split == "test":
        lo, hi = train_seeds * NUM_LEVELS, (train_seeds + test_seeds) * NUM_LEVELS
    else:
        raise ValueError(f"unknown split {split!r}")
    return [f"babyai::{i}" for i in range(lo, hi)]


def parse_task_id(task_id: str) -> int:
    return int(task_id.rsplit("::", 1)[1])


def level_of(task_id: str) -> int:
    """Which of the forty levels this task is, for stratified sampling."""
    return parse_task_id(task_id) % NUM_LEVELS + 1


def available_actions(observation: str) -> list[str]:
    """The quoted actions in the trailing `Available actions: [...]` block."""
    marker = observation.rfind("Available actions:")
    if marker < 0:
        return []
    return re.findall(r'"([^"]*)"', observation[marker:])


def extract_command(content: str, valid: list[str]) -> str | None:
    """Pull one valid action out of a model reply, matched case-insensitively
    against the FULL list the environment accepts."""
    candidates = [content.strip()]
    candidates.extend(line.strip().strip("`").strip() for line in content.splitlines() if line.strip())
    lowered = {action.lower(): action for action in valid}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def _closest_hint(content: str, valid: list[str], limit: int = 6) -> str:
    """Name the valid actions nearest to what the model just tried.

    Same reasoning as `scienceworld_agent._closest_hint`: a bare "not valid"
    leaves nothing to correct toward, and the observed failure is a model
    repeating one near-miss until the episode is killed.
    """
    import difflib

    guess = content.strip().splitlines()[0].strip().strip("`").lower() if content.strip() else ""
    if not guess:
        return ""
    matches = difflib.get_close_matches(guess, [a.lower() for a in valid], n=limit, cutoff=0.5)
    if not matches:
        head = guess.split()[0] if guess.split() else ""
        matches = [a for a in valid if a.lower().startswith(head)][:limit]
    return " -- closest valid actions: " + ", ".join(f"`{m}`" for m in matches) if matches else ""


def run_task(
    client_env: BabyAIEnvClient,
    task_id: str,
    client: ModelClient,
    *,
    memory_lookup: Callable[[str], tuple[str, list[dict[str, Any]]]] | None = None,
    memory_block: str = "",
    max_steps: int = 20,
    success_threshold: float = 0.0,
) -> dict[str, Any]:
    """Drive one BabyAI episode and return its canonical trajectory.

    The goal is only known after the env resets, so retrieval is a callback:
    `memory_lookup(goal) -> (block, selection_log)`, the same convention
    `webshop_agent.run_task` uses and for the same reason. A fixed
    `memory_block` (guided replay's plan) is used as given. Either way the
    block goes on the first user message only. The selection log comes back
    on the trajectory as `retrieved_memory`.

    `max_steps` counts AGENT turns, not primitive gridworld steps: a single
    `go to red ball 1` may consume many of the latter, and the server caps
    those itself (50 per episode by default).
    """
    data_idx = parse_task_id(task_id)
    first = client_env.reset(data_idx)
    observation = first.get("observation") or ""
    goal = observation.split("\n", 1)[0].removeprefix("Your goal: ").strip()

    selection: list[dict[str, Any]] = []
    if memory_lookup is not None:
        memory_block, selection = memory_lookup(goal)

    user_message = observation
    if memory_block:
        user_message += "\n" + memory_block
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": user_message},
    ]
    steps: list[dict[str, Any]] = [{"index": 0, "role": "user", "content": user_message}]
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
    termination = "max_steps"
    ungrounded_turns = 0
    score = float(first.get("score") or 0.0)
    done = bool(first.get("done"))

    for _ in range(max_steps):
        if done:
            break
        try:
            reply = client.chat_messages(messages)
        except Exception as exc:  # noqa: BLE001
            if "context length" in str(exc).lower() or "input_tokens" in str(exc).lower():
                termination = "context_overflow"
                break
            raise
        for key in usage_totals:
            value = (reply.usage or {}).get(key)
            if isinstance(value, int):
                usage_totals[key] += value
        steps.append({"index": len(steps), "role": "assistant", "content": reply.content})
        messages.append({"role": "assistant", "content": reply.content})

        valid = available_actions(observation)
        command = extract_command(reply.content, valid)
        if command is None:
            ungrounded_turns += 1
            if ungrounded_turns >= 3:
                termination = "ungrounded_action"
                break
            nudge = (
                f"`{reply.content.strip()[:120]}` is not a valid action here"
                + _closest_hint(reply.content, valid)
                + ". Reply with exactly one action copied verbatim from the list.\n"
                + observation
            )
            steps.append({"index": len(steps), "role": "user", "content": nudge})
            messages.append({"role": "user", "content": nudge})
            continue
        ungrounded_turns = 0

        result = client_env.step(command)
        observation = result.get("observation") or ""
        score = float(result.get("score") or score)
        done = bool(result.get("done"))
        steps.append({"index": len(steps), "role": "user", "content": observation})
        messages.append({"role": "user", "content": observation})
        if done:
            # BabyAI ends the episode either way; the score separates the
            # cases. Measured over solved episodes it lands in 0.83-0.99 and
            # NEVER reaches 1.0, because reward is discounted by how many
            # primitive steps the high-level action consumed -- while a
            # wrong object scores exactly 0. So success is "finished with
            # any credit", not "finished with full credit"; a 1.0 threshold
            # would report pass_rate 0 on every run and give the ablation
            # grid no resolution at all.
            termination = "solved" if score > success_threshold else "task_ended"
            break

    return {
        "source_task_id": f"babyai.{task_id}",
        "domain": "babyai",
        "task": {"id": task_id, "instruction": goal},
        "success": bool(done and score > success_threshold),
        "reward": score,
        "termination_reason": termination,
        "evaluation": {"score": score, "level": level_of(task_id)},
        "steps": steps,
        "usage": usage_totals,
        "retrieved_memory": selection,
    }
