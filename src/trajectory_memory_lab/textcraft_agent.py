"""Text agent for TextCraft (AgentGym), the seventh benchmark in the suite.

TextCraft is a Minecraft crafting game played entirely in text. Each episode
shows a list of crafting recipes and one goal item; the agent must work the
goal down to base materials, fetch those, and craft back up.

What TextCraft contributes that the other six do not:

- **Hierarchical decomposition.** The goal is the root of a recipe tree of
  depth 1-4, and intermediate items must be crafted before they can be used.
  AppWorld/tau2 chain API calls, ALFWorld/ScienceWorld follow procedures,
  WebShop matches attributes and BabyAI navigates -- none of them require
  building a subgoal tree and executing it bottom-up.
- **Distractor filtering.** Up to 10 recipes that are irrelevant to the goal
  are shuffled into the list every episode, so "which recipes matter" is
  part of the task rather than given.
- **A difficulty axis that is an explicit parameter.** `data_idx` indexes a
  DEPTH-SORTED list of goals, so a train/test split on depth is a genuine
  compositional-generalization test -- the only one in the suite besides
  ALFWorld's `valid_unseen`. See `split_task_ids`.

Three properties shape the harness:

1. **Actions are generated, not chosen from a list.** There is no
   `Available actions:` block as in ALFWorld or BabyAI; the agent composes
   one of three forms, and anything else comes back as
   "Could not execute ...". So `extract_command` validates against a grammar
   rather than matching a menu.
2. **`get` only works for non-craftable base items.** Asking for something
   that has a recipe fails with "Could not find X" -- the same message an
   invalid item name produces, which is a genuine ambiguity the agent has to
   resolve by reading the recipe list.
3. **Reward is binary**: 1 exactly when the goal item enters the inventory,
   and the episode terminates there. Unlike BabyAI's discounted score, so
   `mean_score` and `pass_rate` coincide; both are reported anyway to keep
   the cross-benchmark tables uniform.
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable

from .agentgym_client import AgentGymEnvClient
from .model_client import ModelClient

DEFAULT_ENV_URL = "http://127.0.0.1:36002"

# data_idx ranges per recipe-tree depth, from the depth-sorted goal list the
# server builds (`sorted(item_recipes_min_depth(1), key=depth)`). Verified
# against the live server at every boundary and recorded in
# `textcraft_experiment/task_manifest.json`; `assert_reproduces` is the guard
# if the ordering ever drifts.
DEPTH_RANGES = {1: (0, 131), 2: (132, 416), 3: (417, 532), 4: (533, 543)}
TOTAL_TASKS = 544

SPLIT_SEED = 20260928
POOL_SIZE = 200
TEST_SIZE = 80

AGENT_SYSTEM = """You are an autonomous assistant playing TextCraft, a Minecraft crafting game presented entirely in text.

You are given a list of crafting recipes and one goal item. Work out which recipes lead to the goal, gather the base materials, and craft your way up to it.

You may use EXACTLY these three actions, one per turn, and nothing else:

- `craft <output> using <inputs>` -- e.g. `craft 4 crimson planks using 1 crimson stems`. Copy the recipe verbatim from the list, including the output count and every input with its count.
- `get <count> <item>` -- e.g. `get 1 crimson stems`. This works ONLY for base materials that have NO recipe in the list. If an item can be crafted, `get` fails and you must craft it instead.
- `inventory` -- lists what you are carrying.

Reply with one action on its own line and nothing else: no explanation, no reasoning, no code fences.

Item names come from the recipe list; write them exactly as they appear there (`crimson planks`, not `minecraft:crimson_planks`).

Work bottom-up. A recipe can only be used once you already hold every input in at least the stated quantity, so craft or fetch the inputs first. Watch the counts: crafting `4 crimson planks` once gives you four, which may be enough for several later recipes, while a recipe needing `6 crimson planks` requires crafting that step twice.

Some recipes in the list are irrelevant to your goal -- ignore them.

The episode ends as soon as the goal item is in your inventory."""

_ACTION_PATTERNS = (
    re.compile(r"^craft\s+.+\s+using\s+.+$", re.IGNORECASE),
    re.compile(r"^get\s+\d+\s+.+$", re.IGNORECASE),
    re.compile(r"^inventory$", re.IGNORECASE),
)


class TextCraftEnvClient(AgentGymEnvClient):
    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        super().__init__(base_url, timeout, env_var="TEXTCRAFT_ENV_URL", default_url=DEFAULT_ENV_URL)


def depth_of(data_idx: int) -> int:
    for depth, (lo, hi) in DEPTH_RANGES.items():
        if lo <= data_idx <= hi:
            return depth
    raise ValueError(f"data_idx {data_idx} outside 0..{TOTAL_TASKS - 1}")


def parse_task_id(task_id: str) -> int:
    return int(task_id.rsplit("::", 1)[1])


def task_depth(task_id: str) -> int:
    return depth_of(parse_task_id(task_id))


def split_task_ids(split: str) -> list[str]:
    """Three disjoint splits, all deterministic under `SPLIT_SEED`.

    - `train`  : 200 goals sampled from depth 1-2, stratified to keep the
                 132:285 depth ratio. This is what every pool is built from.
    - `test`   : 80 more goals from depth 1-2, disjoint from train. The
                 same-distribution held-out set, directly comparable with the
                 other benchmarks' test splits.
    - `deep`   : ALL 127 goals of depth 3-4, which no pool ever sees. This is
                 the compositional-generalization line: the agent learned on
                 goals at most two crafting steps deep and is asked for ones
                 three or four deep, where intermediate items must themselves
                 be built from other intermediates.

    Splitting the shallow band rather than sampling across all four depths is
    what makes `deep` meaningful; a stratified split over everything would
    put depth-4 goals on both sides and measure nothing new.
    """
    shallow: dict[int, list[int]] = {}
    for depth in (1, 2):
        lo, hi = DEPTH_RANGES[depth]
        shallow[depth] = list(range(lo, hi + 1))

    rng = random.Random(SPLIT_SEED)
    total_shallow = sum(len(v) for v in shallow.values())
    picked: dict[int, list[int]] = {}
    for depth, ids in shallow.items():
        share = round((POOL_SIZE + TEST_SIZE) * len(ids) / total_shallow)
        picked[depth] = rng.sample(ids, share)

    train: list[int] = []
    test: list[int] = []
    for depth, ids in picked.items():
        cut = round(POOL_SIZE * len(ids) / (POOL_SIZE + TEST_SIZE))
        train.extend(ids[:cut])
        test.extend(ids[cut:])

    if split == "train":
        chosen = train
    elif split == "test":
        chosen = test
    elif split == "deep":
        chosen = [i for d in (3, 4) for i in range(DEPTH_RANGES[d][0], DEPTH_RANGES[d][1] + 1)]
    else:
        raise ValueError(f"unknown split {split!r}")
    return [f"textcraft::{i}" for i in sorted(chosen)]


def goal_of(observation: str) -> str:
    marker = observation.rfind("Goal:")
    if marker < 0:
        return ""
    return observation[marker + len("Goal:"):].strip().rstrip(".")


def recipes_of(observation: str) -> list[str]:
    body = observation.split("Goal:")[0]
    return [ln.strip() for ln in body.splitlines() if ln.strip().lower().startswith("craft ")]


def extract_command(content: str) -> str | None:
    """Pull one grammatically valid action out of a model reply.

    There is no menu to match against here, so validity is the grammar: the
    three forms the environment's own regexes accept. Anything else is
    rejected locally rather than spent as a turn, because the environment
    answers it with "Could not execute ..." and no correction.
    """
    candidates = [content.strip()]
    candidates.extend(ln.strip() for ln in content.splitlines() if ln.strip())
    for candidate in candidates:
        cleaned = candidate.strip().strip("`").strip()
        cleaned = re.sub(r"^(action|Action)\s*:\s*", "", cleaned).strip()
        if any(p.match(cleaned) for p in _ACTION_PATTERNS):
            return cleaned
    return None


def run_task(
    client_env: TextCraftEnvClient,
    task_id: str,
    client: ModelClient,
    *,
    memory_lookup: Callable[[str], tuple[str, list[dict[str, Any]]]] | None = None,
    memory_block: str = "",
    max_steps: int = 40,
) -> dict[str, Any]:
    """Drive one TextCraft episode and return its canonical trajectory.

    The goal is only known after reset, so retrieval is a callback, the same
    convention `babyai_agent` and `webshop_agent` use. `max_steps` counts
    agent turns; a depth-4 goal needs roughly a dozen even when played
    perfectly, so this is the binding budget rather than a formality.
    """
    data_idx = parse_task_id(task_id)
    first = client_env.reset(data_idx)
    observation = first.get("observation") or ""
    goal = goal_of(observation)

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
    reward = float(first.get("reward") or 0.0)
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

        command = extract_command(reply.content)
        if command is None:
            ungrounded_turns += 1
            if ungrounded_turns >= 3:
                termination = "ungrounded_action"
                break
            nudge = (
                f"`{reply.content.strip()[:120]}` is not a valid action. Reply with exactly one of:\n"
                "  craft <output> using <inputs>\n  get <count> <item>\n  inventory\n"
                "and nothing else."
            )
            steps.append({"index": len(steps), "role": "user", "content": nudge})
            messages.append({"role": "user", "content": nudge})
            continue
        ungrounded_turns = 0

        result = client_env.step(command)
        observation = result.get("observation") or ""
        reward = float(result.get("reward") or reward)
        done = bool(result.get("done"))
        steps.append({"index": len(steps), "role": "user", "content": observation})
        messages.append({"role": "user", "content": observation})
        if done:
            termination = "solved" if reward > 0 else "task_ended"
            break

    return {
        "source_task_id": f"textcraft.{task_id}",
        "domain": "textcraft",
        "task": {"id": task_id, "instruction": f"craft {goal}"},
        "success": bool(done and reward > 0),
        "reward": reward,
        "termination_reason": termination,
        "evaluation": {"score": reward, "depth": task_depth(task_id), "goal": goal},
        "steps": steps,
        "usage": usage_totals,
        "retrieved_memory": selection,
    }
