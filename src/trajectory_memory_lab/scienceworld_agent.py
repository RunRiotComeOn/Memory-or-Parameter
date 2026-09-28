"""Text-adventure agent loop for ScienceWorld, recording canonical trajectories.

Third benchmark: an elementary-science-curriculum text environment (30 task
types -- boil water, determine electrical conductivity, mendelian genetics,
measure an inclined plane's angle, ...), chosen alongside ALFWorld as a
second non-AppWorld domain -- household chores vs. science experiments are
different enough that a memory bank built on one should not trivially
transfer to the other, which is useful for testing whether this project's
router/memory machinery is domain-general or ALFWorld-specific.

Action space differs from ALFWorld in a way that changes how the interface
must be built. `info["valid"]` (returned by both `env.reset()` and
`env.step()`) is the full verb x object x object cartesian product for the
CURRENT room graph -- 417 items / ~12,000 characters at reset on a fresh
"boil" task, REGARDLESS of task complexity, because 324 of those 417 are
"connect X to Y" / "disconnect X" room-graph-connectivity actions that most
tasks never use (verified: the "boil" task's own gold action sequence uses
zero of them). But "connect ... to ..." genuinely IS load-bearing for the
electrical-circuit task family (verified: "power-component"'s gold path is
built almost entirely from "connect battery anode to ..." commands), so it
cannot simply be filtered out either.

`info["valid"]` is alphabetically sorted, so a plain char-budget truncation
would keep only "connect ..."/"disconnect ..." entries (all starting with
'c'/'d') and cut every "look"/"open"/"pick up"/"focus on"/... action a
non-circuit task actually needs. This module instead partitions the list --
every non-connect/disconnect action first (verified: only ~93 items / ~1,660
characters across every task type tried, well within budget), then as many
connect/disconnect actions as still fit, with a count of how many were
dropped. A circuit task with more live connect/disconnect options than fit
is told how many were omitted, exactly like ALFWorld's elision markers.

Deliberately no `simplificationStr` (e.g. "teleportAction", "openDoors"):
those exist in ScienceWorld specifically to reduce task difficulty for
weaker agents, but AppWorld and ALFWorld are both run at full difficulty in
this project, and changing that per-benchmark would confound any cross
-benchmark comparison of "how hard is this domain" with "how much was it
simplified."

Canonical trajectory shape matches `alfworld_agent`/`appworld_agent`:
    {"source_task_id", "domain": "scienceworld", "task": {"id", "instruction"},
     "success", "reward", "termination_reason", "evaluation",
     "steps": [{"index", "role", "content"}]}
"""

from __future__ import annotations

from typing import Any

from .model_client import ModelClient

TRAIN_EVAL_METHOD = {
    "train": "get_variations_train",
    "dev": "get_variations_dev",
    "test": "get_variations_test",
}

# Above this many characters of admissible-action text, connect/disconnect
# entries start getting dropped (in alphabetical order, arbitrarily -- there
# is no cheap way to know which specific wire/terminal pairing a given
# circuit task needs without already knowing the solution) rather than
# growing the prompt further. Two separate things were wrong here at first,
# and the fix for each made the other visible:
#
# 1. The cap was applied ONLY to connect/disconnect entries; every other
#    action went in unbounded, on a measurement ("~93 items / ~1,660
#    characters across every task type tried") that does not survive an
#    object-rich room. A real `grow-plant` run produced a single
#    66,547-character turn and 3 of 12 probe tasks died of `context_overflow`
#    by step 17 -- failures of the harness, not of the agent.
#
# 2. Capping the total instead then cost real successes, because the list is
#    the agent's only source of EXACT object names. With the list sampled,
#    `find-plant` went from `focus on adult pea plant` (solved) to
#    `focus on pea plant` -- a plausible name that the environment does not
#    accept -- and `focus on` is the one verb that ends the episode when it
#    is wrong. Showing fewer actions is not a neutral trade here.
#
# What actually decouples the two: only the CURRENT turn needs the action
# list. `_strip_actions` removes it from older turns as the conversation
# grows, so the budget below bounds one turn rather than all thirty. At
# ~3 characters/token, 30 turns of observation (<=1,500) plus one live list
# of 12,000 is ~19k tokens -- well inside the 65,536 window, while leaving
# the list complete in every room measured so far.
MAX_ACTION_LIST_CHARS = 12_000

AGENT_SYSTEM = """You are an autonomous assistant completing a household science-experiment task in a text-adventure environment (ScienceWorld), built around the elementary science curriculum (states of matter, electrical circuits, plant/animal life cycles, mixtures, forces, and similar).

Every turn you are shown the current room description or the result of your last action, plus a list of VALID ACTIONS -- the commands the environment will accept right now. Reply with EXACTLY ONE command copied verbatim from that list, and nothing else: no explanation, no code fence, just the command text on its own.

Copy the command exactly as written, including the full object name. Object names are precise and near-misses are rejected: if the list offers `focus on adult pea plant`, then `focus on pea plant` is not the same command and will not work. Only the latest turn carries a full action list; earlier turns in this conversation have had theirs removed, so work from the list on the most recent turn. If that list says some actions were not shown, `look around` or move closer to what you want rather than guessing at a command you have not seen.

You cannot see the whole house at once -- only what's in the room you last looked at or moved to ("go to <room>" moves you there); more objects and valid actions appear as you explore. Typical task-solving pattern: find the target substance/object, `focus on` it if the task instruction implies it, take it to the relevant device or location (stove, freezer, sink, an electrical circuit, ...), and perform the action that satisfies the goal; `open` a container/door before interacting with what's inside it; `look around` if unsure what is in the current room, `inventory` to see what you are carrying. For electrical-circuit tasks, "connect <part> to <part>" commands wire components together -- read the valid-actions list carefully for the exact terminal/component names.

The environment scores your progress incrementally as you complete sub-goals; the episode ends when the task is fully solved (score reaches 100) or the step limit is reached. There is no explicit "finish" command.

Warning: `focus on <object>` is a commitment, not a look. Using it on the wrong object ends the task immediately with a large score penalty -- it does not just fail quietly and let you try again. Explore first (`look around`, `look at <object>`, `look in <container>`) to confirm you have identified the exact substance or object the task instruction refers to, and only then `focus on` it."""


def list_available_tasks(env: Any, split: str) -> dict[str, tuple[str, int]]:
    """task_id ("<task_name>::<variation_id>") -> (task_name, variation_id),
    for every task type's variations in the given split ("train"/"dev"/"test").
    `env` must be a fresh `ScienceWorldEnv()` -- `.load()` populates that
    task's own variation list, so this loads each task name in turn purely to
    enumerate it (task names are fixed; only variation lists differ per
    task), leaving `env` loaded on the last one afterward.
    """
    method_name = TRAIN_EVAL_METHOD[split]
    tasks: dict[str, tuple[str, int]] = {}
    for task_name in env.get_task_names():
        env.load(task_name, 0, simplificationStr="")
        variations = getattr(env, method_name)()
        for variation_id in variations:
            tasks[f"{task_name}::{variation_id}"] = (task_name, variation_id)
    return tasks


def parse_task_id(task_id: str) -> tuple[str, int]:
    task_name, variation_id = task_id.rsplit("::", 1)
    return task_name, int(variation_id)


def _verb_of(action: str) -> str:
    """Group key for round-robin display: the action's leading verb phrase.

    Two-word openers are kept whole (`look at` vs `look in` vs `look around`,
    `pick up`, `move`, ...) because they are genuinely different affordances
    to the agent; anything else groups by its first word.
    """
    parts = action.split()
    if not parts:
        return action
    if len(parts) >= 2 and parts[0] in {"look", "pick", "put", "turn"}:
        return f"{parts[0]} {parts[1]}"
    return parts[0]


def _round_robin(actions: list[str], budget: int) -> tuple[list[str], int]:
    """Fill `budget` characters taking one action from each verb group in turn.

    A plain prefix cut cannot be used here: `info["valid"]` is alphabetically
    sorted, so cutting at a character count keeps whole verb classes and drops
    others entirely (the original code's own docstring flagged this for
    `connect`/`disconnect`, but the same applies to every verb once the list
    is long). Round-robin guarantees the agent always sees that `look at`,
    `move`, `pour`, `activate`, ... exist, even when it can only be shown a
    fraction of their objects.
    """
    groups: dict[str, list[str]] = {}
    for action in actions:
        groups.setdefault(_verb_of(action), []).append(action)
    order = sorted(groups)
    shown: list[str] = []
    used = 0
    index = 0
    while True:
        progressed = False
        for verb in order:
            bucket = groups[verb]
            if index >= len(bucket):
                continue
            action = bucket[index]
            cost = len(action) + 2
            if used + cost > budget:
                return shown, len(actions) - len(shown)
            shown.append(action)
            used += cost
            progressed = True
        if not progressed:
            return shown, len(actions) - len(shown)
        index += 1


def _ordered_actions(valid: list[str], max_chars: int) -> tuple[list[str], int]:
    """Non-connect/disconnect actions first, then connect/disconnect with
    whatever budget is left, both filled round-robin across verb groups.
    Returns (shown_actions, dropped_count).

    connect/disconnect stay deprioritized for the reason the module docstring
    gives: they are 324 of 417 entries at reset on a task whose gold path uses
    none of them, but they are load-bearing for the circuit task family, so
    they are pushed back rather than filtered out.
    """
    primary = [a for a in valid if not (a.startswith("connect ") or a.startswith("disconnect "))]
    secondary = [a for a in valid if a.startswith("connect ") or a.startswith("disconnect ")]
    shown, _ = _round_robin(primary, max_chars)
    used = sum(len(a) + 2 for a in shown)
    extra, _ = _round_robin(secondary, max(0, max_chars - used))
    shown += extra
    return shown, len(valid) - len(shown)


def extract_command(content: str, valid: list[str]) -> str | None:
    """Pull one valid action out of a model reply -- exact match against the
    FULL valid set (not just what was shown), so a command the model
    correctly inferred but that got dropped from display (see
    `_ordered_actions`) still succeeds; only a genuinely invalid guess
    triggers the retry-with-list path."""
    candidates = [content.strip()]
    candidates.extend(line.strip().strip("`").strip() for line in content.splitlines() if line.strip())
    lowered = {c.lower(): c for c in valid}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def _closest_hint(content: str, valid: list[str], limit: int = 6) -> str:
    """Name the valid actions closest to what the model just tried.

    A bare "that was not valid" leaves the model with nothing to correct
    toward, and the measured failure is not that it cannot see the list but
    that it repeats the same near-miss until the episode is killed
    (`pick up glass cup`, `go to outside`, three times each). Surfacing the
    real strings turns that into a one-token fix when the guess was a naming
    near-miss, which is the common case in this environment.
    """
    import difflib

    guess = content.strip().splitlines()[0].strip().strip("`").lower() if content.strip() else ""
    if not guess:
        return ""
    matches = difflib.get_close_matches(guess, [a.lower() for a in valid], n=limit, cutoff=0.5)
    if not matches:
        head = guess.split()[0]
        matches = [a for a in valid if a.lower().startswith(head)][:limit]
    if not matches:
        return ""
    return " -- closest valid actions: " + ", ".join(f"`{m}`" for m in matches)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...[{len(text) - limit} characters omitted]...\n" + text[-half:]


ACTIONS_MARKER = "\n\nValid actions: "
STRIPPED_MARKER = "\n\n[valid actions for that turn omitted -- see the list on the latest turn]"


def _strip_actions(text: str) -> str:
    """Drop the valid-actions block from a past turn's text.

    The list is only actionable on the turn it was produced, but it is by far
    the largest part of every turn, so keeping thirty copies of it is what
    pushed real episodes past the context window. Older turns keep their
    observation, which is the part that still carries information.
    """
    index = text.find(ACTIONS_MARKER)
    return text if index < 0 else text[:index] + STRIPPED_MARKER


def _feedback_text(observation: str, valid: list[str]) -> str:
    shown, dropped = _ordered_actions(valid, MAX_ACTION_LIST_CHARS)
    text = observation + "\n\nValid actions: " + ", ".join(shown)
    if dropped:
        text += (
            f" ... [{dropped} more valid actions not shown; the list above samples every available "
            "verb. `look around`, or move closer to what you want, to bring the rest into range.]"
        )
    return text


ACTION_LIST_TURNS_KEPT = 2


def _strip_history(messages: list[dict[str, Any]], steps: list[dict[str, Any]]) -> None:
    """Keep the action list on the most recent `ACTION_LIST_TURNS_KEPT` user
    turns and remove it from every older one.

    Applied to `steps` as well as `messages`, deliberately: the trajectory is
    what SFT examples are later built from, so it has to record the text the
    model was actually shown. If the record kept the full lists while the
    served conversation dropped them, every training example would carry a
    prompt shape that never occurs at inference.

    Two rather than one: the caller strips BEFORE appending the new turn, so
    keeping one would leave the agent with only the list it is about to be
    given and no trace of the previous one -- and the exact object strings in
    the previous list are what a follow-up command usually has to reuse.
    """
    for container in (messages, steps):
        seen = 0
        for entry in reversed(container):
            if entry["role"] != "user":
                continue
            seen += 1
            if seen >= ACTION_LIST_TURNS_KEPT:
                entry["content"] = _strip_actions(entry["content"])


def run_task(
    env: Any,
    task_id: str,
    client: ModelClient,
    *,
    memory_block: str = "",
    max_steps: int = 30,
    max_output_chars: int = 1_500,
) -> dict[str, Any]:
    """Drive one ScienceWorld episode and return its canonical trajectory.

    `env` must already have `.load(task_name, variation_id, simplificationStr="")`
    called -- callers resolve `task_id` via `parse_task_id` themselves so the
    same loaded env can be reused across variations of the same task_name
    without reconstructing the JVM-backed environment each time.
    """
    obs, info = env.reset()
    instruction = env.get_task_description()

    user_message = f"{obs}\n\nTask: {instruction}\n\n" + _feedback_text("", info["valid"])
    if memory_block:
        user_message += "\n" + memory_block
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": user_message},
    ]
    steps: list[dict[str, Any]] = [{"index": 0, "role": "user", "content": user_message}]
    termination = "max_steps"
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
    ungrounded_turns = 0
    success = False
    last_score = 0

    for _ in range(max_steps):
        try:
            reply = client.chat_messages(messages)
        except Exception as exc:  # noqa: BLE001
            # A task needing many exploration turns (e.g. "find-animal",
            # which has no shortcut but searching room by room) can outgrow
            # the model's context window before max_steps -- the growing
            # per-turn valid-actions text is repeated on every turn with no
            # history trimming (verified: a real run hit this at ~30 turns).
            # This is a real, distinct failure mode from every other
            # termination reason here and must not crash the whole rollout
            # subprocess -- it is recorded exactly like a graceful stop.
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

        command = extract_command(reply.content, info["valid"])
        if command is None:
            ungrounded_turns += 1
            if ungrounded_turns >= 3:
                termination = "ungrounded_action"
                break
            nudge = (
                f"`{reply.content.strip()[:120]}` is not a valid action here"
                + _closest_hint(reply.content, info["valid"])
                + ". Reply with exactly one command copied verbatim from the valid-actions "
                "list, nothing else.\n" + _feedback_text("", info["valid"])
            )
            _strip_history(messages, steps)
            steps.append({"index": len(steps), "role": "user", "content": nudge})
            messages.append({"role": "user", "content": nudge})
            continue
        ungrounded_turns = 0

        observation, _reward, is_completed, info = env.step(command)
        last_score = info.get("score", last_score)
        feedback = _feedback_text(_truncate(str(observation), max_output_chars), info["valid"])
        _strip_history(messages, steps)
        steps.append({"index": len(steps), "role": "user", "content": feedback})
        messages.append({"role": "user", "content": feedback})

        if is_completed:
            success = isinstance(last_score, (int, float)) and last_score >= 100
            termination = "solved" if success else "task_ended"
            break

    return {
        "source_task_id": f"scienceworld.{task_id}",
        "domain": "scienceworld",
        "task": {"id": task_id, "instruction": instruction},
        "success": success,
        "reward": (float(last_score) / 100.0) if isinstance(last_score, (int, float)) else 0.0,
        "termination_reason": termination,
        "evaluation": {"score": last_score},
        "steps": steps,
        "usage": usage_totals,
    }
