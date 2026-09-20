"""Text-adventure agent loop for ALFWorld, recording canonical trajectories.

Second benchmark, deliberately un-alike AppWorld: ALFWorld is a household
TextWorld game, not a code-execution/API sandbox -- the action space is a
short imperative command picked from a per-turn admissible-commands list
("go to fridge 1", "take apple 1 from countertop 1"), not free-form Python.
This module emits a trajectory in the SAME canonical shape
`alloc_writer_harness`/`router_bank_builder` already consume for AppWorld:

    {"source_task_id", "domain", "task": {"id", "instruction"}, "success",
     "reward", "termination_reason", "evaluation",
     "steps": [{"index", "role", "content"}]}

Role mapping: `user` is the room/task intro and every subsequent environment
observation, `assistant` is the command the model wrote. There is no `tool`
role here (unlike AppWorld's code-output split) because ALFWorld's env.step
returns one text observation per action, indistinguishable in kind from the
initial room description -- both are "what the environment just told you".

Train/eval split (ALFWorld's own, not invented here): `train_eval` is one of
"train" / "eval_in_distribution" / "eval_out_of_distribution", mapping to the
`json_2.1.1/{train,valid_seen,valid_unseen}` directories set in the config's
`dataset.{data_path,eval_id_data_path,eval_ood_data_path}`.
`valid_unseen` ("eval_out_of_distribution") is the one whose room LAYOUTS
never appear in train, not just unseen object/task combinations in an
already-seen layout (that's `valid_seen`) -- the strict analogue of
AppWorld's held-out dev split, and the one `run_alfworld_rollout.py` defaults
`--split` to for anything claiming to be a real evaluation number.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .model_client import ModelClient

TASK_TYPES = (
    "pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
)

SPLIT_TO_TRAIN_EVAL = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}

AGENT_SYSTEM = """You are an autonomous assistant completing a household task in a text-adventure environment (ALFWorld).

Every turn you are shown the current room/object description or the result of your last action, and a list of admissible commands -- the ONLY commands the environment will accept right now. Reply with EXACTLY ONE command from that list, verbatim, and nothing else: no explanation, no code fence, just the command text on its own.

Typical command shapes you will see in the admissible list: "go to <receptacle>", "open <receptacle>", "close <receptacle>", "take <object> from <receptacle>", "put <object> in/on <receptacle>", "use <object>", "clean <object> with <receptacle>", "heat <object> with <receptacle>", "cool <object> with <receptacle>", "examine <object>", "look", "inventory".

You cannot see the whole house at once -- only what's in the room you last looked at or moved to. Objects are usually inside/on a receptacle (cabinet, drawer, fridge, microwave, etc.) that must be opened (or is already open) before you can take from it. Read the task instruction carefully: it tells you the target object and where it ultimately belongs. Do not guess an object's exact name; use exactly the identifiers shown in the admissible commands (e.g. "apple 1", not "the apple" or "an apple").

The episode ends automatically when the task is completed or the step limit is reached -- there is no explicit "finish" command; the last correct placement action ends it."""


def extract_command(content: str, admissible_commands: list[str]) -> str | None:
    """Pull one admissible command out of a model reply.

    Tries an exact (case-insensitive, whitespace-trimmed) match against the
    admissible list first -- the system prompt asks for the command verbatim
    and nothing else, so this is the common case. Falls back to the first
    non-empty line, still validated against the admissible set, in case the
    model wrapped the command in a sentence or a fenced block despite being
    told not to. Returns None (not a guess) if nothing in the reply matches
    an actually-admissible command -- an ungrounded free-form action would
    silently no-op or error in a way that makes the transcript unreadable.
    """
    candidates = [content.strip()]
    candidates.extend(line.strip().strip("`").strip() for line in content.splitlines() if line.strip())
    lowered = {c.lower(): c for c in admissible_commands}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def list_available_tasks(split_dir: Path) -> dict[str, Path]:
    """task_id -> game.tw-pddl path, applying the SAME filters
    `alfworld.agents.environment.alfred_tw_env.AlfredTWEnv.collect_game_files`
    does (solvable, known task_type, no movable/Sliced variants) -- so a
    task_id resolved here is guaranteed loadable, and the train/eval task
    lists this produces match what the env itself would have used.

    task_id is the path relative to `split_dir`, task-folder/trial-folder --
    stable, human-readable, and directly re-derivable from `extra.gamefile`
    for debugging.
    """
    tasks: dict[str, Path] = {}
    for task_folder in sorted(split_dir.iterdir()):
        if not task_folder.is_dir():
            continue
        if "movable" in task_folder.name or "Sliced" in task_folder.name:
            continue
        for trial_folder in sorted(task_folder.iterdir()):
            traj_path = trial_folder / "traj_data.json"
            game_path = trial_folder / "game.tw-pddl"
            if not traj_path.exists() or not game_path.exists():
                continue
            traj_data = json.loads(traj_path.read_text())
            if traj_data.get("task_type") not in TASK_TYPES:
                continue
            game_data = json.loads(game_path.read_text())
            if not game_data.get("solvable"):
                continue
            task_id = f"{task_folder.name}/{trial_folder.name}"
            tasks[task_id] = game_path
    return tasks


def load_task_env(config: dict[str, Any], train_eval: str, game_file: Path, seed: int):
    """One game, one env -- `AlfredTWEnv.__init__` always walks and filters
    the WHOLE split first (`collect_game_files`, a few seconds even for one
    task), then this narrows `game_files` down to the single target before
    `init_env` actually registers/loads it. Not optimized for launching many
    single-task subprocesses back-to-back; matches AppWorld's per-task
    subprocess model (`run_appworld_rollout.py`) rather than ALFWorld's own
    batched-env-per-process style, for consistency with how this repo drives
    every other rollout.
    """
    from alfworld.agents.environment import get_environment

    os.environ.setdefault("ALFWORLD_DATA", str(Path(config["dataset"]["data_path"]).parents[1]))
    wrapper = get_environment(config["env"]["type"])(config, train_eval=train_eval)
    wrapper.game_files = [str(game_file)]
    wrapper.num_games = 1
    env = wrapper.init_env(batch_size=1)
    return env


def run_task(
    env: Any,
    task_id: str,
    client: ModelClient,
    *,
    memory_block: str = "",
    max_steps: int = 40,
    max_output_chars: int = 3_000,
) -> dict[str, Any]:
    """Drive one ALFWorld episode and return its canonical trajectory.

    `memory_block`, when non-empty, is appended to the very first user
    message only -- same convention as `appworld_agent.build_initial_user_message`
    -- so a retrieved bank entry reads as context attached to the task, not
    as an environment observation the agent might mistake for something the
    room just told it.
    """
    obs, infos = env.reset()
    intro = str(obs[0])
    instruction_match = re.search(r"Your task is to: (.+)", intro)
    instruction = instruction_match.group(1).strip() if instruction_match else intro

    admissible = list(infos["admissible_commands"][0])
    user_message = intro + "\n\nAdmissible commands: " + ", ".join(admissible)
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

    for _ in range(max_steps):
        reply = client.chat_messages(messages)
        for key in usage_totals:
            value = (reply.usage or {}).get(key)
            if isinstance(value, int):
                usage_totals[key] += value
        steps.append({"index": len(steps), "role": "assistant", "content": reply.content})
        messages.append({"role": "assistant", "content": reply.content})

        command = extract_command(reply.content, admissible)
        if command is None:
            ungrounded_turns += 1
            if ungrounded_turns >= 3:
                termination = "ungrounded_action"
                break
            nudge = (
                "That was not one of the admissible commands. Reply with exactly one command "
                "copied verbatim from the admissible list, nothing else. Admissible commands: "
                + ", ".join(admissible)
            )
            steps.append({"index": len(steps), "role": "user", "content": nudge})
            messages.append({"role": "user", "content": nudge})
            continue
        ungrounded_turns = 0

        obs, scores, dones, infos = env.step([command])
        observation = _truncate(str(obs[0]), max_output_chars)
        admissible = list(infos["admissible_commands"][0])
        feedback = observation + "\n\nAdmissible commands: " + ", ".join(admissible)
        steps.append({"index": len(steps), "role": "user", "content": feedback})
        messages.append({"role": "user", "content": feedback})

        if bool(infos["won"][0]):
            termination = "solved"
            success = True
            break
        if bool(dones[0]):
            termination = "lost"
            break

    return {
        "source_task_id": f"alfworld.{task_id}",
        "domain": "alfworld",
        "task": {"id": task_id, "instruction": instruction},
        "success": success,
        "reward": 1.0 if success else 0.0,
        "termination_reason": termination,
        "evaluation": {"won": success},
        "steps": steps,
        "usage": usage_totals,
    }


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...[{len(text) - limit} characters omitted]...\n" + text[-half:]
