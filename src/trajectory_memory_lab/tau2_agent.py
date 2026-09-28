"""tau2-bench (airline / retail / telecom) as a benchmark of this project's
router/memory/sft pipeline, with Gemini playing the user.

tau2-bench (Barres et al., 2025) is customer service over tool APIs: the
agent follows a written domain policy, calls domain tools against a real
database, and talks to a simulated customer who has a private goal. Telecom
adds dual control -- the customer has tools of their own (toggling airplane
mode, re-seating a SIM) that the agent must talk them through.

Written against tau2's own layered runner (`tau2.runner`, v1.0.1): the
environment, the user simulator and the orchestrator are tau2's, unmodified;
only the agent is ours (`MemoryAgent`, a thin `LLMAgent` subclass), and it is
passed in directly rather than registered, so nothing in the vendored
checkout is patched.

Three LLMs are in play, deliberately different:
- the agent under study: the local Qwen, via vLLM's OpenAI endpoint with
  native tool calling (`--enable-auto-tool-choice --tool-call-parser qwen3_xml`);
- the user simulator: Gemini (`DEFAULT_USER_LLM`);
- the NL-assertion judge, which only retail uses (40 of 114 tasks carry
  natural-language assertions; airline scores DB + communicated info,
  telecom scores environment assertions -- both deterministic): also Gemini,
  because tau2's default judge is gpt-4.1.

Why gemini-2.5-flash with thinking disabled: the user simulator's
determinism bounds every comparison made on this benchmark. Measured on a
cancellation prompt, three identical temperature-0 calls each:
gemini-2.5-flash gave identical replies (0.5 s); gemini-2.5-pro gave three
different replies (5-12 s, hundreds of thinking tokens); gemini-3.8-flash
gave two different replies and then a 503 "high demand". `reasoning_effort:
"disable"` removes 2.5-flash's thinking tokens entirely and kept it
identical. A single prompt was not a proof for whole conversations, though:
run twice, 2 of 5 airline tasks diverged at the customer's first message --
hence `install_llm_cache`, which memoizes the Gemini side on disk.
(Gemini rejects `seed`; tau2 sets `litellm.drop_params = True` globally, so
the orchestrator's per-run seed silently reaches only the agent.)

Memory goes into the agent's system prompt, after the policy, and is
retrieved from the customer's FIRST message -- the first thing the agent
actually knows. The task's user scenario is the customer's private
information and is never used for retrieval.

Canonical trajectory: the usual shape, plus `sft_example` -- the agent's
own view of the conversation in OpenAI chat format with the tool schemas,
exactly what the agent sent to the model (tau2's `to_litellm_messages`), with
the base system prompt (no memory, no guidance). Flattening tool calls to
text, as the text benchmarks' `training_messages` does, would train on a
prompt shape the served model never sees: vLLM renders `tools` into the
system turn and tool calls as Qwen XML. `steps` is a rendering of the FULL
conversation (including the customer's own tool calls in telecom) for the
writers to read.

`evaluation` reports only whether each scoring component passed -- never
the expected actions, assertion texts or target DB state, which would hand
a writer the answer.

Task ids: `<domain>::<tau2 task id>`. Splits are tau2's own
(`split_tasks.json`): airline 30 train / 20 test, retail 74 / 40, telecom
74 / 40 (of its 114-task `base` set).

Must run under tau2's own venv (`third_party/tau2-bench/.venv`, Python 3.12).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable

TAU2_ROOT = Path(__file__).resolve().parents[2] / "third_party/tau2-bench"
DOMAINS = ("airline", "retail", "telecom")
GEMINI_API_KEY_FILE = Path("/nas04/yixuh/.config/continual-memory/gemini_api_key")

DEFAULT_USER_LLM = "gemini/gemini-2.5-flash"
DEFAULT_USER_LLM_ARGS = {"temperature": 0.0, "reasoning_effort": "disable"}
DEFAULT_JUDGE_LLM = DEFAULT_USER_LLM
DEFAULT_JUDGE_LLM_ARGS = dict(DEFAULT_USER_LLM_ARGS)

DEFAULT_AGENT_LLM = "openai/qwen35-tau"


def agent_llm_args(base_url: str, max_tokens: int = 4096) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "api_base": base_url,
        "api_key": "EMPTY",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def configure_gemini(judge_llm: str = DEFAULT_JUDGE_LLM, judge_llm_args: dict | None = None) -> None:
    """Point litellm at the Gemini key and tau2's NL-assertion judge at Gemini.

    The judge model is a module-level import in tau2's evaluator, so it is
    replaced there -- in-process, nothing in the checkout is edited.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY_FILE.read_text().strip()
    import json as _json
    import re as _re

    import tau2.evaluator.evaluator_nl_assertions as nl

    nl.DEFAULT_LLM_NL_ASSERTIONS = judge_llm
    args = dict(judge_llm_args or DEFAULT_JUDGE_LLM_ARGS)
    # tau2's evaluator does a bare `json.loads` on the judge's reply. Its own
    # default judge is gpt-4.1, which answers that prompt with naked JSON;
    # Gemini wraps it in a ```json fence often enough that 3 of the first 6
    # retail tasks died with JSONDecodeError. That kills the SCORING, not the
    # episode, so those tasks silently carry no reward and drop out of the
    # pass rate -- a selective sample loss, not a visible failure.
    #
    # Belt and braces: ask for a JSON mime type so the fence never appears,
    # and wrap `json.loads` so a fence (or prose around the object) is
    # tolerated if one slips through anyway.
    args.setdefault("response_format", {"type": "json_object"})
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args

    if not getattr(nl, "_tml_json_patched", False):
        _strict_loads = nl.json.loads

        def _lenient_loads(text, *a, **kw):
            try:
                return _strict_loads(text, *a, **kw)
            except Exception:
                fenced = _re.search(r"```(?:json)?\s*(.*?)```", text, _re.S)
                candidate = fenced.group(1) if fenced else text
                start, end = candidate.find("{"), candidate.rfind("}")
                if start >= 0 and end > start:
                    return _strict_loads(candidate[start : end + 1], *a, **kw)
                raise

        nl.json = type(_json)("json_lenient")
        nl.json.loads = _lenient_loads
        nl._tml_json_patched = True


DEFAULT_LLM_CACHE = Path(__file__).resolve().parents[2] / "tau2_experiment/llm_cache.sqlite"


def install_llm_cache(path: Path | str | None = DEFAULT_LLM_CACHE) -> None:
    """Make the Gemini side of every simulation a pure function of its input.

    Measured: with gemini-2.5-flash at temperature 0 and thinking disabled,
    2 of 5 airline tasks run twice already diverged at the user simulator's
    very FIRST message ("I'd like to book a flight please." vs the same plus
    the route), one of them flipping reward 0 -> 1; the agent side was
    byte-identical wherever the user was (a deterministic vLLM server).
    Gemini takes no seed, so this cannot be fixed at the API.

    So the user simulator's and the NL judge's `generate` calls are memoized
    on disk, keyed on (model, args, the full message history, tools). The
    same conversation prefix then always gets the same customer reply and the
    same verdict, and two arms of an experiment differ only where the AGENT
    made the conversation differ -- a memory entry that changes the agent's
    reply still gets a fresh (and from then on fixed) customer response. The
    cost is that each prefix is one draw of the simulator, not an average
    over draws; `--repeat` with the cache off measures that spread.

    Both call sites import `generate` by name, so the wrapper replaces those
    two module attributes in-process; the checkout is not edited. `None`
    disables caching.
    """
    if path is None:
        return
    import pickle
    import sqlite3
    import threading

    import tau2.evaluator.evaluator_nl_assertions as nl_module
    import tau2.user.user_simulator as user_module
    from tau2.utils.llm_utils import generate as real_generate
    from tau2.utils.llm_utils import to_litellm_messages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    connection = sqlite3.connect(str(path), timeout=60, check_same_thread=False)
    # Default rollback journal, not WAL: WAL needs shared memory that does
    # not work across an NFS mount, and this file sits in the repo (on NFS)
    # while rollout and replay subprocesses write it concurrently.
    connection.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)")
    connection.commit()

    def strip_ids(value):
        # Tool-call ids are minted fresh per response (random strings), so a
        # key containing them would never repeat.
        if isinstance(value, dict):
            return {k: strip_ids(v) for k, v in value.items() if k not in ("id", "tool_call_id")}
        if isinstance(value, list):
            return [strip_ids(v) for v in value]
        return value

    def cached_generate(model, messages, tools=None, tool_choice=None, call_name=None, **kwargs):
        key_material = json.dumps({
            "model": model,
            "call_name": call_name,
            # `seed` is left out: the orchestrator writes its per-run seed into
            # the user's llm_args (`set_seed`), but litellm drops it before it
            # reaches Gemini -- keeping it made two runs with identical
            # conversations miss each other's entries (measured: repeats with
            # seeds 300/301 diverged at a customer turn after an identical
            # prefix).
            "kwargs": {k: v for k, v in kwargs.items() if k not in ("api_key", "num_retries", "seed")},
            "tool_choice": tool_choice,
            "messages": strip_ids(to_litellm_messages(messages)),
            "tools": [tool.openai_schema for tool in (tools or [])],
        }, sort_keys=True, default=str)
        key = hashlib.sha256(key_material.encode()).hexdigest()
        with lock:
            row = connection.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return pickle.loads(row[0])
        result = real_generate(
            model=model, messages=messages, tools=tools, tool_choice=tool_choice,
            call_name=call_name, **kwargs,
        )
        # Two threads can miss on the same key at once (repeats of one task
        # start together, so their opening turns collide). The first write
        # wins and BOTH return it; returning our own draw would let the
        # loser's conversation silently diverge from what the cache holds.
        with lock:
            connection.execute("INSERT OR IGNORE INTO cache VALUES (?, ?)", (key, pickle.dumps(result)))
            connection.commit()
            stored = connection.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return pickle.loads(stored[0])

    user_module.generate = cached_generate
    nl_module.generate = cached_generate


def quiet_logs() -> None:
    """tau2 logs every LLM call at DEBUG, and litellm's cost lookup logs an
    ERROR on every call to a model missing from its price table (the local
    Qwen) -- harmless, but it buries real errors. Keep WARNING and above,
    minus that one message."""
    import sys

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING", filter=lambda record: "isn't mapped yet" not in record["message"])


def task_id_for(domain: str, tau2_task_id: str) -> str:
    return f"{domain}::{tau2_task_id}"


def parse_task_id(task_id: str) -> tuple[str, str]:
    domain, tau2_task_id = task_id.split("::", 1)
    return domain, tau2_task_id


def task_file_stem(task_id: str) -> str:
    """Filesystem-safe name. Telecom ids are long `[issue]a|b|c[PERSONA:x]`
    strings, so the readable part is capped and a hash keeps it unique."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id.replace("::", "__"))[:80]
    return f"{safe}_{hashlib.sha1(task_id.encode()).hexdigest()[:8]}"


def split_task_ids(domain: str, split: str) -> list[str]:
    from tau2.runner import load_task_splits

    splits = load_task_splits(domain) or {}
    if split not in splits:
        raise KeyError(f"{domain} has no split {split!r}; available: {sorted(splits)}")
    return [task_id_for(domain, str(t)) for t in splits[split]]


def load_task(task_id: str):
    from tau2.runner import get_tasks

    domain, tau2_task_id = parse_task_id(task_id)
    return get_tasks(domain, task_ids=[tau2_task_id])[0]


def _make_memory_agent_class():
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import UserMessage

    class MemoryAgent(LLMAgent):
        """tau2's `LLMAgent`, plus one extra block in the system prompt.

        The block is decided on the first customer message: retrieved from
        it (`memory_lookup`), or fixed in advance (`fixed_block`, used for a
        guided replay's plan). `base_system_prompt` keeps the prompt without
        the block, for the SFT example.
        """

        def __init__(self, *args, memory_lookup=None, fixed_block: str = "", **kwargs):
            super().__init__(*args, **kwargs)
            self.memory_lookup = memory_lookup
            self.fixed_block = fixed_block
            self.injected = False
            self.selection: list[dict[str, Any]] = []
            self.first_user_message: str | None = None

        def generate_next_message(self, message, state):
            if not self.injected and isinstance(message, UserMessage) and message.content:
                self.injected = True
                self.first_user_message = message.content
                block = self.fixed_block
                if self.memory_lookup is not None:
                    block, self.selection = self.memory_lookup(message.content)
                if block:
                    state.system_messages[0].content += "\n\n" + block.strip()
            return super().generate_next_message(message, state)

    return MemoryAgent


def _render_tool_call(tc) -> str:
    return f"{tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})"


def render_steps(messages: list) -> list[dict[str, Any]]:
    """Full conversation -> canonical steps, for the writers.

    Tool calls are rendered inline as `name({...})`. In telecom the customer
    calls tools too; those are labelled as the customer's so a writer does
    not read them as the agent's actions.
    """
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage, UserMessage

    steps: list[dict[str, Any]] = []

    def add(role: str, content: str) -> None:
        steps.append({"index": len(steps), "role": role, "content": content})

    for message in messages:
        if isinstance(message, MultiToolMessage):
            for tool_message in message.tool_messages:
                who = "customer's tool" if tool_message.requestor == "user" else "tool"
                add("tool", f"[{who} result] {tool_message.content}")
        elif isinstance(message, ToolMessage):
            who = "customer's tool" if message.requestor == "user" else "tool"
            add("tool", f"[{who} result] {message.content}")
        elif isinstance(message, AssistantMessage):
            parts = [message.content] if message.content else []
            parts += [f"[tool call] {_render_tool_call(tc)}" for tc in (message.tool_calls or [])]
            add("assistant", "\n".join(parts))
        elif isinstance(message, UserMessage):
            parts = [message.content] if message.content else []
            parts += [f"[customer tool call] {_render_tool_call(tc)}" for tc in (message.tool_calls or [])]
            add("user", "\n".join(parts))
    return steps


def summarize_reward(reward_info) -> dict[str, Any]:
    """Pass/fail per scoring component, as counts -- no expected values."""
    if reward_info is None:
        return {"reward": 0.0}
    summary: dict[str, Any] = {
        "reward": float(reward_info.reward),
        "reward_basis": [str(getattr(b, "value", b)) for b in (reward_info.reward_basis or [])],
    }
    if reward_info.db_check is not None:
        summary["db_match"] = bool(reward_info.db_check.db_match)
    for name, attr, flag in (
        ("communicate_info", "communicate_checks", "met"),
        ("env_assertions", "env_assertions", "met"),
        ("nl_assertions", "nl_assertions", "met"),
        ("expected_actions", "action_checks", "action_match"),
    ):
        checks = getattr(reward_info, attr, None)
        if checks:
            summary[f"{name}_met"] = f"{sum(bool(getattr(c, flag, False)) for c in checks)}/{len(checks)}"
    return summary


def run_task(
    task_id: str,
    *,
    agent_base_url: str,
    agent_llm: str = DEFAULT_AGENT_LLM,
    agent_max_tokens: int = 4096,
    user_llm: str = DEFAULT_USER_LLM,
    user_llm_args: dict | None = None,
    memory_lookup: Callable[[str], tuple[str, list[dict[str, Any]]]] | None = None,
    memory_block: str = "",
    seed: int = 300,
    max_steps: int = 200,
    max_errors: int = 10,
) -> dict[str, Any]:
    """One tau2 simulation -> canonical trajectory (+ `sft_example`)."""
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner import build_environment, build_user, run_simulation
    from tau2.utils.llm_utils import to_litellm_messages

    domain, _ = parse_task_id(task_id)
    task = load_task(task_id)
    environment = build_environment(domain)
    agent = _make_memory_agent_class()(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=agent_llm,
        llm_args=agent_llm_args(agent_base_url, agent_max_tokens),
        memory_lookup=memory_lookup,
        fixed_block=memory_block,
    )
    base_system_prompt = agent.system_prompt
    user = build_user(
        "user_simulator", environment, task,
        llm=user_llm, llm_args=dict(user_llm_args or DEFAULT_USER_LLM_ARGS),
    )
    orchestrator = Orchestrator(
        domain=domain, agent=agent, user=user, environment=environment, task=task,
        max_steps=max_steps, max_errors=max_errors, seed=seed, simulation_id=str(uuid.uuid4()),
    )
    simulation = run_simulation(orchestrator)

    evaluation = summarize_reward(simulation.reward_info)
    reward = float(evaluation["reward"])
    termination = str(getattr(simulation.termination_reason, "value", simulation.termination_reason))
    agent_state = orchestrator.agent_state
    sft_example = None
    if agent_state is not None:
        sft_example = {
            "messages": [{"role": "system", "content": base_system_prompt}]
            + to_litellm_messages(agent_state.messages),
            "tools": [tool.openai_schema for tool in environment.get_tools()],
        }
    return {
        "source_task_id": f"tau2.{task_id}",
        "domain": f"tau2_{domain}",
        # What the agent was actually told: the customer's opening message.
        "task": {"id": task_id, "instruction": agent.first_user_message or ""},
        "success": reward >= 1.0 - 1e-6,
        "reward": reward,
        "termination_reason": termination,
        "evaluation": evaluation,
        "retrieved_memory": agent.selection,
        "steps": render_steps(simulation.messages or []),
        "sft_example": sft_example,
        "usage": {"agent_cost": simulation.agent_cost, "user_cost": simulation.user_cost},
        "simulation": {"id": simulation.id, "duration": simulation.duration, "seed": seed},
    }
