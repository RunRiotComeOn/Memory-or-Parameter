"""Web-shopping agent loop for WebShop, recording canonical trajectories.

Fourth benchmark. WebShop (Yao et al., 2022) is a simulated e-commerce site
over 1.18M real Amazon products: given a human-written shopping request
("i need a long clip-in hair extension which is natural looking, and price
lower than 40.00 dollars"), the agent searches, opens products, picks options
(size, color, ...) and buys. It is also the benchmark ReAct, Reflexion and
ExpeL report on, which makes this project's numbers directly comparable to
published memory/reflection baselines -- ExpeL in particular extracts
cross-task "insights" from experience, the closest prior work to a memory
bank.

What makes it a memory domain: every task shares ONE catalog and ONE search
engine. How that engine responds to queries (short attribute-heavy queries
beat long sentences; the price bound is not a search filter), where options
hide, and which product-type near-misses the grader punishes are the same
from task to task -- reusable knowledge, not per-task answers.

The environment itself runs in a separate long-lived process
(`scripts/webshop_env_server.py`, under `webshop_venv`) because the catalog
takes minutes and tens of GB to load; this module only talks to it over HTTP
and so needs none of WebShop's dependencies.

Action space: `search[<query>]` when the page has a search bar, otherwise
`click[<text>]` for one of the page's clickables (product ids, options,
"buy now", "next >", "< prev", "back to search", "description", ...). The
available list is shown every turn, as for ALFWorld, and a reply that is not
one of them gets a nudge rather than being passed through (the env would
silently no-op it).

Scoring is WebShop's own: on "buy now" the purchase gets a score in [0, 1]
from product type, attribute, option and price matches. `success` is the
standard "success rate" criterion, score == 1; `reward` keeps the raw score,
so both of WebShop's reported metrics (success rate, average score) come out
of the same trajectories. `evaluation` carries the score's components but
never the target product or its attributes -- the sft/memory writers see
`evaluation`, and a writer that could see the answer would write a plan
that just names it.

Splits are WebShop's own, by index into the (seeded) goal list:
`test` = 0-499, `dev` = 500-1499, `train` = 1500 onward
(`baseline_models/env.py`).

Canonical trajectory shape matches the other agent modules:
    {"source_task_id", "domain": "webshop", "task": {"id", "instruction"},
     "success", "reward", "termination_reason", "evaluation",
     "steps": [{"index", "role", "content"}]}
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from .model_client import ModelClient

WEBSHOP_ENV_URL_DEFAULT = "http://127.0.0.1:3100"

SPLIT_RANGES = {"test": (0, 500), "dev": (500, 1500), "train": (1500, None)}

AGENT_SYSTEM = """You are an autonomous shopping assistant on a web store (WebShop). You are given a customer's shopping request and must find and buy the product that best matches it.

Every turn you are shown the current page as text (sections separated by [SEP]) and a list of AVAILABLE ACTIONS. Reply with EXACTLY ONE action from that list and nothing else -- no explanation, no code fence:
- search[<your query>] -- only on a page with a search bar; you choose the query text.
- click[<item>] -- click one of the listed clickable items, copied exactly: a product id (e.g. click[b07xyz1234]), an option value (a size or color), "buy now", "next >", "< prev", "back to search", "description", "features", "reviews", "attributes".

How the purchase is judged: when you click[buy now], the product you are on is scored against the request -- the product type, the attributes the request mentions, the options you selected on the product page (size, color, flavor, count, ...), and the price limit. Only a purchase matching all of them gets full credit. So on the product page, click every option the request specifies BEFORE buy now; an unselected option counts as a miss. The price limit is not applied by the search engine: check the price yourself.

The episode ends when you click buy now or run out of steps. Buying something reasonable is better than running out of steps and buying nothing."""


class WebShopEnvClient:
    """Thin HTTP client for `scripts/webshop_env_server.py`."""

    def __init__(self, base_url: str | None = None, timeout: float = 120):
        self.base_url = (base_url or os.environ.get("WEBSHOP_ENV_URL") or WEBSHOP_ENV_URL_DEFAULT).rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = json.loads(exc.read() or b"{}").get("error", "")
            raise RuntimeError(f"webshop env server {path} failed: {detail}") from exc

    def info(self) -> dict[str, Any]:
        with urllib.request.urlopen(self.base_url + "/info", timeout=self.timeout) as response:
            return json.loads(response.read())

    def reset(self, goal_idx: int) -> dict[str, Any]:
        return self._post("/reset", {"goal_idx": goal_idx})

    def step(self, env_id: str, action: str) -> dict[str, Any]:
        return self._post("/step", {"env_id": env_id, "action": action})

    def close(self, env_id: str) -> None:
        self._post("/close", {"env_id": env_id})


def task_id_for(goal_idx: int) -> str:
    return f"goal_{goal_idx:05d}"


def parse_task_id(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[1])


def split_task_ids(split: str, num_goals: int) -> list[str]:
    start, end = SPLIT_RANGES[split]
    return [task_id_for(i) for i in range(start, num_goals if end is None else min(end, num_goals))]


def _action_list(available: dict[str, Any]) -> list[str]:
    actions = ["search[<your query>]"] if available.get("has_search_bar") else []
    actions.extend(f"click[{c}]" for c in available.get("clickables", []))
    return actions


def _render(observation: str, available: dict[str, Any], max_chars: int) -> str:
    return _truncate(observation, max_chars) + "\n\nAvailable actions: " + ", ".join(_action_list(available))


_ACTION_PATTERN = re.compile(r"^(search|click)\[(.+)\]$", re.IGNORECASE | re.DOTALL)


def extract_action(content: str, available: dict[str, Any]) -> str | None:
    """Pull one admissible action out of a reply: the whole reply first, then
    each line, as in `alfworld_agent.extract_command`. `search[...]` is
    accepted with any non-empty query when the page has a search bar;
    `click[...]` only for a listed clickable (case-insensitive, returned in
    the env's own spelling). A literal `search[<your query>]` copied from the
    list is rejected -- it would search for the placeholder text."""
    clickables = {c.lower(): c for c in available.get("clickables", [])}
    candidates = [content.strip()]
    candidates.extend(line.strip().strip("`").strip() for line in content.splitlines() if line.strip())
    for candidate in candidates:
        match = _ACTION_PATTERN.match(candidate)
        if not match:
            continue
        verb, argument = match.group(1).lower(), match.group(2).strip()
        if verb == "search":
            if available.get("has_search_bar") and argument and argument != "<your query>":
                return f"search[{argument}]"
        elif argument.lower() in clickables:
            return f"click[{clickables[argument.lower()]}]"
    return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...[{len(text) - limit} characters omitted]...\n" + text[-half:]


def run_task(
    env: WebShopEnvClient,
    task_id: str,
    client: ModelClient,
    *,
    memory_lookup: Callable[[str], tuple[str, list[dict[str, Any]]]] | None = None,
    memory_block: str = "",
    max_steps: int = 15,
    max_output_chars: int = 3_000,
) -> dict[str, Any]:
    """Drive one WebShop episode and return its canonical trajectory.

    The instruction is only known after the env resets, so retrieval is a
    callback: `memory_lookup(instruction) -> (block, selection_log)`. A fixed
    `memory_block` (guided replay's plan) is used as given. Either way the
    block goes on the first user message only, as in every other agent
    module here. The selection log is returned on the trajectory as
    `retrieved_memory`.
    """
    state = env.reset(parse_task_id(task_id))
    env_id = state["env_id"]
    # The env's instruction text carries the page's own "Instruction: " label.
    instruction = re.sub(r"^\s*instruction:\s*", "", state["instruction"], flags=re.IGNORECASE).strip()
    selection: list[dict[str, Any]] = []
    if memory_lookup is not None:
        memory_block, selection = memory_lookup(instruction)
    try:
        available = state["available"]
        user_message = _render(state["observation"], available, max_output_chars)
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
        score = 0.0
        purchase: dict[str, Any] | None = None

        for _ in range(max_steps):
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

            action = extract_action(reply.content, available)
            if action is None:
                ungrounded_turns += 1
                if ungrounded_turns >= 3:
                    termination = "ungrounded_action"
                    break
                feedback = (
                    "That is not one of the available actions. Reply with exactly one action, "
                    "copied from the list (for search, put your own query inside the brackets).\n\n"
                    "Available actions: " + ", ".join(_action_list(available))
                )
            else:
                ungrounded_turns = 0
                result = env.step(env_id, action)
                available = result["available"]
                if result["done"]:
                    score = result["reward"]
                    purchase = result.get("purchase")
                    termination = "purchased"
                    break
                feedback = _render(result["observation"], available, max_output_chars)
            steps.append({"index": len(steps), "role": "user", "content": feedback})
            messages.append({"role": "user", "content": feedback})
    finally:
        env.close(env_id)

    success = score >= 1.0 - 1e-9
    return {
        "source_task_id": f"webshop.{task_id}",
        "domain": "webshop",
        "task": {"id": task_id, "instruction": instruction},
        "success": success,
        "reward": score,
        "termination_reason": termination,
        "evaluation": {
            "score": score,
            "purchased": purchase is not None,
            "purchased_asin": (purchase or {}).get("asin"),
            "selected_options": (purchase or {}).get("options"),
            "score_components": (purchase or {}).get("reward_components"),
        },
        "retrieved_memory": selection,
        "steps": steps,
        "usage": usage_totals,
    }
