"""Code-acting agent loop for AppWorld, recording canonical trajectories.

AppWorld is not a tool-call benchmark: the agent writes Python that calls
`apis.<app>.<api>(...)` and reads back whatever the environment prints.  This
module runs that loop and emits a trajectory in the same shape the v2 allocation
writer already consumes for tau-bench, so `alloc_writer_harness` needs no
benchmark-specific branch:

    {"source_task_id", "domain", "task", "success", "reward",
     "termination_reason", "evaluation", "steps": [{"index", "role", "content"}]}

Role mapping: `user` is the supervisor's instruction, `assistant` is the code
the model wrote, `tool` is the environment's execution output.
"""

from __future__ import annotations

import re
from typing import Any

from .model_client import ModelClient


AGENT_SYSTEM = """You are an autonomous assistant that completes a supervisor's task by writing Python code against a set of app APIs.

Every turn you write exactly one fenced Python block:

```python
# your code here
```

The code runs in a persistent IPython session, so variables and imports survive across turns. Write small steps and inspect results before acting on them; do not write one long script that assumes what an API returns.

You only ever see what your code prints. A bare expression shows you nothing: `apis.supervisor.show_profile()` returns the value but displays it nowhere, and the turn comes back as `Execution successful.` with no data. Wrap every value you want to read in `print(...)`. If an observation is `Execution successful.` and you expected data, you forgot to print it — that turn is wasted.

An `apis` object is already available. You cannot import or install anything to reach the apps; everything goes through `apis`.

Discovering the environment:
- `apis.api_docs.show_app_descriptions()` lists every app.
- `apis.api_docs.show_api_descriptions(app_name="spotify")` lists an app's APIs.
- `apis.api_docs.show_api_doc(app_name="spotify", api_name="login")` shows one API's exact parameters, defaults, and response schema.
- `apis.api_docs.search_api_docs(query="songs I have liked")` searches every app's API docs by intent. Use it the moment you are unsure which API does something.
Read the doc for an API before you first call it. Never guess an API or parameter name: list or search for it instead. If a name you tried does not exist, search for what you want rather than trying variations of the name.

Credentials and identity:
- `apis.supervisor.show_profile()` returns the supervisor's own details.
- `apis.supervisor.show_account_passwords()` returns their app account passwords.
- Most apps need `apis.<app>.login(username=..., password=...)`, which returns an access token you must pass to that app's other APIs.

Paging: list APIs are paginated. When a result may exceed one page, loop with the documented page/limit parameters until you have everything; do not answer from the first page alone.

Finishing:
- Call `apis.supervisor.complete_task(status="success")` when the task asked you to *do* something.
- Call `apis.supervisor.complete_task(answer=<value>, status="success")` when the task asked a question. The answer must be the bare value requested — a number, a name, a comma-separated list — with no sentence around it.
- Do not call `complete_task` until you have verified the work through the APIs.

If an API returns an error, read the message and the API doc again rather than retrying the same call unchanged."""


CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(content: str) -> str | None:
    """Pull the first fenced Python block out of a model reply."""
    match = CODE_BLOCK.search(content)
    if match:
        code = match.group(1).strip()
        return code or None
    return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n...[{len(text) - limit} characters omitted]...\n"
        + text[-half:]
    )


def build_initial_user_message(task: Any, memory_block: str) -> str:
    supervisor = getattr(task, "supervisor", None)
    lines = [f"Task: {task.instruction}"]
    if supervisor is not None:
        lines.append(
            "\nSupervisor: "
            f"{getattr(supervisor, 'first_name', '')} {getattr(supervisor, 'last_name', '')}".rstrip()
        )
        for field in ("email", "phone_number"):
            value = getattr(supervisor, field, None)
            if value:
                lines.append(f"{field}: {value}")
    if memory_block:
        lines.append(memory_block)
    return "\n".join(lines)


def run_task(
    world: Any,
    task: Any,
    client: ModelClient,
    *,
    memory_block: str = "",
    max_steps: int = 40,
    max_output_chars: int = 3_000,
) -> dict[str, Any]:
    """Drive one AppWorld task and return its canonical trajectory."""
    user_message = build_initial_user_message(task, memory_block)
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": user_message},
    ]
    steps: list[dict[str, Any]] = [
        {"index": 0, "role": "user", "content": user_message}
    ]
    termination = "max_steps"
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
    empty_code_turns = 0
    truncated_turns = 0

    for _ in range(max_steps):
        reply = client.chat_messages(messages)
        for key in usage_totals:
            value = (reply.usage or {}).get(key)
            if isinstance(value, int):
                usage_totals[key] += value
        code = extract_code(reply.content)
        steps.append(
            {"index": len(steps), "role": "assistant", "content": reply.content}
        )
        messages.append({"role": "assistant", "content": reply.content})
        if code is None:
            # A reply cut off by the token limit has no closing fence, so it
            # parses as "no code". That is a length problem, not a formatting
            # one, and it must not spend the formatting budget.
            truncated = reply.finish_reason == "length" or (
                reply.content.count("```") == 1
            )
            if truncated:
                truncated_turns += 1
                if truncated_turns >= 3:
                    termination = "repeated_truncation"
                    break
                nudge = (
                    "Your reply was cut off before the closing fence. Write a much "
                    "shorter code block: one or two API calls, no long comment "
                    "blocks, and print only what you need to read next."
                )
                steps.append({"index": len(steps), "role": "tool", "content": nudge})
                messages.append({"role": "user", "content": nudge})
                continue
            empty_code_turns += 1
            if empty_code_turns >= 2:
                termination = "no_code_block"
                break
            nudge = (
                "You did not emit a fenced Python block. Reply with exactly one "
                "```python ... ``` block and nothing else."
            )
            steps.append({"index": len(steps), "role": "tool", "content": nudge})
            messages.append({"role": "user", "content": nudge})
            continue
        empty_code_turns = 0
        try:
            output = world.execute(code)
        except Exception as exc:  # environment-level failure, not agent error
            output = f"Environment error: {exc!r}"
            termination = "environment_error"
            steps.append({"index": len(steps), "role": "tool", "content": output})
            break
        output = _truncate(str(output), max_output_chars)
        steps.append({"index": len(steps), "role": "tool", "content": output})
        messages.append({"role": "user", "content": output})
        if world.task_completed():
            termination = "task_completed"
            break

    report = world.evaluate()
    evaluation = report.to_dict() if hasattr(report, "to_dict") else {}
    success = bool(getattr(report, "success", False))
    return {
        "source_task_id": f"appworld.{task_id_of(world, task)}",
        "domain": "appworld",
        "task": {
            "id": task_id_of(world, task),
            "instruction": task.instruction,
        },
        "success": success,
        "reward": 1.0 if success else 0.0,
        "termination_reason": termination,
        "evaluation": evaluation,
        "steps": steps,
        "usage": usage_totals,
    }


def task_id_of(world: Any, task: Any) -> str:
    for holder in (task, world):
        value = getattr(holder, "task_id", None)
        if isinstance(value, str):
            return value
    return "unknown"
