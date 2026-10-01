"""Text agent for SQLGym / BIRD, the eighth benchmark in the suite.

BIRD is text-to-SQL over real databases: each task gives a natural-language
schema description and a question, and the agent must produce SQL whose
result set matches the reference query's. Correctness is decided by
executing both and comparing sets, so ordering does not matter and there is
no partial credit.

Why this benchmark earns a slot -- it is chosen to DISCRIMINATE between two
explanations the first seven benchmarks left fitted rather than tested:

1. **Composability** (`textcraft_summary.md` conclusion 2). BabyAI found
   more SFT samples worse, TextCraft found more better, and the proposed
   deciding variable is whether a sample's knowledge is reusable across
   instances. Splitting BIRD by DATABASE separates the two kinds cleanly:
   SQL clause patterns transfer across schemas, schema knowledge does not.
   The composability account predicts more-is-better here.
2. **Teacher information visibility** (`textcraft_summary.md` conclusion 4).
   Rescue rate splits into a high group (TextCraft 100%, where the whole
   task is printed in the observation) and a low group (WebShop 18.7%,
   ScienceWorld 11.3%, where it is not). Here the schema is visible but the
   TABLE CONTENTS are not, and BIRD questions frequently turn on actual
   values. The teacher sees only what the agent's own exploration surfaced,
   so the account predicts an INTERMEDIATE rescue rate.

Both predictions are recorded before running; TextCraft's value came
largely from a pre-registered prediction turning out wrong.

## The multi-turn protocol, and why it is not sqlgym's

`SqlGymEnv.step` terminates on the first query -- it is a single-shot
text-to-SQL harness. Scoring one generation would make this incomparable
with the other seven benchmarks, which are all episodic, and would also
make hypothesis 2 untestable: if the agent cannot inspect the data, the
question of what the teacher can see relative to the agent does not arise.

So this harness drives the environment directly:

- `EXPLORE: <sql>` runs on the same read-only connection and returns up to
  `MAX_ROWS` rows. No reward, no termination. The connection is opened by
  sqlgym with `?mode=ro`, so exploration cannot mutate a database.
- `SUBMIT: <sql>` runs the query, scores it with sqlgym's own comparison,
  and ends the episode.

Reward is binary, so `mean_score` equals `pass_rate` by construction; both
are reported to keep the cross-benchmark tables uniform.

Runs under `/nas04/yixuh/sqlgym_venv` (sqlgym is a library, not a server --
unlike the AgentGym environments there is no HTTP hop).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .model_client import ModelClient

DEFAULT_BIRD_PATH = "/nas04/yixuh/bird"
# Budget results by CHARACTERS, not rows. A flat 20-row cap looked safe and
# was the single worst harness bug here: `SELECT name FROM
# pragma_table_info(t)` returns one short string per column -- 49 rows and a
# few hundred bytes for a wide table -- and truncating it to 20 hid exactly
# the information the agent needed. Observed consequence: the agent re-ran
# the identical query over and over, burning a whole 30-turn budget without
# submitting. Rows are capped too, but high enough not to bite introspection.
MAX_ROWS = 200
MAX_CELL_CHARS = 200   # a single cell can hold a whole document; truncate it
MAX_RESULT_CHARS = 6000

AGENT_SYSTEM = """You are an expert SQL analyst answering a question against a real SQLite database.

You are shown a description of the database schema and a question. Your job is to produce one SQL query whose result answers the question exactly.

Each turn, reply with EXACTLY ONE of these two commands and nothing else -- no explanation, no markdown fences:

- `EXPLORE: <sql>` -- run a query and see up to 20 rows of its result. Use this to check what values a column actually contains, how names are spelled, whether a join produces what you expect, or how many rows match a filter. The connection is read-only. Exploring costs you a turn but nothing else.
- `SUBMIT: <sql>` -- submit your final answer. This ends the episode immediately and is scored by comparing your result set against the reference answer, so submit only when you are confident.

Write the query on a single line after the command.

The schema description lowercases and expands column names for readability, but the REAL column names often differ: they may contain spaces, capitals, or punctuation, in which case they must be quoted with backticks (for example `Free Meal Count (K-12)`). When a column name is not a plain identifier, explore first rather than guessing.

Answer the question exactly as asked: return only the requested columns, in the requested order where one is specified, and apply every condition stated. Do not add columns that were not asked for.

Every turn tells you how many remain. Spend the early ones on exploration that reduces real uncertainty, then submit. An episode that ends without a SUBMIT scores zero no matter how good your exploration was, so always leave yourself a turn to submit."""


class SqlGymTaskEnv:
    """Thin wrapper over `sqlgym.SqlGymEnv` that adds the multi-turn loop.

    Holds one dataset per (bird_path, mode) so repeated resets do not re-read
    the BIRD json for every task.
    """

    _datasets: dict[tuple[str, str], Any] = {}

    def __init__(self, bird_path: str = DEFAULT_BIRD_PATH, mode: str = "dev") -> None:
        self.bird_path = bird_path
        self.mode = mode
        self._env = None
        self._item = None

    def _dataset(self):
        key = (self.bird_path, self.mode)
        if key not in self._datasets:
            from sqlgym.datasets import BirdDataset

            self._datasets[key] = BirdDataset(self.bird_path, self.mode)
        return self._datasets[key]

    def __len__(self) -> int:
        return len(self._dataset())

    def reset(self, idx: int) -> dict[str, Any]:
        from sqlgym import SqlGymEnv

        dataset = self._dataset()
        self._env = SqlGymEnv(dataset)
        observation = self._env.reset(idx)
        self._item = dataset[idx]
        return {
            "observation": observation,
            "difficulty": (self._item.info or {}).get("difficulty"),
            "evidence": (self._item.info or {}).get("evidence") or "",
            "db_id": self.db_id(idx),
        }

    def db_id(self, idx: int) -> str:
        return self._dataset()._data[idx]["db_id"]  # noqa: SLF001 - no public accessor

    def explore(self, sql: str) -> str:
        result = self._env._exec_sql(sql)  # noqa: SLF001 - no public read-only exec
        return render_result(result)

    def submit(self, sql: str) -> tuple[str, float]:
        result = self._env._exec_sql(sql)  # noqa: SLF001
        reward = float(self._env._get_reward(result))  # noqa: SLF001
        return render_result(result), reward

    def close(self) -> None:
        if self._env is not None and getattr(self._env, "conn", None) is not None:
            try:
                self._env.conn.close()
            except Exception:  # noqa: BLE001 - closing is best effort
                pass
        self._env = None


def render_result(result: Any) -> str:
    """Format rows (or the database's own error) for the agent.

    A syntax or column error comes back as the real sqlite message, which is
    the actionable feedback the agent needs -- this is the same reasoning
    behind ScienceWorld's and BabyAI's `_closest_hint`, except here the
    environment already produces a good message.
    """
    if isinstance(result, Exception):
        return f"SQL error: {result}"
    if not result:
        return "(0 rows)"
    lines: list[str] = []
    used = 0
    for row in result[:MAX_ROWS]:
        cells = [str(c)[:MAX_CELL_CHARS] for c in (row if isinstance(row, tuple) else (row,))]
        line = " | ".join(cells)
        if used + len(line) > MAX_RESULT_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    shown = len(lines)
    text = "\n".join(lines)
    if shown < len(result):
        text += (f"\n... ({len(result)} rows total, showing {shown}. "
                 f"Re-running this query returns the same rows -- narrow it with "
                 f"WHERE/LIMIT/OFFSET or select fewer columns to see more.)")
    else:
        text += f"\n({len(result)} rows)"
    return text


_EXPLORE_RE = re.compile(r"^explore\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
_SUBMIT_RE = re.compile(r"^submit\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)


def extract_command(content: str) -> tuple[str, str] | None:
    """Return ("explore"|"submit", sql), or None if the reply is neither.

    Validation is a grammar rather than a menu, as in TextCraft: there is no
    list of admissible commands to match against.
    """
    text = content.strip()
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    for line in [text, *(l.strip() for l in text.splitlines() if l.strip())]:
        candidate = line.strip().strip("`").strip()
        for kind, pattern in (("explore", _EXPLORE_RE), ("submit", _SUBMIT_RE)):
            match = pattern.match(candidate)
            if match:
                sql = match.group(1).strip().rstrip(";").strip()
                sql = re.sub(r"\s*```.*$", "", sql, flags=re.DOTALL).strip()
                if sql:
                    return kind, sql
    return None


# A task id carries its BIRD mode because train and dev index independently:
# "sqlgym::train::17" and "sqlgym::dev::17" are different questions against
# different databases. Everything downstream (rollout, replay, memory bank)
# passes task ids around as opaque strings, so the mode has to travel inside.
def parse_task_id(task_id: str) -> tuple[str, int]:
    _, mode, idx = task_id.split("::")
    return mode, int(idx)


def make_task_id(mode: str, idx: int) -> str:
    return f"sqlgym::{mode}::{idx}"


SPLIT_SEED = 20260929
POOL_SIZE = 200
TEST_SIZE = 80
XDB_SIZE = 150


def _mode_data(bird_path: str, mode: str):
    from sqlgym.datasets import BirdDataset

    return BirdDataset(bird_path, mode)._data  # noqa: SLF001 - no public accessor


def split_task_ids(split: str, bird_path: str = DEFAULT_BIRD_PATH) -> list[str]:
    """Three splits, deterministic under `SPLIT_SEED`.

    - `train` (200): sampled from BIRD train, spread across its 69 databases.
      Every pool is built from this.
    - `test` (80): more BIRD train questions, disjoint from `train` but over
      the SAME 69 databases. The same-distribution line.
    - `xdb` (150): sampled from BIRD dev, whose 11 databases do not appear in
      train at all (verified: zero overlap). This is the cross-schema line
      and the one that tests the composability account -- SQL clause
      patterns transfer across schemas, schema knowledge does not.

    BIRD ships `difficulty` on dev only, so the difficulty breakdown is an
    evaluation-side cut of `xdb`, not a property of the pool.
    """
    import random

    rng = random.Random(SPLIT_SEED)
    if split in ("train", "test"):
        data = _mode_data(bird_path, "train")
        # Stratify by database so no schema dominates the pool: take a
        # round-robin over databases rather than a flat sample, which would
        # over-weight the databases that happen to carry more questions.
        by_db: dict[str, list[int]] = {}
        for i, row in enumerate(data):
            by_db.setdefault(row["db_id"], []).append(i)
        for ids in by_db.values():
            rng.shuffle(ids)
        ordered: list[int] = []
        cursors = {db: 0 for db in by_db}
        while len(ordered) < POOL_SIZE + TEST_SIZE:
            progressed = False
            for db in sorted(by_db):
                c = cursors[db]
                if c < len(by_db[db]):
                    ordered.append(by_db[db][c]); cursors[db] = c + 1; progressed = True
                    if len(ordered) >= POOL_SIZE + TEST_SIZE:
                        break
            if not progressed:
                break
        chosen = ordered[:POOL_SIZE] if split == "train" else ordered[POOL_SIZE:POOL_SIZE + TEST_SIZE]
        return [make_task_id("train", i) for i in sorted(chosen)]
    if split == "xdb":
        data = _mode_data(bird_path, "dev")
        # Stratify by difficulty so all three tiers are represented in
        # proportion; the per-tier breakdown is the dose-response measurement.
        by_diff: dict[str, list[int]] = {}
        for i, row in enumerate(data):
            by_diff.setdefault(row.get("difficulty") or "unknown", []).append(i)
        chosen: list[int] = []
        for tier, ids in by_diff.items():
            share = round(XDB_SIZE * len(ids) / len(data))
            chosen.extend(rng.sample(ids, min(share, len(ids))))
        return [make_task_id("dev", i) for i in sorted(chosen)]
    raise ValueError(f"unknown split {split!r}")


def run_task(
    env: SqlGymTaskEnv,
    task_id: str,
    client: ModelClient,
    *,
    memory_lookup: Callable[[str], tuple[str, list[dict[str, Any]]]] | None = None,
    memory_block: str = "",
    max_steps: int = 10,
) -> dict[str, Any]:
    """Drive one BIRD episode and return its canonical trajectory.

    The question is only known after reset, so retrieval is a callback, the
    same convention the other env-server benchmarks use.
    """
    mode, idx = parse_task_id(task_id)
    if env.mode != mode:
        raise ValueError(f"env is in mode {env.mode!r} but {task_id} needs {mode!r}")
    first = env.reset(idx)
    question = first["observation"]
    evidence = first["evidence"]

    selection: list[dict[str, Any]] = []
    if memory_lookup is not None:
        memory_block, selection = memory_lookup(question)

    user_message = question
    if evidence:
        user_message += f"\n\nHint: {evidence}"
    user_message += f"\n\nYou have {max_steps} turns."
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
    reward = 0.0
    submitted_sql = ""
    seen_queries: set[str] = set()

    try:
        for turn in range(max_steps):
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
                    "That is not a valid command. Reply with exactly one of:\n"
                    "  EXPLORE: <sql>\n  SUBMIT: <sql>\n"
                    "and nothing else."
                )
                steps.append({"index": len(steps), "role": "user", "content": nudge})
                messages.append({"role": "user", "content": nudge})
                continue
            ungrounded_turns = 0

            kind, sql = command
            # Re-running an identical query cannot produce new information,
            # and a stuck agent does exactly that until the budget is gone.
            # Say so instead of silently re-executing it.
            normalized = " ".join(sql.lower().split())
            if kind == "explore" and normalized in seen_queries:
                repeat = (
                    "You already ran that exact query and got the result above. "
                    "Running it again returns the same rows. Either narrow it "
                    "(WHERE / LIMIT / OFFSET / fewer columns) or SUBMIT your answer."
                )
                steps.append({"index": len(steps), "role": "user", "content": repeat})
                messages.append({"role": "user", "content": repeat})
                continue
            if kind == "explore":
                seen_queries.add(normalized)
            if kind == "submit":
                rendered, reward = env.submit(sql)
                submitted_sql = sql
                termination = "solved" if reward > 0 else "wrong_answer"
                steps.append({"index": len(steps), "role": "user",
                              "content": f"Submitted. Result:\n{rendered}"})
                break

            # On the last turn an EXPLORE is wasted -- the episode ends with
            # nothing submitted. Say so rather than letting it fall off the
            # end, which reads as an unforced loss in the termination mix.
            # The first probe run ended 4 of 6 episodes in `no_submission`:
            # the agent explored until the budget ran out. The system prompt
            # said turns were limited but never said how many, so it had
            # nothing to budget against. Showing the remaining count -- and
            # demanding a submission on the last turn -- makes the budget
            # something the agent can act on, rather than a hidden cliff.
            rendered = env.explore(sql)
            remaining = max_steps - turn - 1
            if remaining == 0:
                termination = "no_submission"
            elif remaining == 1:
                rendered += ("\n\nThis is your LAST turn. Reply with SUBMIT: <sql> now -- "
                             "an episode with no submission scores zero.")
            else:
                rendered += f"\n\n({remaining} turns remain.)"
            steps.append({"index": len(steps), "role": "user", "content": rendered})
            messages.append({"role": "user", "content": rendered})
    finally:
        env.close()

    return {
        "source_task_id": f"sqlgym.{task_id}",
        "domain": "sqlgym",
        "task": {"id": task_id, "instruction": question},
        "success": bool(reward > 0),
        "reward": reward,
        "termination_reason": termination,
        "evaluation": {
            "score": reward,
            "difficulty": first["difficulty"],
            "db_id": first["db_id"],
            "submitted_sql": submitted_sql,
        },
        "steps": steps,
        "usage": usage_totals,
        "retrieved_memory": selection,
    }
