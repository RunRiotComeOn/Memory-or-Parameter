# Trajectory Memory Lab

This experiment uses the same OpenAI-compatible model for browser acting and a
tool-based retention pipeline:

1. execute a WebArena-style browser task and record the trajectory;
2. a routing-only controller chooses whether to invoke `edit_memory`,
   `build_sft_examples`, both, or neither; it cannot produce either artifact;
3. each invoked specialist independently reloads the authoritative task, full trajectory, environment
   feedback, verifier result, and relevant memory, then calls the model with its own prompt;
4. an optional second model pass audits each proposal before deterministic validation and commit.

Selected context entries are reusable external memories visible during later tasks of the
same kind. For SFT, the reviewer may retain at most one complete episode per trajectory. An
episode must cover every consecutive source state from the initial observation through one
terminal `finish` action; isolated middle actions are rejected. The program reconstructs one
multi-turn system/(environment user turn/browser assistant action)+ conversation and trains
all assistant actions while masking the environment turns.

## Setup

```bash
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pip install -e .
```

Start Qwen in one terminal:

```bash
bash scripts/serve_qwen.sh
```

Run eight read-only browser tasks in another terminal:

```bash
.venv/bin/trajectory-memory-lab --limit 8
```

Use `--base-url` and `--model` to point at any other OpenAI-compatible deployment. The task
file is ordinary JSON, so self-hosted WebArena URLs can replace the public read-only URLs
without changing experiment logic.

`edit_memory` can add, refine, or replace versioned memories. Previous revisions and evidence
step provenance remain in `memory_bank.json`; only active memory content and scope are rendered
to the browser agent. Legacy string-only memory banks are upgraded automatically when loaded.

`build_sft_examples` emits a complete recorded or corrected episode. Deterministic checks
verify full consecutive coverage, a single terminal finish, action schemas, required progress
notes, and page element identifiers. A complete recorded successful episode is written to
`sft_examples.jsonl`. A corrected or otherwise unverified complete episode is quarantined in
`sft_candidates.jsonl` with `validation_status=needs_replay` and is not used for training until
branch replay validates it.

Specialist output is recorded first as a candidate. It is distinct from the controller's
routing decision and from the artifact ultimately accepted after audit and deterministic
validation.

Each run writes trajectories, screenshots, the complete controller/tool/audit record in each
task's `retention_decision.json`, the accumulated memory bank, accepted SFT examples, replay
candidates, and a compact result summary under `runs/<UTC timestamp>/`.

Summarize the latest completed run with:

```bash
.venv/bin/python scripts/summarize_run.py
```

For a semantic audit of legacy single-reviewer runs that also treats a model-produced string
`"null"` as an omitted artifact and emits a cleaned SFT JSONL file:

```bash
.venv/bin/python scripts/audit_run.py runs/<UTC timestamp>
```

The model runs with thinking disabled so short structured browser actions do not spend the
token budget on hidden reasoning. The complete observations, actions, execution results,
screenshots, outputs, and usage counts are still retained in each trajectory.

Use `--no-tool-audit` to skip the specialist audit pass for a cheaper diagnostic run. The
controller owns whether `edit_memory` is called. Once called, the Memory Writer proposes one or
more concrete operations; the independent audit and deterministic validator may still reject all
of them. The SFT specialist may abstain when no complete episode is available.

## Memory Writer utility harness

`generate_tau_memory_writer_candidates.py` is a pre-routing utility-data collector. It samples
successful and failed source trajectories from the tau train split, generates multiple isolated
add/noop candidates, applies deterministic
evidence and identifier checks, and selects related plus low-overlap train-validation tasks.
`run_tau_memory_writer_replay.py` evaluates each accepted candidate with fixed BM25 retrieval,
reuses the existing no-memory base runs as paired controls, and writes harm-weighted utility and
preference pairs. The final tau test split is never used to create these training labels.
Candidates from one source share the same utility tasks: two related baseline failures provide
uplift opportunities, while one related and two scope-control baseline successes expose
regressions. Preference records include the common Writer input as well as chosen and rejected
outputs, so they can be converted to SFT or preference-training data without reconstructing the
source trajectory.

## Memory Writer warm-start SFT

Build SFT data from audited memory operations that also pass deterministic evidence and
application checks:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_tau_memory_writer_sft.py \
  --output training/tau_memory_writer_sft_v2

EPOCHS=3 MAX_LENGTH=13000 \
  bash scripts/train_tau_memory_writer_verl.sh \
  training/tau_memory_writer_sft_v2/train.parquet \
  training/tau_memory_writer_sft_v2_lora
```

The preparation step is conditioned on a correct controller call: writer SFT contains only
audited, executable add/refine/replace operations. Empty outcomes are excluded from writer SFT
and saved separately in `controller_routing_outcomes.jsonl` as candidate routing outcomes, not
trusted negative labels. They require separate controller evaluation because an old writer or
auditor failure can also produce an empty outcome. The preparation step keeps
cited evidence and refine targets visible while compacting long trajectories and creates a
source-disjoint positive validation split. Replace remains in the writer schema but is not used
as an SFT label until an audited, executable replace example is available. Downstream replay
utility is reserved for later preference or GRPO training.

## VERL action SFT

Convert an audited action JSONL file and train a Qwen3.5 LoRA with assistant-action-only loss:

```bash
.venv/bin/python scripts/prepare_verl_sft_data.py \
  runs/<UTC timestamp>/sft_action_examples.cleaned.jsonl \
  training/action_sft.parquet

MAX_LENGTH=16384 bash scripts/train_qwen_verl.sh \
  training/action_sft.parquet \
  training/qwen35_action_verl_lora
```

`src/trajectory_memory_lab/verl_dataset.py` applies the Qwen3.5 chat template to the complete
multi-turn conversation and derives a loss mask for every assistant action by prefix
difference. This avoids VERL's generic per-message tokenizer path, which is incompatible with
Qwen3.5's role-sequence validation.
