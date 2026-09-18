# Router + memory + SFT pipeline — current state (2026-09-18)

One-page map of how a training run actually works right now, after DESIGN.md
section 14's changes. DESIGN.md keeps the chronological narrative/motivation
for every decision; this file is the standing "what does it do today" summary
and should be kept in sync whenever the pipeline shape changes.

## 1. What's being learned

A single `nn.Linear(35, 4)` (`router_policy.RouterPolicy`) mapping a feature
vector to 4 route logits (`memory` / `sft` / `both` / `neither`). Everything
else in the pipeline exists to produce that feature vector, produce a reward
for whatever route gets sampled, and turn a GRPO-style policy gradient into
an update on this ~144-parameter model. No other model is trained by this
loop except, indirectly and now for real, the task agent itself (section 5).

## 2. Per-task decision loop (`router_bank_builder.run_router_chain`)

For each task, in order:

1. **Draft memory content** — `routed_writer_system("memory")`, always. One
   LLM call, produces a candidate memory entry from the trajectory + active
   bank, regardless of what route ends up chosen.
2. **Draft an SFT repair plan** — only if the task's base-agent attempt
   failed. A dedicated writer (`appworld_sft_writer.APPWORLD_SFT_WRITER_SYSTEM`,
   same base model, no separately-trained teacher) reads the failed
   trajectory and writes a natural-language plan: which APIs to call, in
   what order, what the original attempt got wrong.
3. **Build router features** (`router_policy.features_of`, 35-dim):
   - 3 numeric: base-agent success (0/1), base-agent reward, active bank
     size / 20.
   - 16-dim hashed bag-of-words of what changed in the bank over the last
     two batches (`_recent_changes_text`, via each entry's `created_position`).
   - 16-dim hashed bag-of-words of this task's own drafted content from
     steps 1-2 (`_draft_content_text`).
   Hashing is `md5(token) % 16`, deterministic, not learned, not an
   embedding model — the router itself stays a plain linear layer.
4. **Router picks a route** — `sample_action` (training, stochastic) or
   `greedy_action` (validation, argmax) over the 4 logits.
5. **Commit** — whichever of the two drafts the chosen route needs gets
   validated (`alloc_writer_harness.validate_alloc_decision`) and, for
   memory, deduplicated against the active bank
   (`_dedup_against_active_bank`, non-destructive merge) before being
   applied. The other draft is simply discarded. `neither` discards both.

## 3. Per-batch GRPO update (`train_router_selfreward.run_one_batch_update`)

- `K` candidates (default 8) run step 2's loop independently over the same
  ~10-task batch, all starting from the same canonical bank, split across
  two deterministic vLLM replicas for wall-clock parallelism.
- Each candidate's resulting bank is self-scored: replay the batch's own 10
  tasks with that bank attached, get `self_pass_rate`.
- **Advantage** (current, reverted from the K+1 experiment): plain
  group-relative, `advantage_k = pass_k - mean_j(pass_j)`. `self_baseline`
  (the same 10 tasks' original no-memory success rate, known for free) is
  computed and logged for visibility only — it does not reach the gradient.
  Known gap: this means the policy has no anchor against "doing nothing
  would have been better than all K candidates," only against its own
  siblings. Not fixed yet (DESIGN.md section 14.1/14.3 have the failed
  attempt and why it was reverted).
- `loss = -sum(advantage * logprob) - entropy_coef * sum(entropy)`, one
  optimizer step per batch. `entropy_coef` (default 0.01) exists to counter
  the route-distribution collapse documented in section 12.
- One candidate is picked **at random** (not the best-scoring one) as the
  canonical bank the next batch continues from, to avoid survivorship bias.

## 4. SFT actually gets trained now (`router_sft_pipeline`)

This is new as of section 14.4 — before it, `sft_plan` was written and never
consumed, so `sft` and `neither` were reward-equivalent no-ops.

1. From the batch's **chosen** candidate only (not all K — replaying every
   candidate's sft picks would multiply AppWorld-eval cost by K for no
   benefit), collect every committed `sft`/`both` decision's repair plan.
2. **Guided replay** (`run_appworld_guided_replay.py`): re-run the SAME task
   fresh, injecting the plan text into the initial prompt exactly the way
   `--memory-bank` retrieval already does (`memory_block`, plain text, no
   new mechanism). A live agent executes against the real environment —
   this is not a scripted replay of the plan's literal steps.
3. Only a replay AppWorld itself scores `success=True` becomes a training
   example, and the example is **that replay's own real transcript**, not
   the plan text. The plan-augmented first user message is stripped back to
   the plain task instruction before the example is kept
   (`appworld_sft_writer.training_messages`) — otherwise the model would
   train on a prompt shape it never sees again at inference.
4. Verified examples accumulate in `<OUTPUT_ROOT>/sft_pool.jsonl`. Every time
   the pool crosses a new multiple of 8, `router_sft_lora_update.sh` fires:
   - pause `det_server_b` (GPUs 2,3), LoRA-finetune (ms-swift, rank 8) on
     the **whole accumulated pool from scratch** (not incremental — avoids
     compounding drift from repeated small updates);
   - pause `det_server_a` too, merge the LoRA into the base weights on CPU
     (`merge_qwen_lora.py`, no GPU needed);
   - reload **both** replicas onto the merged checkpoint, same
     `served-model-name qwen35-tau`, same ports/TP/GPU layout as before.
   Both replicas must end up identical: they're treated as interchangeable
   for GRPO parallelism (candidates split across them by k-parity, unrelated
   to route sampling — section 11/13), so if only one replica got the
   fine-tuned weights, which replica a candidate happens to land on would
   silently decide whether its score reflects the trained agent at all.
   - on any failure or a 7200s timeout, the whole process group is killed
     and both replicas are forced back onto the **base** (untrained)
     checkpoint rather than being left down.

**Known gap, not fixed**: `self_baseline` (step 3's diagnostic, and the
number DESIGN.md section 14.2 flags) is computed once from the ORIGINAL
untrained agent's recorded success rate and never refreshed. Once an SFT
LoRA update actually fires, every subsequent batch's `self_pass_rate` comes
from a different (fine-tuned) agent than the one that produced
`self_baseline` — the comparison silently stops being apples-to-apples. This
has not caused a problem yet because it hasn't fired in a real run.

**Not yet run end-to-end.** The first time a real training run crosses the
8-example threshold is the first real test of steps 2-4 together.

## 5. Validation (`run_validation_pass`)

Once per iteration: build a bank greedily (argmax route, no exploration)
over all 90 train tasks, then run the REAL 57-task dev split through
whatever agent + bank that produces. This is the only number that should
ever be compared to a no-memory baseline — and only after confirming the
baseline was measured on the SAME replica/model state, per section 14.2's
correction (§13's cross-replica finding: absolute scores are not comparable
across different runs/replicas/model states, only within one).

## 6. What still isn't wired up

- `self_baseline` refresh after an SFT trigger (section 14.2/this doc's
  section 4 gap).
- Any weighting of the no-memory baseline into the GRPO advantage beyond
  logging (section 14.1's revert — reverted for being too weak, nothing
  better proposed yet).
- Cleanup of old `lora_work/merged` checkpoints across many trigger cycles
  (each is a full ~70GB Qwen3.5-35B checkpoint; disk has 4.8TB free as of
  2026-09-18, but this will matter eventually if training runs long).
