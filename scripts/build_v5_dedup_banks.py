"""Rebuild v4's batch-0 candidate banks with v5's non-destructive forced refine.

Replays the writer-LLM outputs already recorded under cheap_train_v4 through
v5's dedup rule, so no LLM call is made. This isolates the dedup fix: the
memory *content* is identical to v4's, only the merge behaviour differs, which
makes any eval difference attributable to the fix alone.

Caveat, deliberately accepted: in a real v5 run the writer would see the merged
bank in its prompt and might write differently downstream. This is a controlled
A/B of the merge logic, not a full v5 run.

v6 note: the forced add->refine rule was removed from the pipeline (the writer
now picks its own operation and target -- see `router_bank_builder`'s module
docstring). This script reproduces V5 banks, so it keeps its own verbatim copy
of the rule below rather than importing one. That is the point: the historical
comparison stays runnable without keeping the rule alive in the code path that
actually builds banks today.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_memory_lab.alloc_writer_harness import (  # noqa: E402
    active_entries,
    apply_memory_operation,
)
from trajectory_memory_lab.alloc_writer_harness import (  # noqa: E402
    REFINE_TOPIC_OVERLAP_MIN,
    topic_overlap,
)
from trajectory_memory_lab.router_bank_builder import _retention  # noqa: E402

# Verbatim copy of v5's `_dedup_against_active_bank`, frozen here so this
# historical replay keeps working after v6 deleted it from the pipeline.
CONTENT_SUBSUMED_MIN = 0.9


def _union(first: list, second: list) -> list:
    merged = list(first)
    seen = {str(item) for item in first}
    for item in second:
        if str(item) not in seen:
            merged.append(item)
            seen.add(str(item))
    return merged


def _v5_dedup_against_active_bank(decision: dict, bank: list) -> None:
    memory = decision.get("memory")
    if not memory:
        return
    candidates = active_entries(bank)
    if not candidates:
        return
    best = max(candidates, key=lambda entry: topic_overlap(memory, entry))
    if topic_overlap(memory, best) < REFINE_TOPIC_OVERLAP_MIN:
        return
    decision["memory_operation"] = "refine"
    decision["target_memory_id"] = best["id"]

    old_text = (best.get("content") or "").strip()
    new_text = (memory.get("content") or "").strip()
    retention = _retention(old_text, new_text)
    decision["dedup_retention"] = retention
    if not old_text or old_text in new_text or retention >= CONTENT_SUBSUMED_MIN:
        decision["dedup_merged"] = False
        return
    memory["content"] = f"{old_text}\n{new_text}"
    memory["conditions"] = _union(best.get("conditions") or [], memory.get("conditions") or [])
    memory["exceptions"] = _union(best.get("exceptions") or [], memory.get("exceptions") or [])
    decision["dedup_merged"] = True

SRC = ROOT / "router_reward_v1/cheap_train_v4/iter1/b0"
OUT = ROOT / "router_reward_v1/dedup_fix_v5/b0"
V4_PASS = [0.4, 0.7, 0.4, 0.7, 0.7, 0.5, 0.4, 0.6]

changed = []
for k in range(8):
    bank: list = []
    merges = []
    for p in sorted(glob.glob(str(SRC / f"k{k}/records/appworld/*.json"))):
        rec = json.loads(Path(p).read_text())
        d = rec.get("decision") or {}
        if rec.get("status") != "committed" or not d.get("memory"):
            continue
        dd = {
            "route": d["route"],
            "memory_operation": "add",
            "target_memory_id": None,
            "memory": json.loads(json.dumps(d["memory"])),
        }
        _v5_dedup_against_active_bank(dd, bank)
        if dd.get("dedup_merged"):
            merges.append((rec["source_task_id"], dd["dedup_retention"]))
        apply_memory_operation(
            bank, dd, entry_id=f"router_appworld_{len(bank):03d}",
            source_task_id=rec["source_task_id"], rubric_id="router_v1",
        )
    out_dir = OUT / f"k{k}" / "banks"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memory_appworld.json").write_text(
        json.dumps(active_entries(bank), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "full_appworld.json").write_text(
        json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")

    v4_bank = json.loads((SRC / f"k{k}/banks/memory_appworld.json").read_text())
    v4_chars = sum(len(e["content"]) for e in v4_bank)
    v5_chars = sum(len(e["content"]) for e in active_entries(bank))
    tag = "CHANGED" if merges else "unchanged"
    if merges:
        changed.append(k)
    print(f"k{k}: v4 active={len(v4_bank)} ({v4_chars} chars)  "
          f"v5 active={len(active_entries(bank))} ({v5_chars} chars)  "
          f"v4_pass={V4_PASS[k]}  [{tag}]")
    for src, rt in merges:
        print(f"      merged at {src}: v4 discarded {(1-rt)*100:.0f}% of the target entry")

print()
print("candidates whose bank the fix changes:", changed)
print("unchanged candidates double as a determinism control (must reproduce v4 pass rate)")
