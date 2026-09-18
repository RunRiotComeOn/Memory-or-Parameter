"""CPU-only smoke test for the v6 removal of the forced add->refine rule.

Through v5, `_dedup_against_active_bank` rewrote every memory-writing decision
whose content resembled an active entry: operation forced to `refine`, target
chosen by a Jaccard threshold, content spliced. v6 deletes that and gives the
operation and target back to the writer, keeping only measurement
(`_duplicate_diagnostics`). This test pins down what that means:

  1. The diagnostic must NOT mutate the decision -- that is the whole point.
  2. It must still SEE the duplicate the old rule fired on, so "did crowding
     come back" stays answerable from the records.
  3. `refine_retention` must catch a destructive writer-chosen refine, since
     that safety moved from a rule into a prompt instruction and measurement
     is now the only thing standing behind it.
  4. Near-duplicate `add`s now really do accumulate. Asserted explicitly, so
     the cost of the removal is recorded rather than assumed away.

Run: PYTHONPATH=src .venv/bin/python scripts/smoke/smoke_nondestructive_refine.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_memory_lab.alloc_writer_harness import (  # noqa: E402
    active_entries,
    apply_memory_operation,
)
from trajectory_memory_lab.router_bank_builder import (  # noqa: E402
    _duplicate_diagnostics,
    _retention,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def commit(bank, memory, scope, task, *, operation="add", target=None):
    """One decision through the v6 path: diagnostics, then commit as written."""
    decision = {"route": "memory", "memory_operation": operation, "target_memory_id": target,
                "memory": {"content": memory, "scope": scope, "conditions": [], "exceptions": []}}
    _duplicate_diagnostics(decision, bank)
    entry = apply_memory_operation(bank, decision, entry_id=f"e{len(bank):03d}",
                                   source_task_id=task, rubric_id="t")
    return decision, entry


# --- 1. the diagnostic observes, it does not intervene ----------------------
print("\n1. same-topic complementary memories (the real v4 failure text)")
# The exact two memories v4 wrote at 229360a_1 and 229360a_2 in batch 0 / k0,
# where the second destroyed the first. Invented text is not a substitute:
# whether the old rule fired at all depends on the real Jaccard.
LOGIC = (
    "When cleaning up a library based on download status where 'album downloaded' is defined as "
    "'all songs in the album are downloaded', first retrieve the full list of downloaded songs. "
    "Then, for each album in the library, verify that every song ID in the album's song list "
    "exists in the downloaded set. Only retain the album if it is liked OR if this 'all songs "
    "downloaded' condition is met."
)
APIS = (
    "When tasked with cleaning up Spotify libraries (songs/albums) based on 'liked' or "
    "'downloaded' status, immediately search for and utilize `show_song_privates` to check "
    "individual song status (liked, downloaded, in_library) and `show_song_library`/"
    "`show_album_library` to retrieve library contents. Do not waste steps searching for generic "
    "'library' endpoints if specific 'privates' or 'library' endpoints are not immediately "
    "obvious; explicitly check `show_song_privates` for song status and `show_album` for album "
    "song lists to verify the 'all songs downloaded' condition."
)

bank: list = []
commit(bank, LOGIC, "spotify_library_cleanup", "229360a_1")
decision, _ = commit(bank, APIS, "spotify_library_cleanup", "229360a_2")

check("writer's `add` survives untouched", decision["memory_operation"] == "add")
check("no target was injected", decision["target_memory_id"] is None)
check("content was not rewritten", decision["memory"]["content"] == APIS)
check("but the duplicate is still SEEN", decision.get("dup_would_have_forced_refine") is True,
      f"overlap = {decision.get('dup_best_overlap', 0):.2f}")
check("both entries stay active (nothing was destroyed)", len(active_entries(bank)) == 2,
      f"{len(active_entries(bank))} active")


# --- 2. retention catches a destructive writer-chosen refine ----------------
print("\n2. refine_retention is the early warning that replaced the rule")
bank2: list = []
_, first = commit(bank2, LOGIC, "spotify_library_cleanup", "229360a_1")
bad, _ = commit(bank2, APIS, "spotify_library_cleanup", "229360a_2",
                operation="refine", target=first["id"])
check("a destructive refine is flagged", bad.get("refine_retention", 1.0) < 0.5,
      f"retention = {bad.get('refine_retention', 1.0):.2f}")
check("it still commits (measurement does not gate)", len(active_entries(bank2)) == 1)

bank3: list = []
_, first3 = commit(bank3, LOGIC, "spotify_library_cleanup", "229360a_1")
good, _ = commit(bank3, f"{LOGIC}\n{APIS}", "spotify_library_cleanup", "229360a_2",
                 operation="refine", target=first3["id"])
check("a content-preserving refine scores ~1.0", good.get("refine_retention", 0.0) == 1.0,
      f"retention = {good.get('refine_retention', 0.0):.2f}")


# --- 3. what removing the rule costs: duplicates now accumulate -------------
print("\n3. the cost of removal, asserted rather than assumed")
DUP = ("When searching Spotify artists by genre, the query parameter does not filter by genre, "
       "so verify the genre field of each result and filter manually.")
bank4: list = []
flagged = 0
for i in range(4):
    d, _ = commit(bank4, DUP + (" " * i), "spotify_artist_search", f"t{i}")
    flagged += bool(d.get("dup_would_have_forced_refine"))
check("4 near-duplicate adds now yield 4 active entries (v5 collapsed them to 1)",
      len(active_entries(bank4)) == 4, f"{len(active_entries(bank4))} active")
check("every one of them is flagged in the records", flagged == 3,
      f"{flagged} flagged (the first had an empty bank to compare against)")


# --- 4. blast radius on real recorded decisions -----------------------------
print("\n4. how much v5 was actually rewriting (real recorded runs, no LLM)")
for run in ("cheap_train_v5_nofeat", "cheap_train_v5"):
    paths = sorted(glob.glob(str(ROOT / f"router_reward_v1/{run}/iter1/b*/k*/records/appworld/*.json")))
    if not paths:
        print(f"  {run}: no records on disk, skipped")
        continue
    writes = would_rewrite = 0
    for path in paths:
        rec = json.loads(Path(path).read_text())
        d = rec.get("decision") or {}
        if rec.get("status") != "committed" or not d.get("memory"):
            continue
        writes += 1
        would_rewrite += bool(d.get("dup_would_have_forced_refine") or d.get("dedup_retention") is not None)
    if writes:
        print(f"  {run}: {would_rewrite}/{writes} committed memories "
              f"({would_rewrite / writes * 100:.0f}%) were or would have been rewritten by the v5 rule")

print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILED: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
