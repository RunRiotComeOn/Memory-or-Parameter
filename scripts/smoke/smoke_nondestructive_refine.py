"""CPU-only smoke test for the non-destructive forced-refine fix (v5).

Two things must both hold:
  1. Complementary entries (same topic, different information) must no longer
     destroy each other -- this is the v4 bug.
  2. Genuinely near-duplicate entries must STILL collapse to one active entry --
     this is what DESIGN.md section 10's dedup was added for, and the fix must
     not regress it.

Also replays the real recorded v4 decisions through the fixed logic to show
what would have changed, using no LLM and no GPU.

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
    _dedup_against_active_bank,
    _retention,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def commit(bank, memory, scope, task):
    decision = {"route": "memory", "memory_operation": "add", "target_memory_id": None,
                "memory": {"content": memory, "scope": scope, "conditions": [], "exceptions": []}}
    _dedup_against_active_bank(decision, bank)
    entry = apply_memory_operation(bank, decision, entry_id=f"e{len(bank):03d}",
                                   source_task_id=task, rubric_id="t")
    return decision, entry


# --- 1. the v4 bug: complementary entries on one topic ----------------------
print("\n1. complementary Spotify-library memories (the real v4 failure)")
# The exact two memories v4 wrote at 229360a_1 and 229360a_2 in batch 0 / k0,
# where the second destroyed the first and k0 then failed all three 229360a
# tasks (k4, which did not trigger a refine, passed all three). Invented text is
# not a substitute here: whether dedup fires at all depends on the real Jaccard.
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

check("dedup still fires on same-topic content", decision["memory_operation"] == "refine")
check("it is recognized as complementary, not a refinement", decision.get("dedup_merged") is True,
      f"retention of old text = {decision.get('dedup_retention', 0):.2f}")
survivors = active_entries(bank)
check("exactly one active entry (no duplicate crowding)", len(survivors) == 1, str(len(survivors)))
merged = survivors[0]["content"]
check("the task-logic memory SURVIVES", "Only retain the album if it is liked" in merged)
check("the API memory is also present", "show_song_privates" in merged)
check("nothing from the old entry is lost", _retention(LOGIC, merged) == 1.0,
      f"retention = {_retention(LOGIC, merged):.2f}")

# what v4 would have produced, for contrast
print(f"  v4 would have kept only: {APIS[:70]}...")
print(f"  v5 keeps both ({len(merged)} chars)")


# --- 2. must NOT regress DESIGN section 10's duplicate collapsing ------------
print("\n2. genuine near-duplicates must still collapse (section 10 regression guard)")
DUP = ("When searching Spotify artists by genre, the query parameter does not filter by genre, "
       "so verify the genre field of each result and filter manually.")
DUP2 = ("When searching Spotify artists by genre, the query parameter does not filter by genre; "
        "verify the genre field of each result and filter manually in code.")
bank2: list = []
commit(bank2, DUP, "spotify_artist_search", "07b42fd_1")
d2, _ = commit(bank2, DUP2, "spotify_artist_search", "07b42fd_2")
check("near-duplicate forced to refine", d2["memory_operation"] == "refine")
check("near-duplicate NOT merged (old text adds nothing)", d2.get("dedup_merged") is False,
      f"retention = {d2.get('dedup_retention', 0):.2f} >= 0.9")
check("still exactly one active entry", len(active_entries(bank2)) == 1)
check("no text duplication in the surviving entry",
      active_entries(bank2)[0]["content"].count("filter manually") == 1)

# section 10's own smoke case: 4 near-duplicate writes -> 1 active entry
bank3: list = []
for i in range(4):
    commit(bank3, DUP + (" " * i), "spotify_artist_search", f"t{i}")
check("4 near-duplicate writes still collapse to 1 active entry",
      len(active_entries(bank3)) == 1, f"{len(active_entries(bank3))} active")


# --- 3. replay the real v4 decisions through the fixed logic -----------------
print("\n3. replay of recorded v4 decisions (no LLM, no GPU)")
recs = [json.loads(l) for l in (ROOT / "router_reward_v1/cheap_train_v4/train_log.jsonl").open()]
chosen = {r["batch_idx"]: r["chosen_k"] for r in recs}
total = merged_n = 0
worst = []
for b in sorted(chosen):
    for k in range(8):
        bank_r: list = []
        if b > 0:
            seed_path = ROOT / f"router_reward_v1/cheap_train_v4/iter1/b{b-1}/k{chosen[b-1]}/banks/full_appworld.json"
            if seed_path.exists():
                bank_r = json.loads(seed_path.read_text())
        for p in sorted(glob.glob(str(ROOT / f"router_reward_v1/cheap_train_v4/iter1/b{b}/k{k}/records/appworld/*.json"))):
            rec = json.loads(Path(p).read_text())
            d = rec.get("decision") or {}
            if rec.get("status") != "committed" or not d.get("memory"):
                continue
            dd = {"route": d["route"], "memory_operation": "add", "target_memory_id": None,
                  "memory": dict(d["memory"])}
            _dedup_against_active_bank(dd, bank_r)
            if dd["memory_operation"] == "refine":
                total += 1
                if dd.get("dedup_merged"):
                    merged_n += 1
                    worst.append((dd["dedup_retention"], b, k, rec["source_task_id"]))
            apply_memory_operation(bank_r, dd, entry_id=f"router_appworld_{len(bank_r):03d}",
                                   source_task_id=rec["source_task_id"], rubric_id="router_v1")

print(f"  forced refines replayed: {total}")
print(f"  now content-preserving merges: {merged_n}  ({merged_n/total*100:.0f}%)")
print(f"  still plain replacements (old text subsumed): {total - merged_n}")
for rt, b, k, src in sorted(worst)[:5]:
    print(f"    b{b} k{k} {src}: v4 kept {rt*100:.0f}% of the old entry, v5 keeps 100%")
check("the fix changes real recorded behavior", merged_n > 0, f"{merged_n} merges")

print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILED: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
