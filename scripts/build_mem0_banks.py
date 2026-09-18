"""Rebuild v4's batch-0 candidate banks in mem0, via the sidecar.

Replays the writer-LLM content already recorded under cheap_train_v4 into a
mem0 store per candidate. The memory *text* is therefore identical to v4's and
v5-dedup's; only the storage, merge and retrieval backend differs, so an eval
difference is attributable to the backend swap.

Same deliberate caveat as build_v5_dedup_banks.py: a real mem0 run would let the
writer see mem0's own bank when composing the next memory, so this is a
controlled backend A/B, not a full end-to-end mem0 run.
"""

from __future__ import annotations

import argparse
import glob
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "router_reward_v1/cheap_train_v4/iter1/b0"
OUT = ROOT / "router_reward_v1/mem0_v6/b0"
SIDECAR = "http://127.0.0.1:8020"


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{SIDECAR}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.loads(response.read())
    if "error" in body:
        raise RuntimeError(f"sidecar error on {path}: {body['error']}")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, nargs="+", default=[0, 2, 4])
    args = parser.parse_args()

    for k in args.candidates:
        store = OUT / f"k{k}" / "store"
        store.mkdir(parents=True, exist_ok=True)
        events: list[str] = []
        for path in sorted(glob.glob(str(SRC / f"k{k}/records/appworld/*.json"))):
            record = json.loads(Path(path).read_text())
            decision = record.get("decision") or {}
            memory = decision.get("memory")
            if record.get("status") != "committed" or not memory:
                continue
            result = post("/add", {
                "store": str(store),
                "content": memory.get("content") or "",
                "scope": memory.get("scope") or "",
                "source_task_id": record["source_task_id"],
            })
            for item in (result.get("result") or {}).get("results") or []:
                events.append(f"{item.get('event')}@{record['source_task_id']}")
        entries = post("/export", {"store": str(store)}).get("entries") or []
        (OUT / f"k{k}" / "bank_export.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"k{k}: {len(entries)} entries   events={events}")
        for entry in entries:
            print(f"      - {entry['content'][:110]}")


if __name__ == "__main__":
    main()
