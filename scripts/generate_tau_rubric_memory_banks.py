#!/usr/bin/env python3
"""Generate four train-only memory banks under fixed writer rubrics."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path
from typing import Any

from generate_tau_memory_writer_candidates import (
    RUBRIC_MEMORY_WRITER_SYSTEM,
    _domain_data,
    _trajectory,
)
from trajectory_memory_lab.memory_writer_harness import (
    normalize_writer_candidate,
    validate_writer_candidate,
)
from trajectory_memory_lab.model_client import ModelClient
from trajectory_memory_lab.writer_rubrics import RUBRIC_IDS, rubric_block


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("airline", "retail", "telecom")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)

    train = {
        (domain, str(task_id))
        for domain in DOMAINS
        for task_id in manifest["rubric_train"][domain]["task_ids"]
    }
    forbidden = {
        (domain, str(task_id))
        for split in ("dev", "test")
        for domain in DOMAINS
        for task_id in manifest[split][domain]["task_ids"]
    }
    if train & forbidden:
        raise ValueError("train/dev/test leakage before memory generation")

    domain_data = {
        domain: _domain_data(ROOT / "third_party/tau2-bench/data/simulations", domain)
        for domain in DOMAINS
    }
    jobs = []
    ordinal = 0
    for rubric_id in RUBRIC_IDS:
        for domain in DOMAINS:
            data = domain_data[domain]
            for task_id in map(str, manifest["rubric_train"][domain]["task_ids"]):
                trajectory = _trajectory(
                    domain,
                    data["tasks"][task_id],
                    data["simulations"][task_id],
                    data["policy"],
                )
                jobs.append((ordinal, rubric_id, domain, task_id, trajectory))
                ordinal += 1

    def run_one(job: tuple[int, str, str, str, dict[str, Any]]) -> str:
        ordinal, rubric_id, domain, task_id, trajectory = job
        record_path = args.output / "candidates" / rubric_id / f"{ordinal:03d}_{domain}.json"
        if record_path.exists():
            existing = read_json(record_path)
            if existing.get("status") in {"accepted", "rejected", "error"}:
                return f"resume-skip {rubric_id} {domain}.{task_id} {existing['status']}"
        client = ModelClient(
            base_url=args.base_url,
            api_key="EMPTY",
            model=args.model,
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_tokens,
            seed=args.seed + ordinal,
            enable_thinking=False,
            timeout=args.timeout,
        )
        reply = None
        try:
            reply = client.json_chat(
                system=RUBRIC_MEMORY_WRITER_SYSTEM + rubric_block("memory", rubric_id),
                user=json.dumps(
                    {
                        "current_memory": [],
                        "trajectory": trajectory,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            candidate = normalize_writer_candidate(reply.parsed)
            validation = validate_writer_candidate(candidate, trajectory)
            status = "accepted" if validation["accepted"] else "rejected"
            record = {
                "protocol": "tau_rubric_memory_candidate_v1",
                "rubric_id": rubric_id,
                "domain": domain,
                "source_task_id": task_id,
                "candidate_id": f"{rubric_id}_{domain}_{ordinal:03d}",
                "candidate": candidate,
                "hard_validation": validation,
                "status": status,
                "usage": reply.usage,
            }
        except Exception as exc:
            record = {
                "protocol": "tau_rubric_memory_candidate_v1",
                "rubric_id": rubric_id,
                "domain": domain,
                "source_task_id": task_id,
                "candidate_id": f"{rubric_id}_{domain}_{ordinal:03d}",
                "status": "error",
                "error": repr(exc),
                "raw_prediction": reply.parsed if reply is not None else None,
                "usage": reply.usage if reply is not None else None,
            }
        write_json(record_path, record)
        return f"{rubric_id} {domain}.{task_id} {record['status']}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            print(f"[{index}/{len(jobs)}] {future.result()}", flush=True)

    records = [
        read_json(path)
        for path in sorted((args.output / "candidates").glob("*/*.json"))
    ]
    if len(records) != len(jobs):
        raise ValueError(f"memory candidate count mismatch: {len(records)}/{len(jobs)}")
    banks = {}
    for rubric_id in RUBRIC_IDS:
        banks[rubric_id] = {}
        for domain in DOMAINS:
            entries = []
            for record in records:
                if record["rubric_id"] != rubric_id or record["domain"] != domain:
                    continue
                if record.get("status") != "accepted":
                    continue
                memory = record["candidate"]["memory"]
                entries.append(
                    {
                        "id": record["candidate_id"],
                        "scope": memory["scope"],
                        "content": memory["content"],
                        "conditions": memory.get("conditions") or [],
                        "exceptions": memory.get("exceptions") or [],
                        "status": "active",
                        "source_task_id": record["source_task_id"],
                        "rubric_id": rubric_id,
                    }
                )
            path = args.output / "banks" / rubric_id / f"memory_{domain}.json"
            write_json(path, entries)
            banks[rubric_id][domain] = len(entries)
    summary = {
        "protocol": "tau_rubric_memory_banks_v1",
        "source_split": "rubric_train",
        "source_tasks": len(train),
        "dev_overlap": 0,
        "test_overlap": 0,
        "candidate_status": dict(Counter(record["status"] for record in records)),
        "bank_entries": banks,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
