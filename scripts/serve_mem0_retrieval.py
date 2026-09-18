"""Sidecar exposing mem0 add/search over HTTP, for callers that cannot import it.

Why this exists: `run_appworld_rollout.py` runs in `appworld_venv`, which is
pinned to pydantic 1.10 / SQLAlchemy 1.4, while mem0 needs pydantic 2 /
SQLAlchemy 2. Installing mem0 there would upgrade pydantic across the 1->2
break and take the AppWorld harness with it. So mem0 lives only in the project
`.venv` and the eval process talks to it over localhost.

Stdlib only on purpose (http.server, json) so the sidecar adds no dependency of
its own, and the client side is urllib, so `appworld_venv` stays untouched.

Run: PYTHONPATH=src .venv/bin/python scripts/serve_mem0_retrieval.py --port 8020
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_memory_lab import mem0_store  # noqa: E402

_LOCK = threading.Lock()
_CACHE: dict[str, object] = {}
ARGS: argparse.Namespace


def _memory(store: str):
    """One Memory per store path. Cached because building it loads the ONNX
    embedder, and serialized because a candidate's store is a single on-disk
    Qdrant that must not be opened concurrently."""
    with _LOCK:
        if store not in _CACHE:
            _CACHE[store] = mem0_store.build_memory(store, ARGS.base_url, ARGS.model)
        return _CACHE[store]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet; the eval log is the record
        pass

    def _reply(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {"ok": True, "stores": len(_CACHE)})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._reply(400, {"error": f"bad json: {exc}"})
            return
        store = req.get("store")
        if not store:
            self._reply(400, {"error": "missing 'store'"})
            return
        try:
            memory = _memory(store)
            if self.path == "/search":
                threshold = req.get("threshold")
                ranked = mem0_store.search_entries(
                    memory, req.get("query") or "", int(req.get("top_k") or 0),
                    threshold=float(threshold) if threshold is not None else None,
                )
                self._reply(200, {"results": [
                    {"entry": entry, "score": score} for entry, score in ranked
                ]})
            elif self.path == "/add":
                result = mem0_store.add_entry(
                    memory, req.get("content") or "",
                    scope=req.get("scope") or "",
                    source_task_id=req.get("source_task_id") or "",
                )
                self._reply(200, {"result": result})
            elif self.path == "/export":
                self._reply(200, {"entries": mem0_store.export_entries(memory)})
            else:
                self._reply(404, {"error": "not found"})
        except Exception as exc:  # surface, never silently return empty
            self._reply(500, {"error": repr(exc)})


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-url", default="http://127.0.0.1:8012/v1")
    parser.add_argument("--model", default="qwen35-tau")
    ARGS = parser.parse_args()
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"mem0 retrieval sidecar on http://{ARGS.host}:{ARGS.port} "
          f"(llm={ARGS.base_url}, model={ARGS.model})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
