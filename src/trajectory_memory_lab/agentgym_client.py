"""Client for AgentGym's environment HTTP contract.

Fourteen AgentGym environments sit behind the same five endpoints
(`/create`, `/reset`, `/step`, `/observation`, `/close`), so this is written
against the contract rather than against any one environment. It was
extracted from `babyai_agent.BabyAIEnvClient` unchanged when TextCraft
became the second AgentGym environment in the suite; `BabyAIEnvClient`
remains as a subclass that only supplies BabyAI's default port, so the
BabyAI results already on disk are unaffected.

What the contract does NOT provide, and every caller has had to work around:

- **No `/info`.** There is no way to ask a server what world or split it is
  serving. The BabyAI probe originally tried to compare a server-side seed
  and crashed, because no such thing exists. The check that does work is to
  reset a task whose first observation is already recorded and require it to
  come back byte-identical -- see `assert_reproduces`.
- **Errors come back as a 200 with an `error` key**, not as an HTTP error,
  so `_call` raises on that key. Without it a failed reset reads as an empty
  observation and the episode silently runs against nothing.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class AgentGymEnvClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120.0,
                 env_var: str | None = None, default_url: str | None = None) -> None:
        resolved = base_url or (os.environ.get(env_var) if env_var else None) or default_url
        if not resolved:
            raise ValueError("no base_url, environment variable or default given")
        self.base_url = resolved.rstrip("/")
        self.timeout = timeout
        self.env_id: int | None = None

    def _call(self, path: str, body: dict[str, Any] | None = None, method: str = "POST") -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body or {}).encode() if method == "POST" else None
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(f"env server error on {path}: {payload['error']}")
        return payload

    def create(self, **body: Any) -> int:
        self.env_id = int(self._call("/create", body or None)["id"])
        return self.env_id

    def reset(self, data_idx: int) -> dict[str, Any]:
        if self.env_id is None:
            self.create()
        return self._call("/reset", {"id": self.env_id, "data_idx": data_idx})

    def step(self, action: str) -> dict[str, Any]:
        return self._call("/step", {"id": self.env_id, "action": action})

    def observation(self) -> Any:
        return self._call(f"/observation?id={self.env_id}", method="GET")

    def close(self) -> None:
        if self.env_id is not None:
            try:
                self._call("/close", {"id": self.env_id})
            except Exception:  # noqa: BLE001 - closing is best effort
                pass
            self.env_id = None

    def assert_reproduces(self, data_idx: int, expected_observation: str) -> None:
        """Require this server to rebuild a recorded task exactly.

        Every sft replay compares a fresh attempt against a stored one, which
        is only meaningful if the world is identical. Use this before a run
        that will replay recorded tasks.
        """
        live = (self.reset(data_idx).get("observation") or "").strip()
        want = (expected_observation or "").strip()
        if live != want:
            raise SystemExit(
                f"env server at {self.base_url} does not reproduce task {data_idx}:\n"
                f"  recorded: {want[:300]!r}\n  live:     {live[:300]!r}"
            )
