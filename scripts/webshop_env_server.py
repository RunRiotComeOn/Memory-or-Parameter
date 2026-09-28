#!/usr/bin/env python3
"""Long-lived WebShop environment server: load the catalog once, serve many episodes.

Why a server, when ALFWorld/ScienceWorld run one env per task subprocess:
WebShop's `SimServer` loads the full 1.18M-product catalog and its Lucene
index on construction -- minutes and tens of GB, versus ~1 s for the other
two. Paying that per task (and per guided replay) is not viable, so one
process holds the world and every rollout/replay talks to it over HTTP
(`webshop_agent.WebShopEnvClient`). Clients therefore need nothing from
WebShop's dependency stack and run under the repo's own .venv; only this
script needs `webshop_venv`.

Every episode gets its own `WebAgentTextEnv` sharing the one `SimServer`
(the constructor's `server=` argument exists for exactly this), with a unique
`session_prefix`: `SimServer.user_sessions` is keyed by session string, and
without the prefix two concurrent episodes of the same goal index would share
one session's state. All env calls are serialized under one lock -- the
shared server mutates `user_sessions` and drives one JVM searcher, steps take
milliseconds, and the LLM calls on the client side dominate wall time anyway.

Determinism: stock WebShop is NOT reproducible across processes. Before any
seed is set, `load_products` draws a random price for every product listed
with a price range (`generate_product_prices`: `random.uniform`), and
`get_human_goals` draws every goal's price ceiling (`random.sample`); only
the later goal shuffle is seeded (233). So the same goal index could carry a
different price bound, and the same product a different price, in two
server processes. `--seed` (default 0) seeds `random` before construction,
which fixes both, and is reported by `/info` so a client can refuse to mix
runs from differently seeded worlds.

Paths: data and index live outside the vendored checkout
(`--data-dir`, default /nas04/yixuh/webshop_data, with `indexes/` inside it);
WebShop hardcodes both relative to its package, so they are patched in
before the server is built.

Endpoints (JSON over POST, except GET /info):
  GET  /info                 -> {num_goals, seed, num_products}
  POST /reset {goal_idx}     -> {env_id, instruction, observation, available}
  POST /step  {env_id, action}
                             -> {observation, reward, done, available, [purchase]}
  POST /close {env_id}       -> {}

  PYTHONPATH=third_party/WebShop /nas04/yixuh/webshop_venv/bin/python -u \\
      scripts/webshop_env_server.py --port 3100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/WebShop"))
# pyserini's Java bridge (pyjnius) looks for `javac` to locate a JDK unless
# JAVA_HOME is set, and this machine has only the Java 11 runtime -- which is
# all the bridge actually needs (libjvm.so).
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")

STATE: dict = {}
LOCK = threading.Lock()


def build_world(data_dir: Path, seed: int, num_products: int | None):
    from pyserini.search.lucene import LuceneSearcher

    import web_agent_site.engine.engine as engine
    import web_agent_site.envs.web_agent_text_env as text_env

    suffix = "" if num_products is None else f"_{num_products}"
    engine.DEFAULT_ATTR_PATH = str(data_dir / f"items_ins_v2{suffix}.json")
    engine.HUMAN_ATTR_PATH = str(data_dir / "items_human_ins.json")
    index_dir = data_dir / ("indexes" if num_products is None else f"indexes{suffix}")
    text_env.init_search_engine = lambda num_products=None: LuceneSearcher(str(index_dir))

    random.seed(seed)
    server = text_env.SimServer(
        "http://127.0.0.1:3000",
        str(data_dir / f"items_shuffle{suffix}.json"),
        filter_goals=None,
        limit_goals=-1,
        num_products=num_products,
        human_goals=1,
        show_attrs=False,
    )
    return server, text_env.WebAgentTextEnv


def available_actions(env) -> dict:
    available = env.get_available_actions()
    # The search button itself shows up as a clickable but `step` refuses to
    # click it (searching goes through `search[...]`), so it is not an action.
    clickables = [c for c in available["clickables"] if c != "search"]
    return {"has_search_bar": available["has_search_bar"], "clickables": clickables}


def handle(path: str, body: dict) -> dict:
    server = STATE["server"]
    envs = STATE["envs"]
    if path == "/reset":
        goal_idx = int(body["goal_idx"])
        if not 0 <= goal_idx < len(server.goals):
            raise ValueError(f"goal_idx {goal_idx} out of range [0, {len(server.goals)})")
        env_id = uuid.uuid4().hex[:12]
        env = STATE["env_cls"](
            observation_mode="text", server=server, session_prefix=f"{env_id}_",
        )
        # The constructor already reset once under a random session name;
        # drop that session so the shared server does not accumulate them.
        server.user_sessions.pop(env.session, None)
        observation, _ = env.reset(session=goal_idx)
        envs[env_id] = env
        return {
            "env_id": env_id,
            "instruction": env.instruction_text,
            "observation": observation,
            "available": available_actions(env),
        }
    if path == "/step":
        env = envs[body["env_id"]]
        observation, reward, done, _ = env.step(body["action"])
        result = {
            "observation": observation,
            "reward": float(reward),
            "done": bool(done),
            "available": available_actions(env),
        }
        if done:
            session = server.user_sessions[env.session]
            # Score components only -- never the goal's ASIN or attributes,
            # which would hand a writer the answer (see webshop_agent).
            result["purchase"] = {
                "asin": session.get("asin"),
                "options": session.get("options"),
                "reward_components": session.get("verbose_info"),
            }
        return result
    if path == "/close":
        env = envs.pop(body["env_id"], None)
        if env is not None:
            server.user_sessions.pop(env.session, None)
        return {}
    raise KeyError(path)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/info":
            self._send(404, {"error": self.path})
            return
        self._send(200, {
            "num_goals": len(STATE["server"].goals),
            "seed": STATE["seed"],
            "num_products": STATE["num_products"],
            "open_envs": len(STATE["envs"]),
        })

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with LOCK:
                result = handle(self.path, body)
            self._send(200, result)
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]})

    def log_message(self, *args) -> None:  # one line per step would drown the log
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3100)
    parser.add_argument("--data-dir", type=Path, default=Path("/nas04/yixuh/webshop_data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-products", type=int, default=None,
        help="omit for the full catalog (the standard setting); 1000 = WebShop's 'small' debug setting",
    )
    args = parser.parse_args()

    server, env_cls = build_world(args.data_dir, args.seed, args.num_products)
    STATE.update(server=server, env_cls=env_cls, envs={}, seed=args.seed, num_products=args.num_products)
    print(f"webshop env server: {len(server.goals)} goals, seed={args.seed}, "
          f"num_products={args.num_products or 'all'}, listening on :{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
