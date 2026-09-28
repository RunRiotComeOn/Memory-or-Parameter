#!/usr/bin/env python3
"""Build WebShop's Lucene search index for the full catalog, outside the vendored checkout.

Replaces WebShop's own `search_engine/convert_product_file_format.py` +
`run_indexing.sh`, which cannot be used as shipped for the full catalog:
the converter reads `web_agent_site.utils.DEFAULT_FILE_PATH`, hardcoded to
the 1000-product `items_shuffle_1000.json`, and writes into the checkout. The
document text is built exactly as the converter builds it (title,
description, first bullet point, options -- lowercased), so search results
match stock WebShop's; only the paths differ, plus `--threads` for the
indexer.

Writes `<data-dir>/resources/documents.jsonl` and `<data-dir>/indexes/`, the
layout `scripts/webshop_env_server.py` expects.

  PYTHONPATH=third_party/WebShop /nas04/yixuh/webshop_venv/bin/python -u scripts/build_webshop_index.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/WebShop"))
# pyserini's Java bridge (pyjnius) looks for `javac` to locate a JDK unless
# JAVA_HOME is set, and this machine has only the Java 11 runtime -- which is
# all the bridge actually needs (libjvm.so).
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/nas04/yixuh/webshop_data"))
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    import web_agent_site.engine.engine as engine

    engine.DEFAULT_ATTR_PATH = str(args.data_dir / "items_ins_v2.json")
    engine.HUMAN_ATTR_PATH = str(args.data_dir / "items_human_ins.json")
    all_products, *_ = engine.load_products(filepath=str(args.data_dir / "items_shuffle.json"))

    resources = args.data_dir / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    with (resources / "documents.jsonl").open("w") as handle:
        for product in all_products:
            option_texts = [
                f"{name}: {', '.join(contents)}" for name, contents in product.get("options", {}).items()
            ]
            doc = {
                "id": product["asin"],
                "contents": " ".join([
                    product["Title"],
                    product["Description"],
                    product["BulletPoints"][0],
                    ", and ".join(option_texts),
                ]).lower(),
                "product": product,
            }
            handle.write(json.dumps(doc) + "\n")
    print(f"wrote {len(all_products)} documents to {resources / 'documents.jsonl'}", flush=True)

    subprocess.run([
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(resources),
        "--index", str(args.data_dir / "indexes"),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(args.threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ], check=True)
    print(f"index built at {args.data_dir / 'indexes'}", flush=True)


if __name__ == "__main__":
    main()
