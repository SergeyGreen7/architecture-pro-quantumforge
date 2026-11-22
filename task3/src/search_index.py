#!/usr/bin/env python3
"""Run semantic search against the persisted QuantumForge Chroma index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from index_utils import load_model, open_collection, result_rows


PROJECT_DIR = Path(__file__).resolve().parent.parent


def search(
    query: str,
    *,
    model: Any,
    collection: Any,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    query_embedding = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )
    raw_result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return result_rows(raw_result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language query in Russian or another language")
    parser.add_argument("--index-dir", type=Path, default=PROJECT_DIR / "chroma_db")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _ = load_model(args.device)
    _, collection = open_collection(args.index_dir.expanduser().resolve())
    rows = search(args.query, model=model, collection=collection, top_k=args.top_k)

    if args.json:
        print(json.dumps({"query": args.query, "results": rows}, ensure_ascii=False, indent=2))
        return

    print(f"Query: {args.query}")
    for row in rows:
        excerpt = " ".join(row["text"].split())
        if len(excerpt) > 400:
            excerpt = excerpt[:397].rstrip() + "..."
        print(f"\n{row['rank']}. score={row['score']:.4f}")
        print(f"   {row['citation']}")
        print(f"   {excerpt}")


if __name__ == "__main__":
    main()
