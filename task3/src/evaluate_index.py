#!/usr/bin/env python3
"""Run reproducible smoke tests and save representative search results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from index_utils import MODEL_NAME, load_model, open_collection, write_json
from search_index import search


PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_CASES = [
    {
        "query": "Как Аркон Вейр путешествует между измерениями и какое устройство использует?",
        "expected_titles": ["Аркон Вейр"],
    },
    {
        "query": "Кто такие Зерониды и какую роль они играют во Вселенной?",
        "expected_titles": ["Зерониды"],
    },
    {
        "query": "Кто такой Тёмный Лио и как он пришёл к власти?",
        "expected_titles": ["Тёмный Лио"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=PROJECT_DIR / "chroma_db")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "test_queries.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, device = load_model(args.device)
    _, collection = open_collection(args.index_dir.expanduser().resolve())
    tests = []
    for case in TEST_CASES:
        rows = search(case["query"], model=model, collection=collection, top_k=args.top_k)
        returned_titles = [row["metadata"]["title"] for row in rows]
        passed = any(title in returned_titles for title in case["expected_titles"])
        tests.append({**case, "passed": passed, "results": rows})
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {case['query']} -> {returned_titles}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "device": device,
        "top_k": args.top_k,
        "passed": sum(test["passed"] for test in tests),
        "total": len(tests),
        "tests": tests,
    }
    write_json(args.output.expanduser().resolve(), report)
    print(f"Saved: {args.output}")
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
