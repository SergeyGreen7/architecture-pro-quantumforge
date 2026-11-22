#!/usr/bin/env python3
"""Build an evaluation index with three intentionally removed entities."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TASK7_DIR = REPOSITORY_ROOT / "task7"
DEFAULT_SOURCE_INDEX = REPOSITORY_ROOT / "task6" / "chroma_db"
DEFAULT_TARGET_INDEX = TASK7_DIR / "chroma_db"
DEFAULT_MANIFEST = TASK7_DIR / "gap_manifest.json"
COLLECTION_NAME = "quantumforge_knowledge_base"
GAP_TITLES = ("Куратор Эйдос", "Союз Зенор", "Криос")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--target-index", type=Path, default=DEFAULT_TARGET_INDEX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def batches(items: list[Any], size: int = 100) -> list[list[Any]]:
    return [items[offset : offset + size] for offset in range(0, len(items), size)]


def main() -> None:
    args = parse_args()
    source_index = args.source_index.expanduser().resolve()
    target_index = args.target_index.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()

    source_client = chromadb.PersistentClient(path=str(source_index))
    source = source_client.get_collection(args.collection)
    raw = source.get(
        limit=source.count(),
        include=["documents", "embeddings", "metadatas"],
    )
    ids = raw["ids"]
    documents = raw["documents"]
    embeddings = raw["embeddings"]
    metadatas = raw["metadatas"]
    if documents is None or embeddings is None or metadatas is None:
        raise RuntimeError("source collection did not return complete data")

    removed: list[dict[str, Any]] = []
    kept: list[tuple[str, str, list[float], dict[str, Any]]] = []
    for chunk_id, document, embedding, metadata in zip(
        ids, documents, embeddings, metadatas, strict=True
    ):
        if metadata["title"] in GAP_TITLES:
            removed.append(
                {
                    "id": chunk_id,
                    "title": metadata["title"],
                    "source": metadata["source"],
                    "chunk_index": metadata["chunk_index"],
                }
            )
        else:
            kept.append((chunk_id, document, embedding, metadata))

    target_index.mkdir(parents=True, exist_ok=True)
    target_client = chromadb.PersistentClient(path=str(target_index))
    existing = {item.name for item in target_client.list_collections()}
    if args.collection in existing:
        if not args.rebuild:
            raise RuntimeError("Task 7 index exists; pass --rebuild to replace it")
        target_client.delete_collection(args.collection)
    target = target_client.create_collection(
        args.collection,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimension": 1024,
            "purpose": "task7_coverage_gap_evaluation",
        },
    )

    for batch in batches(kept):
        target.add(
            ids=[row[0] for row in batch],
            documents=[row[1] for row in batch],
            embeddings=[row[2] for row in batch],
            metadatas=[row[3] for row in batch],
        )

    removed_by_title = {
        title: sum(item["title"] == title for item in removed) for title in GAP_TITLES
    }
    if any(count == 0 for count in removed_by_title.values()):
        raise RuntimeError(f"at least one requested gap was absent: {removed_by_title}")
    for title in GAP_TITLES:
        if target.get(where={"title": title})["ids"]:
            raise RuntimeError(f"gap verification failed for {title}")

    expected_count = source.count() - len(removed)
    if target.count() != expected_count:
        raise RuntimeError(
            f"index count mismatch: stored {target.count()}, expected {expected_count}"
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_index": str(source_index),
        "target_index": str(target_index),
        "collection": args.collection,
        "source_chunk_count": source.count(),
        "removed_entities": list(GAP_TITLES),
        "removed_chunks_by_entity": removed_by_title,
        "removed_chunk_count": len(removed),
        "final_chunk_count": target.count(),
        "removed_chunks": removed,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
