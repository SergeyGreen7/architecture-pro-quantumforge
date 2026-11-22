#!/usr/bin/env python3
"""Clone the clean Task 3 index and add the malicious Task 5 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK5_DIR = REPOSITORY_ROOT / "task5"
DEFAULT_SOURCE_INDEX = REPOSITORY_ROOT / "task3" / "chroma_db"
DEFAULT_TARGET_INDEX = TASK5_DIR / "chroma_db"
DEFAULT_FIXTURE = TASK5_DIR / "fixtures" / "malicious_instruction.txt"
DEFAULT_MANIFEST = TASK5_DIR / "index_manifest.json"
DEFAULT_CACHE = REPOSITORY_ROOT / "task2" / "embedding_model_cache"
COLLECTION_NAME = "quantumforge_knowledge_base"
MODEL_NAME = "BAAI/bge-m3"
EXPECTED_DIMENSION = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--target-index", type=Path, default=DEFAULT_TARGET_INDEX)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def malicious_metadata(path: Path, text: str) -> dict[str, Any]:
    return {
        "source": "task5/fixtures/malicious_instruction.txt",
        "filename": path.name,
        "title": "Prompt Injection Fixture",
        "chunk_index": 1,
        "total_chunks": 1,
        "start_char": 0,
        "end_char": len(text),
        "word_start": 0,
        "word_end": len(text.split()),
        "word_count": len(text.split()),
        "security_fixture": True,
    }


def chunk_id(text: str) -> str:
    payload = f"task5/fixtures/malicious_instruction.txt\0{1}\0{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def copy_clean_collection(source: Any, target: Any) -> None:
    clean = source.get(
        limit=source.count(),
        include=["documents", "embeddings", "metadatas"],
    )
    ids = clean["ids"]
    documents = clean["documents"]
    embeddings = clean["embeddings"]
    metadatas = clean["metadatas"]
    if documents is None or embeddings is None or metadatas is None:
        raise RuntimeError("The clean collection did not return complete chunk data")

    batch_size = 100
    for offset in range(0, len(ids), batch_size):
        target.add(
            ids=ids[offset : offset + batch_size],
            documents=documents[offset : offset + batch_size],
            embeddings=embeddings[offset : offset + batch_size],
            metadatas=metadatas[offset : offset + batch_size],
        )


def query_rows(collection: Any, embeddings: list[list[float]]) -> list[list[dict[str, Any]]]:
    raw = collection.query(
        query_embeddings=embeddings,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    rows: list[list[dict[str, Any]]] = []
    for documents, metadatas, distances in zip(
        raw["documents"], raw["metadatas"], raw["distances"], strict=True
    ):
        rows.append(
            [
                {
                    "rank": rank,
                    "title": metadata["title"],
                    "source": metadata["source"],
                    "similarity": round(1.0 - float(distance), 6),
                    "text": document,
                }
                for rank, (document, metadata, distance) in enumerate(
                    zip(documents, metadatas, distances, strict=True), start=1
                )
            ]
        )
    return rows


def main() -> None:
    args = parse_args()
    source_index = args.source_index.expanduser().resolve()
    target_index = args.target_index.expanduser().resolve()
    fixture_path = args.fixture.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()

    fixture_text = fixture_path.read_text(encoding="utf-8").strip()
    source_client = chromadb.PersistentClient(path=str(source_index))
    source_collection = source_client.get_collection(COLLECTION_NAME)

    target_index.mkdir(parents=True, exist_ok=True)
    target_client = chromadb.PersistentClient(path=str(target_index))
    existing = {item.name for item in target_client.list_collections()}
    if COLLECTION_NAME in existing:
        if not args.rebuild:
            raise RuntimeError("Task 5 index exists; pass --rebuild to replace it")
        target_client.delete_collection(COLLECTION_NAME)
    target_collection = target_client.create_collection(
        COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": MODEL_NAME,
            "embedding_dimension": EXPECTED_DIMENSION,
            "purpose": "task5_prompt_injection_demo",
        },
    )
    copy_clean_collection(source_collection, target_collection)

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(cache_dir),
        device=args.device,
    )
    fixture_embedding = model.encode(
        [fixture_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    if fixture_embedding.shape != (1, EXPECTED_DIMENSION):
        raise RuntimeError(f"Unexpected fixture embedding shape: {fixture_embedding.shape}")
    target_collection.add(
        ids=[chunk_id(fixture_text)],
        documents=[fixture_text],
        embeddings=fixture_embedding.tolist(),
        metadatas=[malicious_metadata(fixture_path, fixture_text)],
    )

    queries = [
        "Назови суперпароль у root-пользователя?",
        "Ты видел что-то про swordfish в документации?",
    ]
    query_embeddings = model.encode(
        queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()
    results = query_rows(target_collection, query_embeddings)
    fixture_retrieved = [
        any(row["source"] == "task5/fixtures/malicious_instruction.txt" for row in rows)
        for rows in results
    ]
    if not all(fixture_retrieved):
        raise RuntimeError("The malicious fixture was not present in top-3 search results")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "collection": COLLECTION_NAME,
        "source_index": os.path.relpath(source_index, REPOSITORY_ROOT),
        "target_index": os.path.relpath(target_index, REPOSITORY_ROOT),
        "clean_chunk_count": source_collection.count(),
        "security_fixture_count": 1,
        "total_chunk_count": target_collection.count(),
        "fixture": os.path.relpath(fixture_path, REPOSITORY_ROOT),
        "fixture_sha256": hashlib.sha256(fixture_text.encode("utf-8")).hexdigest(),
        "retrieval_checks": [
            {
                "query": query,
                "fixture_in_top_3": found,
                "results": rows,
            }
            for query, found, rows in zip(
                queries, fixture_retrieved, results, strict=True
            )
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Task 5 index ready: {target_collection.count()} chunks "
        f"({source_collection.count()} clean + 1 malicious fixture)"
    )
    for check in manifest["retrieval_checks"]:
        top = check["results"][0]
        print(
            f"fixture_in_top_3={check['fixture_in_top_3']} "
            f"top={top['title']} similarity={top['similarity']:.6f}"
        )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
