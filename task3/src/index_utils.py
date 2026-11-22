"""Shared helpers for Task 3 index building and querying scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
MODEL_URL = "https://huggingface.co/BAAI/bge-m3"
EXPECTED_DIMENSION = 1024
COLLECTION_NAME = "quantumforge_knowledge_base"


def choose_device(requested: str) -> str:
    """Resolve ``auto`` to the fastest locally available SentenceTransformer device."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    # BGE-M3 currently triggers an MPSNDArray crash with some macOS/PyTorch
    # combinations. CPU is slower but is the reproducible default on macOS;
    # an explicitly requested ``--device mps`` is still passed through above.
    return "cpu"


def load_model(device: str = "auto") -> tuple[SentenceTransformer, str]:
    """Load the exact embedding model selected in task 1 and verify its dimension."""
    resolved_device = choose_device(device)
    model = SentenceTransformer(MODEL_NAME, device=resolved_device)
    dimension = model.get_embedding_dimension()
    if dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"Unexpected embedding dimension: {dimension}; expected {EXPECTED_DIMENSION}"
        )
    return model, resolved_device


def open_collection(index_dir: Path):
    """Open the persisted Chroma collection without a server process."""
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(name=COLLECTION_NAME)
    return client, collection


def stable_chunk_id(source: str, chunk_index: int, text: str) -> str:
    payload = f"{source}\0{chunk_index}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def result_rows(raw_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn Chroma's nested query result into citation-friendly records."""
    ids = (raw_result.get("ids") or [[]])[0]
    documents = (raw_result.get("documents") or [[]])[0]
    metadatas = (raw_result.get("metadatas") or [[]])[0]
    distances = (raw_result.get("distances") or [[]])[0]

    rows: list[dict[str, Any]] = []
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances, strict=True), start=1
    ):
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "score": round(1.0 - float(distance), 6),
                "distance": round(float(distance), 6),
                "text": document,
                "metadata": metadata,
                "citation": (
                    f"{metadata['source']} — {metadata['title']}, "
                    f"chunk {metadata['chunk_index']}/{metadata['total_chunks']}, "
                    f"chars {metadata['start_char']}–{metadata['end_char']}"
                ),
            }
        )
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
