#!/usr/bin/env python3
"""Build a persistent Chroma index from the pseudonymized text knowledge base."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from index_utils import (
    COLLECTION_NAME,
    EXPECTED_DIMENSION,
    MODEL_NAME,
    MODEL_URL,
    load_model,
    stable_chunk_id,
    write_json,
)


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DEFAULT_SOURCE = PROJECT_DIR.parent / "docs" / "knowledge_base"
DEFAULT_INDEX = PROJECT_DIR / "chroma_db"
DEFAULT_MANIFEST = PROJECT_DIR / "index_manifest.json"
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Replace the existing Chroma collection if it already exists.",
    )
    return parser.parse_args()


def first_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        title = line.strip().strip("=#-* ")
        if title:
            return title
    return fallback


def source_digest(files: list[Path], source_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def create_chunks(source_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    files = sorted(source_dir.glob("*.txt"), key=lambda item: item.name.casefold())
    if not files:
        raise FileNotFoundError(f"No .txt documents found in {source_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_WORDS,
        chunk_overlap=CHUNK_OVERLAP_WORDS,
        length_function=lambda value: len(value.split()),
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
        strip_whitespace=True,
    )

    chunks: list[dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        relative_source = f"docs/knowledge_base/{path.name}"
        title = first_title(text, path.stem)
        split_docs = splitter.create_documents(
            [text], metadatas=[{"source": relative_source, "title": title}]
        )
        total_chunks = len(split_docs)
        for zero_index, document in enumerate(split_docs):
            content = document.page_content.strip()
            start_char = int(document.metadata.get("start_index", 0))
            end_char = min(start_char + len(content), len(text))
            chunk_index = zero_index + 1
            word_start = len(text[:start_char].split())
            word_count = len(content.split())
            metadata = {
                "source": relative_source,
                "filename": path.name,
                "title": title,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "start_char": start_char,
                "end_char": end_char,
                "word_start": word_start,
                "word_end": word_start + word_count,
                "word_count": word_count,
            }
            chunks.append(
                {
                    "id": stable_chunk_id(relative_source, chunk_index, content),
                    "text": content,
                    "metadata": metadata,
                }
            )
    if not chunks:
        raise RuntimeError("All source documents were empty")
    return chunks, files


def package_versions() -> dict[str, str]:
    names = ("sentence-transformers", "chromadb", "langchain-text-splitters", "torch")
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    args = parse_args()
    source_dir = args.source.expanduser().resolve()
    index_dir = args.index_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    started = time.perf_counter()

    chunks, files = create_chunks(source_dir)
    chunking_finished = time.perf_counter()

    model_load_started = time.perf_counter()
    model, device = load_model(args.device)
    model_load_finished = time.perf_counter()
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embedding_finished = time.perf_counter()
    if embeddings.shape != (len(chunks), EXPECTED_DIMENSION):
        raise RuntimeError(f"Unexpected embeddings shape: {embeddings.shape}")

    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir))
    existing_names = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME in existing_names:
        if not args.rebuild:
            raise RuntimeError(
                f"Collection {COLLECTION_NAME!r} already exists; pass --rebuild to replace it"
            )
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": MODEL_NAME,
            "embedding_dimension": EXPECTED_DIMENSION,
        },
    )
    insert_batch_size = 500
    for offset in range(0, len(chunks), insert_batch_size):
        batch = chunks[offset : offset + insert_batch_size]
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            embeddings=embeddings[offset : offset + len(batch)].tolist(),
            documents=[chunk["text"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
        )
    indexing_finished = time.perf_counter()

    if collection.count() != len(chunks):
        raise RuntimeError(
            f"Index verification failed: stored {collection.count()}, expected {len(chunks)}"
        )

    word_counts = [chunk["metadata"]["word_count"] for chunk in chunks]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": MODEL_NAME,
            "url": MODEL_URL,
            "embedding_dimension": EXPECTED_DIMENSION,
            "normalized": True,
            "mode": "dense",
            "device": device,
        },
        "vector_database": {
            "type": "ChromaDB",
            "collection": COLLECTION_NAME,
            "distance_metric": "cosine",
            "path": "chroma_db",
        },
        "knowledge_base": {
            "path": "docs/knowledge_base",
            "document_count": len(files),
            "sha256": source_digest(files, source_dir),
        },
        "chunking": {
            "splitter": "RecursiveCharacterTextSplitter",
            "chunk_size_words": CHUNK_SIZE_WORDS,
            "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
            "chunk_count": len(chunks),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
            "mean_words": round(statistics.mean(word_counts), 2),
            "median_words": round(statistics.median(word_counts), 2),
        },
        "timing_seconds": {
            "chunking": round(chunking_finished - started, 3),
            "model_loading": round(model_load_finished - model_load_started, 3),
            "embedding_generation": round(embedding_finished - model_load_finished, 3),
            "index_writing": round(indexing_finished - embedding_finished, 3),
            "total": round(indexing_finished - started, 3),
        },
        "packages": package_versions(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)

    print(f"Indexed {len(chunks)} chunks from {len(files)} documents")
    print(f"Embedding model: {MODEL_NAME} ({EXPECTED_DIMENSION} dimensions, {device})")
    print(f"Chroma collection: {COLLECTION_NAME} at {index_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total time: {manifest['timing_seconds']['total']} seconds")


if __name__ == "__main__":
    main()
