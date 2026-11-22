#!/usr/bin/env python3
"""Incrementally synchronize text documents into a persistent Chroma index."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TASK6_DIR = REPOSITORY_ROOT / "task6"
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "docs" / "knowledge_base"
DEFAULT_INDEX_DIR = TASK6_DIR / "chroma_db"
DEFAULT_STATE_PATH = TASK6_DIR / "state" / "index_state.json"
DEFAULT_LOG_PATH = TASK6_DIR / "storage" / "logs" / "index_updates.jsonl"
DEFAULT_LOCK_PATH = TASK6_DIR / "storage" / "update_index.lock"
DEFAULT_MODEL_CACHE = REPOSITORY_ROOT / "task2" / "embedding_model_cache"
COLLECTION_NAME = "quantumforge_knowledge_base"
MODEL_NAME = "BAAI/bge-m3"
EXPECTED_DIMENSION = 1024
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 30
STATE_VERSION = 1


class EmbeddingModel(Protocol):
    def get_embedding_dimension(self) -> int | None: ...

    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


ModelFactory = Callable[["UpdateConfig"], EmbeddingModel]


@dataclass(frozen=True, slots=True)
class UpdateConfig:
    source_dir: Path
    index_dir: Path
    state_path: Path
    log_path: Path
    lock_path: Path
    model_cache_dir: Path
    collection_name: str = COLLECTION_NAME
    model_name: str = MODEL_NAME
    device: str = "cpu"
    batch_size: int = 8
    rebuild: bool = False
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class SourceDocument:
    path: Path
    source: str
    sha256: str
    size_bytes: int
    mtime_ns: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the Task 6 collection and rebuild it from the current source.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report detected changes without modifying ChromaDB or state.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> UpdateConfig:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    return UpdateConfig(
        source_dir=args.source.expanduser().resolve(),
        index_dir=args.index_dir.expanduser().resolve(),
        state_path=args.state.expanduser().resolve(),
        log_path=args.log.expanduser().resolve(),
        lock_path=args.lock.expanduser().resolve(),
        model_cache_dir=args.model_cache.expanduser().resolve(),
        collection_name=args.collection,
        device=args.device,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
        dry_run=args.dry_run,
    )


def emit_log(path: Path, event: str, **fields: Any) -> dict[str, Any]:
    record = {"timestamp": utc_now(), "event": event, **fields}
    serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    print(serialized, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(serialized + "\n")
    return record


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another index update is already running") from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def document_source(path: Path, source_dir: Path) -> str:
    relative = path.relative_to(source_dir).as_posix()
    try:
        source_root = source_dir.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        source_root = source_dir.name
    return f"{source_root}/{relative}"


def discover_documents(source_dir: Path) -> dict[str, SourceDocument]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    files = sorted(
        (path for path in source_dir.rglob("*.txt") if path.is_file()),
        key=lambda path: path.relative_to(source_dir).as_posix().casefold(),
    )
    if not files:
        raise FileNotFoundError(f"no .txt documents found in {source_dir}")

    documents: dict[str, SourceDocument] = {}
    for path in files:
        content = path.read_bytes()
        stat = path.stat()
        source = document_source(path, source_dir)
        documents[source] = SourceDocument(
            path=path,
            source=source,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            mtime_ns=stat.st_mtime_ns,
        )
    return documents


def first_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        title = line.strip().strip("=#-* ")
        if title:
            return title
    return fallback


def stable_chunk_id(source: str, chunk_index: int, text: str) -> str:
    payload = f"{source}\0{chunk_index}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_document(document: SourceDocument) -> list[dict[str, Any]]:
    text = document.path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_WORDS,
        chunk_overlap=CHUNK_OVERLAP_WORDS,
        length_function=lambda value: len(value.split()),
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
        strip_whitespace=True,
    )
    title = first_title(text, document.path.stem)
    split_docs = splitter.create_documents(
        [text],
        metadatas=[{"source": document.source, "title": title}],
    )
    chunks: list[dict[str, Any]] = []
    total_chunks = len(split_docs)
    for zero_index, split_doc in enumerate(split_docs):
        content = split_doc.page_content.strip()
        start_char = int(split_doc.metadata.get("start_index", 0))
        end_char = min(start_char + len(content), len(text))
        chunk_index = zero_index + 1
        word_start = len(text[:start_char].split())
        word_count = len(content.split())
        chunks.append(
            {
                "id": stable_chunk_id(document.source, chunk_index, content),
                "text": content,
                "metadata": {
                    "source": document.source,
                    "filename": document.path.name,
                    "title": title,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "start_char": start_char,
                    "end_char": end_char,
                    "word_start": word_start,
                    "word_end": word_start + word_count,
                    "word_count": word_count,
                    "document_sha256": document.sha256,
                },
            }
        )
    return chunks


def empty_state(config: UpdateConfig) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "model": config.model_name,
        "collection": config.collection_name,
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "documents": {},
    }


def load_state(config: UpdateConfig) -> dict[str, Any]:
    if config.rebuild or not config.state_path.exists():
        return empty_state(config)
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    expected = {
        "version": STATE_VERSION,
        "model": config.model_name,
        "collection": config.collection_name,
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
    }
    mismatches = [key for key, value in expected.items() if state.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "index settings changed; run with --rebuild (mismatch: "
            + ", ".join(mismatches)
            + ")"
        )
    if not isinstance(state.get("documents"), dict):
        raise RuntimeError("invalid state: documents must be an object")
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_embedding_model(config: UpdateConfig) -> EmbeddingModel:
    config.model_cache_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(
        config.model_name,
        cache_folder=str(config.model_cache_dir),
        device=config.device,
    )
    dimension = model.get_embedding_dimension()
    if dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"unexpected embedding dimension {dimension}; expected {EXPECTED_DIMENSION}"
        )
    return model


def collection_for_update(
    config: UpdateConfig,
    *,
    state_exists: bool,
) -> tuple[Any, Any]:
    config.index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.index_dir))
    existing = {item.name for item in client.list_collections()}
    if config.rebuild and config.collection_name in existing:
        client.delete_collection(config.collection_name)
        existing.remove(config.collection_name)
    if config.collection_name not in existing:
        collection = client.create_collection(
            config.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": config.model_name,
                "embedding_dimension": EXPECTED_DIMENSION,
                "managed_by": "task6/update_index.py",
            },
        )
    else:
        collection = client.get_collection(config.collection_name)
        if not state_exists and collection.count() and not config.rebuild:
            raise RuntimeError(
                "collection contains data but state file is missing; use --rebuild"
            )
    return client, collection


def update_index(
    config: UpdateConfig,
    *,
    model_factory: ModelFactory = load_embedding_model,
) -> dict[str, Any]:
    source_documents = discover_documents(config.source_dir)
    state_existed = config.state_path.exists() and not config.rebuild
    state = load_state(config)
    old_documents: dict[str, dict[str, Any]] = state["documents"]

    current_sources = set(source_documents)
    previous_sources = set(old_documents)
    new_sources = sorted(current_sources - previous_sources)
    deleted_sources = sorted(previous_sources - current_sources)
    changed_sources = sorted(
        source
        for source in current_sources & previous_sources
        if source_documents[source].sha256 != old_documents[source].get("sha256")
    )
    unchanged_sources = sorted(
        current_sources - set(new_sources) - set(changed_sources)
    )

    prepared_chunks: dict[str, list[dict[str, Any]]] = {}
    for source in [*new_sources, *changed_sources]:
        prepared_chunks[source] = split_document(source_documents[source])

    old_ids_to_delete = [
        chunk_id
        for source in [*changed_sources, *deleted_sources]
        for chunk_id in old_documents[source].get("chunk_ids", [])
    ]
    new_chunks = [chunk for source in new_sources for chunk in prepared_chunks[source]]
    updated_chunks = [
        chunk for source in changed_sources for chunk in prepared_chunks[source]
    ]
    chunks_to_upsert = [*new_chunks, *updated_chunks]

    next_documents: dict[str, dict[str, Any]] = {
        source: old_documents[source] for source in unchanged_sources
    }
    for source in [*new_sources, *changed_sources]:
        document = source_documents[source]
        chunks = prepared_chunks[source]
        next_documents[source] = {
            "sha256": document.sha256,
            "size_bytes": document.size_bytes,
            "mtime_ns": document.mtime_ns,
            "chunk_count": len(chunks),
            "chunk_ids": [chunk["id"] for chunk in chunks],
        }

    expected_count = sum(
        int(document["chunk_count"]) for document in next_documents.values()
    )
    changes_detected = bool(new_sources or changed_sources or deleted_sources)
    metrics: dict[str, Any] = {
        "source_documents": len(source_documents),
        "new_documents": len(new_sources),
        "changed_documents": len(changed_sources),
        "deleted_documents": len(deleted_sources),
        "unchanged_documents": len(unchanged_sources),
        "new_document_names": new_sources,
        "changed_document_names": changed_sources,
        "deleted_document_names": deleted_sources,
        "new_chunks": len(new_chunks),
        "updated_chunks": len(updated_chunks),
        "removed_chunks": len(old_ids_to_delete),
        "embeddings_generated": len(chunks_to_upsert),
        "changes_detected": changes_detected,
        "dry_run": config.dry_run,
        "final_index_size": expected_count,
    }
    if config.dry_run:
        return metrics

    _, collection = collection_for_update(config, state_exists=state_existed)
    if old_ids_to_delete:
        collection.delete(ids=old_ids_to_delete)

    if chunks_to_upsert:
        model = model_factory(config)
        dimension = model.get_embedding_dimension()
        if dimension != EXPECTED_DIMENSION:
            raise RuntimeError(
                f"unexpected embedding dimension {dimension}; expected {EXPECTED_DIMENSION}"
            )
        embeddings = model.encode(
            [chunk["text"] for chunk in chunks_to_upsert],
            batch_size=config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if embeddings.shape != (len(chunks_to_upsert), EXPECTED_DIMENSION):
            raise RuntimeError(f"unexpected embeddings shape: {embeddings.shape}")
        insert_batch_size = 500
        for offset in range(0, len(chunks_to_upsert), insert_batch_size):
            batch = chunks_to_upsert[offset : offset + insert_batch_size]
            collection.upsert(
                ids=[chunk["id"] for chunk in batch],
                embeddings=embeddings[offset : offset + len(batch)].tolist(),
                documents=[chunk["text"] for chunk in batch],
                metadatas=[chunk["metadata"] for chunk in batch],
            )

    actual_count = collection.count()
    if actual_count != expected_count:
        raise RuntimeError(
            f"index verification failed: stored {actual_count}, expected {expected_count}"
        )
    state.update(
        {
            "source_dir": str(config.source_dir),
            "index_dir": str(config.index_dir),
            "last_success_at": utc_now(),
            "index_chunk_count": actual_count,
            "documents": dict(sorted(next_documents.items())),
        }
    )
    write_state(config.state_path, state)
    metrics["final_index_size"] = actual_count
    return metrics


def main() -> None:
    run_id = str(uuid.uuid4())
    started_at = utc_now()
    started = time.perf_counter()
    try:
        config = config_from_args(parse_args())
    except Exception as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    emit_log(
        config.log_path,
        "index_update_started",
        run_id=run_id,
        started_at=started_at,
        source=str(config.source_dir),
        index=str(config.index_dir),
        rebuild=config.rebuild,
        dry_run=config.dry_run,
    )
    try:
        with exclusive_lock(config.lock_path):
            metrics = update_index(config)
        emit_log(
            config.log_path,
            "index_update_completed",
            run_id=run_id,
            started_at=started_at,
            completed_at=utc_now(),
            duration_seconds=round(time.perf_counter() - started, 3),
            error_count=0,
            **metrics,
        )
    except Exception as error:
        emit_log(
            config.log_path,
            "index_update_failed",
            run_id=run_id,
            started_at=started_at,
            completed_at=utc_now(),
            duration_seconds=round(time.perf_counter() - started, 3),
            error_count=1,
            error_type=type(error).__name__,
            error=str(error),
        )
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
