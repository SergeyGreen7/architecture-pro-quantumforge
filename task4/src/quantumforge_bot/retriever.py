"""Local BGE-M3 query embedding and ChromaDB retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

from .constants import EXPECTED_EMBEDDING_DIMENSION
from .models import RetrievedChunk


def _resolve_device(requested: str, torch_module: object) -> str:
    if requested != "auto":
        return requested
    cuda = getattr(torch_module, "cuda")
    if cuda.is_available():
        return "cuda"
    # BGE-M3 has had MPS compatibility issues on some macOS/PyTorch combinations.
    if sys.platform == "darwin":
        return "cpu"
    return "cpu"


class LocalChromaRetriever:
    """Retrieve chunks without sending documents or queries to an embedding API."""

    def __init__(
        self,
        *,
        index_dir: Path,
        collection_name: str,
        embedding_model: str,
        embedding_cache_dir: Path,
        device: str,
    ) -> None:
        try:
            import chromadb
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - installation failure
            raise RuntimeError("Local retrieval dependencies are not installed") from error

        if not index_dir.is_dir():
            raise FileNotFoundError(f"Chroma index directory does not exist: {index_dir}")
        embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        resolved_device = _resolve_device(device, torch)
        self._model = SentenceTransformer(
            embedding_model,
            cache_folder=str(embedding_cache_dir),
            device=resolved_device,
        )
        dimension = self._model.get_embedding_dimension()
        if dimension != EXPECTED_EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Embedding dimension {dimension} does not match index dimension "
                f"{EXPECTED_EMBEDDING_DIMENSION}"
            )

        self._client = chromadb.PersistentClient(path=str(index_dir))
        self._collection = self._client.get_collection(name=collection_name)

    @property
    def count(self) -> int:
        return self._collection.count()

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        raw = self._collection.query(
            query_embeddings=embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        chunks: list[RetrievedChunk] = []
        for document, metadata, distance in zip(
            documents, metadatas, distances, strict=True
        ):
            chunks.append(
                RetrievedChunk(
                    text=document,
                    score=round(1.0 - float(distance), 6),
                    source=str(metadata["source"]),
                    title=str(metadata["title"]),
                    chunk_index=int(metadata["chunk_index"]),
                    total_chunks=int(metadata["total_chunks"]),
                )
            )
        return chunks
