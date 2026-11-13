"""Domain models shared by retrieval, generation, and interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One Chroma search result with citation metadata."""

    text: str
    score: float
    source: str
    title: str
    chunk_index: int
    total_chunks: int

    @property
    def citation(self) -> str:
        return (
            f"{self.title} — {self.source}, "
            f"фрагмент {self.chunk_index}/{self.total_chunks}"
        )


@dataclass(frozen=True, slots=True)
class RAGResult:
    """Final answer plus the chunks that grounded it."""

    answer: str
    chunks: tuple[RetrievedChunk, ...]
    answered_from_context: bool

