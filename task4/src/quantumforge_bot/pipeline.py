"""The retrieval-augmented generation pipeline."""

from __future__ import annotations

from typing import Protocol

from .constants import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    UNKNOWN_ANSWER,
)
from .models import RAGResult, RetrievedChunk
from .prompts import build_messages


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]: ...


class TextGenerator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class RAGPipeline:
    """Retrieve locally, gate weak matches, prompt YandexGPT, and cite sources."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        generator: TextGenerator,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._max_context_chars = max_context_chars
        self._max_query_chars = max_query_chars

    def answer(self, query: str) -> RAGResult:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("Query must not be empty")
        if len(normalized_query) > self._max_query_chars:
            raise ValueError(
                f"Query is too long: maximum is {self._max_query_chars} characters"
            )

        chunks = self._retriever.retrieve(normalized_query, top_k=self._top_k)
        if not chunks or chunks[0].score < self._min_similarity:
            return RAGResult(
                answer=UNKNOWN_ANSWER,
                chunks=tuple(chunks),
                answered_from_context=False,
            )

        messages = build_messages(
            normalized_query,
            chunks,
            max_context_chars=self._max_context_chars,
        )
        generated = self._generator.generate(messages).strip()
        if generated.startswith("Я не знаю"):
            return RAGResult(
                answer=UNKNOWN_ANSWER,
                chunks=tuple(chunks),
                answered_from_context=False,
            )

        citations = _unique_citations(chunks)
        answer = f"{generated}\n\nИсточники:\n" + "\n".join(
            f"• {citation}" for citation in citations
        )
        return RAGResult(
            answer=answer,
            chunks=tuple(chunks),
            answered_from_context=True,
        )


def _unique_citations(chunks: list[RetrievedChunk]) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.citation not in seen:
            citations.append(chunk.citation)
            seen.add(chunk.citation)
    return citations
