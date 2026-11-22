"""The retrieval-augmented generation pipeline."""

from __future__ import annotations

from typing import Protocol

from .constants import (
    DEFAULT_SECURITY_MODE,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    UNKNOWN_ANSWER,
    PROTECTED_SYSTEM_PROMPT,
    UNPROTECTED_SYSTEM_PROMPT,
)
from .models import RAGResult, RetrievedChunk
from .prompts import build_messages
from .security import (
    contains_sensitive_output,
    protect_chunks,
    uses_output_guard,
    uses_pre_prompt,
    validate_security_mode,
)


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
        security_mode: str = DEFAULT_SECURITY_MODE,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._max_context_chars = max_context_chars
        self._max_query_chars = max_query_chars
        self._security_mode = validate_security_mode(security_mode)

    def answer(self, query: str) -> RAGResult:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("Query must not be empty")
        if len(normalized_query) > self._max_query_chars:
            raise ValueError(
                f"Query is too long: maximum is {self._max_query_chars} characters"
            )

        retrieved_chunks = self._retriever.retrieve(
            normalized_query, top_k=self._top_k
        )
        if not retrieved_chunks or retrieved_chunks[0].score < self._min_similarity:
            return RAGResult(
                answer=UNKNOWN_ANSWER,
                chunks=tuple(retrieved_chunks),
                answered_from_context=False,
            )

        chunks = protect_chunks(retrieved_chunks, mode=self._security_mode)
        if not chunks or chunks[0].score < self._min_similarity:
            return RAGResult(
                answer=UNKNOWN_ANSWER,
                chunks=tuple(retrieved_chunks),
                answered_from_context=False,
            )

        messages = build_messages(
            normalized_query,
            chunks,
            max_context_chars=self._max_context_chars,
            system_prompt=(
                PROTECTED_SYSTEM_PROMPT
                if uses_pre_prompt(self._security_mode)
                else UNPROTECTED_SYSTEM_PROMPT
            ),
        )
        generated = _without_verification_steps(
            self._generator.generate(messages).strip()
        )
        if generated.startswith("Я не знаю") or (
            uses_output_guard(self._security_mode)
            and contains_sensitive_output(generated)
        ):
            return RAGResult(
                answer=UNKNOWN_ANSWER,
                chunks=tuple(chunks),
                answered_from_context=False,
            )

        citations = _unique_citations(chunks)
        # answer = f"{generated}\n\nИсточники:\n" + "\n".join(
        #     f"• {citation}" for citation in citations
        # )
        answer = generated
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


def _without_verification_steps(answer: str) -> str:
    """Keep only the final answer if a model returns the retired step block."""

    if answer.startswith("Проверка по шагам:") and "Итог:" in answer:
        return answer.split("Итог:", maxsplit=1)[1].strip()
    return answer
