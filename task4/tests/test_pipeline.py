from __future__ import annotations

from quantumforge_bot.models import RetrievedChunk
from quantumforge_bot.pipeline import RAGPipeline, UNKNOWN_ANSWER


def chunk(*, score: float = 0.63) -> RetrievedChunk:
    return RetrievedChunk(
        text="Аркон использует самодельную фазовую пушку для путешествий.",
        score=score,
        source="docs/knowledge_base/Аркон Вейр.txt",
        title="Аркон Вейр",
        chunk_index=1,
        total_chunks=2,
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.chunks


class FakeGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.messages: list[dict[str, str]] | None = None

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self.answer


def test_grounded_answer_uses_few_shot_cot_context_and_citations() -> None:
    retriever = FakeRetriever([chunk()])
    generator = FakeGenerator(
        "Проверка по шагам:\n1. Найдено устройство.\n2. Найдено назначение.\n"
        "Итог: Аркон использует фазовую пушку."
    )
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    result = pipeline.answer("  Как Аркон путешествует?  ")

    assert result.answered_from_context is True
    assert "Итог: Аркон использует фазовую пушку." in result.answer
    assert "Источники:" in result.answer
    assert "Аркон Вейр" in result.answer
    assert retriever.calls == [("Как Аркон путешествует?", 3)]
    assert generator.messages is not None
    assert generator.messages[0]["role"] == "system"
    assert "Chain-of-Thought" in generator.messages[0]["text"]
    assert len(generator.messages) == 6  # system, two few-shot pairs, final question
    assert "Аркон использует самодельную фазовую пушку" in generator.messages[-1]["text"]
    assert "Как Аркон путешествует?" in generator.messages[-1]["text"]


def test_low_similarity_returns_unknown_without_calling_llm() -> None:
    retriever = FakeRetriever([chunk(score=0.2)])
    generator = FakeGenerator("This must not be used")
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        min_similarity=0.45,
    )

    result = pipeline.answer("Как приготовить тирамису?")

    assert result.answer == UNKNOWN_ANSWER
    assert result.answered_from_context is False
    assert generator.messages is None


def test_empty_results_return_unknown() -> None:
    generator = FakeGenerator("This must not be used")
    pipeline = RAGPipeline(retriever=FakeRetriever([]), generator=generator)

    assert pipeline.answer("Какая погода?").answer == UNKNOWN_ANSWER
    assert generator.messages is None


def test_llm_unknown_is_normalized_and_has_no_misleading_citations() -> None:
    pipeline = RAGPipeline(
        retriever=FakeRetriever([chunk()]),
        generator=FakeGenerator("Я не знаю, данных недостаточно."),
    )

    result = pipeline.answer("Неизвестный факт")

    assert result.answer == UNKNOWN_ANSWER
    assert result.answered_from_context is False
    assert "Источники" not in result.answer


def test_query_length_is_limited() -> None:
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),
        generator=FakeGenerator("unused"),
        max_query_chars=10,
    )

    try:
        pipeline.answer("x" * 11)
    except ValueError as error:
        assert "maximum is 10" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")

