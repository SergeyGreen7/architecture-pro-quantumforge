#!/usr/bin/env python3
"""Evaluate RAG coverage against the Task 7 golden question set."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TASK7_DIR = REPOSITORY_ROOT / "task7"
TASK4_SRC = REPOSITORY_ROOT / "task4" / "src"
sys.path.insert(0, str(TASK4_SRC))

from quantumforge_bot.config import Settings  # noqa: E402
from quantumforge_bot.constants import UNKNOWN_ANSWER  # noqa: E402
from quantumforge_bot.llm import YandexGPTClient  # noqa: E402
from quantumforge_bot.pipeline import RAGPipeline  # noqa: E402
from quantumforge_bot.retriever import LocalChromaRetriever  # noqa: E402


DEFAULT_GOLDEN_SET = TASK7_DIR / "golden_questions.txt"
DEFAULT_INDEX = TASK7_DIR / "chroma_db"
DEFAULT_LOG = TASK7_DIR / "logs.jsonl"
DEFAULT_REPORT = TASK7_DIR / "REPORT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        required = {
            "id",
            "topic",
            "question",
            "expected_answerable",
            "expected_answer",
            "expected_sources",
            "required_keywords",
        }
        missing = required - item.keys()
        if missing:
            raise ValueError(f"line {line_number}: missing fields {sorted(missing)}")
        questions.append(item)
    if not 10 <= len(questions) <= 15:
        raise ValueError("golden set must contain 10 to 15 questions")
    return questions


def keyword_coverage(answer: str, keywords: list[str]) -> tuple[float, list[str]]:
    if not keywords:
        return 1.0, []
    normalized = answer.casefold()
    matched = [keyword for keyword in keywords if keyword.casefold() in normalized]
    return len(matched) / len(keywords), matched


def score_result(
    golden: dict[str, Any],
    *,
    answer: str,
    answered_from_context: bool,
    retrieved_titles: list[str],
) -> dict[str, Any]:
    answer_found = (
        answered_from_context
        and answer != UNKNOWN_ANSWER
        and len(answer.strip()) >= 20
    )
    completeness, matched_keywords = keyword_coverage(
        answer, golden["required_keywords"]
    )
    expected_sources = set(golden["expected_sources"])
    source_relevant = (
        bool(expected_sources & set(retrieved_titles)) if expected_sources else None
    )

    if golden["expected_answerable"]:
        if not answer_found:
            status = "fail_missing_answer"
        elif source_relevant is False:
            status = "fail_irrelevant_source"
        elif completeness < 0.5:
            status = "fail_low_completeness"
        else:
            status = "pass"
    else:
        status = "pass" if not answer_found else "fail_unexpected_answer"

    return {
        "answer_found": answer_found,
        "completeness_score": round(completeness, 3),
        "matched_keywords": matched_keywords,
        "source_relevant": source_relevant,
        "status": status,
        "correct": status == "pass",
    }


def evaluate_question(
    golden: dict[str, Any],
    pipeline: RAGPipeline,
    *,
    min_similarity: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = pipeline.answer(golden["question"])
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    retrieved_titles = [chunk.title for chunk in result.chunks]
    scoring = score_result(
        golden,
        answer=result.answer,
        answered_from_context=result.answered_from_context,
        retrieved_titles=retrieved_titles,
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_id": golden["id"],
        "topic": golden["topic"],
        "query": golden["question"],
        "expected_answerable": golden["expected_answerable"],
        "expected_answer": golden["expected_answer"],
        "chunks_found": bool(result.chunks),
        "chunks_above_threshold": sum(
            chunk.score >= min_similarity for chunk in result.chunks
        ),
        "top_similarity": result.chunks[0].score if result.chunks else None,
        "retrieved_sources": [chunk.citation for chunk in result.chunks],
        "retrieved_titles": retrieved_titles,
        "answer": result.answer,
        "answer_length": len(result.answer),
        "successful_answer": scoring["answer_found"],
        "latency_ms": latency_ms,
        **scoring,
    }


def percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


def markdown_rows(records: list[dict[str, Any]]) -> str:
    lines = [
        "| ID | Тема | Ожидался ответ | Результат | Completeness | Top source |",
        "|---|---|---:|---|---:|---|",
    ]
    for record in records:
        title = record["retrieved_titles"][0] if record["retrieved_titles"] else "—"
        lines.append(
            f"| `{record['question_id']}` | {record['topic']} | "
            f"{'yes' if record['expected_answerable'] else 'no'} | "
            f"{record['status']} | {record['completeness_score']:.2f} | {title} |"
        )
    return "\n".join(lines)


def write_report(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    known = [item for item in records if item["expected_answerable"]]
    absent = [item for item in records if not item["expected_answerable"]]
    passed = [item for item in records if item["correct"]]
    known_passed = [item for item in known if item["correct"]]
    absent_passed = [item for item in absent if item["correct"]]
    relevant_sources = [item for item in known if item["source_relevant"] is True]
    artificial_gaps = [
        item
        for item in absent
        if item["question_id"].startswith("gap_") and not item["successful_answer"]
    ]
    domain_boundaries = [
        item for item in absent if item["question_id"].startswith("absent_")
    ]
    in_domain = [*known, *artificial_gaps]
    weak = [
        item
        for item in known
        if not item["correct"] or item["completeness_score"] < 1.0
    ]
    average_completeness = round(
        statistics.mean(item["completeness_score"] for item in known), 3
    )

    weak_text = (
        "\n".join(
            f"- **{item['topic']}**: `{item['status']}`, completeness "
            f"{item['completeness_score']:.2f}; top sources: "
            + ", ".join(item["retrieved_titles"] or ["нет"])
            for item in weak
        )
        if weak
        else "- Все вопросы golden set прошли заданные правила."
    )
    gap_text = "\n".join(
        f"- **{item['topic']}**: ответ корректно не найден; "
        f"retrieval вернул {', '.join(item['retrieved_titles'] or ['ничего'])}."
        for item in artificial_gaps
    )
    summary = {
        "total_questions": len(records),
        "passed": len(passed),
        "overall_accuracy_percent": percentage(len(passed), len(records)),
        "known_questions": len(known),
        "known_passed": len(known_passed),
        "known_answer_accuracy_percent": percentage(len(known_passed), len(known)),
        "absent_questions": len(absent),
        "correct_refusals": len(absent_passed),
        "refusal_accuracy_percent": percentage(len(absent_passed), len(absent)),
        "relevant_source_rate_percent": percentage(len(relevant_sources), len(known)),
        "average_known_completeness": average_completeness,
        "in_domain_questions": len(in_domain),
        "knowledge_coverage_percent": percentage(len(known), len(in_domain)),
        "identified_gaps": len(artificial_gaps),
        "domain_boundaries": len(domain_boundaries),
    }
    report = f"""# Task 7: отчёт о покрытии и качестве базы знаний

Дата оценки: {datetime.now(timezone.utc).isoformat()}

## Итоговые метрики

- Всего вопросов: **{summary['total_questions']}**.
- Общая точность: **{summary['passed']}/{len(records)} ({summary['overall_accuracy_percent']}%)**.
- Известные темы: **{summary['known_passed']}/{len(known)} ({summary['known_answer_accuracy_percent']}%)**.
- Корректные отказы: **{summary['correct_refusals']}/{len(absent)} ({summary['refusal_accuracy_percent']}%)**.
- Релевантный ожидаемый источник: **{summary['relevant_source_rate_percent']}%** known-вопросов.
- Средняя keyword completeness: **{summary['average_known_completeness']:.3f}**.
- Покрытие in-domain тем: **{len(known)}/{len(in_domain)} ({summary['knowledge_coverage_percent']}%)**.
- Выявлено пробелов базы знаний: **{summary['identified_gaps']}**.
- Границ домена, не являющихся дефектом: **{summary['domain_boundaries']}**.

## Результаты golden set

{markdown_rows(records)}

## Плохо покрытые темы и нерелевантные источники

{weak_text}

## Подтверждённые пробелы

{gap_text}

Тема погоды — естественная граница домена QuantumForge, а не дефект корпуса.

## Оценка актуальности

В исходных `.txt` нет полей `published_at`, `updated_at`, версии или URL первоисточника.
Поэтому текущую актуальность нельзя обоснованно измерить: filesystem mtime показывает
время копирования, а не актуальность факта. Это самостоятельный пробел в metadata.

## Рекомендации

1. Восстановить статьи об Эйдосах, Союзе Зенор и Криосе, затем пересобрать Task 6 index.
2. Добавить к каждому документу `source_url`, `published_at`, `updated_at`, `owner` и срок
   пересмотра; включить freshness SLA в ежедневный отчёт.
3. Хранить в логах top similarity и источники; группировать отказы по темам и частоте.
4. Запускать `golden_questions.txt` после каждого обновления; блокировать deployment при
   падении answer accuracy, source relevance или refusal accuracy.
5. Добавлять в golden set реальные частые вопросы из Telegram-логов после удаления PII.

Машиночитаемый лог: `logs.jsonl`.
"""
    path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    golden = load_golden_set(args.golden_set.expanduser().resolve())
    settings = Settings.from_env(require_telegram=False)
    retriever = LocalChromaRetriever(
        index_dir=args.index_dir.expanduser().resolve(),
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model,
        embedding_cache_dir=settings.embedding_cache_dir,
        device=settings.embedding_device,
    )
    generator = YandexGPTClient(
        folder_id=settings.yandex_folder_id,
        api_key=settings.yandex_api_key,
        model_name=settings.yandex_model,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
        max_context_chars=settings.max_context_chars,
        max_query_chars=settings.max_query_chars,
        security_mode="all",
    )

    records = [
        evaluate_question(item, pipeline, min_similarity=settings.min_similarity)
        for item in golden
    ]
    log_path = args.log.expanduser().resolve()
    with log_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = write_report(args.report.expanduser().resolve(), records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Log: {log_path}")
    print(f"Report: {args.report.expanduser().resolve()}")


if __name__ == "__main__":
    main()
