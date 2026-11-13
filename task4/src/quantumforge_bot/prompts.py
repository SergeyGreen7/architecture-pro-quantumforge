"""Grounded few-shot prompt construction."""

from __future__ import annotations

from .constants import FEW_SHOT_MESSAGES, SYSTEM_PROMPT
from .models import RetrievedChunk


def build_messages(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    max_context_chars: int,
) -> list[dict[str, str]]:
    """Build system + few-shot + grounded user messages for YandexGPT."""

    context_parts: list[str] = []
    used_chars = 0
    for number, chunk in enumerate(chunks, start=1):
        part = (
            f"[{number}] Источник: {chunk.citation}; similarity={chunk.score:.4f}\n"
            f"{chunk.text.strip()}"
        )
        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break
        if len(part) > remaining:
            part = part[:remaining].rstrip()
        context_parts.append(part)
        used_chars += len(part)

    context = "\n\n".join(context_parts)
    final_user_message = (
        "Контекст базы знаний:\n"
        f"{context}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Сформируй ответ строго по контексту и соблюдай заданный формат."
    )
    return [
        {"role": "system", "text": SYSTEM_PROMPT},
        *FEW_SHOT_MESSAGES,
        {"role": "user", "text": final_user_message},
    ]
