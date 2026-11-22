"""Prompt-injection defenses for retrieved knowledge-base chunks."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Final

from .constants import DEFAULT_SECURITY_MODE, SUPPORTED_SECURITY_MODES
from .models import RetrievedChunk


_INJECTION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bignore\s+(?:all|any|the|previous|prior|system)\s+instructions?\b",
        r"\bignore\s+instructions?\b",
        r"\b(?:system|developer|assistant)\s+(?:message|prompt|instruction)\b",
        r"\boutput\s*:",
        r"\breveal\s+(?:a\s+)?(?:secret|password|token|key)\b",
        r"игнорируй\s+(?:все|предыдущие|любые)\s+инструкц",
        r"раскрой\s+(?:секрет|пароль|токен|ключ)",
    )
)
_SENSITIVE_OUTPUT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bswordfish\b",
        r"суперпароль\s+root",
    )
)


def validate_security_mode(mode: str) -> str:
    normalized = mode.strip().lower() or DEFAULT_SECURITY_MODE
    if normalized not in SUPPORTED_SECURITY_MODES:
        choices = ", ".join(sorted(SUPPORTED_SECURITY_MODES))
        raise ValueError(f"security mode must be one of: {choices}")
    return normalized


def uses_pre_prompt(mode: str) -> bool:
    return validate_security_mode(mode) in {"pre_prompt", "all"}


def contains_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def contains_sensitive_output(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SENSITIVE_OUTPUT_PATTERNS)


def filter_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [chunk for chunk in chunks if not contains_prompt_injection(chunk.text)]


def sanitize_text(text: str) -> str:
    """Drop complete instruction-like lines instead of leaving their payload behind."""

    safe_lines = [
        line
        for line in text.splitlines()
        if not contains_prompt_injection(line)
    ]
    return "\n".join(safe_lines).strip()


def sanitize_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    sanitized: list[RetrievedChunk] = []
    for chunk in chunks:
        safe_text = sanitize_text(chunk.text)
        if safe_text:
            sanitized.append(replace(chunk, text=safe_text))
    return sanitized


def protect_chunks(
    chunks: list[RetrievedChunk],
    *,
    mode: str,
) -> list[RetrievedChunk]:
    normalized = validate_security_mode(mode)
    protected = list(chunks)
    if normalized in {"post_filter", "all"}:
        protected = filter_chunks(protected)
    if normalized in {"sanitize", "all"}:
        protected = sanitize_chunks(protected)
    return protected


def uses_output_guard(mode: str) -> bool:
    return validate_security_mode(mode) == "all"
