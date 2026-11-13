"""Minimal asynchronous Telegram Bot API client."""

from __future__ import annotations

from collections.abc import Iterable

import httpx

from .constants import (
    TELEGRAM_API_BASE_URL,
    TELEGRAM_HTTP_TIMEOUT_SECONDS,
    TELEGRAM_SAFE_MESSAGE_LIMIT,
)


class TelegramClient:
    def __init__(
        self,
        token: str,
        *,
        timeout_seconds: float = TELEGRAM_HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = f"{TELEGRAM_API_BASE_URL}/bot{token}"
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        await self._call("sendChatAction", {"chat_id": chat_id, "action": action})

    async def send_message(self, chat_id: int, text: str) -> None:
        for part in split_message(text):
            await self._call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": part,
                    "disable_web_page_preview": True,
                },
            )

    async def _call(self, method: str, payload: dict[str, object]) -> dict[str, object]:
        response = await self._client.post(f"{self._base_url}/{method}", json=payload)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(
                f"Telegram {method} failed: {data.get('description', 'unknown error')}"
            )
        return data

    async def aclose(self) -> None:
        await self._client.aclose()


def split_message(
    text: str,
    *,
    limit: int = TELEGRAM_SAFE_MESSAGE_LIMIT,
) -> Iterable[str]:
    """Split long responses at line boundaries below Telegram's hard limit."""

    remaining = text.strip()
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at <= 0:
            split_at = limit
        yield remaining[:split_at].rstrip()
        remaining = remaining[split_at:].lstrip()
    if remaining:
        yield remaining
