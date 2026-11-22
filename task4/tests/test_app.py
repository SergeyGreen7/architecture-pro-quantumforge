from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from quantumforge_bot.app import create_app
from quantumforge_bot.bootstrap import AppServices
from quantumforge_bot.config import Settings
from quantumforge_bot.models import RAGResult


class FakePipeline:
    def answer(self, query: str) -> RAGResult:
        return RAGResult(
            answer=f"Ответ на: {query}",
            chunks=(),
            answered_from_context=True,
        )


class FakeTelegram:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []
        self.actions: list[int] = []
        self.closed = False

    async def send_message(self, chat_id: int, text: str) -> None:
        self.messages.append((chat_id, text))

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.actions.append(chat_id)

    async def aclose(self) -> None:
        self.closed = True


def settings() -> Settings:
    return Settings(
        telegram_bot_token="token",
        telegram_webhook_secret="test-secret",
        yandex_folder_id="folder",
        yandex_api_key="api-key",
        yandex_model="yandexgpt-5.1",
        index_dir=Path("index"),
        embedding_model="BAAI/bge-m3",
        embedding_cache_dir=Path("cache"),
        embedding_device="cpu",
        collection_name="collection",
        top_k=3,
        min_similarity=0.45,
        max_context_chars=9000,
        max_query_chars=2000,
    )


def test_webhook_rejects_invalid_secret() -> None:
    telegram = FakeTelegram()
    services = AppServices(FakePipeline(), telegram, indexed_chunks=203)  # type: ignore[arg-type]
    app = create_app(settings=settings(), services_factory=lambda _: services)

    with TestClient(app) as client:
        response = client.post(
            "/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
            json={"message": {"chat": {"id": 42}, "text": "Вопрос"}},
        )

    assert response.status_code == 403
    assert telegram.messages == []
    assert telegram.closed is True


def test_webhook_answers_text_message() -> None:
    telegram = FakeTelegram()
    services = AppServices(FakePipeline(), telegram, indexed_chunks=203)  # type: ignore[arg-type]
    app = create_app(settings=settings(), services_factory=lambda _: services)

    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "indexed_chunks": 203,
        }
        response = client.post(
            "/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
            json={"message": {"chat": {"id": 42}, "text": "Кто такой Аркон?"}},
        )

    assert response.status_code == 200
    assert telegram.actions == [42]
    assert telegram.messages == [(42, "Ответ на: Кто такой Аркон?")]

