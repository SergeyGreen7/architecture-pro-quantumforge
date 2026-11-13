"""Yandex Cloud AI Studio text generation adapter."""

from __future__ import annotations

from typing import Any

from .constants import DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE


class YandexGPTClient:
    """Small adapter around the official Yandex AI Studio SDK."""

    def __init__(
        self,
        *,
        folder_id: str,
        api_key: str,
        model_name: str,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    ) -> None:
        try:
            from yandex_ai_studio_sdk import AIStudio
        except ImportError as error:  # pragma: no cover - installation failure
            raise RuntimeError("yandex-ai-studio-sdk is not installed") from error

        sdk = AIStudio(folder_id=folder_id, auth=api_key)
        model = sdk.chat.completions(model_name)
        self._model = model.configure(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        result: Any = self._model.run(messages)
        text = str(result.text).strip()
        if not text:
            raise RuntimeError("YandexGPT returned an empty response")
        return text
