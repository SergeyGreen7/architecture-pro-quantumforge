"""Construct production services from validated settings."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .llm import YandexGPTClient
from .pipeline import RAGPipeline
from .retriever import LocalChromaRetriever
from .telegram import TelegramClient


@dataclass(slots=True)
class AppServices:
    pipeline: RAGPipeline
    telegram: TelegramClient
    indexed_chunks: int


def build_pipeline(settings: Settings) -> tuple[RAGPipeline, int]:
    retriever = LocalChromaRetriever(
        index_dir=settings.index_dir,
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
        security_mode=settings.security_mode,
    )
    return pipeline, retriever.count


def build_services(settings: Settings) -> AppServices:
    pipeline, indexed_chunks = build_pipeline(settings)
    return AppServices(
        pipeline=pipeline,
        telegram=TelegramClient(settings.telegram_bot_token),
        indexed_chunks=indexed_chunks,
    )

