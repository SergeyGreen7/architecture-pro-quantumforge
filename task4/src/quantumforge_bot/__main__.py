"""Local REPL entry point: python -m quantumforge_bot."""

from __future__ import annotations

from .bootstrap import build_pipeline
from .config import ConfigurationError, Settings
from .constants import REPL_EXIT_COMMANDS


def main() -> None:
    try:
        settings = Settings.from_env(require_telegram=False)
        pipeline, indexed_chunks = build_pipeline(settings)
    except (ConfigurationError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"Startup failed: {error}") from error

    print(f"QuantumForge RAG is ready ({indexed_chunks} chunks). Type 'exit' to stop.")
    while True:
        try:
            query = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.casefold() in REPL_EXIT_COMMANDS:
            break
        if not query:
            continue
        try:
            print(f"\n{pipeline.answer(query).answer}")
        except Exception as error:  # keep the interactive session alive
            print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
