# Task 5: защита RAG-бота от prompt injection

В этом каталоге находится воспроизводимая демонстрация атаки через документ базы
знаний и многослойной защиты. Чистый индекс Task 3 не изменяется: для теста создаётся
отдельная ChromaDB с 203 штатными чанками и одним вредоносным.

## Состав

| Артефакт | Назначение |
|---|---|
| `fixtures/malicious_instruction.txt` | злонамеренный документ из задания |
| `scripts/build_security_index.py` | сборка отдельного индекса и проверка поиска |
| `index_manifest.json` | состав индекса и top-3 для двух атакующих запросов |
| `scripts/run_security_evaluation.py` | матрица защиты и серия 5+5 |
| `storage/security_evaluation.jsonl` | полный текстовый лог |
| `REPORT.md` | таблицы результатов, защита и выводы |

## Слои защиты

Переменная `RAG_SECURITY_MODE` поддерживает:

- `off` — только базовое grounding-сообщение, без защиты;
- `pre_prompt` — system message запрещает выполнять команды из контекста;
- `post_filter` — подозрительные чанки отбрасываются до LLM;
- `sanitize` — строки с системными конструкциями удаляются;
- `all` — все слои плюс проверка выхода; безопасное значение по умолчанию.

## Воспроизведение

Можно использовать уже созданное окружение Task 4:

```bash
cd task5
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ../task4/.venv/bin/python scripts/build_security_index.py --rebuild

set -a
source ../task4/.env
set +a
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ../task4/.venv/bin/python scripts/run_security_evaluation.py

../task4/.venv/bin/pytest tests
```

Скрипт оценки вызывает YandexGPT и может создать оплачиваемые запросы. Ни API-ключ,
ни Telegram-токен в логи не попадают.

## Запуск защищённого Telegram-бота

Из `task4` задайте тестовый индекс и финальный режим защиты:

```bash
cd task4
set -a
source .env
set +a
export RAG_INDEX_DIR=../task5/chroma_db
export RAG_SECURITY_MODE=all
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ./.venv/bin/uvicorn quantumforge_bot.app:app --host 127.0.0.1 --port 8080
```

Для режимов `off`, `pre_prompt`, `post_filter` и `sanitize` измените
`RAG_SECURITY_MODE` и перезапустите сервер. Режим `off` предназначен только для лабораторной
демонстрации.
