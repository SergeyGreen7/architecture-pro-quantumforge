# Telegram RAG-бот QuantumForge в Yandex Cloud

Рабочий RAG-бот принимает вопросы из Telegram, строит эмбеддинг запроса локальной
моделью `BAAI/bge-m3`, ищет фрагменты в ChromaDB и передаёт только найденный
контекст в YandexGPT. Бот возвращает короткий прямой ответ без раскрытия
внутренних рассуждений.

## Архитектура

```text
Telegram
   │ HTTPS webhook + secret header
   ▼
FastAPI в Yandex Serverless Container
   │
   ├── BAAI/bge-m3 (локально, task2/embedding_model_cache)
   │          │
   │          ▼
   │    ChromaDB (task3/chroma_db, cosine top-k)
   │          │
   │          ▼
   │    контекст + few-shot
   │
   └──────────────────────► YandexGPT в AI Studio
                              │
                              ▼
                         ответ + источники
```

Поиск выполняется локально: запрос и документы не отправляются во внешний API
для получения эмбеддингов. В Yandex Cloud AI Studio уходят вопрос и только
несколько найденных фрагментов.

### Где находятся модель и индекс

В исходном репозитории `task2` содержит словарь псевдонимизации, а готовый
векторный индекс физически находится в `task3/chroma_db`. Поэтому бот использует:

- `task2/embedding_model_cache` — локальный кеш весов `BAAI/bge-m3`, создаётся
  автоматически при первом запуске и исключён из Git;
- `task3/chroma_db` — готовый индекс из 203 чанков, построенный той же моделью.

Пути можно переопределить через `EMBEDDING_CACHE_DIR` и `RAG_INDEX_DIR`.
Docker-образ заранее загружает модель в `/app/task2/embedding_model_cache`, поэтому
в облаке эмбеддинги работают без Hugging Face и без сетевого скачивания при
холодном старте.

## Что реализовано

- локальное преобразование запроса в нормализованный 1024-мерный вектор BGE-M3;
- поиск top-k чанков в постоянной коллекции ChromaDB по cosine similarity;
- порог уверенности: слабая выдача сразу возвращает «Я не знаю» без вызова LLM;
- два few-shot примера из реальной базы (`Аркон Вейр` и `Зерониды`);
- короткий прямой ответ без блока с пошаговыми рассуждениями;
- запрет использовать знания вне контекста и инструкции внутри документов;
- YandexGPT через официальный `yandex-ai-studio-sdk`;
- Telegram webhook с проверкой `X-Telegram-Bot-Api-Secret-Token`;
- структурированные однострочные JSON-логи в локальном storage и Yandex Cloud Logging;
- `/start`, `/help`, обработка ошибок и разбиение длинных сообщений;
- `/health` для проверки контейнера;
- локальный REPL и unit-тесты без обращения к Telegram, модели и YandexGPT;
- Docker-образ для Yandex Serverless Containers.

## Структура

| Файл | Назначение |
|---|---|
| `src/quantumforge_bot/constants.py` | все константы, значения по умолчанию, тексты и промпты |
| `src/quantumforge_bot/logging_config.py` | JSON formatter, stdout и ротация файлов |
| `src/quantumforge_bot/retriever.py` | BGE-M3 и поиск в ChromaDB |
| `src/quantumforge_bot/prompts.py` | system prompt, few-shot и формат ответа |
| `src/quantumforge_bot/pipeline.py` | полный RAG-пайплайн и confidence gate |
| `src/quantumforge_bot/llm.py` | клиент YandexGPT |
| `src/quantumforge_bot/telegram.py` | клиент Telegram Bot API |
| `src/quantumforge_bot/app.py` | FastAPI webhook и health endpoint |
| `src/quantumforge_bot/__main__.py` | локальный консольный REPL |
| `tests/` | изолированные тесты пайплайна и webhook |
| `examples/dialogues.md` | 3 успешных и 2 неизвестных вопроса |
| `Dockerfile` | воспроизводимый образ с моделью и индексом |
| `storage/logs/` | локальные JSONL-файлы журналов |

## Локальная установка

Нужны Python 3.12 и [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
cd task4
uv sync
cp .env.example .env
```

Заполните в `.env`:

- `YANDEX_FOLDER_ID` — ID каталога Yandex Cloud;
- `YANDEX_API_KEY` — API-ключ сервисного аккаунта;
- `YANDEX_MODEL` — по умолчанию `yandexgpt-5.1`;
- `TELEGRAM_BOT_TOKEN` — токен от `@BotFather`;
- `TELEGRAM_WEBHOOK_SECRET` — случайная строка из латинских букв, цифр, `_`, `-`.

Загрузите переменные и запустите REPL:

```bash
set -a
source .env
set +a
uv run python -m quantumforge_bot
```

Первый локальный запуск скачает BGE-M3 в
`../task2/embedding_model_cache`. YandexGPT является платным API: каждый вопрос,
прошедший порог поиска, создаёт оплачиваемый запрос.

## Телеграм бот

```bash
set -a
source .env
set +a

./.venv/bin/uvicorn quantumforge_bot.app:app \
  --host 127.0.0.1 \
  --port 8080
```

или

```bash
./run_bot.sh
```

## Локальный HTTP-сервер:

```bash
uv run uvicorn quantumforge_bot.app:app --host 0.0.0.0 --port 8080
curl http://localhost:8080/health
```

Для локального Telegram webhook понадобится публичный HTTPS-туннель. Для целевого
варианта используйте URL Serverless Container.

## Тесты

```bash
cd task4
uv run pytest
```

Тесты проверяют:

- наличие few-shot и контекста в успешном запросе;
- ответ «Я не знаю» при пустой или слабой выдаче без вызова YandexGPT;
- нормализацию отказа самой LLM;
- лимит длины вопроса и деление длинных Telegram-сообщений;
- health endpoint, webhook secret и обработку Telegram update.

Интеграционная ручная проверка:

```bash
uv run python -m quantumforge_bot
```

Введите:

```text
Как Аркон Вейр путешествует между измерениями и какое устройство использует?
```

Первым источником должен быть `Аркон Вейр`, а similarity — около `0.63`.

## Настройки качества

| Переменная | По умолчанию | Назначение |
|---|---:|---|
| `RAG_TOP_K` | `3` | число чанков в контексте |
| `RAG_MIN_SIMILARITY` | `0.45` | порог для ответа вместо «Я не знаю» |
| `RAG_MAX_CONTEXT_CHARS` | `9000` | предел контекста для LLM |
| `RAG_MAX_QUERY_CHARS` | `2000` | защита от слишком больших запросов |
| `EMBEDDING_DEVICE` | `auto` | `cpu`, `cuda`, `mps` или `auto` |
| `LOG_DIR` | `./storage/logs` | каталог локальных JSONL-логов |
| `LOG_LEVEL` | `INFO` | минимальный уровень логирования |
| `LOG_MAX_BYTES` | `5242880` | размер файла до ротации, байт |
| `LOG_BACKUP_COUNT` | `5` | число архивных файлов |

Порог `0.45` расположен ниже результатов контрольных вопросов (`0.605–0.653`),
но отсекает слабые совпадения. Перед production его следует откалибровать на
размеченном наборе вопросов.

## Логи

Локально журнал записывается в:

```text
task4/storage/logs/quantumforge-bot.jsonl
```

Это формат JSON Lines: каждая строка является самостоятельным JSON-объектом.
При достижении 5 МиБ файл ротируется, по умолчанию сохраняются пять предыдущих
файлов. Пример записи:

```json
{"timestamp":"2026-11-13T21:19:38.717830+00:00","level":"INFO","stream_name":"quantumforge-rag-bot","message":"rag_query_completed","logger":"quantumforge_bot.app","request_id":"123456","duration_ms":842.17,"answered_from_context":true,"retrieved_chunks":3,"top_similarity":0.62999}
```

Логируются запуск и остановка, отклонённые webhook-запросы, команды, начало и
результат RAG-запроса, время обработки, число чанков, лучший similarity и ошибки.
Токены, API-ключи, полный текст вопроса, prompt и содержимое найденных документов
не записываются.

В Docker локальный файл пишется в `/tmp/quantumforge-rag-bot/logs` и является
временным. Те же однострочные JSON-объекты выводятся в `stdout`; Serverless
Containers автоматически сохраняет их в Yandex Cloud Logging. Для постоянного
облачного хранения следует использовать Cloud Logging, а не файловую систему
контейнера.

Посмотреть облачные записи можно на вкладке **Logs** контейнера или через CLI:

```bash
yc logging read --resource-ids=<CONTAINER_ID>
```

<!--
## Развёртывание в Yandex Cloud

Serverless Container выбран вместо Cloud Functions, потому что PyTorch, BGE-M3
и локальный индекс требуют большого Docker-образа и памяти. Контейнер обязан
слушать порт из переменной `PORT`; Dockerfile это учитывает.

### 1. Подготовьте Yandex Cloud

Установите и инициализируйте `yc`, Docker и Docker Buildx. В каталоге Yandex
Cloud создайте:

1. сервисный аккаунт для контейнера;
2. роль `ai.languageModels.user` для вызова YandexGPT;
3. роль `container-registry.images.puller` для скачивания приватного образа;
4. роль `lockbox.payloadViewer` на секрет;
5. API-ключ сервисного аккаунта с областью
   `yc.ai.foundationModels.execute`;
6. Lockbox secret с тремя ключами: `YANDEX_API_KEY`, `TELEGRAM_BOT_TOKEN`,
   `TELEGRAM_WEBHOOK_SECRET`.

Официальные инструкции:

- [вызов YandexGPT и требования к API-ключу](https://aistudio.yandex.ru/ru/docs/ai-studio/operations/generation/completions-basic);
- [передача Lockbox-секретов в контейнер](https://yandex.cloud/en/docs/lockbox/operations/serverless/containers);
- [создание Serverless Container](https://yandex.cloud/en/docs/serverless-containers/quickstart/container).

### 2. Соберите и отправьте образ

Команды выполняются из корня репозитория. На Apple Silicon обязательно указан
`linux/amd64`.

```bash
export REGISTRY_ID=<идентификатор_Container_Registry>
export IMAGE="cr.yandex/${REGISTRY_ID}/quantumforge-rag-bot:v1"

yc container registry configure-docker
docker buildx build \
  --platform linux/amd64 \
  --file task4/Dockerfile \
  --tag "$IMAGE" \
  --push \
  .
```

Сборка долгая: она устанавливает PyTorch и заранее скачивает BGE-M3. Это
намеренно делает runtime полностью локальным для эмбеддингов.

### 3. Создайте контейнер и ревизию

```bash
yc serverless container create --name quantumforge-rag-bot
```

Подставьте ID сервисного аккаунта, каталога, секрета и версии секрета:

```bash
yc serverless container revision deploy \
  --container-name quantumforge-rag-bot \
  --image "$IMAGE" \
  --cores 2 \
  --memory 4GB \
  --concurrency 1 \
  --execution-timeout 120s \
  --min-log-level INFO \
  --service-account-id <SERVICE_ACCOUNT_ID> \
  --environment YANDEX_FOLDER_ID=<FOLDER_ID>,YANDEX_MODEL=yandexgpt-5.1,EMBEDDING_DEVICE=cpu \
  --secret environment-variable=YANDEX_API_KEY,id=<SECRET_ID>,version-id=<SECRET_VERSION_ID>,key=YANDEX_API_KEY \
  --secret environment-variable=TELEGRAM_BOT_TOKEN,id=<SECRET_ID>,version-id=<SECRET_VERSION_ID>,key=TELEGRAM_BOT_TOKEN \
  --secret environment-variable=TELEGRAM_WEBHOOK_SECRET,id=<SECRET_ID>,version-id=<SECRET_VERSION_ID>,key=TELEGRAM_WEBHOOK_SECRET
```

Telegram должен иметь возможность вызвать URL без IAM-заголовка, поэтому
сделайте контейнер публичным. Доступ к webhook всё равно защищён Telegram secret
header.

```bash
yc serverless container allow-unauthenticated-invoke quantumforge-rag-bot
```

Получите HTTPS invocation URL в выводе контейнера или в консоли Yandex Cloud и
проверьте:

```bash
curl https://<CONTAINER_ID>.containers.yandexcloud.net/health
```

Ожидается:

```json
{"status":"ok","indexed_chunks":203}
```

### 4. Зарегистрируйте Telegram webhook

На локальной машине задайте те же Telegram secrets и URL контейнера:

```bash
export TELEGRAM_BOT_TOKEN=<BOT_TOKEN>
export TELEGRAM_WEBHOOK_SECRET=<WEBHOOK_SECRET>
export CONTAINER_URL=https://<CONTAINER_ID>.containers.yandexcloud.net
```

Зарегистрируйте webhook:

```bash
curl --fail-with-body --silent --show-error \
  "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
  --header 'Content-Type: application/json' \
  --data "{\"url\":\"${CONTAINER_URL}/telegram/webhook\",\"secret_token\":\"${TELEGRAM_WEBHOOK_SECRET}\",\"allowed_updates\":[\"message\",\"edited_message\"],\"drop_pending_updates\":true}"
```

Проверьте состояние:

```bash
curl --fail-with-body --silent --show-error \
  "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo"
```

Telegram добавляет `X-Telegram-Bot-Api-Secret-Token` ко всем запросам webhook;
приложение отклоняет запросы без правильного значения кодом `403`.

## Безопасность и ограничения

- `.env`, API-ключ и Telegram token не должны попадать в Git или Docker-слои;
- секреты production передаются через Yandex Lockbox;
- публичным является только вызов контейнера, webhook защищён secret header;
- документы рассматриваются как недоверенные данные, инструкции в них
  игнорируются system prompt;
- генерация основана на трёх чанках, это не заменяет контроль доступа к
  корпоративным документам;
- один экземпляр обрабатывает один запрос одновременно, чтобы не конкурировать
  за CPU/RAM локальной модели;
- при большом трафике лучше вынести embedding service и ChromaDB в отдельные
  долгоживущие сервисы. -->

## Примеры

Три успешных диалога и два случая «Я не знаю» находятся в
[`examples/dialogues.md`](examples/dialogues.md).
