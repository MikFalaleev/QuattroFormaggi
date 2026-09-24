# Quattro Formaggi

Воспроизводимый проект дообучения и оценки локальной языковой модели для логистики:

- **QF-12B** — QLoRA-адаптер к `mistralai/Mistral-Nemo-Instruct-2407` для одного сценария v0.1: запрос на перевозку → JSON-карточка груза (`card_v1`) + список недостающих полей; экспорт в GGUF для LM Studio.
- **QF-Lab** — отдельный учебный трек: собственный Transformer ~100M параметров, обучение с нуля.

Статус: реализованы **шаги 1–9** (каркас, CLI, `qf doctor`, манифесты, архитектурные проверки; загрузка сырого датасета с provenance; профилирование и проверка целостности таблиц; контракт данных — схема карточки `card_v1`, формат SFT-записи, единицы, правило недостающих полей, российские города (замена американских городов датасета), реестр задач, см. `docs/DATA_SPEC.md`; факты о загрузках `load_facts.jsonl`; генератор учебных заявок и эталонных ответов `qf data build`; независимая проверка датасета `qf validate-data` и отчёт длин; эталонный набор `bench_v1` — проверен и заморожен, `qf bench …`; метрики, статистика и eval-harness `qf eval run` / `qf eval compare`, пока проверенные только на подставных ответах fake-backend'а). По решению пользователя до шага 10 карточка расширяется до `card_v2` — типы транспорта и особые условия (`docs/PLAN_card_v2.md`); реализованы подшаги V1 (контракт, правила и промпт v2), V2 (факты с транспортом и особыми условиями, `qf data facts-v2`), V3 (заявки v2: `qf data build --config configs/data/generate_v2.yaml` → `data/processed/generated_v2/`), V4 (проверка данных v2: `qf validate-data --data-dir data/processed/generated_v2`) V5 (`bench_v2` заморожен: 271 запись, из них 33 заявки-мока, настоящих заявок нет) и V6 (метрики и критические ошибки `card_v2`, `qf eval run` на `bench_v2`). Обучения, весов и результатов модели пока нет.

Документы: `Quattro_Formaggi_DEVELOPMENT_PLAN.md` (план разработки), `IMPLEMENTATION_PLAN.md` (пошаговый план), `docs/PROJECT.md` (паспорт проекта), `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/TODO.md` (отложенные задачи и напоминания), `docs/PLAN_card_v2.md` (утверждённый план карточки `card_v2`: типы транспорта и особые условия, подшаги V1–V6 до шага 10), `AGENTS.md` (правила для coding agents).

## Быстрый старт

Нужны `uv` и `git`. Python 3.11 uv установит сам.

```bash
uv sync --extra dev        # окружение разработки (без torch/CUDA)
uv run qf --version
uv run qf doctor           # read-only отчёт об окружении → runs/doctor/hardware.json
uv run qf data fetch       # скачать закреплённую ревизию датасета (~31 MB) → data/raw/
uv run qf data fetch --verify-only   # проверить скачанные файлы по provenance.json
uv run qf data profile     # 10 проверок целостности → data/processed/profile_report.{json,md}
uv run qf data facts       # факты о 85 410 загрузках с российскими городами → data/processed/load_facts.jsonl
uv run qf data build       # факты + 2 050 учебных записей → data/processed/generated_v1/*.jsonl
uv run qf validate-data    # утечки, дубликаты, схема → generated_v1/validation_report.json (код 1 при ошибках)
uv run qf data report      # состав и длины → generated_v1/length_report.{json,md}
uv run qf bench verify     # замороженный bench_v1 совпадает со своим sha256
uv run qf bench verify --config configs/eval/benchmark_v2.yaml   # замороженный bench_v2 (card_v2) совпадает со своим sha256
```

Проверка eval-harness без модели (fake-backend отвечает эталоном с внесёнными ошибками; это **не** результат модели):

```bash
uv run python -m tests.fake_eval   # ответы и конфиги → data/processed/fake_eval/
uv run qf eval run --config data/processed/fake_eval/fake_bench_v1.yaml       # → runs/<run_id>/report.md
uv run qf eval run --config data/processed/fake_eval/fake_bench_v1_gold.yaml
uv run qf eval compare runs/<run_id A> runs/<run_id B>                        # → runs/<run_id>/compare.md
uv run python -m tests.fake_eval --bench data/splits_v2/bench_v2.jsonl        # то же для bench_v2 (card_v2)
uv run qf eval run --config data/processed/fake_eval/fake_bench_v2.yaml
```

Ручная проверка в браузере (без модели; сервер только на этом компьютере): `uv run qf ui`, затем открыть http://127.0.0.1:8765/.

Поддерживаемое оборудование: разработка, данные и inference — macOS на Apple Silicon (LM Studio); QLoRA-обучение — удалённая машина Linux + NVIDIA (≥24 GB VRAM). См. `docs/PROJECT.md`, раздел 4.

## Команды

| Команда | Шаг | Статус |
|---|---|---|
| `qf doctor [--out PATH]` | 1 | реализована |
| `qf data fetch [--config PATH] [--verify-only]` | 2 | реализована |
| `qf data profile [--source-config PATH] [--expectations PATH]` | 3 | реализована |
| `qf data facts [--config PATH] [--source-config PATH]` | 5 | реализована |
| `qf data facts-v2 [--table PATH]` | V2 | реализована: транспорт и особые условия для `card_v2` → `data/processed/load_facts_v2.jsonl` и отчёт `facts_v2_report.md` |
| `qf data build [--config PATH] [--facts-config PATH] [--source-config PATH]` | 6, V3 | реализована; с `--config configs/data/generate_v2.yaml` строит факты v2 и `generated_v2` |
| `qf validate-data [--data-dir DIR] [--bench FILE]` | 7, V4 | реализована; для `card_v2` — `--data-dir data/processed/generated_v2` |
| `qf data split [--data-dir DIR]`, `qf data split --input FILE --out-dir DIR [--config PATH]` | 7 | реализована |
| `qf data report [--data-dir DIR]` (`--tokenizer` — шаг 11) | 7 | реализована |
| `qf bench export-review / import-review / freeze [--manual FILE] / verify` (`--config PATH`) | 8, V5 | реализованы; порядок — `docs/EVAL_SPEC.md`; для `card_v2` — `--config configs/eval/benchmark_v2.yaml` → `data/splits_v2/` |
| `qf eval run --config PATH [--resume RUN_DIR]`, `qf eval compare RUN_A RUN_B` | 9, V6 | реализованы для `card_v1` и `card_v2` (backend `fake`; LM Studio — шаг 10); метрики — `docs/EVAL_SPEC.md` |
| `qf ui [--port 8765]` | — | реализована: локальный веб-интерфейс для ручной проверки — заявки и эталоны, оценка ответа, генератор, отчёты прогонов (D-107) |
| `qf eval-baseline --config PATH --json-schema on\|off [--resume RUN_DIR]` | 10 | реализована: baseline модели в LM Studio в одном из двух режимов (backend `openai_local`); нижняя граница — `qf eval run` с `configs/eval/baseline_{empty,rules}_v2.yaml` |
| `qf tokens audit` | 11 | не реализована |
| `qf train estimate`, `qf train run` | 12 | не реализованы |
| `qf export verify` | 15 | не реализована |
| `qf export merge` | 16 | не реализована |
| `qf export gguf` | 17 | не реализована |
| `qf extract` | 18 | не реализована |
| `qf release check` | 19 | не реализована |
| `qf lab …` | L1–L6 | не реализована |

Нереализованная команда всегда завершается кодом 2 с сообщением `Stage <N> is not implemented yet` — она никогда не имитирует успешный результат.

## Проверки качества (после каждого шага)

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/qf
uv run lint-imports
uv run pytest -q -m "not slow and not gpu and not network and not lmstudio and not needs_tokenizer"
uv run pytest --cov=qf --cov-report=term-missing   # покрытие ≥85% для лёгких пакетов
```

Тесты с маркерами `network`, `gpu`, `lmstudio` без явного флага пропускаются: `--run-network` (скачивает данные — только с разрешения), `--run-gpu`, `--run-lmstudio`.

## Данные и веса

Каталоги `data/`, `artifacts/`, `runs/` и файлы весов (`*.gguf`, `*.safetensors`, `*.bin`) не хранятся в Git. Ничего не загружается во внешние сервисы автоматически.
