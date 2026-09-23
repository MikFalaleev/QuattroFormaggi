# Quattro Formaggi

Воспроизводимый проект дообучения и оценки локальной языковой модели для логистики:

- **QF-12B** — QLoRA-адаптер к `mistralai/Mistral-Nemo-Instruct-2407` для одного сценария v0.1: запрос на перевозку → JSON-карточка груза (`card_v1`) + список недостающих полей; экспорт в GGUF для LM Studio.
- **QF-Lab** — отдельный учебный трек: собственный Transformer ~100M параметров, обучение с нуля.

Статус: реализованы **шаги 1–2** (каркас, CLI, `qf doctor`, манифесты, архитектурные проверки; загрузка сырого датасета с provenance). Обучения и весов пока нет.

Документы: `Quattro_Formaggi_DEVELOPMENT_PLAN.md` (план разработки), `IMPLEMENTATION_PLAN.md` (пошаговый план), `docs/PROJECT.md` (паспорт проекта), `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `AGENTS.md` (правила для coding agents).

## Быстрый старт

Нужны `uv` и `git`. Python 3.11 uv установит сам.

```bash
uv sync --extra dev        # окружение разработки (без torch/CUDA)
uv run qf --version
uv run qf doctor           # read-only отчёт об окружении → runs/doctor/hardware.json
uv run qf data fetch       # скачать закреплённую ревизию датасета (~31 MB) → data/raw/
uv run qf data fetch --verify-only   # проверить скачанные файлы по provenance.json
```

Поддерживаемое оборудование: разработка, данные и inference — macOS на Apple Silicon (LM Studio); QLoRA-обучение — удалённая машина Linux + NVIDIA (≥24 GB VRAM). См. `docs/PROJECT.md`, раздел 4.

## Команды

| Команда | Шаг | Статус |
|---|---|---|
| `qf doctor [--out PATH]` | 1 | реализована |
| `qf data fetch [--config PATH] [--verify-only]` | 2 | реализована |
| `qf data profile` | 3 | не реализована (код выхода 2) |
| `qf data build` | 5–6 | не реализована |
| `qf data split`, `qf data report`, `qf validate-data` | 7 | не реализованы |
| `qf bench export-review / import-review / freeze / verify` | 8 | не реализованы |
| `qf eval run`, `qf eval compare` | 9 | не реализованы |
| `qf eval-baseline` | 10 | не реализована |
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
