# Quattro Formaggi

Воспроизводимый проект дообучения и оценки локальной языковой модели для логистики:

- **QF-12B** — QLoRA-адаптер к `mistralai/Mistral-Nemo-Instruct-2407` для одного сценария v0.1: текст заявки на перевозку → JSON-карточка груза (`card_v2`: маршрут, даты, груз, масса, места, тип транспорта, особые условия) + список недостающих полей и противоречий; экспорт в GGUF для LM Studio.
- **QF-Lab** — отдельный учебный трек: собственный Transformer ~100M параметров, обучение с нуля (ещё не начат).

## Статус (30.09.2026)

Реализованы **шаги 1–18** из `IMPLEMENTATION_PLAN.md` (шаг 18 — команда `qf extract` и приёмка Q6_K в LM Studio; исключения по П3/П3м приняты, D-128); впереди шаг 19 (выпуск v0.1).

- **Данные** (шаги 2–8, подшаги V1–V6): закреплённый табличный датасет `yogape/logistics-operations` (MIT) → факты о 85 410 загрузках с российскими городами → синтетические заявки по детерминированным шаблонам, эталоны вычисляет код (не LLM); `generated_v2`: 1 500 записей train / 150 val / 600 test / 300 test_ood; замороженный бенчмарк `bench_v2` — 271 запись, из них 33 заявки-макета, настоящих заявок нет.
- **Оценка** (шаги 9–10): 26 метрик, критические ошибки, bootstrap-интервалы, McNemar; пороги релиза П1–П6 утверждены **до обучения** и не меняются (`docs/EVAL_SPEC.md`, D-112).
- **Обучение** (шаги 11–13, протокол эксперимента D-118): QLoRA r=16 (0,46% параметров), одна эпоха, три seed на арендованной H100 NVL, ~24 мин и ~150 ₽ на seed.
- **Экспорт** (шаги 14–17): оценка адаптера в HF, резервная копия, слияние в bf16 (побайтно воспроизводимо), GGUF закреплённым llama.cpp `42916d83` — тем же, что у движка LM Studio 0.4.25.

**Результаты на `bench_v2`** (271 заявка, жадная генерация, без JSON-схемы):

| модель | backend | key_field_accuracy | json_valid_rate | критических ошибок |
|---|---|---|---|---|
| правила на регулярных выражениях (нижняя граница) | — | 0,874 | 1,0 | — |
| исходная база, GGUF Q4_K_M (E100) | LM Studio | 0,005 | 0,026 | — |
| исходная база, 4 бит | HF | 0,014 | 0,063 | 14 |
| адаптер seed 42 + база 4 бит (E303) | HF | 0,951 | 0,993 | 3 |
| слитая модель bf16 (E304) | HF | 0,960 | 0,982 | 6 |
| слитая модель, GGUF BF16 (E305) | llama.cpp | 0,951 | 0,978 | 7 |
| **слитая модель, GGUF Q6_K — кандидат в релиз** | llama.cpp | **0,960** | **0,985** | **6** |
| слитая модель, GGUF Q4_K_M | llama.cpp | 0,960 | 0,985 | 11 |
| исходная база, GGUF Q4_K_M нашей сборки (E306) | LM Studio | 0,0045 | 0,037 | — |
| **слитая модель, GGUF Q6_K — окончательная проверка (E306)** | LM Studio | **0,9505** | **0,989** | **7** |

- **Окончательная проверка в LM Studio** (E306, Q6_K, контекст 4096, режим без схемы): П1 0,9889, П2 0,9505 (запас ноль кейсов), П4 +0,946, П6 (0,9505; моки 0,741) — выполнены; **П3 (1 пропущенное условие в трудном срезе) и П3м (4 ошибки на макетах при пределе 2) не выполнены — пользователь принял исключения и результат описывается как есть** (D-128); стабильность 28 из 30 принята с допущением. Скорость на M4 Max: ≈ 25 ток/с, TTFT 0,5 с, память процесса ~10 ГиБ.
- **П4** (+5 п. п. к базе в том же backend) — пройден с запасом ~94 п. п. в HF, в llama.cpp и в LM Studio.
- **П5** (потеря квантования к BF16-GGUF ≤ 2 п. п. и ни одной новой критической ошибки трёх видов П3) — **у Q6_K выполнен**, у Q8_0, Q5_K_M и Q4_K_M — нет. По ключевым полям все форматы в пределах ±1 п. п.; новых ошибок трёх видов П3 по заявкам: Q6_K 0, Q8_0 1, Q4_K_M 1, Q5_K_M 3. Если считать все виды критических ошибок, как по ошибке делала первая проверка (D-123), П5 не выполняет ни один формат: у Q6_K одна новая ошибка «пропущено противоречие» на заявке-макете. Пользователь выбрал утверждённый текст порога (D-125). Кандидат в релиз — Q6_K (D-124).
- **П3м** (≤ 2 ошибки на 33 макетах) не выполнен ни одной дообученной моделью: 3–5 ошибок во всех системах. Таблица всех моделей и систем — `docs/EVAL_SPEC.md`, «Результаты проверки порогов».
- **Слабые места:** заявки-макеты, похожие на живые письма (key 0,67–0,82 в зависимости от seed и формата); незнакомый шаблон заявки (test_ood T7 — 0,86: «7 718кг» читается как 718 кг, сленг «борт», «реф»). Находки для следующей версии данных — `docs/TODO.md`.

Все решения — `docs/DECISIONS.md` (D-012…D-128); журнал исследования и карточки экспериментов ведутся вне Git.

Документы: `Quattro_Formaggi_DEVELOPMENT_PLAN.md` (план разработки), `IMPLEMENTATION_PLAN.md` (пошаговый план), `docs/PROJECT.md` (паспорт проекта), `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/TODO.md` (отложенные задачи и напоминания), `docs/PLAN_card_v2.md` (утверждённый план карточки `card_v2`: типы транспорта и особые условия, подшаги V1–V6 до шага 10), `AGENTS.md` (правила для coding agents).

## Быстрый старт

Нужны `uv` и `git`. Python 3.11 uv установит сам. Закреплённый llama.cpp (шаг 17) — git-подмодуль: `git clone --recurse-submodules …` или `git submodule update --init` в уже склонированном репозитории.

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

Поддерживаемое оборудование: разработка, данные и inference — macOS на Apple Silicon (LM Studio); обучение QLoRA, оценка в HF, слияние и сборка GGUF — арендованная машина Linux + NVIDIA (использовалась 1× H100 NVL 94 ГБ). См. `docs/PROJECT.md`, раздел 4.

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
| `qf eval run --config PATH [--resume RUN_DIR]`, `qf eval compare RUN_A RUN_B` | 9, V6 | реализованы для `card_v1` и `card_v2`; backends `fake`, `openai_local` (LM Studio / `llama-server`), `hf_local` (шаг 14); метрики — `docs/EVAL_SPEC.md` |
| `qf ui [--port 8765]` | — | реализована: локальный веб-интерфейс для ручной проверки — заявки и эталоны, оценка ответа, генератор, отчёты прогонов (D-107) |
| `qf eval-baseline --config PATH --json-schema on\|off [--resume RUN_DIR]` | 10 | реализована: baseline модели в LM Studio в одном из двух режимов (backend `openai_local`); нижняя граница — `qf eval run` с `configs/eval/baseline_{empty,rules}_v2.yaml` |
| `qf tokens fetch [--config PATH]`, `qf tokens audit [--data DIR] [--max-len 2048]` | 11 | реализованы: файлы токенизатора закреплённой ревизии без весов; аудит маски обучения → `runs/<id>/mask_audit.md` (D-113) |
| `qf train estimate [--tps N] [--rate R]`, `qf train run [--config PATH] [--max-steps N] [--save-every N] [--eval-every N]`, `qf train run --resume runs/<id>/checkpoint-N`, `qf train fetch-base` | 12 | реализованы: оценка шагов, токенов, памяти и стоимости → `runs/<id>/estimate.md`; обучение QLoRA с контрольными точками и продолжением; веса базы (~24,5 ГБ) — только на GPU-машине после одобрения (D-114…D-117). Репетиция на CPU: `qf train run --config configs/train/tiny_cpu_test.yaml`. Протокол (D-118): платный прогон — `qf train run --experiment E302 [--seed N] --approval "…"` (зарегистрированный эксперимент, чистое git-дерево); `qf train verify runs/<id>` — проверка скопированного прогона |
| `qf eval run --config configs/eval/hf_*_v2.yaml` | 14 | реализована: оценка генерацией в HF на GPU-машине — база в 4 бит с адаптером или без, пакетами; целые `test`/`test_ood` (`bench.kind: sft_dataset`) (D-120) |
| `qf export adapter runs/<id> --name NAME`, `qf export verify --adapter PATH` | 15 | реализованы: резервная копия принятого адаптера в `artifacts/adapters/<name>/` со своим манифестом и её проверка; копию вне компьютера делает пользователь (D-121) |
| `qf export merge --adapter artifacts/adapters/<name> [--out DIR]`, `qf export verify --merged DIR [--files-only]` | 16 | реализованы: адаптер вливается в базу bf16 → `artifacts/merged/…` (HF safetensors, токенизатор базы без изменений, манифест с хэшами); проверка слитой модели против «база + адаптер» или только по файлам (D-122). Нужна машина с GPU и ≥48 ГБ RAM |
| `qf export gguf --source DIR --name NAME [--qtypes Q4_K_M,Q5_K_M]`, `qf export verify --gguf FILE` | 17 | реализованы: HF-модель → GGUF BF16 → квантованные GGUF закреплённым llama.cpp (`third_party/llama.cpp` @42916d83 — как в LM Studio 0.4.25), метаданные сверяются с HF; качество — `qf eval run` против `llama-server` (`configs/eval/llamacpp_*`) (D-123, D-124) |
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
uv run pytest -q -m "not slow and not gpu and not network and not lmstudio and not needs_tokenizer and not llama_server"
uv run pytest --cov=qf --cov-report=term-missing   # покрытие ≥85% для лёгких пакетов
```

Тесты с маркерами `network`, `gpu`, `lmstudio`, `llama_server` без явного флага пропускаются: `--run-network` (скачивает данные — только с разрешения), `--run-gpu`, `--run-lmstudio`, `--run-llama-server` (ворота шага 17 с запущенным `llama-server`).

## Данные и веса

Каталоги `data/`, `artifacts/` (`adapters/`, `merged/`, `gguf/`, `base_model/`), `runs/`, venv конвертера `.venv-convert/`, сборка `third_party/llama.cpp-build/` и файлы весов (`*.gguf`, `*.safetensors`, `*.bin`) не хранятся в Git. Ничего не загружается во внешние сервисы автоматически; веса модели и GGUF публикуются только решением человека (шаг 19).
