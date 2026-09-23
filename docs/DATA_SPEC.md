# DATA_SPEC.md — спецификация данных

**Статус:** заполнены разделы «Источник» (шаг 2) и «Факты о таблицах» (шаг 3). Остальное заполняется на шагах 4–8 `IMPLEMENTATION_PLAN.md`:

- шаг 4 — схема `card_v1`, правило `missing_fields`, единицы и округление, таблица городов RU/EN, формат SFT-записи, `group_id`;
- шаги 6–7 — генератор, сплиты и защита от утечек;
- шаг 8 — состав замороженного benchmark.

Краткий контракт и факты о датасете до заполнения: `IMPLEMENTATION_PLAN.md`, Часть B.

## Источник

| Параметр | Значение |
|---|---|
| Репозиторий | Hugging Face dataset [`yogape/logistics-operations`](https://huggingface.co/datasets/yogape/logistics-operations) |
| Ревизия (закреплена) | `54e7d1d1a437ac9d9b287d3ce3ad0edea6aa7a07` — полный commit sha; ветки, теги и короткие sha конфиг отклоняет |
| Лицензия | MIT (по README датасета). Атрибуция: Yogape Rodriguez, «Synthetic Logistics Operations Database (2022–2024)» |
| Природа данных | Синтетическая табличная база вымышленной американской транспортной компании, 2022–2024; естественного текста запросов нет |
| Конфиг | `configs/data/source.yaml` (реализация `hf_dataset`, `qf.data.fetch:HFDatasetSource`) |
| Локальное размещение | `data/raw/logistics-operations/<revision>/` (вне Git), рядом — манифест артефакта `<revision>.manifest.json` (kind `raw_dataset`, версия `logistics_ops_csv_v1`) |
| Команда | `uv run qf data fetch`; проверка без загрузки — `uv run qf data fetch --verify-only` |

### Загружаемые файлы

| Файл | Назначение |
|---|---|
| `loads.csv` | Загрузки: вес, места, тип прицепа, маршрут, клиент, дата |
| `routes.csv` | Маршруты: города и штаты отправления/назначения — источник истины для маршрута |
| `customers.csv` | Клиенты: используются только `customer_name` и `primary_freight_type` |
| `delivery_events.csv` | События Pickup/Delivery: используются только `event_type`, `scheduled_datetime`, `location_city`, `location_state` |
| `README.md`, `DATABASE_SCHEMA.txt` | Описание и схема датасета — для provenance |

Загружаются **только** эти файлы. Для каждого в `provenance.json` записаны sha256 и размер в байтах, а также repo, ревизия, лицензия, время загрузки (UTC) и список исключённых таблиц с причинами. `--verify-only` пересчитывает хэши и сообщает об изменённых, пропавших и лишних файлах; `.DS_Store` игнорируется.

### Исключённые таблицы

| Таблица | Причина |
|---|---|
| `drivers.csv` | Поля в форме персональных данных (имена, номера водительских прав, даты рождения); для сценария не нужны. Не скачивается |
| `trucks`, `trailers`, `trips`, `fuel_purchases`, `maintenance_records`, `safety_incidents`, `driver_monthly_metrics`, `truck_utilization_metrics`, `facilities` | Не нужны для `card_v1`. `facilities` дополнительно ненадёжна: город площадки совпадает с городом события только в 2 932 из 85 410 загрузок (`IMPLEMENTATION_PLAN.md`, B.1) |

Ожидаемые объёмы (проверяются `test_fetch_real` и на шаге 3): `loads` 85 410 строк, `routes` 58, `customers` 200, `delivery_events` 170 820.

## Факты о таблицах (подтверждено `qf data profile`)

Проверено 23.09.2026 командой `uv run qf data profile` на ревизии `54e7d1d1a437ac9d9b287d3ce3ad0edea6aa7a07` (sha256 артефакта `8092292393cf…`): все 10 проверок PASS. Полный отчёт — `data/processed/profile_report.{json,md}` (вне Git; перезаписывается при каждом запуске). Ожидания, с которыми сверяются данные, — `configs/data/expectations.yaml`: они сняты с этой же ревизии и работают как регрессионная защита (если данные изменятся, pipeline остановится на этом шаге), а не как независимый эталон.

| Факт | Значение | Проверка |
|---|---|---|
| Объёмы | `loads` 85 410, `routes` 58, `customers` 200, `delivery_events` 170 820 | `row_counts` |
| Первичные ключи | уникальны во всех четырёх таблицах | `primary_keys` |
| Внешние ключи | каждый `loads.route_id` есть в `routes`, `loads.customer_id` — в `customers`, `delivery_events.load_id` — в `loads` | `foreign_keys` |
| События | ровно одно `Pickup` и одно `Delivery` на каждую загрузку (по 85 410); других типов нет | `event_pairs` |
| Маршрут | город и штат отправления/назначения из `routes` совпадают с `delivery_events.location_*` в 100% загрузок | `route_matches_events` |
| Дата забора | дата `Pickup.scheduled_datetime` равна `load_date` во всех 85 410 загрузках | `pickup_date_equals_load_date` |
| Срок доставки (дней, Delivery − Pickup) | 0: 16 835; 1: 44 386; 2: 23 546; 3: 643 | `transit_days` |
| `weight_lbs` | 10 000–45 000; квантили 5% 11 721, 25% 18 724, медиана 27 482, 75% 36 190, 95% 43 240 | `value_ranges` |
| `pieces` | 1–28 (все 28 значений встречаются) | `value_ranges` |
| `load_type` | Dry Van 42 464, Refrigerated 42 946 | `value_ranges` |
| `primary_freight_type` (клиенты) | Automotive 41, Food/Beverage 37, Retail 35, Electronics 33, Consumer Goods 30, General 24 | `value_ranges` |
| Города | 20, у каждого ровно один штат: Atlanta GA, Charlotte NC, Chicago IL, Columbus OH, Dallas TX, Denver CO, Detroit MI, Houston TX, Indianapolis IN, Kansas City MO, Las Vegas NV, Los Angeles CA, Memphis TN, Miami FL, Minneapolis MN, New York NY, Philadelphia PA, Phoenix AZ, Portland OR, Seattle WA | `cities` |
| Пропуски | 0% во всех используемых колонках | `null_rates` |
| Период | `load_date` 2022-01 … 2024-12: 36 месяцев, от 2 136 до 2 496 загрузок в месяц | распределение `load_month` |
| Маршруты в загрузках | используются все 58, от 1 386 до 1 628 загрузок на маршрут | распределение `route_id` |

Сведения из B.1, которые этой командой **не** перепроверялись (таблицы не скачиваются): несогласованность `facilities` с городом события (2 932 из 85 410) и статус `Completed` у всех `trips`. Статус `loads.load_status` в проверки не входит.

Как читаются таблицы (`qf.data.raw_tables`): набор колонок сверяется с ожидаемым (порядок не важен); идентификаторы и категории — строки; количества — nullable-целые (`Int64`), чтобы пропуск ловила проверка `null_rates`, а не падение загрузчика; `load_date` и `scheduled_datetime` разбираются по явным форматам `%Y-%m-%d` и `%Y-%m-%d %H:%M:%S.%f`, строка с другим форматом — ошибка с ID записи.
