# DATA_SPEC.md — спецификация данных

**Статус:** заполнен раздел «Источник» (шаг 2). Остальное заполняется на шагах 3–8 `IMPLEMENTATION_PLAN.md`:

- шаг 3 — факты о таблицах, подтверждённые `qf data profile`;
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
