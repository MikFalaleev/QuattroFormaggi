# DATA_SPEC.md — спецификация данных

**Статус:** заполнены разделы «Источник» (шаг 2), «Факты о таблицах» (шаг 3) и контракт данных (шаг 4): схема `card_v1`, правило `missing_fields`, единицы и округление, сериализация, российские города и замена американских городов источника (D-047), задачи, формат SFT-записи, `group_id`. Остальное заполняется на шагах 6–8 `IMPLEMENTATION_PLAN.md`:

- шаги 6–7 — генератор, сплиты и защита от утечек;
- шаг 8 — состав замороженного benchmark.

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

## Контракт ответа `card_v1`

Код: `qf/contracts/card_v1.py` (`ShipmentCard`, `Quantity`, `Place`, `Conflict`, `ExtractionTarget`). Ответ ассистента — **один JSON-объект** из трёх ключей `card`, `missing_fields`, `conflicts`, без markdown, пояснений и пробелов вокруг. Новая версия схемы (например, `card_v2` с габаритами) — новый файл рядом; `card_v1.py` не меняется (`IMPLEMENTATION_PLAN.md`, C.7).

### Поля карточки

Все 11 ключей `card` присутствуют **всегда**; неизвестное значение — `null`. Лишние ключи — ошибка.

| Поле | Тип | Допустимые значения | Откуда в синтетике |
|---|---|---|---|
| `shipper_name` | строка или `null` | непустая строка | `customers.customer_name` |
| `cargo_category` | строка или `null` | `general`, `retail`, `consumer_goods`, `food_beverage`, `automotive`, `electronics` | `customers.primary_freight_type` — **прокси**: это основной тип грузов клиента, а не категория конкретного груза |
| `equipment_type` | строка или `null` | `dry_van`, `reefer` | `loads.load_type`: `Dry Van` → `dry_van`, `Refrigerated` → `reefer` |
| `pieces` | целое или `null` | ≥ 1 | `loads.pieces` |
| `weight_total` | `{value, unit}` или `null` | `value` > 0, `unit`: `kg`, `t`, `lb` | масса из текста, как написана (см. «Единицы»); в синтетических заявках только `kg` и `t` |
| `weight_per_piece` | `{value, unit}` или `null` | как у `weight_total` | масса одного места из текста; в одной записи задан либо `weight_total`, либо `weight_per_piece` |
| `origin` | `{city, region}` или `null` | каноническое русское название города и официальное название субъекта РФ из справочника (раздел «Города») | `routes.origin_city`, заменённый российским городом по `configs/data/city_map_ru_v1.yaml` |
| `destination` | `{city, region}` или `null` | то же | `routes.destination_city`, так же |
| `pickup_date` | строка `YYYY-MM-DD` или `null` | ISO-дата | дата `Pickup.scheduled_datetime` (= `load_date`) |
| `delivery_date` | строка `YYYY-MM-DD` или `null` | ISO-дата | дата `Delivery.scheduled_datetime` |
| `temperature_c` | число или `null` | — | **в `card_v1` всегда `null`**: в источнике нет температурного режима. Для `reefer` поэтому всегда попадает в `missing_fields` |

Правила значений:

- **Масса копируется из текста** значением и единицей («12,6 т» → `{"value":12.6,"unit":"t"}`). Модель не пересчитывает единицы: это делает код (D-004).
- **Города** — каноническое русское название и регион из справочника, даже если в тексте город написан по-английски или без региона: «Kazan» → `{"city":"Казань","region":"Республика Татарстан"}`, «из Москвы» → `{"city":"Москва","region":"Москва"}`. Регион модель выдаёт всегда: для 20 городов справочника он однозначен, даже если в заявке его нет (D-048).
- **Даты** — ISO `YYYY-MM-DD`. Относительные даты («завтра») считаются от даты запроса, которая всегда стоит в первой строке сообщения пользователя.
- **Конфликты.** Если в тексте у поля два разных значения, поле в `card` равно `null`, а в `conflicts` добавляется `{"field": <имя поля>, "values": [<значение 1>, <значение 2>, …]}`: не меньше двух **различных** значений; одно поле — не больше одного раза. Обязательное поле с конфликтом попадает в `missing_fields` по общему правилу.

### Строгие типы (D-041)

Схема ничего не приводит молча: `"22"`, `true` и `22.0` не принимаются как `pieces`; строка — как масса; `2022-1-1`, `2022-01-01T00:00:00` и число — как дата; `NaN`/`Infinity` — ни в каком поле. Так ответ модели с неверными типами считается невалидным, а не «почти правильным». Контракт проверяет **форму** ответа; каноничность города, порядок дат и диапазоны проверяет `qf.domain` (D-046).

### Примеры (каноническая сериализация)

Полная карточка, ничего не пропущено:

```json
{"card":{"shipper_name":"National Retail","cargo_category":"retail","equipment_type":"dry_van","pieces":22,"weight_total":{"value":12592,"unit":"kg"},"weight_per_piece":null,"origin":{"city":"Пермь","region":"Пермский край"},"destination":{"city":"Самара","region":"Самарская область"},"pickup_date":"2022-01-01","delivery_date":"2022-01-02","temperature_c":null},"missing_fields":[],"conflicts":[]}
```

Рефрижератор из Санкт-Петербурга в Казань, масса одного места («18 паллет по 572 кг»), без отправителя и даты доставки — не хватает только температуры:

```json
{"card":{"shipper_name":null,"cargo_category":"food_beverage","equipment_type":"reefer","pieces":18,"weight_total":null,"weight_per_piece":{"value":572,"unit":"kg"},"origin":{"city":"Санкт-Петербург","region":"Санкт-Петербург"},"destination":{"city":"Казань","region":"Республика Татарстан"},"pickup_date":"2023-03-14","delivery_date":null,"temperature_c":null},"missing_fields":["temperature_c"],"conflicts":[]}
```

Город отправления не указан, а масса указана дважды по-разному («12,6 т … по накладной 14,1 т»):

```json
{"card":{"shipper_name":null,"cargo_category":null,"equipment_type":"dry_van","pieces":22,"weight_total":null,"weight_per_piece":null,"origin":null,"destination":{"city":"Самара","region":"Самарская область"},"pickup_date":"2022-01-01","delivery_date":"2022-01-02","temperature_c":null},"missing_fields":["origin","weight_total"],"conflicts":[{"field":"weight_total","values":[{"value":12.6,"unit":"t"},{"value":14.1,"unit":"t"}]}]}
```

## Правило `missing_fields`

`missing_fields` вычисляет **только** функция `qf.domain.compute_missing_fields(card)` — никогда LLM или человек. Порядок элементов фиксирован (`REQUIRED_ORDER`): `origin`, `destination`, `pickup_date`, `equipment_type`, `pieces`, `weight_total`, `temperature_c`.

| Поле | Попадает в `missing_fields`, когда |
|---|---|
| `origin`, `destination`, `pickup_date`, `equipment_type`, `pieces` | значение `null` |
| `weight_total` | не заданы **ни** `weight_total`, **ни** `weight_per_piece` (D-042) |
| `temperature_c` | `equipment_type == "reefer"` и `temperature_c` равно `null`. При `equipment_type = null` температура не требуется |
| `delivery_date`, `shipper_name`, `cargo_category` | никогда (необязательные) |

**Количество мест по умолчанию — 1, но его уточняют у пользователя (решение пользователя, D-050).** Если в заявке количество мест не указано, модель пишет `pieces: null`, а `pieces` попадает в `missing_fields` — пользователя спрашивают. Единицу подставляет **код**, а не модель: `effective_pieces(card)` возвращает указанное число или `DEFAULT_PIECES = 1`. Так в карточке остаётся только то, что написано в заявке, а допущение явно помечено (задача 4 §9 плана разработки — отделять данные документа от предположений).

Если задан только `weight_per_piece`, а `pieces` нет, в списке будет **только** `pieces` (D-042): вес одного места известен, а мест по умолчанию одно. Общую массу в кг даёт `total_weight_kg(card)`: `weight_total`, иначе `weight_per_piece × effective_pieces(card)`, иначе `null`; при невыясненном количестве мест это предварительная масса. Runtime (шаг 18) показывает её вместе с допущением «мест: 1 (по умолчанию, уточнить)».

Согласованность ответа проверяет `check_target_consistency(target)`: `missing_fields` равен правилу; каждое поле из `conflicts` равно `null` в карточке и указано один раз; `weight_total` и `weight_per_piece` не заданы одновременно.

## Единицы и округление

Код: `qf/domain/units.py`.

| Константа или функция | Значение |
|---|---|
| `LB_TO_KG` | `0.45359237` — точное определение международного фунта |
| `T_TO_KG` | `1000` |
| `to_kg(q)` | масса в кг без округления |
| `format_kg(kg)` | округление для показа: `round(kg, 1)` |
| `from_kg(kg, unit)` | обратный пересчёт без округления (для генератора) |
| `normalize_number(x)` | целое число с плавающей точкой → целое (`13.0` → `13`) |

Число массы **в тексте запроса** получается из исходных фунтов функцией `render_value_from_lbs(lbs, unit)`; единицу выбирает генератор (шаг 6). **В синтетических заявках только килограммы и тонны** (D-049): в российских заявках фунтов не бывает. Схема `lb` по-прежнему допускает — реальная заявка может прийти в фунтах, и модель должна переписать её как есть:

| Единица | Правило | Тип `value` | Пример для 27 761 lb |
|---|---|---|---|
| `lb` | целое, как в источнике (в синтетике не используется) | целое | `27761` |
| `kg` | округление до целого | целое | `12592` (точно 12 592,19 кг) |
| `t` | округление до 0,1 т; целая тонна пишется без `.0` | дробное или целое | `12.6` (28 660 lb → `13`) |

Округление — «половина вверх» по точному десятичному произведению `lbs × 0.45359237`, без погрешности float (D-044). **Gold берётся из отрендеренного значения**, а не из исходных фунтов: если в тексте «12,6 т», то в ответе `{"value":12.6,"unit":"t"}`.

## Сериализация и разбор ответа

Код: `qf/domain/serialization.py` (D-043).

- **`serialize_target(target)`** — единственный способ получить текст ответа ассистента: компактный JSON (`separators=(",", ":")`), ключи в порядке объявления полей (без сортировки), кириллица без `\u`-экранирования, целые числа без `.0`.
- **`parse_target(text)`** — строгий разбор для `json_valid_rate`. Ошибка формата (`TargetParseError`): пробелы или перевод строки вокруг объекта, обёртка ```` ```json ````, повторяющиеся ключи, `NaN`/`Infinity`, несоответствие схеме.
- **`parse_target_lenient(text)`** — для полевых метрик (шаг 9): снимает пробелы вокруг и одну обёртку ```` ``` ````, затем разбирает строго. JSON из окружающего текста не извлекает.
- Ответ записи **канонический**, если `text == serialize_target(parse_target(text))`.

## Города

### Российские города карточки

По решению пользователя (23.09.2026, D-047) модель работает с **российскими** городами. Справочник — 20 городов (`qf/domain/geo.py`, `CITIES`): все 16 городов-миллионников и Тула, Томск, Кемерово, Барнаул. Для каждого:

- каноническое русское название — значение `city` в карточке;
- регион — официальное название субъекта РФ (Конституция РФ, ст. 65) — значение `region`; у городов федерального значения регион совпадает с городом;
- английские названия города и региона (заголовки статей английской Википедии) — для английских заявок;
- формы «из …» и «в …» (родительный и винительный падежи) — для русских текстов;
- координаты — для проверки расстояний.

`canonical_place(city)` возвращает значение для карточки, а для неизвестного города — `KeyError`. Названия, английские варианты и координаты взяты из русской Википедии; принадлежность каждого города своему региону сверена с Wikidata (свойство P131) 23.09.2026: совпали все 20. Падежные формы написаны вручную.

| Город | Регион | Город (EN) | Регион (EN) | Откуда | Куда |
|---|---|---|---|---|---|
| Москва | Москва | Moscow | Moscow | из Москвы | в Москву |
| Санкт-Петербург | Санкт-Петербург | Saint Petersburg | Saint Petersburg | из Санкт-Петербурга | в Санкт-Петербург |
| Тула | Тульская область | Tula | Tula Oblast | из Тулы | в Тулу |
| Воронеж | Воронежская область | Voronezh | Voronezh Oblast | из Воронежа | в Воронеж |
| Ростов-на-Дону | Ростовская область | Rostov-on-Don | Rostov Oblast | из Ростова-на-Дону | в Ростов-на-Дону |
| Краснодар | Краснодарский край | Krasnodar | Krasnodar Krai | из Краснодара | в Краснодар |
| Волгоград | Волгоградская область | Volgograd | Volgograd Oblast | из Волгограда | в Волгоград |
| Нижний Новгород | Нижегородская область | Nizhny Novgorod | Nizhny Novgorod Oblast | из Нижнего Новгорода | в Нижний Новгород |
| Казань | Республика Татарстан | Kazan | Tatarstan | из Казани | в Казань |
| Самара | Самарская область | Samara | Samara Oblast | из Самары | в Самару |
| Уфа | Республика Башкортостан | Ufa | Bashkortostan | из Уфы | в Уфу |
| Пермь | Пермский край | Perm | Perm Krai | из Перми | в Пермь |
| Екатеринбург | Свердловская область | Yekaterinburg | Sverdlovsk Oblast | из Екатеринбурга | в Екатеринбург |
| Челябинск | Челябинская область | Chelyabinsk | Chelyabinsk Oblast | из Челябинска | в Челябинск |
| Омск | Омская область | Omsk | Omsk Oblast | из Омска | в Омск |
| Новосибирск | Новосибирская область | Novosibirsk | Novosibirsk Oblast | из Новосибирска | в Новосибирск |
| Барнаул | Алтайский край | Barnaul | Altai Krai | из Барнаула | в Барнаул |
| Томск | Томская область | Tomsk | Tomsk Oblast | из Томска | в Томск |
| Кемерово | Кемеровская область — Кузбасс | Kemerovo | Kemerovo Oblast | из Кемерово | в Кемерово |
| Красноярск | Красноярский край | Krasnoyarsk | Krasnoyarsk Krai | из Красноярска | в Красноярск |

### Замена американских городов источника

Сырой датасет не меняется: в `routes.csv` остаются американские города, их проверяет `qf data profile`. На шаге 5, при сборке фактов о загрузке, каждый город источника заменяется российским по фиксированной таблице `configs/data/city_map_ru_v1.yaml` (класс `qf.data.CityMap`; хэш файла пишется в манифест запуска). Таблица взаимно однозначна: 20 городов источника → 20 городов справочника, поэтому маршрут никогда не превращается в «из города в тот же город», а `route_id` и отложенные для `test_ood` маршруты сохраняются.

| Город источника | Город карточки |
|---|---|
| Atlanta | Воронеж |
| Charlotte | Ростов-на-Дону |
| Chicago | Казань |
| Columbus | Волгоград |
| Dallas | Екатеринбург |
| Denver | Омск |
| Detroit | Уфа |
| Houston | Санкт-Петербург |
| Indianapolis | Самара |
| Kansas City | Пермь |
| Las Vegas | Кемерово |
| Los Angeles | Красноярск |
| Memphis | Нижний Новгород |
| Miami | Краснодар |
| Minneapolis | Челябинск |
| New York | Тула |
| Philadelphia | Москва |
| Phoenix | Томск |
| Portland | Барнаул |
| Seattle | Новосибирск |

**Почему именно так.** Срок доставки в источнике растёт с длиной маршрута (корреляция срока с расстоянием 0,85: до ~800 км — в основном тот же день, 3 000–5 000 км — 2–3 дня). Поэтому соответствие подобрано так, чтобы расстояния между российскими городами повторяли расстояния американских маршрутов: оценка «расстояние по прямой × 1,2» против дорожного расстояния источника по 58 маршрутам — корреляция 0,94, медианное отклонение 16%, худшее — 43%. Это проверяет `tests/test_city_map.py`.

**Ограничения (записываются в модельную карточку).** Даты и сроки доставки взяты из американского источника; реальные сроки перевозки по России они не отражают (в источнике около 2 000 км в сутки). Реального дорожного расстояния между российскими городами в данных нет, поэтому оно не используется ни в карточке, ни в тексте заявок.

## Факты о загрузке (`load_facts_v1`, шаг 5)

Код: контракт `qf/contracts/facts.py` (`LoadFacts`), сборщик `qf/data/facts.py` (`LogisticsOpsFactsBuilder`, в реестре `FACTS_BUILDERS` под именем `logistics_operations_v1`), конфиг `configs/data/facts.yaml`. Команда: `uv run qf data facts`. Результат — `data/processed/load_facts.jsonl` (вне Git): одна загрузка — одна строка JSON, сортировка по `load_id`; артефакт `load_facts@load_facts_v1`, родитель — sha256 сырого датасета. В `run_manifest.json` записаны sha256 сырого датасета, таблицы замены городов и результата. Факты — **единственный вход генератора** (шаг 6): он не читает CSV. Читать строку — `LoadFacts.model_validate_json(line)`.

| Поле | Откуда |
|---|---|
| `load_id`, `route_id`, `customer_id` | `loads` |
| `shipper_name` | `customers.customer_name` |
| `cargo_category` | `customers.primary_freight_type`: `General`→`general`, `Retail`→`retail`, `Consumer Goods`→`consumer_goods`, `Food/Beverage`→`food_beverage`, `Automotive`→`automotive`, `Electronics`→`electronics` |
| `equipment_type` | `loads.load_type`: `Dry Van`→`dry_van`, `Refrigerated`→`reefer` |
| `pieces`, `weight_lbs` | `loads` (вес — исходные фунты; число в тексте заявки получается из них правилами раздела «Единицы») |
| `origin`, `destination` | города `routes`, заменённые российскими по `configs/data/city_map_ru_v1.yaml` (`{city, region}`) |
| `pickup_date`, `delivery_date` | календарные даты событий `Pickup` и `Delivery` (`delivery_events.scheduled_datetime`) |
| `load_month` | месяц `loads.load_date` (`YYYY-MM`), для стратифицированной выборки на шаге 6 |

**Намеренно нет** выручки, надбавок, ставок и расстояний: их нет в `card_v1`, и они не должны попасть в эталонный ответ. Неизвестное значение категории, загрузка без маршрута, клиента или ровно одной пары событий, город вне таблицы замены — ошибка с `load_id` (D-054); сборщик не полагается на то, что `qf data profile` уже запускался.

Результат на ревизии `54e7d1d…` (23.09.2026, запуск `20260923-115142-facts-2be782`): **85 410** фактов, sha256 `d8e8a92e5163fe51be46ee75a41ec5ad3df26619de5a0248e5c77a08eb59348a`; `dry_van` 42 464 / `reefer` 42 946; используются все 20 городов (от 4 406 до 14 562 упоминаний) и все 58 маршрутов; срок доставки 0/1/2/3 дня — 16 835 / 44 386 / 23 546 / 643.

**Особенность источника.** Названия клиентов синтетические и повторяются у разных клиентов, а категория — это основной тип грузов клиента, а не груза: встречаются сочетания вроде «ABC Foods» + `automotive`. Для извлечения это не ошибка (модель переписывает из текста то, что там написано), но тексты заявок на шаге 6 будут выглядеть так же условно.

## Задачи модели

Набор задач открыт (D-040): имя задачи записи (`SFTRecord.task`) проверяется по реестру `TASKS` (`qf/domain/tasks.py`). Задача связывает системный промпт и схему ответа. Новая задача (этап v0.2) — новая запись в реестре; формат записи не меняется.

| Задача | Системный промпт | Схема ответа | С шага |
|---|---|---|---|
| `shipment_extraction` | `system_extract_v1` (файл промпта появится на шаге 6) | `card_v1` | 4 |

## Формат SFT-записи (`sft_record_v1`)

Код: `qf/contracts/records.py` (`SFTRecord`, `VariantInfo`). Файл набора — **JSONL** в UTF-8: одна запись — одна строка JSON, без отступов. Читать строку только так: `SFTRecord.model_validate_json(line)` — модели строгие, и `json.loads` + `model_validate` отвергнет даты-строки (D-041).

| Поле | Тип | Смысл |
|---|---|---|
| `id` | строка | уникальный ID записи, например `qf-train-LOAD00001-0` |
| `task` | строка | задача из реестра `TASKS` (`shipment_extraction`) |
| `group_id` | строка | группа для разбиения на сплиты (см. ниже) |
| `source_id` | строка | откуда факты: `yogape/logistics-operations@<rev[:12]>:<load_id>` |
| `language` | `ru` \| `en` | язык запроса |
| `reviewed` | bool | проверена ли запись человеком |
| `synthetic` | bool | синтезирована ли запись из таблиц |
| `template_family` | строка | семейство шаблона (`T1`…`T8`, шаг 6) |
| `variant` | объект `VariantInfo` | как отрендерен текст (ниже) |
| `schema_version` | `card_v1` | схема ответа ассистента |
| `split` | `train` \| `val` \| `test` \| `test_ood` \| `bench` \| `null` | сплит; `null` — ещё не назначен |
| `messages` | список | ровно три сообщения `system`, `user`, `assistant` в этом порядке; у каждого `role` и `content` |

`VariantInfo`: `weight_unit` (`kg`/`t`/`lb`/`null`; в синтетике `lb` не встречается), `weight_mode` (`total`/`per_piece`/`none`), `dropped_fields` (поля, убранные из текста), `hard_cases` (имена трудных случаев из реестра `HARD_CASES`, шаг 6), `request_date` (синтетическая дата запроса; в источнике её нет), `date_style` (`iso`/`text`/`relative`), `city_lang` (`ru`/`en` — на каком языке город написан в тексте). На шаге 6 добавится `ood_reason`.

Сообщение пользователя начинается с даты запроса (`Дата запроса: YYYY-MM-DD` или `Request date: YYYY-MM-DD`, шаг 6). Содержимое ответа ассистента — каноническая сериализация `ExtractionTarget`.

Пример записи (в файле — одна строка; системный промпт и текст запроса сокращены):

```json
{"id":"qf-train-LOAD00001-0","task":"shipment_extraction","group_id":"load:LOAD00001","source_id":"yogape/logistics-operations@54e7d1d1a437:LOAD00001","language":"ru","reviewed":false,"synthetic":true,"template_family":"T1","variant":{"weight_unit":"kg","weight_mode":"total","dropped_fields":[],"hard_cases":[],"request_date":"2021-12-28","date_style":"iso","city_lang":"ru"},"schema_version":"card_v1","split":"train","messages":[{"role":"system","content":"…"},{"role":"user","content":"Дата запроса: 2021-12-28\n\n…"},{"role":"assistant","content":"{\"card\":{\"shipper_name\":\"National Retail\",…},\"missing_fields\":[],\"conflicts\":[]}"}]}
```

Проверки одной записи — `qf.domain.check_record(record)`; коды: `UNKNOWN_TASK` (задачи нет в `TASKS`), `EMPTY_ASSISTANT`, `TARGET_PARSE` (ответ не разбирается строго), `NOT_CANONICAL`, `TARGET_INCONSISTENT` (нарушено правило `missing_fields` или конфликтов). Межзаписные проверки (утечки, дубликаты) добавит `qf validate-data` на шаге 7.

### `group_id`

`group_id` объединяет записи, которые нельзя разносить по разным сплитам. Для синтетики из этого датасета — `load:<load_id>`: все варианты текста, построенные из одной загрузки, имеют общий `group_id`. Разбиение на сплиты делается по `group_id` (D-006): одна загрузка никогда не попадает одновременно в train и test. Для будущих ручных или реальных записей `group_id` — идентификатор исходной заявки или переписки.
