---
license: apache-2.0
base_model: mistralai/Mistral-Nemo-Instruct-2407
language:
- ru
- en
library_name: gguf
pipeline_tag: text-generation
tags:
- logistics
- information-extraction
- structured-output
- qlora
---

![Quattro Formaggi: from requests to validated cargo records](qf-banner-en.png)

# Quattro-Formaggi-12B-Logistics-v0.1

A **derived model**: `mistralai/Mistral-Nemo-Instruct-2407` (revision `04d8a90549d23fc6bd7f642064003592df51e9b3`, Apache-2.0) fine-tuned with a QLoRA adapter for **one task**: turning a free-text freight request (Russian or English) into a structured shipment card (JSON) plus a list of missing fields and conflicts. Renaming does not create new weights: the only change is the LoRA adapter merged into the base model. The tokenizer, chat template and vocabulary of the base model are unchanged.

Status: **v0.1, research release.** It was trained and measured on **synthetic** requests only (see Limitations). Do not use it for legal, compliance or financial decisions, and always let code check the values it returns.

## Files and hashes

| Role | What | sha256 (first 12) |
|---|---|---|
| gguf | `Quattro-Formaggi-12B-Logistics-v0.1-Q6_K.gguf` — release candidate, 9.4 GiB | f0de62136581 |
| gguf_bf16 | `Quattro-Formaggi-12B-Logistics-v0.1-BF16.gguf` — parent of the quantized file, 22.8 GiB | 9d6b9b486b6b |
| merged | merged bf16 HF package (safetensors) | 88e51a679e6e |
| adapter | LoRA adapter, seed 42 (r=16) | ef058b72eadc |

GGUF files were built with llama.cpp `42916d83f4a225e56709f873aa8050ac11f5b6a4`, the commit of LM Studio's llama.cpp runtime 2.44.0. Rebuilding the BF16 GGUF on a different machine gave a byte-identical file.

## Intended use

- Extract a shipment card from a request: route (city and region), pickup and delivery dates, equipment type, cargo category, number of pieces, weight, shipper, special conditions (temperature, securing, packaging, oversize dimensions, sensors).
- List the fields that are still missing and the fields stated with conflicting values.
- **Not** intended for: arithmetic (code converts units and sums weights), legal or compliance conclusions, documents or long free-form text, anything outside the card schema.

## Output format and how to call it

The answer is one JSON object `{"card": {...}, "missing_fields": [...], "conflicts": [...]}` (schema `card_v2`). An unknown value is `null`; weight is copied as written (`{"value": 12.6, "unit": "t"}`) and converted by code. The model must be called with the **exact system prompt it was trained with** (`system_extract_v2`, sha256 `fa3297296346666443f60d0318f01eed7c669032ffcacc534e8fd166544f4e41`; file `src/qf/domain/prompts/system_extract_v2.txt` in the project repository) and the request date as the first line of the user message (`Request date: YYYY-MM-DD`, in Russian `Дата запроса: YYYY-MM-DD`). Use temperature 0 and a context of at least 4096. JSON-schema constrained decoding is optional (see results).

With llama.cpp (the Hugging Face repository lists BF16 first; ask for the Q6_K file explicitly):

```bash
llama-server -hf MikhailSAI/Quattro-Formaggi-12B-Logistics-v0.1-GGUF:Q6_K -c 4096
```

The system prompt is not stored in the GGUF file: send it as the system message of every request (file `system_extract_v2.txt` in the repositories).

With LM Studio:

```bash
lms import --copy --user-repo LOCAL/Quattro-Formaggi-12B-Logistics-v0.1-GGUF Quattro-Formaggi-12B-Logistics-v0.1-Q6_K.gguf
lms load quattro-formaggi-12b-logistics-v0.1 --context-length 4096 --gpu max --parallel 1
lms server start --port 1234 --bind 127.0.0.1
```

The project command `qf extract` does the rest: it builds the messages exactly as in training, parses the answer strictly, **replaces the model's `missing_fields` with the rule computed by code**, derives the total weight and checks values independently (unknown city, dates, weight above 25 t). Its status is `ok`, `invalid_output` or `backend_error`.

## Training data

Fully synthetic. Requests are rendered by deterministic templates (8 template families, Russian 60% and English 40%) from rows of four tables of `yogape/logistics-operations` (revision `54e7d1d1a437ac9d9b287d3ce3ad0edea6aa7a07`, MIT; Yogape Rodriguez, 2025): `loads`, `routes`, `customers`, `delivery_events`. The source is a fictional US trucking company; its cities are mapped to 20 Russian cities by a fixed, versioned table. For template records, gold answers are computed by code from the rows and the rendered text, never by an LLM or by hand; the 33 mock requests of `bench_v2` are the exception (written by an AI assistant, see Limitations). Equipment types and special conditions are assigned by a versioned table, not taken from the source. The `drivers` table (personal-data-shaped) was never downloaded.

| Split | Records | sha256 (first 12) | Use |
|---|---:|---|---|
| train | 1500 | 968c144affef | training |
| val | 150 | 731a1c5badc2 | validation during training |
| test | 600 | 7b93cc8f276a | held-out (not used in training) |
| test_ood | 300 | f1c9fbf8f599 | held-out routes and template families |
| bench_v2 | 271 | 964c5c157d80 | frozen benchmark, never used in training |

`bench_v2` contains 238 synthetic records and **33 requests written by the project's coding assistant to look like live emails ("mocks")**; it holds **no real customer requests**. Special conditions appear in about 44% of the training records (660 of 1500) and in 56% of `bench_v2` (152 of 271).

**Human review: all 271 records of `bench_v2` were reviewed by the author; the training splits were reviewed only as samples.** For template records gold answers are computed by code; the cards of the 33 mock requests were written by the assistant, and only their canonicalisation and missing-field list are computed by code. The formal review-verdicts file (276 ok, 0 fixed, 0 dropped) holds the coding assistant's preliminary verdicts, copied without changes (D-102), so no per-record human verdict protocol exists among the artifacts; the 33 mock requests and their gold cards were written by the assistant (D-103).

## Training

QLoRA on top of a 4-bit (nf4, double quantization, bf16 compute) base: LoRA rank 16, alpha 32, dropout 0.05 on all attention and MLP projections (0.46% of the parameters), learning rate 1e-4 with cosine decay and 3% warmup, effective batch 16, one epoch = 94 optimizer steps, loss on assistant tokens only, seed 42. One NVIDIA H100 NVL, 24 to 29 minutes per seed. Three seeds were trained (42, 43, 44). Seed 42 is the primary seed of the evaluation plan recorded in the project log before the runs (not an external pre-registration) and was accepted by the author after the pilot evaluation; it is the released adapter. It was **not** the best of the three on `bench_v2` in HF: key accuracy 0.9505 (seed 42), 0.973 (seed 43), 0.964 (seed 44), i.e. the spread between seeds is about two points.

## Evaluation

Benchmark `bench_v2` (271 requests), greedy decoding, **no JSON schema**, the release GGUF in LM Studio 0.4.25+1 on an Apple M4 Max (context 4096; llama.cpp runtime 2.47.0, `llama-server` commit `6c7a87f`, a different llama.cpp commit from the one the files were built with). Intervals are 95% bootstrap. The release gates were approved before training and were not changed afterwards.

| Gate | Measured | Verdict |
|---|---|---|
| P1 json_valid_rate >= 0.98 | 0.9889 [0.974; 1.000] | met |
| P2 key_field_accuracy >= 0.95 | 0.9505 [0.918; 0.978], 211 of 222 cases, zero margin | met |
| P3 zero hallucinated_required_field, hallucinated_condition, missed_condition in the hard slice | 0 / 0 / 1 (one unscored failed answer in the slice) | not met (D-128) |
| P3m at most 2 of those errors on the 33 mock requests | 4 | not met (D-128) |
| P4 gain over the base model >= 5 pp (same backend and mode) | +0.946 [0.912; 0.973] (0.0045 to 0.9505) | met |
| P5 quantization loss vs BF16 GGUF <= 2 pp and no new critical errors of the three P3 kinds | key +0.009, 0 new of the three kinds (Q6_K, llama.cpp); counting all kinds 1 new (`missed_conflict`, one mock request) | met as approved (three kinds); not met if all kinds are counted |
| P6 better than the regex baseline: key >= 0.874, mocks > 0.704 | 0.9505 and 0.741 (27 cases) | met |

The two unmet gates (P3, P3m) are reported as they are (D-128) and compared across systems below. P5 is counted as approved in the plan, over the three P3 error kinds (decision D-125); counting every kind of critical error, as an earlier check did by mistake, Q6_K has one new error (a missed conflict on a mock request) and P5 would not be met. Both readings are shown on purpose.

Other metrics of the same run (95% CI): field accuracy (micro) 0.9781 [0.965; 0.988]; special conditions exactly right 0.9631 [0.937; 0.985]; condition precision 0.9792 [0.955; 0.996], recall 0.9874 [0.971; 1.000]; `missing_fields` F1 0.8877 [0.825; 0.938], exact 0.9336 [0.900; 0.963]; conflict recall 1.0 (19 cases); hallucination rate 0.041 [0.019; 0.067]. Key-field accuracy by slice: clean 1.000 (51), conditions 0.987 (77), hard 0.953 (64), missing 1.000 (3), mock requests 0.741 (27); Russian 0.937 (142), English 0.975 (80); in-distribution 0.947, out-of-distribution 0.958.

**The same model in different systems** (`bench_v2`, no schema; P3 = the three error kinds in the hard slice; P3m = mock requests):

| Model, system | json | key | P3 | P3m | key on mocks |
|---|---:|---:|---|---:|---:|
| base model, LM Studio (own Q4_K_M build) | 0.037 | 0.0045 | 0 / 6 / 0 | 1 | 0 |
| regex rules (lower bound, no model) | 1.000 | 0.874 | 0 / 2 / 1 | 1 | 0.704 |
| adapter on 4-bit base, HF | 0.993 | 0.950 | 0 / 0 / 0 | 3 | 0.667 |
| merged bf16, HF | 0.982 | 0.959 | 0 / 0 / 0 | 4 | 0.741 |
| merged BF16 GGUF, llama.cpp | 0.978 | 0.950 | 0 / 1 / 1 | 3 | 0.741 |
| **merged Q6_K GGUF, llama.cpp** | 0.985 | 0.959 | 0 / 0 / 0 | 3 | 0.741 |
| **merged Q6_K GGUF, LM Studio** | 0.989 | 0.9505 | 0 / 0 / 1 | 4 | 0.741 |
| merged Q6_K GGUF, LM Studio, JSON schema on | 1.000 | 0.955 | 0 / 0 / 1 | 5 | 0.778 |

Near-equivalent variants of the same model differ by about one percentage point in key accuracy and by 0 to 2 in P3, so the thresholds on rare events are sensitive to the configuration; random noise was not separated from the effect of the engine, platform or weight representation, and the cause of the difference between engines was not investigated (the LM Studio runtime differs from the `llama-server` runs both in the llama.cpp commit and in the compute platform). Quantization (llama.cpp, vs BF16 GGUF): Q8_0, Q6_K and Q4_K_M are within 1 pp on key accuracy; Q6_K was chosen as the release format (Q5_K_M and Q4_K_M add new critical errors of the three kinds: 3 and 1; Q8_0: 1). On held-out splits (adapter, HF, seed 42): test 0.994, test_ood 0.946 (template T7 0.86: a thousands separator glued to the unit, "7 718кг", is read as 718).

## Performance (Apple M4 Max, 36 GB, LM Studio, Q6_K, one request at a time)

Latency p50 6.4 s, p95 9.7 s for a request of about 1130 prompt tokens (p95 1182) and about 150 answer tokens; time to first token p50 0.50 s, p95 0.77 s; about 25 tokens/s while generating. The LM Studio process used 9.4 to 10.8 GiB. Stability: 30 requests at short, p95 and about 3.4k-token prompts ran without server errors; 28 of the 30 answers parsed strictly (the two that did not are mock requests; accepted with a caveat for the purposes of the experiment pipeline). Not load-tested with concurrent users.

## Limitations

- **Synthetic texts only.** All training and benchmark requests come from templates or were written by a coding assistant; quality on real requests was **not measured** (on the 33 mock requests key accuracy is 0.741, far below the 0.95 of the template-based benchmark, and gate P3m is not met).
- **20 Russian cities.** The source data is a US trucking company mapped to a fixed catalog of 20 Russian cities; routes and distances are not real. In a live check the model copied a real city outside the catalog (Tver) correctly, but the code check reports it as unknown. Shipper names stay as in the US dataset.
- **Synthetic labels.** `cargo_category` is a proxy derived from the customer; equipment types and special conditions come from a table, not from real shipments.
- **Known weak spots:** unfamiliar request styles (glued thousands separators, slang such as "борт" or "реф"), requests written like informal emails (two mock requests are answered with values outside the allowed lists and fail strict parsing).
- **The model does not calculate.** Unit conversion, total weight, the missing-fields rule and plausibility checks are done by code; the model's own `missing_fields` should not be trusted.
- **Gates.** P3 and P3m are not met; P2 passes with zero margin; results shift by about one point between inference engines.
- Context tested at 4096 tokens (prompts up to about 3.4k). Single user; no concurrency test.
- The pipeline code, the data generator and the mock requests were written with an AI coding assistant (Claude Code) under the author's direction; each step's results were accepted by the author; all benchmark records were reviewed by the author, training records only as samples.
- Not for legal, compliance or financial conclusions.

## Reproduction

Commands of the project repository (each stage writes a run manifest with code commit, hashes and versions):

```bash
uv sync --extra dev --extra tokenizer          # on the GPU machine: --locked --extra train-cuda --extra dev --extra tokenizer
uv run qf data fetch                     # dataset at the pinned revision (network)
uv run qf data facts-v2                  # facts with equipment and special conditions
uv run qf data build --config configs/data/generate_v2.yaml
uv run qf validate-data --data-dir data/processed/generated_v2
uv run qf bench verify --config configs/eval/benchmark_v2.yaml
uv run qf tokens fetch                   # pinned tokenizer and config files (no weights)
uv run qf tokens audit                   # tokenizer, template and loss-mask audit
uv run qf train fetch-base --approval "WHO, WHEN, WHAT"   # ~24.5 GB base weights, GPU machine
uv run qf train run --experiment E302 --seed 42 --approval "WHO, WHEN, WHAT"   # GPU machine; seed 42 is the released adapter
uv run qf export adapter RUN_DIR --name qf-12b-logistics-v0.1-seed42
uv run qf export merge --adapter artifacts/adapters/qf-12b-logistics-v0.1-seed42
uv run qf export gguf --source artifacts/merged/Quattro-Formaggi-12B-Logistics-v0.1 --name Quattro-Formaggi-12B-Logistics-v0.1 --qtypes none   # the BF16 GGUF (byte-identical on two machines)
# Q6_K: quantize that BF16 file with the pinned llama-quantize (llama.cpp 42916d83)
uv run qf eval-baseline --config configs/eval/lmstudio_merged_q6k_bench_v2.yaml --json-schema off
uv run qf release check                  # artifacts, lineage to the raw dataset, this card
```

The Q6_K file was made by quantizing the BF16 GGUF with the pinned `llama-quantize` through a small helper script of the project, which is **not part of the repository's commands** and is not published; `qf export gguf --qtypes Q4_K_M,Q5_K_M` uses the same quantizer for other types, but reproducing Q6_K with a `qf` command itself has not been verified. Configs are in `configs/`; the decisions behind every choice are in `docs/DECISIONS.md` (D-012 to D-131) and the metric definitions in `docs/EVAL_SPEC.md`.

## License and attribution

Apache-2.0, as the base model. Base model: Mistral AI, `Mistral-Nemo-Instruct-2407` (Apache-2.0). Data source: Yogape Rodriguez (2025), `yogape/logistics-operations`, MIT licence. Published by the author on Hugging Face: weights and adapter `MikhailSAI/Quattro-Formaggi-12B-Logistics-v0.1`, GGUF files `MikhailSAI/Quattro-Formaggi-12B-Logistics-v0.1-GGUF`, synthetic data (dataset) `MikhailSAI/Quattro-Formaggi-Logistics-Synthetic-v0.1`.
