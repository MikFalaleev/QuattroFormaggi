"""QF-12B adapter training: features, collator, estimate, trainer adapters (from step 11).

Nothing here imports torch or transformers at import time: the tokenizer comes as an argument,
`transformers` and `torch` are imported inside the functions that need them (plan C.3). Trainer
adapters (`qf.training.trainers.*`) are registered lazily in `qf.cli.wiring`.
"""

from typing import Any

from qf.common import Registry
from qf.contracts import AdapterTrainer
from qf.training.audit import AuditExample, AuditReport, audit_masks, render_audit
from qf.training.collator import PadCollator, choose_pad_token
from qf.training.config import (
    TRAIN_CONFIG_COPY,
    base_model_for,
    check_budget,
    check_max_steps,
    data_refs,
    load_records,
    load_train_config,
    system_prompt_hash,
    with_overrides,
)
from qf.training.estimate import (
    LORA_BYTES_PER_PARAM,
    PLAN_BASE_PARAMS,
    MistralDims,
    base_weight_bytes,
    estimate,
    lora_param_count,
    module_shapes,
    steps_for_epochs,
)
from qf.training.events import (
    CONSOLE_FILE,
    EVENTS_FILE,
    PREFLIGHT_FILE,
    append_event,
    append_preflight,
    read_events,
    read_preflights,
)
from qf.training.experiments import (
    EXPERIMENTS_DIR,
    experiment_config_hash,
    load_experiment,
    registered_config,
    replicates_done,
)
from qf.training.features import (
    IGNORE_INDEX,
    ChatTokenizer,
    Features,
    MaskConstructionError,
    Skip,
    TemplateCheck,
    build_features,
    chat_ids,
    check_template,
    token_stats,
)
from qf.training.tokenizer_io import (
    PROVENANCE_FILE,
    BaseModelConfig,
    chat_template_hash,
    download_tokenizer_files,
    load_tokenizer,
    tokenizer_hash,
    verify_tokenizer_files,
)
from qf.training.verify import (
    INTEGRITY_FILE,
    AdapterCheck,
    read_safetensors,
    verify_adapter,
    verify_run,
)
from qf.training.weights_io import (
    WEIGHTS_PROVENANCE_FILE,
    download_base_weights,
    hub_files,
    verify_base_weights,
)

TRAINERS: Registry[Any] = Registry("trainer", port=AdapterTrainer)

__all__ = [
    "CONSOLE_FILE",
    "EVENTS_FILE",
    "PREFLIGHT_FILE",
    "append_preflight",
    "read_preflights",
    "EXPERIMENTS_DIR",
    "INTEGRITY_FILE",
    "AdapterCheck",
    "append_event",
    "read_events",
    "read_safetensors",
    "verify_adapter",
    "verify_run",
    "IGNORE_INDEX",
    "LORA_BYTES_PER_PARAM",
    "PLAN_BASE_PARAMS",
    "PROVENANCE_FILE",
    "TRAINERS",
    "TRAIN_CONFIG_COPY",
    "WEIGHTS_PROVENANCE_FILE",
    "MistralDims",
    "base_model_for",
    "base_weight_bytes",
    "check_budget",
    "check_max_steps",
    "data_refs",
    "download_base_weights",
    "estimate",
    "experiment_config_hash",
    "load_experiment",
    "registered_config",
    "replicates_done",
    "hub_files",
    "load_records",
    "load_train_config",
    "lora_param_count",
    "module_shapes",
    "steps_for_epochs",
    "system_prompt_hash",
    "token_stats",
    "verify_base_weights",
    "with_overrides",
    "AuditExample",
    "AuditReport",
    "BaseModelConfig",
    "ChatTokenizer",
    "Features",
    "MaskConstructionError",
    "PadCollator",
    "Skip",
    "TemplateCheck",
    "audit_masks",
    "build_features",
    "chat_ids",
    "chat_template_hash",
    "check_template",
    "choose_pad_token",
    "download_tokenizer_files",
    "load_tokenizer",
    "render_audit",
    "tokenizer_hash",
    "verify_tokenizer_files",
]
