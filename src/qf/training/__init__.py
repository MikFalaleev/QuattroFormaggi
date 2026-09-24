"""QF-12B adapter training: features, collator, estimate, trainer adapters (from step 11).

Nothing here imports torch or transformers at import time: the tokenizer comes as an argument,
`transformers` and `torch` are imported inside the functions that need them (plan C.3).
"""

from qf.training.audit import AuditExample, AuditReport, audit_masks, render_audit
from qf.training.collator import PadCollator, choose_pad_token
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

__all__ = [
    "IGNORE_INDEX",
    "PROVENANCE_FILE",
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
