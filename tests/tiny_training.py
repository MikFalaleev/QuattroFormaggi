"""A tiny training project for CPU tests of step 12: a word-level tokenizer with a Mistral-like
chat template, short records and their `sft_dataset` artifacts in a fake project root.

Nothing is downloaded; the model is a randomly initialised 2-layer Mistral (loader
`tiny_random_cpu`). Needs transformers and tokenizers (import this module after importorskip).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from qf.common import sha256_file, write_artifact
from qf.contracts import Message, SFTRecord, TrainConfig
from tests.factories import make_record

SYSTEM = "извлеки карточку груза"
REVISION = "a" * 40
MISTRAL_LIKE = (
    "{{ bos_token }}{% for m in messages %}{% if m.role == 'user' %}"
    "{% if loop.last and messages[0].role == 'system' %}[INST] {{ messages[0].content }} "
    "{{ m.content }} [/INST] {% else %}[INST] {{ m.content }} [/INST] {% endif %}"
    "{% elif m.role == 'assistant' %}{{ m.content }} {{ eos_token }}{% endif %}{% endfor %}"
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "config.json")


def user_text(i: int) -> str:
    return f"заявка {i} : груз {i % 5} мест в город {i % 3}"


def answer_text(i: int) -> str:
    return f'{{ "pieces" : {i % 5} , "city" : {i % 3} }}'


def tiny_records(n: int, split: str, offset: int = 0) -> list[SFTRecord]:
    records = []
    for i in range(offset, offset + n):
        messages = [Message(role="system", content=SYSTEM),
                    Message(role="user", content=user_text(i)),
                    Message(role="assistant", content=answer_text(i))]  # fmt: skip
        records.append(make_record(id=f"qf-{split}-LOAD{i:05d}-0", group_id=f"load:LOAD{i:05d}",
                                   split=split, messages=messages))  # fmt: skip
    return records


def tiny_tokenizer() -> Any:
    words = set(SYSTEM.split()) | {"заявка", ":", "груз", "мест", "в", "город", "{", "}", ",",
                                   '"pieces"', '"city"', *map(str, range(100))}  # fmt: skip
    specials = ["<unk>", "<s>", "</s>", "[INST]", "[/INST]", "<pad>"]
    vocab = {word: i for i, word in enumerate([*specials, *sorted(words)])}
    model = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    model.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=model, bos_token="<s>", eos_token="</s>",
                                  unk_token="<unk>",
                                  additional_special_tokens=["[INST]", "[/INST]"])  # fmt: skip
    tok.chat_template = MISTRAL_LIKE
    return tok


@dataclass(frozen=True)
class TinyProject:
    root: Path
    base_config: Path  # relative to root
    train_path: Path
    val_path: Path
    train_sha256: str
    val_sha256: str


def _write_split(root: Path, name: str, records: list[SFTRecord]) -> tuple[Path, str]:
    path = root / "data" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    ref = write_artifact(path, "sft_dataset", "sft_record_v1", "test-run", root=root)
    return path.relative_to(root), ref.sha256


def tiny_project(root: Path, *, n_train: int = 16, n_val: int = 4) -> TinyProject:
    """The fake base model (tokenizer files + provenance) and train/val artifacts in `root`."""
    local = Path("artifacts/base_model/fake-tiny")
    directory = root / local
    tiny_tokenizer().save_pretrained(directory)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    provenance = {"model": "fake/tiny", "revision": REVISION, "license": "apache-2.0",
                  "files": {name: {"sha256": sha256_file(directory / name),
                                   "bytes": (directory / name).stat().st_size}
                            for name in TOKENIZER_FILES}}  # fmt: skip
    (directory / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    base_config = Path("configs/train/base_model_tiny.yaml")
    (root / base_config).parent.mkdir(parents=True, exist_ok=True)
    (root / base_config).write_text(
        f"id: fake/tiny\nrevision: {REVISION}\nlicense: apache-2.0\nfix_mistral_regex: false\n"
        f"tokenizer_files: [{', '.join(TOKENIZER_FILES)}]\nlocal_dir: {local.as_posix()}\n",
        encoding="utf-8")  # fmt: skip
    train_path, train_sha = _write_split(root, "train", tiny_records(n_train, "train"))
    val_path, val_sha = _write_split(root, "val", tiny_records(n_val, "val", offset=1000))
    return TinyProject(root, base_config, train_path, val_path, train_sha, val_sha)


def tiny_config(project: TinyProject, *, dropout: float = 0.0, lora: dict[str, Any] | None = None,
                budget: dict[str, Any] | None = None, trainer: dict[str, Any] | None = None,
                **training: Any) -> TrainConfig:  # fmt: skip
    settings = {"max_sequence_length": 128, "micro_batch_size": 2,
                "gradient_accumulation_steps": 1, "learning_rate": 5e-3, "epochs": 1,
                "max_steps": 8, "max_wall_time_minutes": 60, "warmup_ratio": 0.0,
                "max_grad_norm": 1.0, "gradient_checkpointing": True,
                "checkpoint_every_optimizer_steps": 100, "evaluate_every_optimizer_steps": 100,
                "keep_checkpoints": 3, **training}  # fmt: skip
    return TrainConfig.model_validate({
        "run_name": "tiny", "seed": 7,
        "model": {"id": "fake/tiny", "revision": REVISION,
                  "base_model_config": project.base_config.as_posix()},
        "quantization": {"load_in_4bit": False, "compute_dtype": "bf16"},
        "lora": {"rank": 8, "alpha": 16, "dropout": dropout,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                    "up_proj", "down_proj"], **(lora or {})},
        "data": {"train_path": project.train_path.as_posix(),
                 "val_path": project.val_path.as_posix(),
                 "expected_sha256": {"train": project.train_sha256, "val": project.val_sha256}},
        "training": settings,
        "budget": budget or {},
        "trainer": {"name": "hf_trainer_qlora", "loader": "tiny_random_cpu", **(trainer or {})},
    })  # fmt: skip


def git_init(root: Path) -> None:
    """A committed git repository in `root` (the manifest records the commit)."""

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    (root / ".gitignore").write_text("runs/\n", encoding="utf-8")
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "-qm", "tiny")


def train_log(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]
