"""Features, loss mask, template check, collator and audit on a tiny fake tokenizer (step 11):
a WordLevel `tokenizers` model with jinja chat templates, built in memory, nothing downloaded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from qf.cli.main import main  # noqa: E402
from qf.contracts import Message, SFTRecord  # noqa: E402
from qf.training import (  # noqa: E402
    IGNORE_INDEX,
    Features,
    MaskConstructionError,
    PadCollator,
    Skip,
    audit_masks,
    build_features,
    check_template,
    choose_pad_token,
    render_audit,
)
from tests.factories import make_record  # noqa: E402

SYSTEM = "извлеки карточку"
USER = "Дата запроса : 2026-09-22 нужна фура из Казани"
ANSWER = '{"card" : 1 }'
WORDS = sorted({*SYSTEM.split(), *USER.split(), *ANSWER.split(), "лишнее"})
# the Mistral way: the system prompt goes into the LAST user turn only
MISTRAL_LIKE = (
    "{{ bos_token }}{% for m in messages %}{% if m.role == 'user' %}"
    "{% if loop.last and messages[0].role == 'system' %}[INST] {{ messages[0].content }} "
    "{{ m.content }} [/INST] {% else %}[INST] {{ m.content }} [/INST] {% endif %}"
    "{% elif m.role == 'assistant' %}{{ m.content }} {{ eos_token }}{% endif %}{% endfor %}"
)
# the system prompt is kept in the finished dialogue too: variant B equals the template
KEEPS_SYSTEM = (
    "{{ bos_token }}{% for m in messages %}{% if m.role == 'user' %}"
    "[INST] {{ messages[0].content }} {{ m.content }} [/INST] "
    "{% elif m.role == 'assistant' %}{{ m.content }} {{ eos_token }}{% endif %}{% endfor %}"
)
# the user part changes once the answer is added: variant B cannot be trusted
CHANGES_PREFIX = KEEPS_SYSTEM.replace("{% elif", "{% if messages|length > 2 %}лишнее {% endif %}"
                                      "{% elif")  # fmt: skip
DOUBLE_BOS = "{{ bos_token }}" + KEEPS_SYSTEM


def fake_tokenizer(template: str = MISTRAL_LIKE, *, pad: str | None = None,
                   with_pad_word: bool = True) -> Any:  # fmt: skip
    specials = ["<unk>", "<s>", "</s>", "[INST]", "[/INST]", *(["<pad>"] if with_pad_word else [])]
    vocab = {word: i for i, word in enumerate([*specials, *WORDS])}
    model = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    model.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=model, bos_token="<s>", eos_token="</s>",
                                  unk_token="<unk>", pad_token=pad,
                                  additional_special_tokens=["[INST]", "[/INST]"])  # fmt: skip
    tok.chat_template = template
    return tok


def record(answer: str = ANSWER, n: int = 0) -> SFTRecord:
    return make_record(id=f"qf-train-LOAD{n:05d}-0", group_id=f"load:LOAD{n:05d}", split="train",
                       messages=[Message(role="system", content=SYSTEM),
                                 Message(role="user", content=USER),
                                 Message(role="assistant", content=answer)])  # fmt: skip


def built(tok: Any, rec: SFTRecord | None = None, max_len: int = 512) -> Features:
    result = build_features(rec or record(), tok, max_len)
    assert isinstance(result, Features)
    return result


def test_labels_mask_prompt() -> None:
    tok = fake_tokenizer()
    features = built(tok)
    prompt = features.input_ids[: features.prompt_tokens]
    assert features.labels[: features.prompt_tokens] == [IGNORE_INDEX] * len(prompt)
    trainable = [t for t in features.labels if t != IGNORE_INDEX]
    assert tok.decode(trainable) == ANSWER + " </s>"
    assert features.answer_tokens == len(trainable) == len(ANSWER.split()) + 1


def test_last_label_is_eos() -> None:
    tok = fake_tokenizer()
    features = built(tok)
    assert features.labels[-1] == tok.eos_token_id == features.input_ids[-1]
    assert features.input_ids.count(tok.eos_token_id) == 1


def test_eos_not_duplicated() -> None:
    with pytest.raises(MaskConstructionError, match="doubled"):
        build_features(record(ANSWER + " </s>"), fake_tokenizer(), 512)


def test_single_bos() -> None:
    assert built(fake_tokenizer()).input_ids.count(1) == 1
    with pytest.raises(MaskConstructionError, match="one BOS"):
        build_features(record(), fake_tokenizer(DOUBLE_BOS), 512)


def test_too_long_is_skipped_not_truncated() -> None:
    tok = fake_tokenizer()
    full = len(built(tok).input_ids)
    skipped = build_features(record(), tok, full - 1)
    assert skipped == Skip(record().id, "too_long", full)
    assert isinstance(build_features(record(), tok, full), Features)


def test_zero_trainable_tokens_raises() -> None:
    with pytest.raises(MaskConstructionError, match="no trainable"):
        build_features(record(""), fake_tokenizer(), 512)


def test_template_dropping_system_does_not_drop_it_from_input() -> None:
    tok = fake_tokenizer(MISTRAL_LIKE)
    check = check_template(record(), tok)
    assert not check.identical and not check.system_in_full and check.system_in_input
    assert check.answer_tokens_match and "system prompt" in check.difference
    assert SYSTEM in tok.decode(built(tok).input_ids)


def test_identical_when_the_template_keeps_the_system_prompt() -> None:
    check = check_template(record(), fake_tokenizer(KEEPS_SYSTEM))
    assert check.identical and check.system_in_full and check.difference == ""


def test_prefix_mismatch_raises() -> None:
    with pytest.raises(MaskConstructionError, match="beyond the system prompt"):
        check_template(record(), fake_tokenizer(CHANGES_PREFIX))


def test_roles_are_checked() -> None:
    bad = record().model_copy(update={"messages": record().messages[1:] + record().messages[:1]})
    with pytest.raises(MaskConstructionError, match="expected system, user, assistant"):
        build_features(bad, fake_tokenizer(), 512)


def test_collator_padding_labels_minus100() -> None:
    tok = fake_tokenizer()
    short, long = built(tok), built(tok, record('{"card" : 1 } 1'))
    collator = PadCollator(pad_token_id=choose_pad_token(tok)[0], pad_to_multiple_of=4)
    batch = collator.pad([short.as_dict(), long.as_dict()])
    width = len(batch["input_ids"][0])
    assert width % 4 == 0 and width >= len(long.input_ids)
    gap = width - len(short.input_ids)
    assert batch["labels"][0][-gap:] == [IGNORE_INDEX] * gap
    assert batch["attention_mask"][0] == [1] * len(short.input_ids) + [0] * gap
    assert batch["input_ids"][0][-gap:] == [collator.pad_token_id] * gap


def test_collator_does_not_add_tokens() -> None:
    for tok, reason in ((fake_tokenizer(pad="<pad>"), "pad token"),
                        (fake_tokenizer(), "<pad>"),
                        (fake_tokenizer(with_pad_word=False), "EOS")):  # fmt: skip
        before = len(tok)
        pad_id, why = choose_pad_token(tok)
        assert reason in why and len(tok) == before
        if reason == "EOS":
            assert pad_id == tok.eos_token_id


def test_collator_tensors_need_torch() -> None:
    pytest.importorskip("torch")
    tok = fake_tokenizer()
    batch = PadCollator(pad_token_id=0)([built(tok).as_dict()])
    assert batch["input_ids"].shape[0] == 1


def test_audit_report() -> None:
    tok = fake_tokenizer()
    records = [record(n=i) for i in range(3)]
    report = audit_masks(records, tok, max_len=512, n_examples=2)
    summary = report.summary()
    assert summary["records"] == 3 and summary["too_long"] == 0
    assert summary["examples_trainable_is_answer"] is False  # the fake decodes with spaces
    assert summary["per_split"]["train"]["records"] == 3
    assert summary["template"] == {"the template drops the system prompt from the finished "
                                   "dialogue": 3}  # fmt: skip
    text = render_audit(report, {"model": "fake/model", "revision": "0" * 40,
                                 "tokenizer_hash": "t", "chat_template_hash": "c",
                                 "data": "`fake`"})  # fmt: skip
    assert "шаблон теряет системный промпт" in text and "## Пример 2" in text
    skipped = audit_masks(records, tok, max_len=5)
    assert skipped.summary()["too_long"] == 3 and skipped.examples == []


def test_cli_fetch_and_audit(fake_project: Path, monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    """`qf tokens fetch` with a fake Hub, then `qf tokens audit` on a small dataset."""
    saved = fake_project / "hub"
    fake_tokenizer().save_pretrained(saved)
    # like the real repository: the chat template lives in tokenizer_config.json
    # (transformers 5 saves it to chat_template.jinja, which `qf tokens fetch` does not fetch)
    config_file = saved / "tokenizer_config.json"
    tokenizer_config = json.loads(config_file.read_text(encoding="utf-8"))
    tokenizer_config["chat_template"] = MISTRAL_LIKE
    config_file.write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (saved / "chat_template.jinja").unlink(missing_ok=True)
    for name in ("config.json", "generation_config.json", "special_tokens_map.json"):
        if not (saved / name).exists():  # transformers 5 no longer writes special_tokens_map
            (saved / name).write_text("{}", encoding="utf-8")

    def fake_download(repo_id: str, filename: str, revision: str, local_dir: Path) -> str:
        target = Path(local_dir) / filename
        target.write_bytes((saved / filename).read_bytes())
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    config = fake_project / "configs/train/base_model.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "id: fake/model\nrevision: " + "a" * 40 + "\nlicense: apache-2.0\nfix_mistral_regex: "
        "false\ntokenizer_files: [tokenizer.json, tokenizer_config.json, special_tokens_map.json,"
        " config.json, generation_config.json]\nlocal_dir: artifacts/base_model/fake\n",
        encoding="utf-8")  # fmt: skip
    assert main(["tokens", "fetch"]) == 0
    data = fake_project / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("".join(record(n=i).model_dump_json() + "\n"
                                              for i in range(4)), encoding="utf-8")  # fmt: skip
    assert main(["tokens", "audit", "--data", str(data), "--examples", "1"]) == 0
    out = capsys.readouterr().out
    assert "4 records: too long 0" in out
    run = next((fake_project / "runs").glob("*-tokens-audit-*"))
    summary = json.loads((run / "mask_audit.json").read_text(encoding="utf-8"))
    assert summary["records_changed_by_regex_fix"] == 0 and summary["pad_token_id"] == 5
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tokenizer_hash"] and manifest["chat_template_hash"]
    assert manifest["base_revision"] == "a" * 40
    (fake_project / "artifacts/base_model/fake/tokenizer.json").write_text("{}")
    assert main(["tokens", "audit", "--data", str(data)]) == 1  # changed files are refused
    assert main(["tokens", "audit", "--data", str(fake_project / "nothing")]) == 1
