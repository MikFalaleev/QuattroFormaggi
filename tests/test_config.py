from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from qf.common import ComponentConfig, QFError, StrictConfig, config_hash, load_yaml_config


class TrainLike(StrictConfig):
    seed: int
    name: str = "run"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_valid_config(tmp_path: Path) -> None:
    cfg = load_yaml_config(_write(tmp_path, "seed: 42\nname: пилот\n"), TrainLike)
    assert cfg == TrainLike(seed=42, name="пилот")


def test_extra_field_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "seed: 1\nlearning_rat: 0.1\n")
    with pytest.raises(QFError) as info:
        load_yaml_config(path, TrainLike)
    assert str(path) in str(info.value)
    assert "learning_rat" in str(info.value)


def test_config_hash_stable() -> None:
    assert config_hash(TrainLike(seed=1, name="a")) == config_hash(TrainLike(name="a", seed=1))
    assert config_hash(TrainLike(seed=1)) != config_hash(TrainLike(seed=2))


def test_non_strict_model_refused(tmp_path: Path) -> None:
    class Loose(BaseModel):
        seed: int

    with pytest.raises(QFError, match="extra='forbid'"):
        load_yaml_config(_write(tmp_path, "seed: 1\n"), Loose)


@pytest.mark.parametrize(
    ("text", "message"),
    [("seed: [1\n", "invalid YAML"), ("- 1\n- 2\n", "must be a mapping"), ("", "seed")],
)
def test_bad_yaml_content(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(QFError, match=message):
        load_yaml_config(_write(tmp_path, text), TrainLike)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(QFError, match="cannot read config"):
        load_yaml_config(tmp_path / "absent.yaml", TrainLike)


def test_component_config_keeps_implementation_params() -> None:
    selector = ComponentConfig(name="openai_local", base_url="http://127.0.0.1:1234/v1")
    assert selector.name == "openai_local"
    assert selector.model_extra == {"base_url": "http://127.0.0.1:1234/v1"}
