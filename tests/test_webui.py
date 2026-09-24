"""The local web interface `qf ui` (D-107): the app behind the page and its HTTP layer."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from qf.cli.main import main
from qf.cli.webui import GENERATED, WebApp, WebError, make_server
from qf.common import QFError
from qf.domain import get_target_schema
from tests.generation import repo_config_v2, synthetic_facts_v2
from tests.test_eval_v2 import bench_v2

BENCH_V2 = Path("data/splits_v2/bench_v2.jsonl")


@pytest.fixture
def project(fake_project: Path) -> Path:
    """bench_v2 of five records, card_v2 facts and generator config, one eval run, one
    comparison."""
    root = fake_project
    (root / BENCH_V2).parent.mkdir(parents=True)
    (root / BENCH_V2).write_text("".join(r.model_dump_json() + "\n" for r in bench_v2()),
                                 encoding="utf-8")  # fmt: skip
    facts = root / "data/processed/load_facts_v2.jsonl"
    facts.parent.mkdir(parents=True)
    facts.write_text("".join(f.model_dump_json() + "\n" for f in synthetic_facts_v2()),
                     encoding="utf-8")  # fmt: skip
    config = root / "configs/data/generate_v2.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(repo_config_v2().model_dump(mode="json"), allow_unicode=True))
    run = root / "runs" / "20260924-101707-eval-2da09f"
    run.mkdir(parents=True)
    (run / "eval_state.json").write_text(json.dumps({"name": "fake_bench_v2", "backend": "fake",
                                                     "schema_version": "card_v2"}))  # fmt: skip
    (run / "report.md").write_text("# Оценка: fake_bench_v2\n", encoding="utf-8")
    compare = root / "runs" / "20260924-101714-eval-compare-8ff4fe"
    compare.mkdir()
    (compare / "compare.md").write_text("# Сравнение прогонов\n", encoding="utf-8")
    (root / "runs" / "20260924-000000-doctor-aaaaaa").mkdir()  # neither: not listed
    return root


def gold_text(app: WebApp, record_id: str) -> str:
    text: str = app.record("bench_v2", record_id)["gold_text"]
    return text


def test_datasets_and_records(project: Path) -> None:
    app = WebApp(project)
    assert [d["id"] for d in app.datasets()] == ["bench_v2"]
    rows = app.records("bench_v2")
    assert [r["kind"] for r in rows] == ["clean", "conditions", "hard", "missing", "manual"]
    assert rows[4]["hard_case"] == "manual" and rows[1]["conditions"] == ["temperature", "sensors"]
    detail = app.record("bench_v2", rows[1]["id"])
    assert detail["special_conditions_flag"] == "да" and detail["text"] == "Заявка 1"
    assert app.record("bench_v2", rows[0]["id"])["special_conditions_flag"] == "нет"
    with pytest.raises(WebError, match="неизвестный набор"):
        app.records("../etc")
    with pytest.raises(WebError, match="нет файла"):
        app.records("bench_v1")
    with pytest.raises(WebError, match="нет записи"):
        app.record("bench_v2", "nope")


def test_score_and_check(project: Path) -> None:
    app = WebApp(project)
    record_id = app.records("bench_v2")[1]["id"]  # the reefer with a temperature and a sensor
    gold = gold_text(app, record_id)
    right = app.score("bench_v2", record_id, gold)
    assert not right["failed"] and right["score"]["critical_errors"] == []
    assert right["score"]["key_fields_correct"] is True and right["kind"] == "conditions"
    assert {f["field"] for f in right["fields"]} >= {"special_conditions", "origin"}
    answer = json.loads(gold)
    answer["card"]["special_conditions"] = answer["card"]["special_conditions"][:1]
    wrong = app.score("bench_v2", record_id, json.dumps(answer, ensure_ascii=False))
    assert wrong["score"]["critical_errors"] == ["missed_condition"]
    assert app.score("bench_v2", record_id, "{")["failed"] is True
    fenced = app.check("card_v2", f"```json\n{gold}\n```")
    assert not fenced["strict"]["ok"] and fenced["lenient"]["ok"]
    assert fenced["consistency"] == [] and fenced["canonical"] == gold
    reordered = json.loads(gold)
    reordered["card"]["special_conditions"].reverse()
    check = app.check("card_v2", json.dumps(reordered, ensure_ascii=False))
    assert check["consistency"] and check["canonical"] == gold  # order is a rule, fixed by code
    assert app.check("card_v2", "not json")["lenient"]["ok"] is False
    with pytest.raises(WebError):
        app.check("card_v9", gold)


def test_generate(project: Path) -> None:
    app = WebApp(project)
    options = app.generator_options()
    assert [f["name"] for f in options["families"]][:2] == ["T1", "T2"]
    assert "oversize_partial" in options["hard_cases"] and options["loads"] > 0
    lowbed = app.generate("", "", "oversize_partial", seed=3)
    assert lowbed["hard_case"] == "oversize_partial" and lowbed["warning"] is None
    assert lowbed["equipment"] == "lowbed" and lowbed["dataset"] == GENERATED
    tent_id = next(f.load_id for f in synthetic_facts_v2() if f.equipment_type == "tent")
    by_hand = app.generate(tent_id, "T7", "oversize_partial", seed=1)
    assert by_hand["warning"] and by_hand["hard_case"] != "oversize_partial"
    assert by_hand["family"] == "T7" and by_hand["variant"]["ood_reason"] == "family"
    assert app.generate(tent_id, "T1", "", seed=1)["family"] == "T1"
    assert GENERATED in [d["id"] for d in app.datasets()]
    scored = app.score(GENERATED, lowbed["id"], lowbed["gold_text"])
    assert scored["score"]["critical_errors"] == []
    for args, message in ((("NOPE", "", ""), "нет загрузки"), ((tent_id, "T9", ""), "семейства"),
                          ((tent_id, "", "weird"), "вида заявки")):  # fmt: skip
        with pytest.raises(WebError, match=message):
            app.generate(*args, seed=1)


def test_generator_needs_the_data(fake_project: Path) -> None:
    with pytest.raises(WebError, match="load_facts_v2"):
        WebApp(fake_project).generate("", "", "", seed=1)


def test_runs(project: Path) -> None:
    app = WebApp(project)
    runs = app.runs()
    assert [r["id"] for r in runs] == ["20260924-101714-eval-compare-8ff4fe",
                                       "20260924-101707-eval-2da09f"]  # fmt: skip
    assert runs[1]["name"] == "fake_bench_v2" and runs[0]["type"] == "compare"
    assert app.run_report(runs[1]["id"])["markdown"].startswith("# Оценка")
    assert app.run_report(runs[0]["id"])["file"] == "compare.md"
    for bad in ("../../etc/passwd", "20260924-000000-doctor-aaaaaa"):
        with pytest.raises(WebError):
            app.run_report(bad)


# --- HTTP ----------------------------------------------------------------------------------


@pytest.fixture
def server(project: Path) -> Iterator[tuple[str, int, WebApp]]:
    app = WebApp(project)
    httpd = make_server(app, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "127.0.0.1", httpd.server_address[1], app
    httpd.shutdown()
    httpd.server_close()


def request(server: tuple[str, int, WebApp], method: str, path: str, body: Any = None,
            raw: bytes | None = None) -> tuple[int, Any]:  # fmt: skip
    conn = http.client.HTTPConnection(server[0], server[1], timeout=10)
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    payload = response.read()
    conn.close()
    kind = response.getheader("Content-Type") or ""
    return response.status, json.loads(payload) if "json" in kind else payload.decode()


def test_http_routes(server: tuple[str, int, WebApp]) -> None:
    status, page = request(server, "GET", "/")
    assert status == 200 and "Quattro Formaggi" in page and "<script src" not in page
    assert request(server, "GET", "/api/datasets")[1][0]["id"] == "bench_v2"
    rows = request(server, "GET", "/api/records?dataset=bench_v2")[1]
    record_id = rows[0]["id"]
    detail = request(server, "GET", f"/api/record?dataset=bench_v2&id={record_id}")[1]
    status, scored = request(server, "POST", "/api/score",
                             {"dataset": "bench_v2", "id": record_id,
                              "answer": detail["gold_text"]})  # fmt: skip
    assert status == 200 and scored["score"]["critical_errors"] == []
    status, checked = request(server, "POST", "/api/check", {"schema": "card_v2", "answer": "{"})
    assert status == 200 and checked["lenient"]["ok"] is False
    status, made = request(server, "POST", "/api/generate", {"kind": "clean", "seed": 2})
    assert status == 200 and made["kind"] in ("clean", "conditions")
    assert request(server, "GET", "/api/generator")[0] == 200
    assert request(server, "GET", "/api/runs")[1][0]["type"] == "compare"
    assert request(server, "GET", "/api/run?id=20260924-101707-eval-2da09f")[0] == 200


def test_http_errors(server: tuple[str, int, WebApp], monkeypatch: pytest.MonkeyPatch) -> None:
    assert request(server, "GET", "/api/records")[0] == 400  # no dataset
    assert request(server, "GET", "/api/nothing")[0] == 404
    assert request(server, "POST", "/api/nothing", {})[0] == 404
    assert request(server, "POST", "/api/score", raw=b"{not json")[0] == 400
    assert request(server, "POST", "/api/score", raw=b"[1, 2]")[0] == 400
    assert request(server, "POST", "/api/generate", {"seed": -1})[0] == 400
    assert request(server, "POST", "/api/check", {"schema": 2})[0] == 400
    status, body = request(server, "POST", "/api/check", {"schema": "card_v9", "answer": "{}"})
    assert status == 400 and "card_v9" in body["error"]
    conn = http.client.HTTPConnection(server[0], server[1], timeout=10)
    conn.putrequest("POST", "/api/check")
    conn.putheader("Content-Length", str(10**7))
    conn.endheaders()
    assert conn.getresponse().status == 413
    conn.close()

    def broken() -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(server[2], "datasets", broken)
    status, body = request(server, "GET", "/api/datasets")
    assert status == 500 and "boom" in body["error"]


def test_only_loopback(project: Path) -> None:
    with pytest.raises(QFError, match="this computer only"):
        make_server(WebApp(project), host="0.0.0.0", port=0)


def test_cli_command(project: Path, monkeypatch: pytest.MonkeyPatch,
                     capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    def interrupted(self: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("http.server.ThreadingHTTPServer.serve_forever", interrupted)
    assert main(["ui", "--port", "0"]) == 0
    out = capsys.readouterr().out
    assert "http://127.0.0.1:" in out and "остановлено" in out
    assert main(["ui", "--host", "0.0.0.0", "--port", "0"]) == 1


def test_gold_of_every_page_record_is_canonical(project: Path) -> None:
    """The page shows the gold as the model must write it: the canonical serialization."""
    app = WebApp(project)
    serialize = get_target_schema("card_v2").serialize
    for row in app.records("bench_v2"):
        detail = app.record("bench_v2", row["id"])
        target = get_target_schema("card_v2").parse(detail["gold_text"])
        assert serialize(target) == detail["gold_text"]
