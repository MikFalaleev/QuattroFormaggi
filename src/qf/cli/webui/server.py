"""HTTP layer of the local web interface (`qf ui`, D-107): the page and a small JSON API.

Standard library only (`http.server`), bound to a loopback address: the interface is for one
person on their own computer, like LM Studio's server. It does not load anything from the
internet (the page has no external scripts or fonts).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Final
from urllib.parse import parse_qs, urlsplit

from qf.cli.webui.app import WebApp, WebError
from qf.common import QFError

__all__ = ["LOOPBACK_HOSTS", "MAX_BODY_BYTES", "make_server"]

LOOPBACK_HOSTS: Final = ("127.0.0.1", "localhost", "::1")
MAX_BODY_BYTES: Final = 1_000_000
_log = logging.getLogger(__name__)

Query = dict[str, str]
Body = dict[str, Any]


def _page() -> bytes:
    return resources.files("qf.cli.webui").joinpath("static/index.html").read_bytes()


def _text(params: Query | Body, name: str, *, required: bool = True) -> str:
    value = params.get(name)
    if value is None or value == "":
        if required:
            raise WebError(f"нет параметра {name!r}")
        return ""
    if not isinstance(value, str):
        raise WebError(f"параметр {name!r} должен быть строкой")
    return value


def _seed(body: Body) -> int:
    value = body.get("seed", 0)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**31:
        raise WebError("seed должен быть целым числом от 0")
    return value


def _get_routes(app: WebApp) -> dict[str, Callable[[Query], Any]]:
    return {
        "/api/datasets": lambda q: app.datasets(),
        "/api/records": lambda q: app.records(_text(q, "dataset")),
        "/api/record": lambda q: app.record(_text(q, "dataset"), _text(q, "id")),
        "/api/generator": lambda q: app.generator_options(),
        "/api/runs": lambda q: app.runs(),
        "/api/run": lambda q: app.run_report(_text(q, "id")),
    }


def _post_routes(app: WebApp) -> dict[str, Callable[[Body], Any]]:
    return {
        "/api/check": lambda b: app.check(_text(b, "schema"), _text(b, "answer", required=False)),
        "/api/score": lambda b: app.score(_text(b, "dataset"), _text(b, "id"),
                                          _text(b, "answer", required=False)),
        "/api/generate": lambda b: app.generate(
            _text(b, "load_id", required=False), _text(b, "family", required=False),
            _text(b, "kind", required=False), _seed(b)),
    }  # fmt: skip


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib signature)
        _log.debug("%s - " + format, self.address_string(), *args)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _answer(self, handler: Callable[[Any], Any], params: Any) -> None:
        try:
            self._json(HTTPStatus.OK, handler(params))
        except WebError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except QFError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # the page shows the error instead of a dead request
            _log.exception("web ui request failed")
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": f"внутренняя ошибка: {type(exc).__name__}: {exc}"})  # fmt: skip

    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        url = urlsplit(self.path)
        if url.path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, _page(), "text/html; charset=utf-8")
            return
        handler = _get_routes(self.server.app).get(url.path)
        if handler is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"нет адреса {url.path}"})
            return
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        self._answer(handler, query)

    def do_POST(self) -> None:  # noqa: N802 (stdlib name)
        handler = _post_routes(self.server.app).get(urlsplit(self.path).path)
        if handler is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"нет адреса {self.path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "слишком большой запрос"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "тело запроса — не JSON"})
            return
        if not isinstance(body, dict):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "тело запроса — не JSON-объект"})
            return
        self._answer(handler, body)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: WebApp) -> None:
        self.app = app
        super().__init__(address, _Handler)


def make_server(app: WebApp, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """The server, not started; `port=0` picks a free one. Only loopback hosts are allowed."""
    if host not in LOOPBACK_HOSTS:
        allowed = ", ".join(LOOPBACK_HOSTS)
        raise QFError(f"the web interface listens on this computer only ({allowed}), "
                      f"not {host!r}")  # fmt: skip
    return _Server((host, port), app)
