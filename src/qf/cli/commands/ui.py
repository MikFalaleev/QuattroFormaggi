"""`qf ui`: the local web interface for trying the pipeline by hand (D-107)."""

from __future__ import annotations

import argparse

from qf.cli.webui import WebApp, make_server
from qf.common import project_root

__all__ = ["configure", "run"]


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="loopback address to listen on (default: 127.0.0.1)")  # fmt: skip


def run(args: argparse.Namespace) -> int:
    server = make_server(WebApp(project_root()), args.host, args.port)
    print(f"Quattro Formaggi UI: http://{args.host}:{server.server_port}/  (Ctrl+C — остановить)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("остановлено")
    finally:
        server.server_close()
    return 0
