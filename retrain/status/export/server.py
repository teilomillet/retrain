"""HTTP server for live run exports."""

from __future__ import annotations

import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from retrain.status.export.prometheus import render_prometheus_text
from retrain.status.export.runs import _render_runs_json
from retrain.status.export.scan import collect_run_snapshots


class _ExporterHandler(BaseHTTPRequestHandler):
    root: Path = Path(".")

    def do_GET(self) -> None:
        snapshots = collect_run_snapshots(self.root)
        if self.path in ("/metrics", "/metrics/"):
            payload = render_prometheus_text(snapshots).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path in ("/v1/runs", "/v1/runs/"):
            payload = _render_runs_json(
                self.root, snapshots, generated_at=time.time()
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path in ("/healthz", "/healthz/"):
            payload = json.dumps(
                {
                    "status": "ok",
                    "root": str(self.root),
                    "run_count": len(snapshots),
                }
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args: object) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expose live retrain run progress.")
    parser.add_argument(
        "--root", default="logs", help="Root directory containing retrain run dirs"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9108)
    args = parser.parse_args(argv)

    handler_cls = type(
        "RetrainProgressHandler",
        (_ExporterHandler,),
        {"root": Path(args.root).resolve()},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(
        f"retrain progress exporter listening on http://{args.host}:{args.port} "
        f"(root={handler_cls.root})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
