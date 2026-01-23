import json
import os
import sys
import time
import uuid
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


TOKEN = os.getenv("REMOTE_RUNNER_TOKEN")
LOG_DIR = Path(os.getenv("REMOTE_RUNNER_LOG_DIR", "logs"))
PORT = int(os.getenv("PORT", "8000"))


def _auth_ok(handler: BaseHTTPRequestHandler) -> bool:
    if not TOKEN:
        return True
    auth = handler.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1].strip() == TOKEN
    return handler.headers.get("X-Token") == TOKEN


class RemoteRunnerHandler(BaseHTTPRequestHandler):
    server_version = "RemoteRunner/1.0"

    def log_message(self, format, *args):  # noqa: A003 - keep default signature
        # Render captures stdout; keep logs concise.
        sys.stdout.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path in ("/", "/health"):
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/run":
            self._send_json(404, {"error": "not_found"})
            return
        if not _auth_ok(self):
            self._send_json(401, {"error": "unauthorized"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b""
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid_json"})
            return

        args = payload.get("args")
        if not isinstance(args, list) or not all(isinstance(arg, str) and arg for arg in args):
            self._send_json(400, {"error": "args_must_be_list_of_strings"})
            return

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"run_{timestamp}_{uuid.uuid4().hex[:8]}.log"

        with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
            process = subprocess.Popen(
                [sys.executable, "callrail_tool.py", *args],
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )

        self._send_json(202, {"status": "started", "pid": process.pid, "log_path": str(log_path)})


def main() -> None:
    server = ThreadingHTTPServer(("", PORT), RemoteRunnerHandler)
    print(f"Remote runner listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
