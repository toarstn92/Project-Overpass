"""
Local web launcher for Project Overpass.

Serves ``launcher/index.html`` on 127.0.0.1 and exposes a small JSON API that
walks the user through:

1. Configuring the OpenAI API key (written to ``.env``)
2. Creating ``.venv``, installing ``requirements.txt`` and passing ``pip-audit``
3. Running the FastAPI backend, the Streamlit frontend, or the pytest suite
   (pip-audit is re-run and must pass before every launch)

Standard library only, so it runs before any dependency is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from launcher.audit import ROOT, pip_audit_command

LAUNCHER_DIR = Path(__file__).resolve().parent
INDEX_HTML = LAUNCHER_DIR / "index.html"
ENV_FILE = ROOT / ".env"
ENV_TEMPLATE = ROOT / "_env"
REQUIREMENTS = ROOT / "requirements.txt"
VENV_DIR = ROOT / ".venv"
SETUP_MARKER = VENV_DIR / ".overpass-setup"
CHROMA_SQLITE = ROOT / "data" / "chroma" / "chroma.sqlite3"

PLACEHOLDER_KEY = "sk-...your-key-here"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_API_URL = "http://localhost:8000"
FRONTEND_PORT = 8501
MAX_LOG_LINES = 5000
MAX_BODY_BYTES = 64 * 1024
API_STARTUP_TIMEOUT = 15 * 60  # first start downloads the embedding model


# ── Environment / .env helpers ──────────────────────────────────────────────

def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def write_env_values(updates: dict[str, str]) -> None:
    """Update keys in .env (created from the _env template if missing), keeping other lines."""
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    elif ENV_TEMPLATE.exists():
        lines = ENV_TEMPLATE.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    seen: set[str] = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            lines[i] = f"{key}={updates[key]}"
            seen.add(key)
    lines += [f"{k}={v}" for k, v in updates.items() if k not in seen]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def api_key_configured() -> bool:
    key = read_env_file().get("OPENAI_API_KEY", "")
    return bool(key) and key != PLACEHOLDER_KEY


def mask_key(key: str) -> str:
    return f"{key[:3]}…{key[-4:]}" if len(key) > 8 else "••••"


def api_base_url() -> str:
    return read_env_file().get("API_URL") or DEFAULT_API_URL


def api_port() -> int:
    return urlparse(api_base_url()).port or 8000


def requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def setup_complete() -> bool:
    return (
        venv_python().exists()
        and SETUP_MARKER.exists()
        and SETUP_MARKER.read_text(encoding="utf-8").strip() == requirements_hash()
    )


def verify_openai_key(key: str) -> tuple[bool | None, str]:
    """Return (True|False|None, message). None means OpenAI could not be reached."""
    request = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15):
            return True, "API key verified with OpenAI."
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "OpenAI rejected this key (401 Unauthorized)."
        if exc.code == 429:
            return True, "Key accepted, but OpenAI reports a rate-limit/quota issue (429)."
        return False, f"OpenAI returned HTTP {exc.code} while verifying the key."
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return None, f"Could not reach OpenAI to verify the key: {reason}"


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


# ── Jobs (log-capturing subprocess pipelines) ───────────────────────────────

class JobStopped(Exception):
    pass


class Job:
    """A named unit of work with a log buffer and an optional child process."""

    def __init__(self, name: str):
        self.name = name
        self.state = "idle"   # idle | running | ready | succeeded | failed | stopped
        self.message = ""
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self._lines: list[str] = []
        self._offset = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.state in ("running", "ready")

    def log(self, text: str) -> None:
        with self._lock:
            self._lines.extend(text.splitlines() or [""])
            overflow = len(self._lines) - MAX_LOG_LINES
            if overflow > 0:
                del self._lines[:overflow]
                self._offset += overflow

    def read_log(self, since: int) -> tuple[list[str], int]:
        with self._lock:
            start = max(since - self._offset, 0)
            return self._lines[start:], self._offset + len(self._lines)

    def set(self, state: str, message: str = "") -> None:
        if self._stop.is_set() and state != "running":
            state, message = "stopped", "Stopped."
            if self.state == "stopped":
                return
        self.state, self.message = state, message
        if message:
            self.log(f"── {message}")

    def check_stop(self) -> None:
        if self._stop.is_set():
            raise JobStopped()

    def start(self, target: Callable[["Job"], None]) -> bool:
        if self.busy:
            return False
        self._stop.clear()
        with self._lock:
            self._offset += len(self._lines)
            self._lines = []
        self.set("running")
        self.thread = threading.Thread(target=self._run, args=(target,), daemon=True)
        self.thread.start()
        return True

    def _run(self, target: Callable[["Job"], None]) -> None:
        try:
            target(self)
        except JobStopped:
            self.set("stopped", "Stopped.")
        except Exception as exc:  # surface unexpected launcher errors in the UI
            self.set("failed", f"Launcher error: {exc}")

    def stop(self) -> None:
        if not self.busy:
            return
        self._stop.set()
        was_ready = self.state == "ready"
        if was_ready:
            self.set("stopped")
        proc = self.proc
        if proc and proc.poll() is None:
            self.log("── Stopping process…")
            kill_tree(proc)


def kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _spawn(job: Job, cmd: list[str]) -> subprocess.Popen:
    job.check_stop()
    shown = " ".join(Path(cmd[0]).name if i == 0 else c for i, c in enumerate(cmd))
    job.log(f"$ {shown}")
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=child_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    job.proc = proc
    return proc


def run_step(job: Job, cmd: list[str]) -> int:
    """Run a command to completion, streaming its output into the job log."""
    proc = _spawn(job, cmd)
    assert proc.stdout is not None
    for line in proc.stdout:
        job.log(line.rstrip("\n"))
    code = proc.wait()
    job.proc = None
    job.check_stop()
    return code


def start_service(job: Job, cmd: list[str]) -> subprocess.Popen:
    """Start a long-running command whose output is streamed in the background."""
    proc = _spawn(job, cmd)

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            job.log(line.rstrip("\n"))
        code = proc.wait()
        if job.state in ("running", "ready"):
            job.set("failed" if code else "stopped", f"Process exited with code {code}.")

    threading.Thread(target=pump, daemon=True).start()
    return proc


def wait_until_up(job: Job, proc: subprocess.Popen, url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job.check_stop()
        if proc.poll() is not None:
            return False
        if http_ok(url):
            return True
        time.sleep(1.5)
    return False


# ── Launcher state & pipelines ──────────────────────────────────────────────

class Launcher:
    def __init__(self, open_browser: bool = True):
        self.open_browser = open_browser
        self.jobs = {name: Job(name) for name in ("setup", "api", "frontend", "tests")}
        self._api_lock = threading.Lock()

    # Step 2 – environment + dependency audit
    def setup(self, job: Job) -> None:
        py = venv_python()
        if not py.exists():
            job.log(f"Creating virtual environment in {VENV_DIR} …")
            if run_step(job, [sys.executable, "-m", "venv", str(VENV_DIR)]) != 0:
                return job.set("failed", "Could not create the virtual environment.")
        if run_step(job, [str(py), "-m", "pip", "install", "--upgrade", "pip"]) != 0:
            return job.set("failed", "Upgrading pip failed.")
        if run_step(job, [str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)]) != 0:
            hint = ""
            if os.name == "nt" and len(str(VENV_DIR)) > 80:
                hint = (" The project path is long — enable Windows long-path support or restart with "
                        "'python init.py --venv C:\\overpass-venv'.")
            return job.set("failed", "Installing requirements.txt failed." + hint)
        if not self.audit(job):
            SETUP_MARKER.unlink(missing_ok=True)
            return job.set("failed", "pip-audit found vulnerabilities — update requirements.txt and retry.")
        SETUP_MARKER.write_text(requirements_hash(), encoding="utf-8")
        job.set("succeeded", "Environment ready and pip-audit passed.")

    def audit(self, job: Job) -> bool:
        job.log("Running pip-audit (must pass before anything runs)…")
        ok = run_step(job, pip_audit_command(venv_python())) == 0
        job.log("── pip-audit passed." if ok else "── pip-audit FAILED.")
        return ok

    # Step 3 – run targets
    def run_api(self, job: Job) -> None:
        if not self.audit(job):
            return job.set("failed", "Blocked: pip-audit failed.")
        self._start_api(job)

    def _start_api(self, job: Job) -> bool:
        health = f"{api_base_url()}/health"
        if http_ok(health):
            job.set("ready", f"API is already running at {api_base_url()}.")
            return True
        if not CHROMA_SQLITE.exists():
            job.log("Wisdom corpus not indexed yet — ingesting local sources (first run only)…")
            if run_step(job, [str(venv_python()), "-m", "data.ingest", "--local"]) != 0:
                job.set("failed", "Corpus ingestion failed.")
                return False
        proc = start_service(job, [
            str(venv_python()), "-m", "uvicorn", "api.main:app",
            "--host", "127.0.0.1", "--port", str(api_port()),
        ])
        job.log("Waiting for the API to become healthy (first start downloads the embedding model)…")
        if wait_until_up(job, proc, health, API_STARTUP_TIMEOUT):
            job.set("ready", f"API running — docs at {api_base_url()}/docs")
            if self.open_browser:
                webbrowser.open(f"{api_base_url()}/docs")
            return True
        if proc.poll() is None:
            kill_tree(proc)
        job.set("failed", "API did not become healthy.")
        return False

    def ensure_api(self, job: Job) -> bool:
        """Make sure the backend is up (starting it in the 'api' job if needed)."""
        with self._api_lock:
            api_job = self.jobs["api"]
            if http_ok(f"{api_base_url()}/health"):
                job.log(f"API is running at {api_base_url()}.")
                return True
            if api_job.busy and api_job.thread and api_job.thread.is_alive():
                job.log("Waiting for the API that is already starting…")
            else:
                job.log("Starting the API (see the API log)…")
                api_job.start(self._start_api)
        while api_job.thread and api_job.thread.is_alive():
            job.check_stop()
            time.sleep(1)
        ok = api_job.state == "ready"
        if not ok:
            job.log("API failed to start — check the API log.")
        return ok

    def run_frontend(self, job: Job) -> None:
        if not self.audit(job):
            return job.set("failed", "Blocked: pip-audit failed.")
        if not self.ensure_api(job):
            return job.set("failed", "Frontend needs the API, which failed to start.")
        url = f"http://localhost:{FRONTEND_PORT}"
        proc = start_service(job, [
            str(venv_python()), "-m", "streamlit", "run", "frontend/app.py",
            "--server.headless", "true", "--server.port", str(FRONTEND_PORT),
            "--server.address", "localhost", "--browser.gatherUsageStats", "false",
        ])
        if wait_until_up(job, proc, url, 180):
            job.set("ready", f"Frontend running at {url}")
            if self.open_browser:
                webbrowser.open(url)
        else:
            if proc.poll() is None:
                kill_tree(proc)
            job.set("failed", "Frontend did not start.")

    def run_tests(self, job: Job) -> None:
        if not self.audit(job):
            return job.set("failed", "Blocked: pip-audit failed.")
        if not self.ensure_api(job):
            return job.set("failed", "Tests need the API, which failed to start.")
        code = run_step(job, [str(venv_python()), "-m", "pytest", "tests", "-v", "-s"])
        job.set("succeeded" if code == 0 else "failed", f"pytest finished with exit code {code}.")

    def stop_all(self) -> None:
        for job in self.jobs.values():
            job.stop()

    def status(self) -> dict:
        env = read_env_file()
        key = env.get("OPENAI_API_KEY", "")
        configured = api_key_configured()
        return {
            "python": sys.version.split()[0],
            "key": {
                "configured": configured,
                "masked": mask_key(key) if configured else "",
                "model": env.get("LLM_MODEL") or DEFAULT_MODEL,
            },
            "setup": {"complete": setup_complete(), "venv": venv_python().exists()},
            "jobs": {n: {"state": j.state, "message": j.message} for n, j in self.jobs.items()},
            "urls": {
                "api": api_base_url(),
                "docs": f"{api_base_url()}/docs",
                "frontend": f"http://localhost:{FRONTEND_PORT}",
            },
        }


# ── HTTP layer ──────────────────────────────────────────────────────────────

def make_handler(launcher: Launcher, token: str, allowed_hosts: set[str], shutdown: Callable[[], None]):
    runners = {
        "api": launcher.run_api,
        "frontend": launcher.run_frontend,
        "tests": launcher.run_tests,
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "OverpassLauncher/1.0"

        def log_message(self, *args) -> None:  # keep the console quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _guard(self, api: bool) -> bool:
            # Host check blocks DNS-rebinding; the token (sent as a custom header,
            # which forces a CORS preflight we never approve) blocks cross-site requests.
            if self.headers.get("Host", "") not in allowed_hosts:
                self._json({"error": "Forbidden host."}, 403)
                return False
            if api and not secrets.compare_digest(self.headers.get("X-Launcher-Token", ""), token):
                self._json({"error": "Invalid launcher token."}, 403)
                return False
            return True

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ValueError("Request body too large.")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            return data

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                if not self._guard(api=False):
                    return
                html = INDEX_HTML.read_text(encoding="utf-8").replace("__LAUNCHER_TOKEN__", token)
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            if not self._guard(api=True):
                return
            if url.path == "/api/status":
                return self._json(launcher.status())
            if url.path == "/api/logs":
                query = parse_qs(url.query)
                job = launcher.jobs.get(query.get("job", [""])[0])
                if not job:
                    return self._json({"error": "Unknown job."}, 404)
                since = int(query.get("since", ["0"])[0] or 0)
                lines, nxt = job.read_log(since)
                return self._json({"lines": lines, "next": nxt, "state": job.state})
            self._json({"error": "Not found."}, 404)

        def do_POST(self) -> None:
            if not self._guard(api=True):
                return
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json({"error": str(exc)}, 400)
            path = urlparse(self.path).path

            if path == "/api/key":
                return self._save_key(body)

            if path == "/api/setup":
                if not api_key_configured():
                    return self._json({"error": "Configure the OpenAI API key first."}, 409)
                if not launcher.jobs["setup"].start(launcher.setup):
                    return self._json({"error": "Setup is already running."}, 409)
                return self._json({"ok": True})

            if path == "/api/run":
                target = body.get("target")
                if target not in runners:
                    return self._json({"error": "Unknown target."}, 400)
                if not api_key_configured():
                    return self._json({"error": "Configure the OpenAI API key first."}, 409)
                if not setup_complete():
                    return self._json({"error": "Complete step 2 (install + pip-audit) first."}, 409)
                if launcher.jobs["setup"].busy:
                    return self._json({"error": "Wait for setup to finish."}, 409)
                if not launcher.jobs[target].start(runners[target]):
                    return self._json({"error": f"{target} is already running."}, 409)
                return self._json({"ok": True})

            if path == "/api/stop":
                job = launcher.jobs.get(body.get("target", ""))
                if not job:
                    return self._json({"error": "Unknown target."}, 400)
                job.stop()
                return self._json({"ok": True})

            if path == "/api/shutdown":
                self._json({"ok": True})
                threading.Thread(target=shutdown, daemon=True).start()
                return
            self._json({"error": "Not found."}, 404)

        def _save_key(self, body: dict) -> None:
            key = str(body.get("api_key", "")).strip()
            model = str(body.get("model", "")).strip() or DEFAULT_MODEL
            if not key.startswith("sk-") or len(key) < 20 or any(c.isspace() for c in key):
                return self._json({"error": "That does not look like an OpenAI API key (expected 'sk-…')."}, 400)
            if any(c.isspace() or c in "\"'#" for c in model):
                return self._json({"error": "Invalid model name."}, 400)
            valid, message = verify_openai_key(key)
            if valid is False:
                return self._json({"error": message}, 400)
            if valid is None and not body.get("skip_verify"):
                return self._json({"error": message, "can_skip": True}, 502)
            write_env_values({"OPENAI_API_KEY": key, "LLM_MODEL": model})
            if valid is None:
                message = "Saved without verification (OpenAI was unreachable)."
            self._json({"ok": True, "message": message, "masked": mask_key(key)})

    return Handler


class LauncherHTTPServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server silently share a busy port.
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open the Project Overpass setup page.")
    parser.add_argument("--port", type=int, default=8765, help="Launcher port (default 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser windows automatically")
    parser.add_argument("--venv", type=Path, default=os.environ.get("OVERPASS_VENV"),
                        help="Virtual environment location (default: .venv in the project folder)")
    args = parser.parse_args(argv)

    if args.venv:
        global VENV_DIR, SETUP_MARKER
        VENV_DIR = Path(args.venv).expanduser().resolve()
        SETUP_MARKER = VENV_DIR / ".overpass-setup"

    if sys.version_info < (3, 10):
        print("Project Overpass needs Python 3.10+ (developed with 3.12).", file=sys.stderr)
        return 1

    launcher = Launcher(open_browser=not args.no_browser)
    token = secrets.token_urlsafe(32)
    allowed_hosts: set[str] = set()
    server_ref: dict[str, ThreadingHTTPServer] = {}

    def shutdown() -> None:
        launcher.stop_all()
        server_ref["server"].shutdown()

    handler = make_handler(launcher, token, allowed_hosts, shutdown)
    try:
        server = LauncherHTTPServer(("127.0.0.1", args.port), handler)
    except OSError:
        server = LauncherHTTPServer(("127.0.0.1", 0), handler)  # port busy: pick a free one
    server_ref["server"] = server
    port = server.server_address[1]
    allowed_hosts.update({f"127.0.0.1:{port}", f"localhost:{port}"})

    url = f"http://127.0.0.1:{port}/"
    print(f"Project Overpass launcher running at {url}  (Ctrl+C to quit)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        launcher.stop_all()
        server.server_close()
    return 0
