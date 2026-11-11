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
import re
import secrets
import shutil
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
from launcher.environments import (
    WINDOWS, EnvironmentLock, linked_directory, load_selection, previous_environments,
    project_name, python_in, save_selection, suggested_environment, valid_environment,
)

LAUNCHER_DIR = Path(__file__).resolve().parent
INDEX_HTML = LAUNCHER_DIR / "index.html"
ENV_FILE = ROOT / ".env"
ENV_TEMPLATE = ROOT / "_env"
REQUIREMENTS = ROOT / "requirements.txt"
VENV_DIR = ROOT / ".venv"
SETUP_MARKER = VENV_DIR / ".overpass-setup"
SELECTION_FILE = ROOT / ".overpass-launcher.json"
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
    return python_in(VENV_DIR)


def select_environment(directory: Path) -> None:
    global VENV_DIR, SETUP_MARKER
    VENV_DIR = directory.expanduser().resolve()
    SETUP_MARKER = VENV_DIR / ".overpass-setup"


def initial_environment(explicit: Path | None) -> Path:
    saved = load_selection(SELECTION_FILE)
    return Path(explicit or os.environ.get("OVERPASS_VENV") or saved.get("venv") or ROOT / ".venv")


def read_env_file(path: Path | None = None) -> dict[str, str]:
    path = ENV_FILE if path is None else path
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
        self.admission_lock = threading.Lock()
        self._environment_lock: EnvironmentLock | None = None
        self._locked_directory: Path | None = None
        self.failed_local = load_selection(SELECTION_FILE).get("failed_local", False)
        self.recovery_reason = ""

    def claim_environment(self) -> None:
        if self._locked_directory == VENV_DIR:
            return
        lease = EnvironmentLock(VENV_DIR)
        lease.acquire()
        self.release_environment()
        self._environment_lock = lease
        self._locked_directory = VENV_DIR

    def release_environment(self) -> None:
        if self._environment_lock:
            self._environment_lock.release()
        self._environment_lock = None
        self._locked_directory = None

    def recovery_options(self) -> dict:
        result = {
            "available": WINDOWS and bool(self.recovery_reason or self.failed_local),
            "reason": self.recovery_reason,
            "suggested": "", "previous": [], "error": "",
        }
        if WINDOWS:
            try:
                name = project_name(read_env_file(), read_env_file(ENV_TEMPLATE))
                result["suggested"] = str(suggested_environment(ROOT, name))
                result["previous"] = previous_environments(name)
            except (OSError, ValueError) as exc:
                result["error"] = str(exc)
        return result

    def start_setup(self, directory: str | None = None) -> None:
        """Caller holds admission_lock across environment selection and job admission."""
        if any(job.busy for job in self.jobs.values()):
            raise ValueError("Stop running services/tests and wait for setup before changing the environment.")
        if directory is not None:
            if not self.recovery_options()["available"]:
                raise ValueError("Short-path retry is not available for this setup.")
            if not isinstance(directory, str) or not directory.strip():
                raise ValueError("Choose an absolute environment path.")
            destination = Path(directory).expanduser()
            if not destination.is_absolute():
                raise ValueError("Choose an absolute environment path.")
            if destination.exists() and linked_directory(destination):
                raise ValueError("The replacement environment must not be a symlink or junction.")
            destination = destination.resolve()
            if destination == VENV_DIR or len(str(destination)) >= len(str(ROOT / ".venv")):
                raise ValueError("Choose a different environment path shorter than the project's .venv path.")
            if destination == ROOT or destination in ROOT.parents:
                raise ValueError("The replacement must not be the project folder or its parent.")
            if destination.exists() and not valid_environment(destination):
                raise ValueError("The selected folder exists but is not a Python virtual environment.")
            lease = EnvironmentLock(destination)
            lease.acquire()
            try:
                save_selection(SELECTION_FILE, destination, self.failed_local)
            except OSError:
                lease.release()
                raise
            self.release_environment()
            select_environment(destination)
            self._environment_lock, self._locked_directory = lease, destination
        else:
            self.claim_environment()
        if not self.jobs["setup"].start(self.setup):
            raise ValueError("Setup is already running.")

    def record_install_failure(self, job: Job, message: str = "Installing requirements.txt failed.") -> None:
        output = "\n".join(job.read_log(0)[0])
        explicit = bool(re.search(r"winerror\s*206|filename.{0,30}too long|file name.{0,30}too long", output, re.I))
        if WINDOWS and (explicit or len(str(VENV_DIR)) > 80):
            self.recovery_reason = (
                "Windows reported a path-length error." if explicit
                else "The environment path is long; this may have caused the installation failure. See the pip log."
            )
            local = ROOT / ".venv"
            if VENV_DIR == local.resolve() and local.exists() and not linked_directory(local):
                self.failed_local = secrets.token_hex(16)
                (local / ".overpass-failed").write_text(self.failed_local, encoding="utf-8")
            save_selection(SELECTION_FILE, VENV_DIR, self.failed_local)
            job.set("failed", message + " " + self.recovery_reason + " Try a shorter folder below.")
        else:
            job.set("failed", message + " See the setup log.")

    def cleanup_failed_local(self, job: Job) -> None:
        if not self.failed_local:
            return
        local = ROOT / ".venv"
        if local.exists():
            lease = EnvironmentLock(local)
            lease.acquire()
            try:
                if linked_directory(local) or local.resolve() == VENV_DIR or not valid_environment(local):
                    raise ValueError("Refusing to delete .venv: it is linked, selected, or not a valid environment.")
                if (local / ".overpass-failed").read_text(encoding="utf-8") != self.failed_local:
                    raise ValueError("Refusing to delete .venv: it no longer matches the failed environment.")
                job.log(f"Removing the failed environment at {local}")
                target = str(local.resolve())
                if os.name == "nt":
                    # The failed installation can itself contain paths longer than MAX_PATH.
                    target = "\\\\?\\UNC\\" + target[2:] if target.startswith("\\\\") else "\\\\?\\" + target
                shutil.rmtree(target)
            finally:
                lease.release()
        save_selection(SELECTION_FILE, VENV_DIR, False)
        self.failed_local = False

    # Step 2 – environment + dependency audit
    def setup(self, job: Job) -> None:
        SETUP_MARKER.unlink(missing_ok=True)
        py = venv_python()
        if not py.exists():
            job.log(f"Creating virtual environment in {VENV_DIR} …")
            if run_step(job, [sys.executable, "-m", "venv", str(VENV_DIR)]) != 0:
                return self.record_install_failure(job, "Could not create the virtual environment.")
        if run_step(job, [str(py), "-m", "pip", "install", "--upgrade", "pip"]) != 0:
            return self.record_install_failure(job, "Upgrading pip failed.")
        if run_step(job, [str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)]) != 0:
            return self.record_install_failure(job)
        if not self.audit(job):
            SETUP_MARKER.unlink(missing_ok=True)
            return job.set("failed", "pip-audit failed — see the log and resolve the audit error before retrying.")
        save_selection(SELECTION_FILE, VENV_DIR, self.failed_local)
        SETUP_MARKER.write_text(requirements_hash(), encoding="utf-8")
        if self.failed_local and VENV_DIR != (ROOT / ".venv").resolve():
            try:
                self.cleanup_failed_local(job)
            except (OSError, ValueError, RuntimeError) as exc:
                return job.set("succeeded", f"Environment ready and pip-audit passed, but cleanup is incomplete: {exc}")
        elif self.failed_local:
            save_selection(SELECTION_FILE, VENV_DIR, False)
            self.failed_local = False
        self.recovery_reason = ""
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
            "setup": {
                "complete": setup_complete(), "venv": venv_python().exists(),
                "path": str(VENV_DIR), "recovery": self.recovery_options(),
            },
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
                try:
                    with launcher.admission_lock:
                        if "venv" in body and not isinstance(body["venv"], str):
                            raise ValueError("Choose an absolute environment path.")
                        launcher.start_setup(body.get("venv"))
                except (OSError, ValueError, RuntimeError) as exc:
                    return self._json({"error": str(exc)}, 409)
                return self._json({"ok": True})

            if path == "/api/run":
                target = body.get("target")
                if target not in runners:
                    return self._json({"error": "Unknown target."}, 400)
                if not api_key_configured():
                    return self._json({"error": "Configure the OpenAI API key first."}, 409)
                with launcher.admission_lock:
                    if not setup_complete():
                        return self._json({"error": "Complete step 2 (install + pip-audit) first."}, 409)
                    if launcher.jobs["setup"].busy:
                        return self._json({"error": "Wait for setup to finish."}, 409)
                    try:
                        launcher.claim_environment()
                    except (OSError, RuntimeError) as exc:
                        return self._json({"error": str(exc)}, 409)
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
    parser.add_argument("--venv", type=Path,
                        help="Virtual environment location (default: .venv in the project folder)")
    args = parser.parse_args(argv)

    try:
        select_environment(initial_environment(args.venv))
    except (OSError, ValueError) as exc:
        print(f"Launcher configuration error: {exc}", file=sys.stderr)
        return 1

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
        launcher.release_environment()
        server.server_close()
    return 0
