"""Local environment selection and cross-launcher environment locks (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

WINDOWS = os.name == "nt"
ENVIRONMENTS_ROOT = Path("C:\\venvs")


def python_in(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if WINDOWS else "bin/python")


def valid_environment(directory: Path) -> bool:
    return (directory / "pyvenv.cfg").is_file() and python_in(directory).is_file()


def linked_directory(directory: Path) -> bool:
    return directory.is_symlink() or bool(
        getattr(directory.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def project_name(values: dict[str, str], template: dict[str, str]) -> str:
    name = values.get("PROJECT_NAME", template.get("PROJECT_NAME", ""))
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
    if (
        not name or len(name) > 64
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
        or name in (".", "..") or name.endswith((".", " "))
        or name.split(".")[0].upper() in reserved
    ):
        raise ValueError("PROJECT_NAME must be a safe Windows folder name (1-64 characters).")
    return name


def suggested_environment(root: Path, name: str) -> Path:
    code = hashlib.sha256(os.path.normcase(str(root.resolve())).encode()).hexdigest()[:12]
    return ENVIRONMENTS_ROOT / name / code


def previous_environments(name: str) -> list[str]:
    parent = ENVIRONMENTS_ROOT / name
    if not parent.exists():
        return []
    return sorted(
        str(path) for path in parent.iterdir()
        if path.is_dir() and not linked_directory(path)
        and valid_environment(path)
    )


def load_selection(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read launcher settings {path}: {exc}") from exc
    if (
        not isinstance(data, dict) or set(data) != {"venv", "failed_local"}
        or not isinstance(data["venv"], str) or not data["venv"]
        or not Path(data["venv"]).is_absolute()
        or not (
            data["failed_local"] is False
            or isinstance(data["failed_local"], str)
            and re.fullmatch(r"[0-9a-f]{32}", data["failed_local"])
        )
    ):
        raise ValueError(f"Invalid launcher settings in {path}. Expected an absolute venv path and a failed-environment identifier.")
    return data


def save_selection(path: Path, directory: Path, failed_local: str | bool) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(
            json.dumps({"venv": str(directory), "failed_local": failed_local}) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class EnvironmentLock:
    """Keep other launcher processes from modifying an environment in use."""

    def __init__(self, directory: Path):
        self.path = directory.parent / f".{directory.name}.overpass.lock"
        self.file = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            if WINDOWS:
                import msvcrt
                stream.seek(0, 2)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError(
                f"Environment is in use by another launcher or cannot be locked: {self.path}. "
                "Stop the other launcher or choose a different folder."
            ) from exc
        self.file = stream

    def release(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
