"""
Shared pip-audit configuration used by the launcher and the test-suite.

pip-audit has no config file, so the list of accepted (justified) vulnerability
IDs lives in ``pip-audit-ignore.txt`` at the repository root.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORE_FILE = ROOT / "pip-audit-ignore.txt"


def load_ignored_vulns(path: Path = IGNORE_FILE) -> list[str]:
    """Return the vulnerability IDs listed in the ignore file (comments stripped)."""
    if not path.exists():
        return []
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            ids.append(entry)
    return ids


def pip_audit_command(python: str | Path) -> list[str]:
    """
    Build the pip-audit command that audits the environment of ``python``.

    The installed environment is audited (rather than re-resolving
    requirements.txt) so that every transitive dependency actually in use is
    checked, and the check is fast enough to run before every app launch.
    """
    cmd = [
        str(python), "-m", "pip_audit",
        "--local",
        "--skip-editable",
        "--progress-spinner", "off",
    ]
    for vuln_id in load_ignored_vulns():
        cmd += ["--ignore-vuln", vuln_id]
    return cmd
