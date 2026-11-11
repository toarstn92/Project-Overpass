"""
Dependency security test — the installed environment must pass pip-audit.

Run with: pytest tests/test_pip_audit.py -v
Requires: pip-audit installed (it is in requirements.txt) and network access
to the PyPI / OSV vulnerability databases.
"""

from __future__ import annotations

import subprocess
import sys

from launcher.audit import IGNORE_FILE, load_ignored_vulns, pip_audit_command


def test_ignore_file_entries_are_justified():
    """Every ignored vulnerability must carry an inline justification comment."""
    for line in IGNORE_FILE.read_text(encoding="utf-8-sig").splitlines():
        entry, _, comment = line.partition("#")
        if entry.strip():
            assert comment.strip(), f"Ignored vulnerability {entry.strip()} has no justification"


def test_load_ignored_vulns_strips_comments():
    ids = load_ignored_vulns()
    assert ids, "pip-audit-ignore.txt is expected to list at least one ID"
    assert all(" " not in i and "#" not in i for i in ids)


def test_pip_audit_passes():
    result = subprocess.run(
        pip_audit_command(sys.executable),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
    )
    output = result.stdout + result.stderr
    print(output)
    assert result.returncode == 0, f"pip-audit reported vulnerabilities or failed:\n{output}"
