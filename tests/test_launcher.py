"""Launcher recovery tests; no API key, dependencies, or network services required."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from launcher import environments as envs
from launcher import server


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / ("deep-checkout-" * 8)
        self.root.mkdir()
        self.local = self.root / ".venv"
        self.replacement = self.base / "venvs" / "project" / "short-code"
        self.settings = self.root / ".overpass-launcher.json"
        self.requirements = self.root / "requirements.txt"
        self.requirements.write_text("example==1\n", encoding="utf-8")
        for name, value in {
            "ROOT": self.root, "SELECTION_FILE": self.settings,
            "REQUIREMENTS": self.requirements, "WINDOWS": True,
            "VENV_DIR": self.local, "SETUP_MARKER": self.local / ".overpass-setup",
        }.items():
            override = patch.object(server, name, value)
            override.start()
            self.addCleanup(override.stop)
        environment = patch.dict(os.environ)
        environment.start()
        os.environ.pop("OVERPASS_VENV", None)
        self.addCleanup(environment.stop)
        self.launcher = server.Launcher(open_browser=False)
        self.addCleanup(self.launcher.release_environment)
        self.job = self.launcher.jobs["setup"]

    def make_environment(self, path):
        py = envs.python_in(path)
        py.parent.mkdir(parents=True, exist_ok=True)
        py.touch()
        (path / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")

    def mark_failure(self, output="OSError: [WinError 206] The filename is too long"):
        self.make_environment(self.local)
        self.job.log(output)
        self.launcher.record_install_failure(self.job)

    def run_pipeline(self, install_code=0, audit=True):
        commands = []

        def execute(job, command):
            commands.append(command)
            if command[1:3] == ["-m", "venv"]:
                self.make_environment(Path(command[3]))
            if "-r" in command:
                return install_code
            return 0

        with patch.object(server, "run_step", side_effect=execute), patch.object(
            self.launcher, "audit", return_value=audit
        ):
            self.launcher.setup(self.job)
        return commands

    def choose_replacement(self):
        with patch.object(self.job, "start", return_value=True):
            self.launcher.start_setup(str(self.replacement))

    def test_default_and_override_precedence(self):
        self.assertEqual(server.initial_environment(None), self.local)
        envs.save_selection(self.settings, self.replacement, False)
        self.assertEqual(server.initial_environment(None), self.replacement)
        os.environ["OVERPASS_VENV"] = str(self.base / "environment-override")
        self.assertEqual(server.initial_environment(None), Path(os.environ["OVERPASS_VENV"]))
        self.assertEqual(server.initial_environment(self.base / "cli"), self.base / "cli")

    def test_state_validation(self):
        for data in ("not json", "[]", '{"venv":"relative","failed_local":false}'):
            self.settings.write_text(data, encoding="utf-8")
            with self.assertRaises(ValueError):
                envs.load_selection(self.settings)

    def test_project_name_validation_and_template_fallback(self):
        self.assertEqual(envs.project_name({}, {"PROJECT_NAME": "Project-Overpass"}), "Project-Overpass")
        self.assertEqual(envs.project_name({"PROJECT_NAME": "override"}, {}), "override")
        for name in ("", "..", "a\\b", "a/b", "NUL", "con.txt", "a:", "name.", "name ", "x" * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                envs.project_name({"PROJECT_NAME": name}, {})

    def test_suggested_path_stable_and_worktree_specific(self):
        with patch.object(envs, "ENVIRONMENTS_ROOT", self.base / "venvs"):
            first = envs.suggested_environment(self.root, "Project-Overpass")
            self.assertEqual(first, envs.suggested_environment(self.root, "Project-Overpass"))
            self.assertNotEqual(first, envs.suggested_environment(self.root / "another", "Project-Overpass"))
            self.assertEqual(first.parent, self.base / "venvs" / "Project-Overpass")

    def test_lists_only_valid_immediate_project_environments(self):
        self.make_environment(self.replacement)
        self.make_environment(self.base / "venvs" / "other-project" / "other")
        (self.replacement.parent / "unrelated").mkdir()
        with patch.object(envs, "ENVIRONMENTS_ROOT", self.base / "venvs"):
            self.assertEqual(envs.previous_environments("project"), [str(self.replacement)])

    def test_confirmed_and_suspected_path_failures(self):
        self.mark_failure()
        self.assertIn("Windows reported", self.launcher.recovery_reason)
        self.assertTrue(self.launcher.failed_local)
        self.launcher.record_install_failure(server.Job("setup"))
        self.assertIn("may have caused", self.launcher.recovery_reason)

    def test_unrelated_failure_under_short_path_has_no_retry(self):
        server.select_environment(self.base / "v")
        job = server.Job("setup")
        job.log("Network connection failed")
        self.launcher.record_install_failure(job)
        self.assertFalse(self.launcher.recovery_options()["available"])
        self.assertIn("See the setup log", job.message)

    def test_non_windows_failure_has_no_recovery(self):
        with patch.object(server, "WINDOWS", False):
            self.mark_failure()
            self.assertFalse(self.launcher.recovery_options()["available"])

    def test_retry_uses_selected_environment_and_remembers_it(self):
        self.mark_failure()
        self.choose_replacement()
        commands = self.run_pipeline()
        self.assertEqual(self.job.state, "succeeded")
        self.assertFalse(self.local.exists())
        self.assertTrue(server.setup_complete())
        for command in commands:
            if command[1:3] != ["-m", "venv"]:
                self.assertEqual(command[0], str(envs.python_in(self.replacement)))
        self.assertEqual(server.initial_environment(None), self.replacement)
        restarted = server.Launcher(False)
        self.assertFalse(restarted.failed_local)

    def test_existing_environment_is_reused_and_audited(self):
        self.mark_failure()
        self.make_environment(self.replacement)
        self.choose_replacement()
        commands = self.run_pipeline()
        self.assertFalse(any(command[1:3] == ["-m", "venv"] for command in commands))
        self.assertEqual(self.job.state, "succeeded")
        self.assertFalse(self.local.exists())

    def test_failures_preserve_local_and_keep_step_three_locked(self):
        for failure in ("install", "audit"):
            with self.subTest(failure=failure):
                self.launcher.release_environment()
                server.select_environment(self.local)
                self.mark_failure()
                self.choose_replacement()
                self.replacement.mkdir(parents=True, exist_ok=True)
                server.SETUP_MARKER.write_text(server.requirements_hash(), encoding="utf-8")
                self.run_pipeline(install_code=int(failure == "install"), audit=failure != "audit")
                self.assertEqual(self.job.state, "failed")
                self.assertTrue(self.local.exists())
                self.assertFalse(server.setup_complete())

    def test_same_environment_failed_reinstall_invalidates_marker(self):
        self.make_environment(self.local)
        server.SETUP_MARKER.write_text(server.requirements_hash(), encoding="utf-8")
        self.run_pipeline(install_code=1)
        self.assertFalse(server.setup_complete())

    def test_invalid_destinations_and_busy_jobs_are_rejected(self):
        self.mark_failure()
        unrelated = self.base / "not-a-venv"
        unrelated.mkdir()
        for destination in ("relative", str(unrelated), str(self.root), str(self.local)):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                self.launcher.start_setup(destination)
        self.launcher.jobs["api"].state = "ready"
        with self.assertRaises(ValueError):
            self.launcher.start_setup(str(self.replacement))

    def test_path_cannot_be_supplied_before_failure(self):
        with self.assertRaises(ValueError):
            self.launcher.start_setup(str(self.replacement))

    def test_persistence_failure_prevents_switch(self):
        self.mark_failure()
        with patch.object(server, "save_selection", side_effect=PermissionError("read only")):
            with self.assertRaises(PermissionError):
                self.launcher.start_setup(str(self.replacement))
        self.assertEqual(server.VENV_DIR, self.local)
        self.assertTrue(self.local.exists())

    def test_locked_existing_environment_cannot_be_selected(self):
        self.mark_failure()
        self.make_environment(self.replacement)
        other = envs.EnvironmentLock(self.replacement)
        other.acquire()
        try:
            with self.assertRaises(RuntimeError):
                self.launcher.start_setup(str(self.replacement))
            self.assertEqual(server.VENV_DIR, self.local)
        finally:
            other.release()

    def test_cleanup_failure_is_reported_without_hiding_ready_environment(self):
        self.mark_failure()
        self.choose_replacement()
        with patch.object(server.shutil, "rmtree", side_effect=PermissionError("in use")):
            self.run_pipeline()
        self.assertTrue(server.setup_complete())
        self.assertTrue(self.local.exists())
        self.assertIn("cleanup is incomplete", self.job.message)

    def test_cleanup_refuses_changed_or_linked_environment(self):
        self.mark_failure()
        self.choose_replacement()
        (self.local / ".overpass-failed").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.launcher.cleanup_failed_local(self.job)
        with patch.object(server, "linked_directory", return_value=True):
            with self.assertRaises(ValueError):
                self.launcher.cleanup_failed_local(self.job)
        self.assertTrue(self.local.exists())

    @unittest.skipUnless(os.name == "nt", "Windows extended-length paths")
    def test_cleanup_handles_long_files_left_by_failed_install(self):
        self.mark_failure()
        self.choose_replacement()
        long_folder = Path("\\\\?\\" + str(self.local)) / ("package-" * 12) / ("module-" * 12)
        long_folder.mkdir(parents=True)
        (long_folder / "source.py").write_text("pass\n", encoding="utf-8")
        self.launcher.cleanup_failed_local(self.job)
        self.assertFalse(self.local.exists())

    def test_cleanup_refuses_environment_locked_by_another_launcher(self):
        self.mark_failure()
        self.choose_replacement()
        original_lock = envs.EnvironmentLock(self.local)
        original_lock.acquire()
        try:
            with self.assertRaises(RuntimeError):
                self.launcher.cleanup_failed_local(self.job)
            self.assertTrue(self.local.exists())
        finally:
            original_lock.release()

    def test_post_audit_persistence_failure_prevents_cleanup(self):
        self.mark_failure()
        self.choose_replacement()
        with patch.object(server, "save_selection", side_effect=PermissionError("read only")):
            with self.assertRaises(PermissionError):
                self.run_pipeline()
        self.assertTrue(self.local.exists())
        self.assertFalse(server.setup_complete())

    def test_failed_creation_and_pip_upgrade_stay_locked(self):
        self.mark_failure()
        self.choose_replacement()
        with patch.object(server, "run_step", return_value=1):
            self.launcher.setup(self.job)
        self.assertIn("Could not create", self.job.message)
        self.assertFalse(server.setup_complete())
        self.make_environment(self.replacement)
        with patch.object(server, "run_step", return_value=1):
            self.launcher.setup(self.job)
        self.assertIn("Upgrading pip failed", self.job.message)
        self.assertTrue(self.local.exists())
        self.assertFalse(server.setup_complete())

    def test_long_path_creation_failure_also_offers_recovery(self):
        with patch.object(server, "run_step", return_value=1):
            self.launcher.setup(self.job)
        self.assertIn("Could not create", self.job.message)
        self.assertTrue(self.launcher.recovery_options()["available"])

    def test_external_failed_environment_is_not_marked_for_deletion(self):
        server.select_environment(self.root / "external-environment")
        self.make_environment(server.VENV_DIR)
        self.launcher.record_install_failure(self.job)
        self.assertFalse(self.launcher.failed_local)

    def test_restart_preserves_pending_cleanup_identity(self):
        self.mark_failure()
        identifier = self.launcher.failed_local
        self.choose_replacement()
        restarted = server.Launcher(False)
        self.assertEqual(restarted.failed_local, identifier)

    def test_cross_process_environment_lock(self):
        lease = envs.EnvironmentLock(self.replacement)
        lease.acquire()
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 "from pathlib import Path; from launcher.environments import EnvironmentLock; "
                 "import sys; lock = EnvironmentLock(Path(sys.argv[1])); lock.acquire()",
                 str(self.replacement)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("in use", result.stderr)
        finally:
            lease.release()
        unlocked = envs.EnvironmentLock(self.replacement)
        unlocked.acquire()
        unlocked.release()

    def test_audit_and_launch_targets_use_selected_python(self):
        self.make_environment(self.replacement)
        server.select_environment(self.replacement)
        with patch.object(server, "run_step", return_value=0) as execute:
            self.assertTrue(self.launcher.audit(self.job))
            self.assertEqual(execute.call_args.args[1][0], str(envs.python_in(self.replacement)))
        for target in ("api", "frontend", "tests"):
            job = self.launcher.jobs[target]
            with patch.object(self.launcher, "audit", return_value=False), patch.object(
                self.launcher, "_start_api"
            ) as start_api, patch.object(self.launcher, "ensure_api") as ensure_api:
                getattr(self.launcher, "run_" + target)(job)
                self.assertEqual(job.state, "failed")
                start_api.assert_not_called()
                ensure_api.assert_not_called()
        with patch.object(self.launcher, "audit", return_value=True), patch.object(
            self.launcher, "ensure_api", return_value=True
        ), patch.object(server, "run_step", return_value=0) as execute:
            self.launcher.run_tests(self.launcher.jobs["tests"])
            self.assertEqual(execute.call_args.args[1][0], str(envs.python_in(self.replacement)))

    def test_api_ingestion_and_frontend_commands_use_selected_python(self):
        self.make_environment(self.replacement)
        server.select_environment(self.replacement)
        selected = str(envs.python_in(self.replacement))
        with patch.object(server, "http_ok", return_value=False), patch.object(
            server, "CHROMA_SQLITE", self.base / "not-ingested"
        ), patch.object(server, "run_step", return_value=0) as execute, patch.object(
            server, "start_service"
        ) as service, patch.object(server, "wait_until_up", return_value=True):
            self.assertTrue(self.launcher._start_api(self.launcher.jobs["api"]))
            self.assertEqual(execute.call_args.args[1][0], selected)
            self.assertEqual(service.call_args.args[1][0], selected)
        with patch.object(self.launcher, "audit", return_value=True), patch.object(
            self.launcher, "ensure_api", return_value=True
        ), patch.object(server, "start_service") as service, patch.object(
            server, "wait_until_up", return_value=True
        ):
            self.launcher.run_frontend(self.launcher.jobs["frontend"])
            self.assertEqual(service.call_args.args[1][0], selected)

    def test_protected_http_retry_and_status(self):
        self.mark_failure()
        hosts = set()
        handler = server.make_handler(self.launcher, "test-token", hosts, lambda: None)
        httpd = server.LauncherHTTPServer(("127.0.0.1", 0), handler)
        hosts.add(f"127.0.0.1:{httpd.server_port}")
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{httpd.server_port}"
        try:
            with self.assertRaises(urllib.error.HTTPError) as forbidden:
                urllib.request.urlopen(url + "/api/status")
            self.assertEqual(forbidden.exception.code, 403)
            with patch.object(server, "api_key_configured", return_value=True), patch.object(
                self.job, "start", return_value=True
            ):
                request = urllib.request.Request(
                    url + "/api/setup", json.dumps({"venv": str(self.replacement)}).encode(),
                    {"X-Launcher-Token": "test-token", "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request) as response:
                    self.assertTrue(json.load(response)["ok"])
            request = urllib.request.Request(url + "/api/status", headers={"X-Launcher-Token": "test-token"})
            with urllib.request.urlopen(request) as response:
                data = json.load(response)
            self.assertEqual(data["setup"]["path"], str(self.replacement))
            self.assertFalse(data["setup"]["complete"])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
