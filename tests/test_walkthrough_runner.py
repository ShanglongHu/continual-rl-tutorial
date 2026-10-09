"""Tests use local fixtures, not a stochastic training suite."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "scripts/run_walkthrough.py"
spec = importlib.util.spec_from_file_location("walkthrough_runner", MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class WalkthroughRunner(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="crl-runner-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        (self.root / "tutorials").mkdir()
        self.name = "sample.py"
        self.body = b"import sys\nprint('sample', sys.argv[1:])\n"
        (self.root / "tutorials" / self.name).write_bytes(self.body)
        self.manifest = {"files": [{"file": "walkthrough--sample.py",
                        "target": "tutorials/sample.py",
                        "sha256": hashlib.sha256(self.body).hexdigest()}]}
        self.index = {"groups": [{"title": "经典", "chapters": [{"title": "例子",
            "scripts": [{"file": self.name, "path": "/crl-code/tutorials/sample.py",
                         "sectionTitle": "手算"}]}]}]}
        self.write_index()

    def write_index(self):
        (self.root / "data/site-export.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.root / "data/walkthroughs.json").write_text(json.dumps(self.index), encoding="utf-8")

    def test_same_script_keeps_multiple_chapter_entries(self):
        self.index["groups"][0]["chapters"] *= 2
        self.write_index()
        self.assertEqual(len(runner.catalog(self.root)[self.name]["chapters"]), 2)

    def test_wrong_hash_refuses_execution_without_overwriting_learner_edit(self):
        file = self.root / "tutorials" / self.name
        file.write_bytes(b"# reader's edit\n")
        with patch.object(runner.subprocess, "run") as process:
            with self.assertRaisesRegex(ValueError, "不同于教材"):
                runner.run_one(self.root, self.name, runner.catalog(self.root)[self.name], [], 30)
            process.assert_not_called()
        self.assertEqual(file.read_bytes(), b"# reader's edit\n")

    def test_path_traversal_and_index_mismatch_fail(self):
        self.manifest["files"][0]["target"] = "tutorials/../outside.py"
        self.write_index()
        with self.assertRaisesRegex(ValueError, "不合法"):
            runner.catalog(self.root)
        self.manifest["files"][0]["target"] = "tutorials/sample.py"
        self.index["groups"][0]["chapters"][0]["scripts"][0]["path"] = "/wrong.py"
        self.write_index()
        with self.assertRaisesRegex(ValueError, "不符"):
            runner.catalog(self.root)

    def test_external_symlink_is_rejected(self):
        file = self.root / "tutorials" / self.name
        file.unlink()
        outside = self.root / "outside.py"
        outside.write_bytes(self.body)
        file.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "超出"):
            runner.source_bytes(self.root, self.name, runner.catalog(self.root)[self.name])

    def test_real_run_works_in_empty_directory(self):
        self.body = b"import pathlib,sys\nassert len(list(pathlib.Path.cwd().iterdir())) == 1\nassert sys.argv[1:] == ['--example']\n"
        (self.root / "tutorials" / self.name).write_bytes(self.body)
        self.manifest["files"][0]["sha256"] = hashlib.sha256(self.body).hexdigest()
        self.write_index()
        self.assertEqual(runner.run_one(self.root, self.name, runner.catalog(self.root)[self.name], ["--example"], 30), 0)

    def test_nonzero_and_timeout_are_reported(self):
        entry = runner.catalog(self.root)[self.name]
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)):
            self.assertEqual(runner.run_one(self.root, self.name, entry, [], 30), 7)
        with patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 1)):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.run_one(self.root, self.name, entry, [], 1), 124)

    def test_runner_status_does_not_pollute_script_stdout(self):
        entry = runner.catalog(self.root)[self.name]
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()) as status:
                self.assertEqual(runner.run_one(self.root, self.name, entry, [], 30), 0)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("运行 sample.py", status.getvalue())

    def test_listing_never_runs_code(self):
        with patch.object(runner, "ROOT", self.root), patch.object(runner.subprocess, "run") as process:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(runner.main(["--list"]), 0)
            self.assertIn("sample.py", output.getvalue())
            self.assertIn("经典 / 例子 / 手算", output.getvalue())
            process.assert_not_called()

    def test_unregistered_name_is_not_executed(self):
        with patch.object(runner, "ROOT", self.root), patch.object(runner.subprocess, "run") as process:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(["../outside.py"]), 2)
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
