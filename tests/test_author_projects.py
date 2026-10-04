"""作者工程身份、清洁状态与非覆盖约定；不联网安装或训练。"""
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("author_project", ROOT / "scripts/author_project.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AuthorProjectTests(unittest.TestCase):
    def test_registry_keeps_real_identity_and_entrypoints(self):
        registry = module.projects()
        self.assertEqual(set(registry), {"dreamerv3", "tdmpc2", "disco_rl"})
        self.assertIn("reimplementation", registry["dreamerv3"]["identity"])
        self.assertEqual(registry["tdmpc2"]["entrypoint"]["cwd"], "tdmpc2")
        self.assertEqual(len(registry["disco_rl"]["entrypoint"]["notebooks"]), 2)
        self.assertTrue(all("not_trained" in p["status"] for p in registry.values()))

    def test_prepare_refuses_existing_directory_before_network(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                module.prepare(module.projects()["dreamerv3"], temp)

    def test_verify_rejects_wrong_commit_and_dirty_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            module.git(directory, "init")
            module.git(directory, "config", "user.name", "Adapter Test")
            module.git(directory, "config", "user.email", "test@example.invalid")
            module.git(directory, "remote", "add", "origin", "https://github.com/example/project.git")
            module.git(directory, "commit", "--allow-empty", "-m", "fixture")
            project = dict(id="fixture", repository="https://github.com/example/project", commit=module.git(directory, "rev-parse", "HEAD"), reading_order=[])
            self.assertTrue(module.verify(project, directory)["clean"])
            project["commit"] = "0" * 40
            with self.assertRaises(ValueError):
                module.verify(project, directory)
            project["commit"] = module.git(directory, "rev-parse", "HEAD")
            marker = directory / "untracked-test-marker"
            marker.touch()
            with self.assertRaises(ValueError):
                module.verify(project, directory)
            marker.unlink()
            project["reading_order"] = ["missing.py"]
            with self.assertRaises(ValueError):
                module.verify(project, directory)
            project["reading_order"] = []
            module.git(directory, "remote", "set-url", "origin", "https://github.com/another/project.git")
            with self.assertRaises(ValueError):
                module.verify(project, directory)


if __name__ == "__main__":
    unittest.main()
