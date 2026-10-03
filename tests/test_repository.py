"""Repository-level checks: portable docs, licenses, and the reproduction entrypoint."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_all", ROOT/"scripts/run_all.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Documentation(unittest.TestCase):
    def pages(self):
        return [p for folder in ("tutorials", "algorithms", "docs", "textbook", "foundations")
                for p in (ROOT/folder).rglob("*.md")] + list(ROOT.glob("*.md"))

    def test_chapter_inventory(self):
        self.assertEqual(len(list((ROOT/"tutorials").glob("[0-9][0-9]-*.md"))), 8)
        self.assertEqual(len(list((ROOT/"algorithms").glob("[0-9][0-9]-*.md"))), 13)
        for path in (ROOT/"algorithms").glob("[0-9][0-9]-*.md"):
            body = path.read_text(encoding="utf-8")
            for marker in ("预备知识与符号", "下载与运行", "参考文献与实现", "$$", "```python"):
                self.assertIn(marker, body, str(path.relative_to(ROOT)))

    def test_complete_textbook_and_prerequisites(self):
        inventory=json.loads((ROOT/'data/site-export.json').read_text())
        self.assertEqual(len(inventory['chapters']),22)
        self.assertGreaterEqual(len(inventory['lessons']),19)
        for chapter in inventory['chapters']:
            path=ROOT/'textbook'/(chapter['id']+'.md')
            self.assertTrue(path.exists())
            self.assertGreaterEqual(path.read_text().count('\n## '),7)
        for lesson in inventory['lessons']:
            path=ROOT/(lesson['path'].rstrip('/')+'.md')
            body=path.read_text()
            self.assertGreaterEqual(body.count('\n## '),8)
            self.assertIn('```python',body)
            self.assertIn('$$',body)
            self.assertTrue((ROOT/'examples'/lesson['file']).is_file())

    def test_site_sync_receipt(self):
        receipt=json.loads((ROOT/'data/site-sync.json').read_text())
        self.assertEqual(receipt['source_export_sha256'],hashlib.sha256((ROOT/'data/site-export.json').read_bytes()).hexdigest())
        for name,info in receipt['files'].items():
            self.assertEqual(info['sha256'],hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        for item in json.loads((ROOT/'data/site-export.json').read_text())['files']:
            if item['file'].endswith('.py'):
                self.assertEqual(item['sha256'],hashlib.sha256((ROOT/item['target']).read_bytes()).hexdigest())

    def test_relative_file_links(self):
        fence = chr(96)*3
        for path in self.pages():
            body = re.sub(fence+r".*?"+fence, "", path.read_text(encoding="utf-8"), flags=re.S)
            for href in re.findall(r"\]\(([^)\s]+)\)", body):
                parts = urlsplit(href)
                if parts.scheme or not parts.path:
                    continue
                dest = (path.parent/unquote(parts.path)).resolve()
                self.assertTrue(dest.is_relative_to(ROOT), str(path)+": "+href)
                self.assertTrue(dest.exists(), str(path)+": "+href)
                if parts.fragment and re.fullmatch(r"track-\d+", parts.fragment):
                    self.assertIn('id="'+parts.fragment+'"',
                                  dest.read_text(encoding="utf-8"))

    def test_no_private_paths_or_preview_dependencies(self):
        paths = self.pages()+list((ROOT/"examples").glob("*.py"))
        for path in paths:
            body = path.read_text(encoding="utf-8")
            for marker in ("/Users/", "/private/tmp/", "127.0.0.1:", "file://", "codex://",
                           "](/zh/continual-rl/", "src/data/crl/"):
                self.assertNotIn(marker, body, str(path.relative_to(ROOT)))

    def test_program_commands_resolve(self):
        for path in self.pages():
            for script in re.findall(r"python3\s+([A-Za-z0-9_./-]+\.py)",
                                     path.read_text(encoding="utf-8")):
                self.assertTrue((ROOT/script).is_file(), str(path)+": "+script)

    def test_license_boundaries(self):
        self.assertIn("MIT License", (ROOT/"LICENSE-CODE").read_text())
        self.assertIn("CC BY 4.0", (ROOT/"LICENSE-DOCS.md").read_text())
        self.assertIn("Third-party", (ROOT/"LICENSE-DOCS.md").read_text())


class Reproduction(unittest.TestCase):
    def test_unknown_git_status_is_not_clean(self):
        result=mock.Mock(returncode=128,stdout='')
        with mock.patch.object(runner.subprocess,'run',return_value=result):
            self.assertEqual(runner.git_info(),{'commit':None,'dirty':None})

    def test_manifest_and_all_outputs(self):
        with tempfile.TemporaryDirectory(prefix="crl-suite-") as temporary:
            out = Path(temporary)/"run"
            manifest = runner.run_suite(out, quick=True)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["mode"], "smoke")
            self.assertEqual(len(manifest["commands"]), 10)
            self.assertEqual(len(manifest["artifacts"]), 15)
            self.assertEqual(len(list(out.glob("*.csv"))), 9)
            self.assertEqual(len(list(out.glob("*.html"))), 6)
            for name, info in manifest["artifacts"].items():
                self.assertEqual(info["sha256"],
                                 hashlib.sha256((out/name).read_bytes()).hexdigest())
            for name, checksum in manifest["source_sha256"].items():
                self.assertEqual(checksum, hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
            with (out/"memory.csv").open() as handle:
                self.assertEqual({float(row["value"]) for row in csv.DictReader(handle)}, {.5, 1.})
            self.assertNotIn(temporary, (out/"manifest.json").read_text())
            for path in out.glob("*.csv"):
                self.assertNotIn(temporary, path.read_text())
            self.assertEqual(json.loads((out/"manifest.json").read_text())["status"], "complete")

    def test_refuses_existing_directory_without_mutation(self):
        with tempfile.TemporaryDirectory(prefix="crl-suite-") as temporary:
            marker = Path(temporary)/"user-data.txt"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                runner.run_suite(Path(temporary), quick=True)
            self.assertEqual(list(Path(temporary).iterdir()), [marker])
            self.assertEqual(marker.read_text(), "keep")

    def test_quick_and_teaching_configs_are_different(self):
        smoke, full = runner.experiment_commands(True), runner.experiment_commands(False)
        self.assertEqual([x[0] for x in smoke], [x[0] for x in full])
        self.assertNotEqual(smoke, full)
        for commands in [smoke, full]:
            for name, _, args in commands:
                if name in ("memory", "retention", "consolidation"):
                    self.assertEqual(args[args.index("--seeds")+1], "1")


if __name__ == "__main__":
    unittest.main()
