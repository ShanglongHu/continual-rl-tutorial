import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('site_sync',ROOT/'scripts/sync_site.py')
sync=importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)

class PublicExport(unittest.TestCase):
    def test_relative_download_and_code_paths(self):
        text='[code](test_lab.py)\n[chapter](https://yingwen.io/zh/continual-rl/algorithms/value/)\npython3 test_lab.py test'
        output=sync.rewrite_markdown(text,'chapter-value.md','textbook/value.md',{'download/test_lab.py':'examples/test_lab.py','algorithms/value/':'textbook/value.md'},{'test_lab.py'})
        self.assertIn('[code](../examples/test_lab.py)',output)
        self.assertIn('[chapter](value.md)',output)
        self.assertIn('python3 examples/test_lab.py test',output)

    def test_external_links_are_not_rewritten(self):
        text='[original](https://example.org/paper#equation)'
        self.assertEqual(sync.rewrite_markdown(text,'a.md','docs/a.md',{},set()),text)

    def test_unsafe_paths_are_rejected(self):
        for target in ('../../README.md','/tmp/x','examples/../../x','.','unknown/x'):
            with self.assertRaises(ValueError):
                sync.safe_target(target)

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            (Path(root)/'examples').symlink_to(outside,target_is_directory=True)
            with patch.object(sync,'ROOT',Path(root)), self.assertRaises(ValueError):
                sync.safe_target('examples/test.py')

    def test_query_reference_keeps_online_semantics(self):
        output=sync.rewrite_markdown('[resources](/zh/continual-rl/library/?chapter=gvf)','chapter-gvf.md','textbook/gvf.md',{},set())
        self.assertEqual(output,'[resources](https://yingwen.io/zh/continual-rl/library/?chapter=gvf)')

    def test_only_existing_markdown_anchors_are_rewritten(self):
        href='https://yingwen.io/zh/continual-rl/algorithms/value/'
        links={'algorithms/value/':'textbook/value.md'}
        anchors={'textbook/value.md':{'lesson-derive'}}
        result=sync.rewrite_markdown('[derive]('+href+'#lesson-derive)\n[alias]('+href+'#algorithm-example)','chapter-value.md','textbook/value.md',links,set(),anchors)
        self.assertIn('[derive](value.md#lesson-derive)',result)
        self.assertIn('[alias]('+href+'#algorithm-example)',result)

    def test_sync_preserves_untracked_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'examples').mkdir()
            own=root/'examples/owned.py'
            own.write_text('user work')
            with patch.object(sync,'ROOT',root), patch.object(sync,'expected_files',return_value=({'examples/owned.py':b'generated'}, {'files':{}})), patch.object(sync.subprocess,'check_output',side_effect=['','examples/owned.py\n']):
                with self.assertRaises(RuntimeError):
                    sync.sync(root)
            self.assertEqual(own.read_text(),'user work')

if __name__=='__main__':
    unittest.main()
