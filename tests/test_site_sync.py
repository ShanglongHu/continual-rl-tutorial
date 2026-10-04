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

    def test_published_source_assets_resolve_to_existing_repository_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('implementations/group/update.py','tests/test_example.py','tests/test space.py'):
                file=root/name
                file.parent.mkdir(parents=True,exist_ok=True)
                file.write_text('# source\n')
            cases={
                '/crl-code/implementations/group/update.py':'../implementations/group/update.py',
                'https://yingwen.io/crl-code/tests/test_example.py#L4':'../tests/test_example.py#L4',
                '/crl-code/tests/test%20space.py':'../tests/test%20space.py',
            }
            with patch.object(sync,'ROOT',root):
                for href,expected in cases.items():
                    with self.subTest(href=href):
                        output=sync.rewrite_markdown('[source]('+href+')','a.md','docs/a.md',{},set())
                        self.assertEqual(output,'[source]('+expected+')')

    def test_published_downloads_queries_and_missing_sources_stay_online(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'tests').mkdir()
            (root/'tests/example.py').write_text('# source\n')
            (root/'tests/not-code.txt').write_text('not a Python asset\n')
            with patch.object(sync,'ROOT',root):
                for href in ('/crl-code/learning-code.zip',
                             '/crl-code/results/example/curves.svg#plot',
                             '/crl-code/tests/missing.py',
                             '/crl-code/tests/',
                             '/crl-code/tests/not-code.txt',
                             '/crl-code/tests/example.py?raw=1#L2'):
                    with self.subTest(href=href):
                        output=sync.rewrite_markdown('[asset]('+href+')','a.md','docs/a.md',{},set())
                        self.assertEqual(output,'[asset](https://yingwen.io'+href+')')

    def test_published_source_paths_do_not_rewrite_external_hosts(self):
        for href in ('https://example.org/crl-code/tests/test_example.py',
                     'https://yingwen.io.example.org/crl-code/tests/test_example.py',
                     '//example.org/crl-code/implementations/update.py'):
            text='[external]('+href+')'
            with self.subTest(href=href):
                self.assertEqual(sync.rewrite_markdown(text,'a.md','docs/a.md',{},set()),text)

    def test_published_source_paths_reject_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root=Path(directory)
            (root/'tests').mkdir()
            (root/'tests/example.py').write_text('# source\n')
            (root/'private.py').write_text('# not a published source root\n')
            (Path(outside)/'outside.py').write_text('# outside repository\n')
            (root/'tests/escape.py').symlink_to(Path(outside)/'outside.py')
            (root/'tests/external').symlink_to(outside,target_is_directory=True)
            paths=('/crl-code/tests/%2e%2e/private.py',
                   '/crl-code/tests/../tests/example.py',
                   '/crl-code/tests/%2e%2e%2fprivate.py',
                   '/crl-code/%2ftests/example.py',
                   '/crl-code/tests/%5c..%5cprivate.py',
                   '/crl-code/tests/%00.py',
                   '/crl-code/tests/%ff.py',
                   '/crl-code/tests/escape.py',
                   '/crl-code/tests/external/outside.py')
            with patch.object(sync,'ROOT',root):
                for href in paths:
                    with self.subTest(href=href):
                        output=sync.rewrite_markdown('[unsafe]('+href+')','a.md','docs/a.md',{},set())
                        self.assertTrue(output.startswith('[unsafe](https://yingwen.io/crl-code/'),output)
                        self.assertNotIn(outside,output)

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
