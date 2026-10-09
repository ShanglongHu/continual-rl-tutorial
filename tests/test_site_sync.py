import importlib.util
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('site_sync',ROOT/'scripts/sync_site.py')
sync=importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)

class PublicExport(unittest.TestCase):
    def test_original_figure_first_sync_rewrites_without_an_existing_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sync, 'ROOT', Path(directory)):
                figures={'/crl-figures/example.svg':'assets/crl-figures/example.svg'}
                result=sync.rewrite_markdown('![图](https://yingwen.io/crl-figures/example.svg#panel)', 'a.md', 'foundations/tabular/a.md', {}, set(), published_figures=figures)
                self.assertEqual(result, '![图](../../assets/crl-figures/example.svg#panel)')
                self.assertEqual(sync.rewrite_markdown(result, 'a.md', 'foundations/tabular/a.md', {}, set(), published_figures=figures),result)

    def test_figure_queries_external_hosts_and_unsafe_paths_stay_online(self):
        figures={'/crl-figures/example.svg':'assets/crl-figures/example.svg'}
        cases=['https://example.org/crl-figures/example.svg', '/crl-figures/example.svg?raw=1', '/crl-figures/../crl-figures/example.svg', '/crl-figures/%2e%2e/crl-figures/example.svg']
        for href in cases:
            with self.subTest(href=href):
                result=sync.rewrite_markdown('![图]('+href+')','a.md','docs/a.md',{},set(),published_figures=figures)
                self.assertNotIn('../assets/',result)

    def test_clean_sync_keeps_exact_svg_bytes_and_source_hash(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as source:
            root,export=Path(directory),Path(source)
            svg='<svg xmlns="http://www.w3.org/2000/svg"><text>中文图</text></svg>\n'.encode()
            entries=[]
            for name,target,body in [('figure--example.svg','assets/crl-figures/example.svg',svg), ('a.md','docs/a.md',b'![diagram](/crl-figures/example.svg)')]:
                (export/name).write_bytes(body)
                entries.append({'file':name,'target':target,'sha256':sync.digest(body)})
            (export/'repository-export.json').write_text(json.dumps({'schema_version':1,'files':entries,'chapters':[],'lessons':[]}))
            with patch.object(sync,'ROOT',root), patch.object(sync,'LEGACY',[]), patch.object(sync.subprocess,'check_output',return_value=''):
                sync.sync(export)
                self.assertEqual(sync.sync(export,check=True),[])
            self.assertEqual((root/'assets/crl-figures/example.svg').read_bytes(),svg)
            self.assertEqual((root/'docs/a.md').read_text(),'![diagram](../assets/crl-figures/example.svg)')
            receipt=json.loads((root/'data/site-sync.json').read_text())
            self.assertEqual(receipt['files']['assets/crl-figures/example.svg']['source_sha256'],sync.digest(svg))

    def test_figure_declaration_and_symlink_cannot_escape_asset_root(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as outside:
            root,export=Path(directory),Path(source)
            (root/'assets').symlink_to(outside,target_is_directory=True)
            entry={'file':'figure--example.svg','target':'assets/crl-figures/example.svg','sha256':'unused'}
            (export/'repository-export.json').write_text(json.dumps({'schema_version':1,'files':[entry]}))
            with patch.object(sync,'ROOT',root), self.assertRaisesRegex(ValueError,'escapes'):
                sync.expected_files(export)
            for target in ['assets/private.txt','assets/crl-figures/nested/a.svg','assets/crl-figures/a.png']:
                with self.subTest(target=target),self.assertRaises(ValueError):
                    sync.safe_target(target)
            entry['target']='docs/example.svg'
            (export/'repository-export.json').write_text(json.dumps({'schema_version':1,'files':[entry]}))
            with patch.object(sync,'ROOT',root), self.assertRaisesRegex(ValueError,'Invalid figure declaration'):
                sync.expected_files(export)

    def test_first_sync_resolves_declared_tutorial_before_it_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            text='[源文件](/crl-code/tutorials/new-example.py)\n\n在文件所在目录运行：\n\n```bash\npython3 new-example.py\n```'
            with patch.object(sync,'ROOT',root):
                result=sync.rewrite_markdown(text,'a.md','docs/a.md',{},set(),published_tutorials={'tutorials/new-example.py'})
                self.assertIn('[源文件](../tutorials/new-example.py)',result)
                self.assertIn('python3 tutorials/new-example.py',result)
                self.assertIn('在仓库根目录运行',result)
                self.assertFalse((root/'tutorials/new-example.py').exists())

    def test_manifest_declaration_does_not_allow_external_symlink(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root=Path(directory)
            (root/'tutorials').symlink_to(outside,target_is_directory=True)
            text='[source](/crl-code/tutorials/new.py)\npython3 new.py'
            with patch.object(sync,'ROOT',root):
                result=sync.rewrite_markdown(text,'a.md','docs/a.md',{},set(),published_tutorials={'tutorials/new.py'})
                self.assertIn('https://yingwen.io/crl-code/tutorials/new.py',result)
                self.assertIn('\npython3 new.py',result)

    def test_clean_sync_includes_source_hashes_and_local_links_in_one_pass(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as source:
            root,export=Path(directory),Path(source)
            inputs=[('example.md','docs/example.md',b'[code](/crl-code/tutorials/new-example.py)\npython3 new-example.py\n'),
                    ('walkthrough--new-example.py','tutorials/new-example.py','# 中文\nprint(1)\n'.encode())]
            entries=[]
            for name,target,body in inputs:
                (export/name).write_bytes(body)
                entries.append({'file':name,'target':target,'sha256':sync.digest(body)})
            (export/'repository-export.json').write_text(json.dumps({'schema_version':1,'files':entries,'chapters':[],'lessons':[]}))
            with patch.object(sync,'ROOT',root), patch.object(sync,'LEGACY',[]), patch.object(sync.subprocess,'check_output',return_value=''):
                changed=sync.sync(export)
                self.assertIn('tutorials/new-example.py',changed)
                self.assertEqual(sync.sync(export,check=True),[])
            self.assertEqual((root/'tutorials/new-example.py').read_bytes(),inputs[1][2])
            self.assertIn('[code](../tutorials/new-example.py)',(root/'docs/example.md').read_text())
            self.assertIn('python3 tutorials/new-example.py',(root/'docs/example.md').read_text())
            receipt=json.loads((root/'data/site-sync.json').read_text())
            self.assertEqual(receipt['files']['tutorials/new-example.py']['source_sha256'],entries[1]['sha256'])

    def test_bad_source_checksum_prevents_any_sync_write(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as source:
            root,export=Path(directory),Path(source)
            (export/'walkthrough--new.py').write_text('# corrupted source\n')
            entry={'file':'walkthrough--new.py','target':'tutorials/new.py','sha256':sync.digest(b'# expected source\n')}
            (export/'repository-export.json').write_text(json.dumps({'schema_version':1,'files':[entry]}))
            with patch.object(sync,'ROOT',root), self.assertRaisesRegex(ValueError,'Source checksum mismatch'):
                sync.sync(export)
            self.assertEqual(list(root.iterdir()),[])

    def test_previously_synced_tutorial_edits_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'tutorials').mkdir();(root/'data').mkdir()
            source=root/'tutorials/example.py'
            source.write_text('# local changes\n')
            previous={'files':{'tutorials/example.py':{'sha256':sync.digest(b'# prior generated\n')}}}
            (root/'data/site-sync.json').write_text(json.dumps(previous))
            with patch.object(sync,'ROOT',root), patch.object(sync,'expected_files',return_value=({'tutorials/example.py':b'# new generated\n'},{'files':{}})), self.assertRaisesRegex(RuntimeError,'Modified generated file'):
                sync.sync(root)
            self.assertEqual(source.read_text(),'# local changes\n')

    def test_chapter_roles_and_display_numbers_are_independent_of_sort_keys(self):
        lessons=[dict(id=f"core-{i}", title=f"Core {i}", track="deep", order=i,
                      kind="core", displayOrdinal=i, path=f"foundations/deep/core-{i}/") for i in range(1, 7)]
        lessons += [dict(id="systems", title="Systems", track="deep", order=6.5,
                         kind="core", displayOrdinal=7, path="foundations/deep/systems/"),
                    dict(id="branch", title="Research", track="deep", order=.5,
                         kind="branch", displayOrdinal=None, path="foundations/deep/research/")]
        groups=sync.foundation_groups(list(reversed(lessons)), "deep")
        self.assertEqual([l["id"] for l in groups[0][1]], [f"core-{i}" for i in range(1, 7)]+["systems"])
        self.assertEqual([l["id"] for l in groups[1][1]], ["branch"])
        text=sync.foundation_links(groups[0][1])+"\n"+sync.foundation_links(groups[1][1])
        self.assertIn("[第 7 章 · Systems](systems.md)", text)
        self.assertIn("[Research](research.md)", text)
        self.assertNotIn("6.5", text)

    def test_stale_or_inconsistent_chapter_metadata_is_rejected(self):
        good=dict(id="systems", title="Systems", track="deep", order=6.5,
                  kind="core", displayOrdinal=1, path="foundations/deep/systems/")
        for bad in ({k:v for k,v in good.items() if k!="kind"},
                    {k:v for k,v in good.items() if k!="displayOrdinal"},
                    {**good,"displayOrdinal":6.5}, {**good,"displayOrdinal":2},
                    {**good,"displayOrdinal":True}, {**good,"kind":"branch"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sync.foundation_groups([bad], "deep")

    def test_relative_download_and_code_paths(self):
        text='[code](test_lab.py)\n[chapter](https://yingwen.io/zh/continual-rl/algorithms/value/)\npython3 test_lab.py test'
        output=sync.rewrite_markdown(text,'chapter-value.md','textbook/value.md',{'download/test_lab.py':'examples/test_lab.py','algorithms/value/':'textbook/value.md'},{'test_lab.py'})
        self.assertIn('[code](../examples/test_lab.py)',output)
        self.assertIn('[chapter](value.md)',output)
        self.assertIn('python3 examples/test_lab.py test',output)

    def test_external_links_are_not_rewritten(self):
        text='[original](https://example.org/paper#equation)'
        self.assertEqual(sync.rewrite_markdown(text,'a.md','docs/a.md',{},set()),text)

    def test_html_relative_chapter_routes_use_manifest_targets_and_anchors(self):
        links={'algorithms/streaming/':'textbook/streaming.md',
               'algorithms/credit-assignment/':'textbook/credit.md'}
        anchors={'textbook/streaming.md':{'lesson-learner-state'},'textbook/credit.md':set()}
        source=('[restore](../../algorithms/streaming/#lesson-learner-state)\n'
                '[credit](../../algorithms/credit-assignment/)\n'
                '[alias](../../algorithms/streaming/#browser-only)\n'
                '[query](../../algorithms/streaming/?mode=full)')
        expected=('[restore](../textbook/streaming.md#lesson-learner-state)\n'
                  '[credit](../textbook/credit.md)\n'
                  '[alias](https://yingwen.io/zh/continual-rl/algorithms/streaming/#browser-only)\n'
                  '[query](https://yingwen.io/zh/continual-rl/algorithms/streaming/?mode=full)')
        for source_name,target in [('chapter-state.md','algorithms/11-state.md'),
                                   ('algorithm-tutorials.md','docs/algorithm-tutorials.md')]:
            with self.subTest(target=target):
                self.assertEqual(sync.rewrite_markdown(source,source_name,target,links,set(),anchors),expected)

    def test_website_tutorial_command_paths_use_only_safe_published_sources(self):
        with tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as outside:
            root=Path(directory)
            (root/'tutorials').mkdir()
            (root/'tutorials/escape.py').symlink_to(Path(outside)/'escape.py')
            source=('python3 public/crl-code/tutorials/offline_support_walkthrough.py --test\n'
                    'python public/crl-code/tutorials/offline_support_walkthrough.py\n'
                    'python3 public/crl-code/tutorials/missing.py\n'
                    'python3 public/crl-code/tutorials/escape.py\n'
                    'python3 public/crl-code/tutorials/../unknown.py')
            expected=source.replace('public/crl-code/tutorials/offline_support_walkthrough.py',
                                    'tutorials/offline_support_walkthrough.py')
            with patch.object(sync,'ROOT',root):
                result=sync.rewrite_markdown(source,'a.md','foundations/deep/a.md',{},
                                             {'offline_support_walkthrough.py'},
                                             published_tutorials={'tutorials/offline_support_walkthrough.py','tutorials/escape.py'})
                self.assertEqual(result,expected)
                self.assertEqual(sync.rewrite_markdown(result,'a.md','foundations/deep/a.md',{},set()),result)

    def test_single_file_tutorial_commands_run_from_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'tutorials').mkdir()
            (root/'tutorials/gae_ppo_walkthrough.py').write_text('# example\n')
            text='python3 gae_ppo_walkthrough.py --test\npython gae_ppo_walkthrough.py\npython3 tutorials/gae_ppo_walkthrough.py --test\npython3 missing.py'
            with patch.object(sync,'ROOT',root):
                result=sync.rewrite_markdown(text,'a.md','foundations/deep/a.md',{},set())
                self.assertEqual(result, 'python3 tutorials/gae_ppo_walkthrough.py --test\npython tutorials/gae_ppo_walkthrough.py\npython3 tutorials/gae_ppo_walkthrough.py --test\npython3 missing.py')
                self.assertEqual(sync.rewrite_markdown(result,'a.md','foundations/deep/a.md',{},set()),result)

    def test_offline_parity_check_downloads_the_unbundled_website_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=('第一、二行在保存教程文件的目录运行；第三个命令在仓库根目录运行。'
                    '计算给定目标与解析位移，不采样或训练。\n\n'
                    '```bash\npython3 offline_support_walkthrough.py --test\n'
                    'python3 offline_support_walkthrough.py\n'
                    '# 在网站仓库根目录核对公开图中的数据：\n'
                    'python3 public/crl-code/tutorials/offline_support_walkthrough.py \\\n'
                    '  --compare public/crl-figures/offline-support-data.json\n```')
            with patch.object(sync,'ROOT',root):
                result=sync.rewrite_markdown(source,'study-deep-offline.md','foundations/deep/offline.md',
                                             {},set(),published_tutorials={'tutorials/offline_support_walkthrough.py'})
            self.assertIn('以下命令在教程仓库根目录运行。计算给定目标与解析位移，不采样或训练。',result)
            self.assertIn('[网站公开图数据](https://yingwen.io/crl-figures/offline-support-data.json)',result)
            self.assertIn('保存为教程仓库根目录的 `offline-support-data.json`',result)
            self.assertIn('该 JSON 由网站提供，不随教程仓库分发。',result)
            self.assertIn('python3 tutorials/offline_support_walkthrough.py --compare offline-support-data.json',result)
            self.assertNotIn('public/crl-',result)
            self.assertEqual(result.count('```'),4)
            self.assertEqual(list(root.iterdir()),[])

    def test_hyphenated_tutorial_commands_match_existing_files_without_partial_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'tutorials').mkdir()
            (root/'tutorials/offpolicy-geometry-walkthrough.py').write_text('# example\n')
            source='python3 offpolicy-geometry-walkthrough.py\npython3 tutorials/offpolicy-geometry-walkthrough.py\npython3 missing-walkthrough.py'
            expected='python3 tutorials/offpolicy-geometry-walkthrough.py\npython3 tutorials/offpolicy-geometry-walkthrough.py\npython3 missing-walkthrough.py'
            with patch.object(sync,'ROOT',root):
                self.assertEqual(sync.rewrite_markdown(source,'a.md','docs/a.md',{},set()),expected)
                self.assertEqual(sync.rewrite_markdown(expected,'a.md','docs/a.md',{},set()),expected)

    def test_tutorial_working_directory_prose_follows_only_the_local_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'tutorials').mkdir()
            (root/'tutorials/example.py').write_text('# example\n')
            text=('[单文件](/crl-code/tutorials/example.py)。将这个单文件保存为 example.py，在文件所在目录运行：\n\n'
                  '```bash\npython3 example.py\n```\n\n'
                  '在保存文件的目录运行下方命令。\n\n```bash\npython3 example.py\n```\n\n'
                  '[作者项目](https://example.org/project)在文件所在目录运行 `python3 author.py`。')
            with patch.object(sync,'ROOT',root):
                result=sync.rewrite_markdown(text,'a.md','docs/a.md',{},set())
                self.assertIn('仓库中的文件为 `tutorials/example.py`，在仓库根目录运行',result)
                self.assertIn('在仓库根目录运行下方命令',result)
                self.assertIn('在文件所在目录运行 `python3 author.py`',result)
                self.assertNotIn('tutorials/tutorials/',result)

    def test_protocol_directory_prose_with_intervening_table(self):
        with tempfile.TemporaryDirectory() as directory:
            text=('在文件所在目录运行 python3 continual_protocol.py。\n\n'
                  '| 事件 | 用途 |\n| --- | --- |\n| run_start | 契约 |\n\n'
                  '```sh\npython3 continual_protocol.py --mode fixture --events\n```')
            with patch.object(sync,'ROOT',Path(directory)):
                result=sync.rewrite_markdown(text,'a.md','docs/a.md',{},set(),published_tutorials={'tutorials/continual_protocol.py'})
            self.assertIn('在仓库根目录运行 python3 tutorials/continual_protocol.py',result)
            self.assertIn('python3 tutorials/continual_protocol.py --mode fixture --events',result)
            self.assertNotIn('在文件所在目录',result)

    def test_empty_download_directory_instruction_matches_first_sync_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=('[教程](/crl-code/tutorials/belief-planning-walkthrough.py)。保存到空目录后运行下面的命令。\n\n'
                    '```bash\npython3 belief-planning-walkthrough.py --test\n```\n\n'
                    '[作者代码](https://example.org/code)保存到空目录后运行。\n\n```bash\npython3 author.py\n```')
            with patch.object(sync,'ROOT',root):
                result=sync.rewrite_markdown(source,'a.md','docs/a.md',{},set(),published_tutorials={'tutorials/belief-planning-walkthrough.py'})
            self.assertIn('在仓库根目录运行下面的命令',result)
            self.assertIn('python3 tutorials/belief-planning-walkthrough.py --test',result)
            self.assertIn('作者代码](https://example.org/code)保存到空目录后运行',result)

    def test_tutorial_commands_reject_external_symlinks_and_preserve_example_priority(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root=Path(directory)
            (root/'tutorials').mkdir()
            external=Path(outside)/'escape.py'
            external.write_text('# outside\n')
            (root/'tutorials/escape.py').symlink_to(external)
            (root/'tutorials/shared.py').write_text('# tutorial\n')
            with patch.object(sync,'ROOT',root):
                result=sync.rewrite_markdown('python3 escape.py\npython3 shared.py','a.md','docs/a.md',{}, {'shared.py'})
                self.assertEqual(result,'python3 escape.py\npython3 examples/shared.py')

    def test_published_figures_and_data_stay_online(self):
        cases={
            '/crl-figures/example.svg':'https://yingwen.io/crl-figures/example.svg',
            '/crl-figures/example%20data.json?download=1#values':'https://yingwen.io/crl-figures/example%20data.json?download=1#values',
            'https://yingwen.io/crl-figures/example.png#panel':'https://yingwen.io/crl-figures/example.png#panel',
            'http://yingwen.io/crl-figures/example.json?raw=1':'https://yingwen.io/crl-figures/example.json?raw=1',
        }
        for href,expected in cases.items():
            for label in ('[data]', '![figure]'):
                with self.subTest(href=href,label=label):
                    output=sync.rewrite_markdown(label+'('+href+')','a.md','docs/a.md',{},set())
                    self.assertEqual(output,label+'('+expected+')')

    def test_figure_paths_do_not_rewrite_external_hosts(self):
        for href in ('https://example.org/crl-figures/example.svg',
                     'https://yingwen.io.example.org/crl-figures/example.json',
                     '//example.org/crl-figures/example%20data.json?raw=1#values'):
            text='[external]('+href+')'
            with self.subTest(href=href):
                self.assertEqual(sync.rewrite_markdown(text,'a.md','docs/a.md',{},set()),text)

    def test_published_source_assets_resolve_to_existing_repository_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('implementations/group/update.py','tests/test_example.py','tests/test space.py','tutorials/selfplay_search_labels.py'):
                file=root/name
                file.parent.mkdir(parents=True,exist_ok=True)
                file.write_text('# source\n')
            cases={
                '/crl-code/implementations/group/update.py':'../implementations/group/update.py',
                'https://yingwen.io/crl-code/tests/test_example.py#L4':'../tests/test_example.py#L4',
                '/crl-code/tests/test%20space.py':'../tests/test%20space.py',
                '/crl-code/tutorials/selfplay_search_labels.py':'../tutorials/selfplay_search_labels.py',
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
