"""Complete population and source evidence gates run before website writes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from implementations import runtime

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('build_learning_gallery',ROOT/'scripts/build_learning_gallery.py')
gallery=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


class GalleryTests(unittest.TestCase):
    def test_clear_retention_plot_uses_same_complete_population(self):
        selected={key:path for key,path in runtime.discover().items() if key in ('extended-clear','extended-fresh_ac')}
        self.assertEqual(len(selected),2)
        with tempfile.TemporaryDirectory() as temp,patch.object(runtime,'discover',return_value=selected):
            out=Path(temp)
            runtime.experiment(list(selected),[0,1],40,out/'clear')
            diagnostics=gallery.verified_gallery(out)[4]
            for key in selected:
                values=[]
                for seed in (0,1):
                    rows=[json.loads(line) for line in (out/'clear'/key/('seed-'+str(seed))/'events.jsonl').read_text().splitlines()]
                    values.append(rows[-1]['old_return'])
                points=diagnostics[(key,'old_return')]
                self.assertEqual(points[-1]['n'],2)
                self.assertAlmostEqual(points[-1]['mean'],sum(values)/2)

    def test_retention_diagnostic_reuses_all_registered_seeds(self):
        selected={key:path for key,path in runtime.discover().items() if key in ('ewc','online_sgd','reservoir_replay')}
        with tempfile.TemporaryDirectory() as temp,patch.object(runtime,'discover',return_value=selected):
            out=Path(temp)
            manifest=runtime.experiment(list(selected),[0,1],20,out/'retention')
            diagnostics=gallery.verified_gallery(out)[4]
            self.assertEqual(set(diagnostics),set(selected))
            for key,points in diagnostics.items():
                self.assertTrue(all(point['n']==2 for point in points))
                old=[]
                for seed in (0,1):
                    rows=[json.loads(line) for line in (out/'retention'/key/('seed-'+str(seed))/'events.jsonl').read_text().splitlines()]
                    old.append(rows[-1]['old_task_mse'])
                self.assertAlmostEqual(points[-1]['mean'],sum(old)/2)
            site=out/'site'
            (site/'src/components/crl').mkdir(parents=True)
            (site/'src/data/crl').mkdir(parents=True)
            gallery.export_gallery(out,site)
            catalog=json.loads((site/'src/data/crl/code-gallery.json').read_text())
            for name in ('implementation-coverage.md','author-projects.md','learning-code.md'):
                published=site/'public/crl-code'/name
                self.assertTrue(published.read_bytes().startswith(b'\xef\xbb\xbf'))
                self.assertEqual(published.read_text(encoding='utf-8-sig'),
                                 (ROOT/'docs'/name).read_text(encoding='utf-8'))
            with zipfile.ZipFile(site/'public/crl-code/learning-code.zip') as archive:
                members=set(archive.namelist())
                required={'README.md','docs/learning-code.md','examples/deep_requirements.txt','scripts/author_project.py',
                          'integrations/author_projects.json','docs/author-projects.md','docs/implementation-coverage.md','LICENSE','LICENSE-CODE','LICENSE-DOCS.md'}
                self.assertTrue(required<=members)
                for name in ('LICENSE','LICENSE-CODE','LICENSE-DOCS.md'):
                    self.assertEqual(archive.read(name),(ROOT/name).read_bytes())
                self.assertTrue({'tests/test_extended_classic.py','tests/test_extended_adaptation.py',
                                 'tests/test_extended_knowledge.py','tests/test_extended_multiagent.py'}<=members)
                self.assertFalse(any('.git' in Path(name).parts or Path(name).parts[0]=='results' for name in members))
                unpack=out/'unpacked-code'
                archive.extractall(unpack)
                listed=subprocess.run([sys.executable,'implementations/runtime.py','--list'],cwd=unpack,check=True,text=True,capture_output=True)
                self.assertIn('td_lambda',listed.stdout)
            for entry in catalog['entries']:
                diagnostic=entry['diagnostic_plots'][0]
                self.assertEqual(diagnostic['key'],'old-task-mse')
                path=site/'public/crl-code/results'/entry['id']/diagnostic['json']
                self.assertEqual(json.loads(path.read_text()),diagnostic['curves'])
                self.assertTrue((path.parent/diagnostic['svg']).is_file())
                with zipfile.ZipFile(path.parent/'raw-runs.zip') as archive:
                    saved=json.loads(archive.read('manifest.json'))
                    self.assertEqual(set(archive.namelist()),set(saved['artifacts'])|{'manifest.json'})

    def test_missing_seed_is_rejected_even_with_complete_status(self):
        selected={key:path for key,path in runtime.discover().items() if key in ('td0','mc_prediction')}
        with tempfile.TemporaryDirectory() as temp,patch.object(runtime,'discover',return_value=selected):
            out=Path(temp)
            runtime.experiment(['td0','mc_prediction'],[0,1],20,out/'prediction')
            self.assertEqual(set(gallery.verified_gallery(out)[2]),set(selected))
            path=out/'prediction/manifest.json'
            data=json.loads(path.read_text())
            data['runs'].pop()
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'planned algorithm/seed'):
                gallery.verified_gallery(out)

    def test_edited_plot_rejected_even_if_artifact_digest_is_refreshed(self):
        selected={key:path for key,path in runtime.discover().items() if key in ('td0','mc_prediction')}
        with tempfile.TemporaryDirectory() as temp,patch.object(runtime,'discover',return_value=selected):
            out=Path(temp)
            runtime.experiment(['td0','mc_prediction'],[0,1],20,out/'prediction')
            path=out/'prediction/curves.json'
            curves=json.loads(path.read_text())
            curves['td0'][0]['mean']=12345
            path.write_text(json.dumps(curves))
            manifest_path=out/'prediction/manifest.json'
            manifest=json.loads(manifest_path.read_text())
            import hashlib
            manifest['artifacts']['curves.json']=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'complete raw population'):
                gallery.verified_gallery(out)


if __name__=='__main__':
    unittest.main()
