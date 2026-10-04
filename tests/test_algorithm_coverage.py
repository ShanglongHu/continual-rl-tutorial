"""Coverage names must route to real chapters and preserve claim boundaries."""
import json
from pathlib import Path
import unittest
from implementations import runtime

ROOT=Path(__file__).resolve().parents[1]


class CoverageTests(unittest.TestCase):
    def test_actual_independent_population_and_real_chapters(self):
        coverage=json.loads((ROOT/'integrations/algorithm_coverage.json').read_text())
        entries=coverage['entries']
        self.assertEqual(len({e['id'] for e in entries}),len(entries))
        independent=[e for e in entries if e['status'] in ('independent_implementation','independent_component')]
        self.assertEqual({e['implementation_id'] for e in independent},set(runtime.discover()))
        self.assertEqual(len(independent),len(runtime.discover()))
        valid=set()
        for name,key in (('algorithm-tutorials','chapters'),('foundations','lessons'),('bridge-chapters','chapters')):
            valid.update(e['path'] for e in json.loads((ROOT/'data'/f'{name}.json').read_text())[key])
        valid.update('foundations/'+e['id']+'/' for e in json.loads((ROOT/'data/foundations.json').read_text())['tracks'])
        for entry in entries:
            self.assertIn(entry['status'],coverage['status_enum'])
            self.assertTrue(entry['chapter_paths'])
            self.assertTrue(set(entry['chapter_paths'])<=valid,entry['id'])
            if entry.get('file'):
                self.assertTrue((ROOT/entry['file']).is_file(),entry['id'])
        for status,count in coverage['counts'].items():
            self.assertEqual(count,sum(e['status']==status for e in entries))

    def test_components_preserve_original_formula_provenance(self):
        from scripts.build_algorithm_coverage import build
        coverage=json.loads((ROOT/'integrations/algorithm_coverage.json').read_text())
        self.assertEqual(build(coverage),coverage)
        originals=[e for e in coverage['entries'] if 'formula_origin' in e or e['status']=='formula_core']
        self.assertEqual(len(originals),54)
        for e in coverage['entries']:
            if e['status']=='independent_component':
                self.assertEqual(e['metadata']['implementation_kind'],'component_experiment')
                self.assertTrue(e['limitations'])
            if 'formula_origin' in e:
                self.assertEqual(e['id'],e['formula_origin']['id'])
                self.assertIn(e['id'],e['metadata']['coverage_refs'])
                self.assertEqual(e['formula_origin']['status'],'formula_core')

    def test_author_sources_are_not_claimed_as_trained(self):
        registry=json.loads((ROOT/'integrations/author_projects.json').read_text())
        coverage=json.loads((ROOT/'integrations/algorithm_coverage.json').read_text())
        authors=[e for e in coverage['entries'] if e['status']=='author_project']
        self.assertEqual({e['author_project_id'] for e in authors},{p['id'] for p in registry['projects']})
        self.assertTrue(all('not_trained' in p['status'] for p in registry['projects']))
        self.assertNotIn('coverage_percentage',coverage)


if __name__=='__main__':
    unittest.main()
