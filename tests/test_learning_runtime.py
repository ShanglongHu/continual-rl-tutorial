"""The experiment wrapper must not turn incomplete evidence into a success plot."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from implementations import runtime


class RuntimeTests(unittest.TestCase):
    def test_invalid_rows(self):
        for rows in ([], [{'step':1,'value':math.nan,'phase':'x'}],
                     [{'step':1,'value':1,'phase':'x'}],
                     [{'step':10,'value':True,'phase':'x'}],
                     [{'step':2,'value':1,'phase':'x'},{'step':1,'value':2,'phase':'x'}]):
            with self.assertRaises(ValueError): runtime.validate_rows(rows,10)

    def test_compatibility(self):
        a = dict(family='classic',task='a',metric='reward',unit='return',higher_better=True,budget='steps')
        runtime.comparable([a,dict(a)])
        with self.assertRaises(ValueError): runtime.comparable([a,{**a,'task':'b'}])

    def test_sample_std_not_sem(self):
        records = [dict(algorithm='a',status='complete',rows=[dict(step=1,value=x)]) for x in [1,3]]
        p = runtime.summarize(records)['a'][0]
        self.assertEqual(p['mean'],2)
        self.assertAlmostEqual(p['std'],math.sqrt(2))
        self.assertEqual(p['n'],2)

    def test_independent_file_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            m = runtime.experiment(['td0','mc_prediction'],[0,1],40,out)
            self.assertEqual(m['status'],'complete')
            self.assertTrue((out/'learning-curves.svg').is_file())
            with self.assertRaises(FileExistsError): runtime.experiment(['td0'],[0],40,out)

    def test_failure_is_retained_and_not_plotted(self):
        def bad(**kwargs): raise RuntimeError('intentional test')
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime,'load',return_value=SimpleNamespace(run=bad)):
            out=Path(temp)/'run'
            m=runtime.experiment(['td0'],[0,1],40,out)
            self.assertEqual(m['status'],'failed')
            self.assertEqual(len(m['runs']),2)
            self.assertFalse((out/'learning-curves.svg').exists())
            self.assertEqual(json.loads((out/'manifest.json').read_text())['runs'][0]['status'],'failed')

    def test_emitted_and_returned_measurements_must_match(self):
        def bad(seed,steps,emit):
            emit(dict(step=steps,value=1.,phase='test'))
            return [dict(step=steps,value=2.,phase='test')]
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime,'load',return_value=SimpleNamespace(run=bad)):
            out=Path(temp)/'run'
            manifest=runtime.experiment(['td0'],[0],10,out)
            self.assertEqual(manifest['status'],'failed')
            self.assertTrue((out/'source/runtime.py').is_file())
            self.assertTrue((out/'td0/seed-0/config.json').is_file())
            self.assertEqual(json.loads((out/'td0/seed-0/events.jsonl').read_text())['value'],1.)
            self.assertFalse((out/'learning-curves.svg').exists())

    def test_failed_population_cannot_be_summarized(self):
        with self.assertRaises(ValueError):
            runtime.summarize([dict(algorithm='a',status='failed')])


if __name__ == '__main__': unittest.main()
