import importlib.util
import csv
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('crl_labs',Path(__file__).resolve().parents[1]/'examples/crl_labs.py')
labs=importlib.util.module_from_spec(spec)
spec.loader.exec_module(labs)
analyze_spec = importlib.util.spec_from_file_location('analyze_crl',Path(labs.__file__).with_name('analyze_crl.py'))
analyzer=importlib.util.module_from_spec(analyze_spec)
analyze_spec.loader.exec_module(analyzer)

class TeachingDiagnostics(unittest.TestCase):
    def test_determinism(self):
        self.assertEqual(labs.bandit(seed=7),labs.bandit(seed=7))
        self.assertNotEqual(labs.bandit(seed=7),labs.bandit(seed=8))

    def test_bandit_valid_windows(self):
        rows=labs.bandit(steps=201)
        self.assertEqual(len(rows),24)
        self.assertTrue(all(0<=r['value']<=1 for r in rows))
        self.assertEqual(max(r['step'] for r in rows),201)

    def test_sample_mean_does_not_depend_on_constant_alpha(self):
        low=[r for r in labs.bandit(seed=7,alpha=.02) if r['method']=='sample_mean']
        high=[r for r in labs.bandit(seed=7,alpha=.5) if r['method']=='sample_mean']
        self.assertEqual(low,high)

    def test_windows_never_mix_regimes(self):
        rows=labs.bandit(steps=401,switch=125)
        rewards=[r for r in rows if r['method']=='sample_mean' and r['metric']=='window_reward']
        self.assertEqual(sum(r['window_steps'] for r in rewards),401)
        self.assertTrue(any(r['step']==125 for r in rewards))
        self.assertTrue(all(not(r['step']-r['window_steps']<125<r['step']) for r in rewards))

    def test_ema_derivation(self):
        alpha=.1
        q=.8
        rewards=[0,0,0]
        for r in rewards:q+=alpha*(r-q)
        unrolled=(1-alpha)**len(rewards)*.8+sum(alpha*(1-alpha)**(len(rewards)-1-i)*r for i,r in enumerate(rewards))
        self.assertAlmostEqual(q,.5832)
        self.assertAlmostEqual(q,unrolled)
        h=math.log(.5)/math.log(1-alpha)
        self.assertAlmostEqual((1-alpha)**h,.5)

    def test_linear_loss_contraction(self):
        for start in (1.,0.):
            target,alpha=-1.,.05
            updated=start-alpha*(start-target)
            self.assertAlmostEqual(((updated-target)/(start-target))**2,(1-alpha)**2)

    def test_report_end_to_end_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='crl-lab-test-') as d:
            csv_path=Path(d)/'bandit.csv'
            report=Path(d)/'report.html'
            cmd=[sys.executable,labs.__file__,'bandit','--seeds','2','--steps','401','--switch','125','--out',str(csv_path)]
            subprocess.run(cmd,check=True,capture_output=True)
            rows=analyzer.load_rows(csv_path)
            self.assertEqual(len(rows),96)
            render_cmd=[sys.executable,analyzer.__file__,str(csv_path),'--out',str(report)]
            subprocess.run(render_cmd,check=True,capture_output=True)
            html=report.read_text()
            self.assertEqual(html.count('<svg '),4)
            self.assertIn('样本标准差，不是置信区间',html)
            self.assertIn('<main>',html)
            self.assertIn('@media print',html)
            self.assertNotEqual(subprocess.run(render_cmd,capture_output=True).returncode,0)
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertEqual(report.read_text(),html)

    def test_information_limit(self):
        for seed in range(5):
            rows=labs.memory(seed=seed,steps=1000)
            self.assertEqual(rows[0]['value'],0.5)
            self.assertEqual(rows[1]['value'],1.)

    def test_credit_finite(self):
        for seed in range(3):
            rows=labs.credit(seed=seed,steps=2000)
            self.assertTrue(all(math.isfinite(r['value']) and r['value']>=0 for r in rows))

    def test_memory_requires_balanced_trials(self):
        for steps in (0, 1, 3, -2):
            with self.assertRaises(ValueError):
                labs.memory(steps=steps)

    def test_cli_defaults_and_invalid_parameters(self):
        with tempfile.TemporaryDirectory(prefix='crl-cli-test-') as d:
            for lab in ('memory', 'retention'):
                out=Path(d)/(lab+'.csv')
                cmd=[sys.executable,labs.__file__,lab,'--steps','100','--out',str(out)]
                result=subprocess.run(cmd,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                with out.open() as handle:
                    self.assertEqual({row['seed'] for row in csv.DictReader(handle)},{'0'})
            for args in (['memory','--steps','101'], ['retention','--seeds','2'],
                         ['credit','--noise','nan'], ['credit','--meta','inf'],
                         ['memory','--switch','20']):
                out=Path(d)/'invalid.csv'
                result=subprocess.run([sys.executable,labs.__file__,*args,'--out',str(out)],
                                      capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(out.exists())

    def test_forgetting_not_inability_to_learn(self):
        rows=labs.retention(steps=1000)
        last={r['metric']:r['value'] for r in rows if r['step']==1000}
        self.assertGreater(last['old_task_error'],3.9)
        self.assertLess(last['new_task_error_old'],1e-20)
        self.assertLess(last['new_task_error_fresh'],1e-20)

if __name__=='__main__':unittest.main()
