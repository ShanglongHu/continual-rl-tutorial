"""真实更新的手算检查、终点约定与所有独立实验的预算检查。"""
import importlib
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DIRECTORY = ROOT / "implementations" / "classic"


def module(name):
    return importlib.import_module("implementations.classic." + name)


class ClassicTests(unittest.TestCase):
    def test_every_experiment_runs_reproducibly_and_emits(self):
        modules = {p.stem: module(p.stem) for p in DIRECTORY.glob("*.py")
                   if not p.name.startswith("_")}
        self.assertEqual(len(modules), 23)
        for name, m in modules.items():
            with self.subTest(name=name):
                emitted = []
                rows = m.run(seed=3, steps=120, emit=emitted.append)
                self.assertEqual(rows, emitted)
                self.assertEqual(rows, m.run(seed=3, steps=120))
                self.assertEqual(rows[-1]["step"], 120)
                self.assertLessEqual(len(rows), 100)
                self.assertTrue(all(math.isfinite(r["value"]) for r in rows))
                baseline = modules[m.META["baseline"]]
                for field in ("task", "metric", "unit", "higher_better", "budget"):
                    self.assertEqual(m.META[field], baseline.META[field])
                with self.assertRaises(ValueError):
                    m.run(steps=0)

    def test_sample_average_and_constant_step(self):
        self.assertAlmostEqual(module("bandit_sample_average").update(.2,1.,4),.4)
        self.assertAlmostEqual(module("bandit_constant_step").update(.2,1.,4),.28)

    def test_exact_discounted_control_cycle(self):
        m=module("q_learning")
        left={s:[1.,0.] for s in range(5)}
        right={s:[0.,1.] for s in range(5)}
        self.assertAlmostEqual(m.control_score(left),-.2,places=14)
        expected=sum(-.01*.95**k for k in range(4))+.95**4
        self.assertAlmostEqual(m.control_score(right),expected,places=14)
        left[0]=[0.,1.]
        self.assertAlmostEqual(m.control_score(left),-.2,places=14)

    def test_gradient_bandit_conserves_preference_sum(self):
        h = module("bandit_gradient").update([0.,0.],0,1.,0.,[.5,.5])
        self.assertEqual(h,[.05,-.05])

    def test_bellman_synchronous_scan_and_exact_planning(self):
        m = module("value_iteration")
        v = m.update(dict.fromkeys(range(5),0.))
        self.assertEqual(v[4],1.)
        self.assertEqual(v[3],-.01)
        self.assertLess(m.run(steps=100)[-1]["value"],1e-12)
        self.assertLess(module("policy_iteration").run(steps=1000)[-1]["value"],1e-8)

    def test_td_terminal_and_nstep_targets(self):
        v = dict.fromkeys(range(1,6),0.)
        module("td0").update(v,5,1.,None)
        self.assertAlmostEqual(v[5],.1)
        target = module("nstep_td").update(v,3,[0.,1.],None)
        self.assertEqual(target,1.)
        self.assertAlmostEqual(v[3],.1)

    def test_first_visit_mc(self):
        v, counts = {1:0.,2:0.},{1:0,2:0}
        module("mc_prediction").update(v,[1,2,1],[0.,0.,1.],counts)
        self.assertEqual(counts,{1:1,2:1})
        self.assertEqual(v,{1:1.,2:1.})

    def test_true_online_lambda_zero_equals_td(self):
        v, u = {1:.2,2:.4},{1:.2,2:.4}
        module("td0").update(v,1,.3,2)
        module("true_online_td").update(u,{1:0.,2:0.},1,.3,2,.8,lam=0.)
        self.assertAlmostEqual(u[1],v[1])
        self.assertEqual(u[2],v[2])

    def test_lambda_trace_updates_prior_state(self):
        v, e = {1:0.,2:0.},{1:1.,2:0.}
        module("td_lambda").update(v,e,2,1.,None)
        self.assertAlmostEqual(v[1],.08)
        self.assertAlmostEqual(v[2],.1)

    def test_control_targets_and_terminal_masks(self):
        for name in ("sarsa","expected_sarsa","q_learning","dyna_q","prioritized_sweeping"):
            q = {0:[0.,0.],1:[2.,4.]}
            m = module(name)
            if name in ("dyna_q","prioritized_sweeping"):
                m.update(q,0,0,1.,None)
            else:
                m.update(q,0,0,1.,None,1)
            self.assertAlmostEqual(q[0][0],.2)
        q = {0:[0.,0.],1:[2.,4.]}
        self.assertAlmostEqual(module("sarsa").update(q,0,0,1.,1,0),2.9)
        q = {0:[0.,0.],1:[2.,4.]}
        self.assertAlmostEqual(module("expected_sarsa").update(q,0,0,1.,1),4.705)
        q = {0:[0.,0.],1:[2.,4.]}
        self.assertAlmostEqual(module("nstep_sarsa").update(q,0,0,[1.,2.],None,0),2.9)

    def test_double_q_uses_other_table_for_evaluation(self):
        first = {0:[0.,0.],1:[2.,4.]}
        second = {0:[0.,0.],1:[9.,3.]}
        target = module("double_q").update(first,second,0,0,1.,1)
        self.assertAlmostEqual(target,3.85)
        self.assertEqual(second[0],[0.,0.])

    def test_linear_updates_read_old_auxiliary_weights(self):
        w, delta = module("linear_td").update([0.,0.],[1.,.5],1.,[0.,0.])
        self.assertEqual(w,[.03,.015])
        self.assertEqual(delta,1.)
        for name in ("gtd2","tdc"):
            w,h,delta = module(name).update([0.,0.],[0.,0.],[1.,.5],1.,[0.,0.])
            self.assertEqual(h,[.1,.05])
            self.assertEqual(w,[0.,0.] if name == "gtd2" else [.03,.015])
        w,delta = module("emphatic_td").update([0.,0.],[1.,.5],1.,[0.,0.],2.)
        self.assertEqual(w,[.02,.01])


if __name__ == "__main__":
    unittest.main()
