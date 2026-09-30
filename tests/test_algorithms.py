#!/usr/bin/env python3
"""Mechanism and output checks for the standalone RL foundations teaching pack."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/rl_foundations.py"
spec = importlib.util.spec_from_file_location("foundations", SCRIPT)
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)


class Mechanisms(unittest.TestCase):
    def test_terminal_target(self):
        self.assertEqual(rl.td_target(1, .9, 100, True), 1)
        self.assertAlmostEqual(rl.td_target(0, .9, .4, False), .36)

    def test_sarsa_vs_q(self):
        for action, expected in [(0, 1.8), (None, 4.5)]:
            q = [[0., 0.], [2., 5.]]
            rl.q_step(q, 0, 0, 0, 1, False, 1., .9, action)
            self.assertAlmostEqual(q[0][0], expected)

    def test_q_terminal_bootstrap_mask(self):
        q = [[0., 0.], [100., 200.]]
        rl.q_step(q, 0, 0, 1, 1, True, 1., .9)
        self.assertEqual(q[0][0], 1.)

    def test_softmax_and_gradient(self):
        logits = [0., 0.]
        rl.policy_step(logits, [.5, .5], 1, 1, .1)
        self.assertEqual(logits, [-.05, .05])
        self.assertAlmostEqual(sum(logits), 0)
        self.assertAlmostEqual(rl.softmax(logits)[1], .52497918747894)
        self.assertEqual(rl.softmax([10000., 10000.]), [.5, .5])

    def test_log_prob_gradient_finite_difference(self):
        logits, action, eps = [.3, -.2], 1, 1e-6
        p = rl.softmax(logits)
        for j in range(2):
            above, below = logits.copy(), logits.copy()
            above[j] += eps
            below[j] -= eps
            finite = (math.log(rl.softmax(above)[action]) -
                      math.log(rl.softmax(below)[action])) / (2 * eps)
            self.assertAlmostEqual(finite, float(j == action)-p[j], places=7)

    def test_td_lambda_zero_equals_td(self):
        rows = rl.prediction(seed=12, episodes=101, lam=0.)
        td = [(r["index"], r["value"]) for r in rows if r["method"] == "td0"]
        traces = [(r["index"], r["value"]) for r in rows if r["method"] == "td_lambda"]
        self.assertEqual(td, traces)

    def test_planning_zero_equals_q_learning(self):
        rows = rl.control(seed=11, episodes=101, switch=47, planning=0, experiment="dyna")
        curves = lambda method: [(r["index"], r["metric"], r["value"])
                                 for r in rows if r["method"] == method]
        self.assertEqual(curves("q_learning"), curves("dyna_q"))

    def test_planning_update_budget(self):
        rows = rl.control(episodes=30, planning=3, experiment="dyna")
        last = {(r["method"], r["metric"]): r["value"] for r in rows}
        self.assertEqual(last["dyna_q", "update_count"],
                         4 * last["dyna_q", "environment_steps"])
        self.assertEqual(last["q_learning", "update_count"],
                         last["q_learning", "environment_steps"])

    def test_reward_switch_only_changes_terminal_bonus(self):
        self.assertEqual(rl.corridor(3, 1, False), rl.corridor(3, 1, True))
        old, new = rl.corridor(1, 0, False), rl.corridor(1, 0, True)
        self.assertEqual(old[0], new[0])
        self.assertTrue(old[2] and new[2])
        self.assertAlmostEqual(new[1]-old[1], .9)

    def test_ewc_known_optimum(self):
        for strength in [0., 1., 4.]:
            rows = rl.consolidation(strength=strength)
            final = {r["metric"]: r["value"] for r in rows
                     if r["method"] == "quadratic_ewc" and r["index"] == 200}
            w = (strength-1)/(strength+1)
            self.assertAlmostEqual(final["old_loss"], .5*(w-1)**2, delta=.0002)
            self.assertAlmostEqual(final["new_loss"], .5*(w+1)**2, delta=.0002)

    def test_rehearsal_same_optimum_different_step_scale(self):
        rows = rl.consolidation(strength=1.)
        lookup = {(r["method"], r["index"], r["metric"]): r["value"] for r in rows}
        self.assertNotEqual(lookup["equal_rehearsal", 5, "old_loss"],
                            lookup["quadratic_ewc", 5, "old_loss"])
        for method in ["equal_rehearsal", "quadratic_ewc"]:
            self.assertAlmostEqual(lookup[method, 200, "old_loss"], .5, delta=.0001)

    def test_deterministic_and_finite(self):
        for fn in [rl.prediction, rl.control, rl.policy, rl.consolidation]:
            rows = fn(seed=7, episodes=65, switch=31)
            self.assertEqual(rows, fn(seed=7, episodes=65, switch=31))
            self.assertTrue(rows)
            self.assertTrue(all(math.isfinite(r["value"]) for r in rows))


class CommandLine(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                              capture_output=True, text=True, timeout=60)

    def test_all_five_commands_and_report(self):
        with tempfile.TemporaryDirectory(prefix="crl-algorithm-test-") as tmp:
            for experiment in ["prediction", "control", "policy", "dyna", "consolidation"]:
                prefix = Path(tmp) / experiment
                completed = self.run_script(experiment, "--seeds", 1, "--episodes", 60,
                                            "--out", prefix)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                with Path(str(prefix)+".csv").open() as f:
                    rows = list(csv.DictReader(f))
                config = json.loads(rows[0]["config"])
                self.assertEqual(config["experiment"], experiment)
                document = Path(str(prefix)+".html").read_text()
                self.assertEqual(document.count("<svg "), len({r["metric"] for r in rows}))
                self.assertIn("完整运行配置", document)
                self.assertNotIn("NaN", document)

    def test_refuses_either_existing_output(self):
        with tempfile.TemporaryDirectory(prefix="crl-algorithm-test-") as tmp:
            for suffix in [".csv", ".html"]:
                prefix = Path(tmp) / suffix[1:]
                existing = Path(str(prefix)+suffix)
                existing.write_text("user result", encoding="utf-8")
                completed = self.run_script("control", "--out", prefix)
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(existing.read_text(), "user result")

    def test_rejects_nonfinite_parameters(self):
        with tempfile.TemporaryDirectory(prefix="crl-algorithm-test-") as tmp:
            completed = self.run_script("control", "--alpha", "nan", "--out", Path(tmp)/"bad")
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_deterministic_experiment_not_fake_replicates(self):
        with tempfile.TemporaryDirectory(prefix="crl-algorithm-test-") as tmp:
            completed = self.run_script("consolidation", "--seeds", 5,
                                        "--out", Path(tmp)/"bad")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("deterministic", completed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
