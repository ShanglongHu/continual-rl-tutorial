"""Deterministic model semantics; these checks do not launch training."""
import unittest
from tutorials import option_model_walkthrough as w


class OptionWalkthroughTests(unittest.TestCase):
    def setUp(self):
        self.options = w.make_options()
        self.option = next(o for o in self.options if o.name == "NW:3,1")

    def test_same_four_step_trajectory_and_arrival_termination(self):
        rows = w.execute(self.option, (0, 0))
        self.assertEqual([w.NAMES[t.action] for t in rows], ["down", "right", "right", "right"])
        self.assertEqual([t.stop for t in rows], [False, False, False, True])
        self.assertTrue(all(not t.terminal for t in rows))
        self.assertTrue(self.option.beta((1, 3)))
        self.assertGreater(len(w.execute(self.option, (1, 3))), 0)
        with self.assertRaises(ValueError):
            w.execute(self.option, (3, 1))

    def test_complete_and_incomplete_outcomes(self):
        rows = w.execute(self.option, (0, 0))
        sample = w.outcome(rows)
        self.assertAlmostEqual(sample["reward"], -3.439)
        self.assertAlmostEqual(sample["kernel"]["3,1"], .6561)
        self.assertAlmostEqual(w.backup(sample, {"3,1": 10}), 3.122)
        with self.assertRaises(ValueError):
            w.outcome(rows[:-1])

    def test_terminal_reward_is_retained_without_endpoint_mass(self):
        # Construct a one-step primitive option that enters Z.
        primitive = w.Option("right", frozenset(), frozenset({(5, 0)}), {(5, 0): 1}, w.GOAL)
        rows = w.execute(primitive, (5, 0), "goal")
        sample = w.outcome(rows)
        self.assertEqual(sample["reward"], 1)
        self.assertEqual(sample["kernel"], {})
        rewards, kernels = {}, {}
        w.intra_pass(primitive, rows, rewards, kernels)
        self.assertEqual(rewards["5,0"], 1)
        self.assertEqual(kernels["5,0"], {})

    def test_intra_option_uses_old_successor_before_writing(self):
        # A self-loop makes read/write order observable.
        s = (0, 0)
        o = w.Option("self", frozenset({s}), frozenset({s}), {s: 3}, (3, 1))
        row = w.Transition(s, 3, 2, s, False, False)
        rewards, kernels = {"0,0": 4}, {"0,0": {"3,1": .5}}
        w.intra_pass(o, [row], rewards, kernels, alpha=.5)
        self.assertAlmostEqual(rewards["0,0"], 4.8)
        self.assertAlmostEqual(kernels["0,0"]["3,1"], .475)

    def test_one_pass_is_not_full_trajectory_smoothing(self):
        data = w.calculate()
        first, last = data["intra"][0], data["intra"][-1]
        self.assertEqual(first["rewards"]["0,0"], -1)
        self.assertEqual(first["kernels"]["0,0"], {})
        self.assertAlmostEqual(first["kernels"]["2,1"]["3,1"], .9)
        self.assertAlmostEqual(last["rewards"]["0,0"], -3.439)
        self.assertAlmostEqual(last["kernels"]["0,0"]["3,1"], .6561)

    def test_wrong_gamma_and_stale_policy_change_the_answer(self):
        data = w.calculate()
        model = data["sample"]
        self.assertNotAlmostEqual(model["reward"] + .9 * 10, w.backup(model, {"3,1": 10}))
        changed = data["version_change"]
        self.assertEqual(changed["old"]["end"], changed["new"]["end"])
        self.assertAlmostEqual(changed["old_target"] - changed["new_target"], 2.49318)


if __name__ == "__main__":
    unittest.main()
