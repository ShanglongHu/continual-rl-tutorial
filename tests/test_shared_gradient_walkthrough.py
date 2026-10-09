"""Independent hand calculations and prefix semantics; no training."""
from fractions import Fraction as F
import unittest
from tutorials import shared_gradient_walkthrough as w


class SharedGradientTests(unittest.TestCase):
    def vector_close(self, actual, expected):
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, float(b), places=12)

    def test_mc_geometry_and_total_error_are_distinct(self):
        data = w.calculate()
        self.vector_close(data["mc"][1]["after"], (F(-3, 50), F(-21, 50)))
        self.vector_close(data["mc"][1]["predictions"], (F(-3, 50), F(-3, 10)))
        self.assertAlmostEqual(data["mc"][0]["objective"], float(F(221, 400)))
        self.assertAlmostEqual(data["mc"][1]["objective"], float(F(2017, 5000)))
        # Current A error shrinks, while total error initially grows.
        self.assertGreater(data["mc"][0]["objective"], data["initial_objective"])

    def test_corridor_trace_has_shared_not_statewise_credit(self):
        data = w.calculate()
        self.vector_close(data["accumulating"][1]["trace"], (F(13, 10), F(3, 5)))
        self.vector_close(data["credit_to_predictions"], (F(13, 10), F(7, 5)))
        self.vector_close(data["accumulating"][1]["after"], (F(-17, 100), F(-27, 50)))
        self.vector_close(data["true_online"][1]["trace"], (F(57, 50), F(12, 25)))
        self.vector_close(data["true_online"][1]["correction"], (F(17, 125), F(-6, 125)))
        self.vector_close(data["true_online"][1]["after"], (F(11, 100), F(-12, 25)))
        self.assertEqual(data["true_online"][1]["saved_old_value"], 0)
        self.assertAlmostEqual(data["true_online"][1]["value"], .8)

    def test_forward_rebuilds_each_prefix_including_nonterminal_bootstrap(self):
        xs = ((1., .2), (.8, -.5), (1., .2), (.3, .7), (0., 0.))
        rewards = (2., -1., .4, .9)
        for alpha in (0., .03, .5):
            for lam in (0., .5, 1.):
                for gamma in (.4, 1.):
                    with self.subTest(alpha=alpha, lam=lam, gamma=gamma):
                        a = w.backward(xs, rewards, alpha, lam, gamma, (.1, -.2))
                        b = w.online_forward(xs, rewards, alpha, lam, gamma, (.1, -.2))
                        for backward, forward in zip(a, b):
                            self.vector_close(backward["after"], forward["after"])

    def test_lambda_zero_reduces_to_td_zero(self):
        true = w.backward(lam=0)
        accum = w.backward(lam=0, true_online=False)
        for a, b in zip(true, accum):
            self.vector_close(a["after"], b["after"])
            self.vector_close(a["trace"], w.FEATURES[a["t"]])

    def test_nonlinear_trace_stores_derivatives_at_their_original_versions(self):
        data = w.nonlinear_memory()
        self.assertEqual(data["theta_after"], 1.625)
        self.assertEqual(data["remembered_trace"], 1.5)
        self.assertEqual(data["recomputed_trace"], 2.625)
        eps = 1e-6
        for theta, derivative in ((.5, 1), (1.625, 3.25)):
            numerical = ((theta + eps) ** 2 - (theta - eps) ** 2) / (2 * eps)
            self.assertAlmostEqual(numerical, derivative, places=8)

    def test_no_input_state_mutation_and_empty_prefix(self):
        xs, initial = [[1., 0.], [.8, .6], [0., 0.]], [0., 0.]
        w.backward(xs=xs, initial=initial)
        self.assertEqual(xs, [[1., 0.], [.8, .6], [0., 0.]])
        self.assertEqual(initial, [0., 0.])
        self.assertEqual(w.online_forward(xs=((0., 0.),), rewards=()), [])


if __name__ == "__main__":
    unittest.main()
