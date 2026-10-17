"""Check the scalar SAA recursion against an independent scenario-tree QP."""

import contextlib
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lqr import histograms_qqplots, value_functions


def scenario_tree_value(params, noise, x):
    """Optimize all controls on a small, equally weighted scenario tree."""
    A, B, Q, R = (params[name] for name in ("A", "B", "Q", "R"))
    states = cp.Constant(np.array([x], dtype=float))
    objective = 0
    for samples in noise:
        controls = cp.Variable(states.shape)
        objective += cp.sum(Q * cp.square(states) + R * cp.square(controls)) / states.size
        next_states = [A * states + B * controls + sample for sample in samples]
        states = cp.hstack(next_states)
    objective += Q * cp.sum_squares(states) / states.size
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve()
    if problem.status != cp.OPTIMAL:
        raise AssertionError(f"Independent scenario-tree solve failed: {problem.status}")
    return problem.value


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.params = {"A": 0.8, "B": -0.6, "Q": 1.7, "R": 0.9, "Sigma": 0.7, "T": 2}

    def test_numerical_recursion_matches_full_scenario_tree(self):
        N = 3
        params = self.params
        noise = np.random.default_rng(12345).uniform(
            -np.sqrt(3) * params["Sigma"], np.sqrt(3) * params["Sigma"], (params["T"], N)
        )
        P, k, q = value_functions.compute_saa_value_functions_backward(params, N)
        for t in range(params["T"]):
            for x in (-1.4, 0.0, 0.9):
                expected = scenario_tree_value(params, noise[t:], x)
                self.assertAlmostEqual(P[t] * x**2 + k[t] * x + q[t], expected, places=8)

    def test_true_recursion_matches_exact_two_point_quadrature(self):
        params = self.params
        # For quadratic costs, equally weighted +/-Sigma exactly integrates
        # every conditional expectation under the centered uniform law.
        noise = np.tile([-params["Sigma"], params["Sigma"]], (params["T"], 1))
        P, q = value_functions.compute_true_value_function(params)
        for t in range(params["T"]):
            for x in (-0.8, 0.0, 1.3):
                self.assertAlmostEqual(P[t] * x**2 + q[t], scenario_tree_value(params, noise[t:], x), places=8)

    def test_histogram_recursion_matches_scenario_tree_and_is_reproducible(self):
        params = {**self.params, "N": 3, "REPLICATIONS": 3, "X_EVAL": 0.6, "HISTOGRAM_TIMES": [1, 2]}
        with contextlib.redirect_stdout(io.StringIO()):
            results = histograms_qqplots.compute_lq_simulation_results(params)
            repeated = histograms_qqplots.compute_lq_simulation_results(params)
        self.assertEqual(results, repeated)
        streams = np.random.default_rng(12345).spawn(params["REPLICATIONS"])
        true_P, true_q = value_functions.compute_true_value_function(params)
        for r, stream in enumerate(streams):
            noise = stream.uniform(-np.sqrt(3) * params["Sigma"], np.sqrt(3) * params["Sigma"], (params["T"], params["N"]))
            for t in params["HISTOGRAM_TIMES"]:
                x = params["X_EVAL"]
                expected = np.sqrt(params["N"]) * (
                    scenario_tree_value(params, noise[t - 1:], x)
                    - true_P[t - 1] * x**2 - true_q[t - 1]
                )
                self.assertAlmostEqual(results[t][r], expected, places=8)

    def test_no_control_and_zero_noise(self):
        for B, Sigma in [(0.0, 0.7), (-0.6, 0.0)]:
            params = {**self.params, "B": B, "Sigma": Sigma, "N": 1,
                      "REPLICATIONS": 2, "X_EVAL": 0.6, "HISTOGRAM_TIMES": [1, 2]}
            with contextlib.redirect_stdout(io.StringIO()):
                results = histograms_qqplots.compute_lq_simulation_results(params)
            self.assertTrue(all(np.isfinite(values).all() for values in results.values()))
            if Sigma == 0:
                np.testing.assert_allclose(list(results.values()), 0.0, atol=1e-12)

    def test_failed_solver_is_not_used_as_a_value_function(self):
        with mock.patch.object(cp.Problem, "solve"), mock.patch.object(
            cp.Problem, "status", new_callable=mock.PropertyMock, return_value=cp.INFEASIBLE
        ):
            with self.assertRaisesRegex(RuntimeError, "status=infeasible"):
                value_functions._compute_saa_value_at_x_numerically(1.0, 1.7, 0.0, 0.0, np.array([0.0]), self.params)

    def test_invalid_sample_counts_and_stages_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "N must be a positive integer"):
            value_functions.compute_saa_value_functions_backward(self.params, 0)
        params = {**self.params, "N": 1, "REPLICATIONS": 2, "X_EVAL": 1.0, "HISTOGRAM_TIMES": [0]}
        with self.assertRaisesRegex(ValueError, "HISTOGRAM_TIMES"):
            histograms_qqplots.compute_lq_simulation_results(params)

    def test_plotting_creates_requested_directory_and_closes_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plots"
            with mock.patch.object(value_functions, "save_path", output):
                coeffs = np.ones(3)
                value_functions.plot_value_functions(coeffs, coeffs * 0, coeffs * 0, coeffs, coeffs * 0, [1, 2], 3)
            self.assertEqual(len(list(output.glob("*.pdf"))), 2)
            self.assertFalse(plt.get_fignums())

    def test_import_does_not_create_output_directories(self):
        with mock.patch("os.makedirs") as makedirs, mock.patch.object(Path, "mkdir") as mkdir:
            importlib.reload(value_functions)
            makedirs.assert_not_called()
            mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
