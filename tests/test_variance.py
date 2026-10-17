"""Independent integration checks for the scalar uniform-noise CLT variance."""

import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from lqr.problem_data import system_parameters
from lqr.true_variance import (
    calculate_variance_decomposition,
    compute_coefficients,
    uniform_squared_noise_variance,
)


def uniform_quadrature(sigma):
    """Integrate polynomials of degree at most nine under the stated law."""
    nodes, weights = leggauss(5)
    return np.sqrt(3.0) * sigma * nodes, weights / 2.0


def integrated_reference(params):
    """Build a reference from integrated objectives and quadratic minimization.

    The control objective is sampled at -1, 0, 1 to find its minimum;
    current-stage variances are integrated directly.  This avoids using the
    production Riccati and fourth-moment variance formulas as an oracle.
    """
    T, A, B, Q, R, sigma = (params[k] for k in ("T", "A", "B", "Q", "R", "Sigma"))
    noise, weights = uniform_quadrature(sigma)
    P = np.zeros(T + 1)
    q = np.zeros(T + 1)
    K = np.zeros(T)
    S = np.zeros(T + 1)
    v = np.zeros(T + 1)
    P[-1] = Q

    for t in range(T - 1, -1, -1):
        def minimum_at(x):
            def objective(u):
                future = A * x + B * u + noise
                return Q * x**2 + R * u**2 + weights @ (P[t + 1] * future**2 + q[t + 1])

            minus, zero, plus = (objective(u) for u in (-1.0, 0.0, 1.0))
            quadratic = (minus + plus - 2.0 * zero) / 2.0
            linear = (plus - minus) / 2.0
            u_star = -linear / (2.0 * quadratic)
            return zero - linear**2 / (4.0 * quadratic), u_star

        q[t], _ = minimum_at(0.0)
        value_one, K[t] = minimum_at(1.0)
        P[t] = value_one - q[t]

        def variance_at(x):
            mean_next_state = A * x + B * K[t] * x
            future_values = P[t + 1] * (mean_next_state + noise)**2
            centered = future_values - weights @ future_values
            local_variance = weights @ centered**2
            propagated_variance = S[t + 1] * mean_next_state**2 + v[t + 1]
            return local_variance + propagated_variance

        v[t] = variance_at(0.0)
        S[t] = variance_at(1.0) - v[t]

    return P, K, S, v


class UniformVarianceTests(unittest.TestCase):
    def test_squared_noise_variance_matches_integration(self):
        for sigma in (0.0, 0.2, 1.0, 2.4):
            with self.subTest(sigma=sigma):
                noise, weights = uniform_quadrature(sigma)
                centered_squares = noise**2 - weights @ noise**2
                integrated_variance = weights @ centered_squares**2
                self.assertAlmostEqual(
                    uniform_squared_noise_variance(sigma), integrated_variance, places=12
                )

    def test_backward_coefficients_match_integrated_reference(self):
        cases = [
            system_parameters(),
            dict(T=1, A=1.4, B=0.7, Q=0.6, R=2.0, Sigma=0.3),
            dict(T=4, A=-0.8, B=1.2, Q=2.0, R=0.9, Sigma=1.5),
            dict(T=6, A=1.1, B=-0.4, Q=0.2, R=1.7, Sigma=0.4),
            dict(T=3, A=0.0, B=0.8, Q=0.9, R=1.1, Sigma=1.8),
            dict(T=4, A=0.7, B=0.0, Q=1.6, R=0.4, Sigma=0.9),
            dict(T=3, A=0.9, B=0.6, Q=0.0, R=1.2, Sigma=0.7),
            dict(T=5, A=1.1, B=-0.5, Q=0.7, R=0.8, Sigma=0.0),
        ]
        for params in cases:
            with self.subTest(params=params):
                actual = compute_coefficients(params)
                reference = integrated_reference(params)
                for name, coefficients, expected in zip(("P", "K", "S", "v"), actual, reference):
                    with self.subTest(coefficients=name):
                        np.testing.assert_allclose(coefficients, expected, rtol=1e-11, atol=1e-11)

    def test_decomposition_matches_direct_current_stage_integration(self):
        params = dict(T=5, A=-0.9, B=0.7, Q=1.3, R=0.8, Sigma=1.2)
        P, K, S, v = compute_coefficients(params)
        noise, weights = uniform_quadrature(params["Sigma"])
        for x in (-2.3, -0.5, 0.0, 1.0, 2.7):
            with self.subTest(x=x):
                local, propagated, total = calculate_variance_decomposition(params, P, K, S, v, x)
                for t in range(params["T"]):
                    mean_next_state = (params["A"] + params["B"] * K[t]) * x
                    future_values = P[t + 1] * (mean_next_state + noise)**2
                    centered = future_values - weights @ future_values
                    self.assertAlmostEqual(local[t], weights @ centered**2, places=11)
                    self.assertAlmostEqual(
                        propagated[t], S[t + 1] * mean_next_state**2 + v[t + 1], places=11
                    )
                np.testing.assert_allclose(total, S[:-1] * x**2 + v[:-1], rtol=1e-13, atol=1e-13)

    def test_default_last_stage_variance_is_one_point_eight(self):
        params = system_parameters()
        P, K, S, v = compute_coefficients(params)
        local, propagated, total = calculate_variance_decomposition(params, P, K, S, v, 1.0)
        self.assertAlmostEqual(local[-1], 1.8)
        self.assertEqual(propagated[-1], 0.0)
        self.assertAlmostEqual(total[-1], 1.8)
        self.assertEqual(S[-1], 0.0)
        self.assertEqual(v[-1], 0.0)

    def test_zero_noise_has_zero_variance_at_every_stage(self):
        params = {**system_parameters(), "Sigma": 0.0}
        P, K, S, v = compute_coefficients(params)
        np.testing.assert_array_equal(S, np.zeros(params["T"] + 1))
        np.testing.assert_array_equal(v, np.zeros(params["T"] + 1))
        for component in calculate_variance_decomposition(params, P, K, S, v, 2.5):
            np.testing.assert_array_equal(component, np.zeros(params["T"]))


if __name__ == "__main__":
    unittest.main()
