"""Scalar LQR asymptotic variances for centered, variance-Sigma**2 uniform noise."""

import numpy as np
from typing import Dict, Any, Tuple


def uniform_squared_noise_variance(sigma: float) -> float:
    """Return Var(xi**2) for xi uniform on [-sqrt(3)*sigma, sqrt(3)*sigma].

    E[xi**2] = sigma**2 and E[xi**4] = (9/5)*sigma**4,
    so the variance is (4/5)*sigma**4, rather than the Gaussian value 2*sigma**4.
    """
    return (4.0 / 5.0) * sigma**4

def compute_coefficients(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the backward recursions to compute the full series for P, K, S, and v.

    Arrays use index t-1 for decision stage t; index T is the terminal cost.
    Sigma denotes the standard deviation of the centered uniform disturbance.
    """
    T, A, B, Q, R, Sigma = (params[k] for k in ["T", "A", "B", "Q", "R", "Sigma"])

    P = np.zeros(T + 1)
    K = np.zeros(T)
    S = np.zeros(T + 1)
    v = np.zeros(T + 1)

    # Set terminal conditions
    P[T] = Q
    S[T] = 0.0
    v[T] = 0.0

    # Backward recursion from t = T-1 down to 0
    for t in range(T - 1, -1, -1):
        P_next, S_next, v_next = P[t+1], S[t+1], v[t+1]
        K[t] = - (A * B * P_next) / (R + B**2 * P_next)
        P[t] = Q + A**2 * P_next - K[t]**2 * (R + B**2 * P_next)
        M = A + B * K[t]
        var_xi_term = P_next**2 * uniform_squared_noise_variance(Sigma)
        v[t] = v_next + var_xi_term
        S[t] = M**2 * (S_next + 4 * P_next**2 * Sigma**2)

    return P, K, S, v

def calculate_variance_decomposition(params: Dict[str, Any], P: np.ndarray, K: np.ndarray, S: np.ndarray, v: np.ndarray, x_test: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the decomposed variance series for a single x_test value.
    """
    T, A, B, Sigma = (params[k] for k in ["T", "A", "B", "Sigma"])
    local_variance_series = np.zeros(T)
    propagated_variance_series = np.zeros(T)

    for t_idx in range(T):
        P_next, S_next, v_next = P[t_idx + 1], S[t_idx + 1], v[t_idx + 1]
        M_t = A + B * K[t_idx]
        local_S_comp = 4 * x_test**2 * M_t**2 * P_next**2 * Sigma**2
        local_v_comp = P_next**2 * uniform_squared_noise_variance(Sigma)
        local_variance_series[t_idx] = local_S_comp + local_v_comp
        propagated_S_comp = x_test**2 * M_t**2 * S_next
        propagated_v_comp = v_next
        propagated_variance_series[t_idx] = propagated_S_comp + propagated_v_comp

    total_variance_series = local_variance_series + propagated_variance_series
    return local_variance_series, propagated_variance_series, total_variance_series
