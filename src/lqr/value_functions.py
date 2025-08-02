import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


import os
save_path = "../../data/value_functions"
os.makedirs(save_path, exist_ok=True)

def _compute_saa_value_at_x_numerically(x, P_next, k_next, q_next, noise_t, params):
    """
    Helper function to compute the SAA value for a single state x
    by solving the inner minimization problem numerically with CVXPY.
    """
    A, B, Q, R = params["A"], params["B"], params["Q"], params["R"]

    # Define the optimization variable (control u)
    u = cp.Variable()

    # Define the cost components. The objective is convex in u.
    stage_cost = Q * x**2 + R * cp.power(u, 2)

    # Define the expression for the next states based on the SAA samples
    next_states_expr = A * x + B * u + noise_t

    # Define the future cost based on the next-stage value function
    future_cost_samples = P_next * cp.power(next_states_expr, 2) + k_next * next_states_expr + q_next
    expected_future_cost = cp.mean(future_cost_samples)

    # Define the convex optimization problem
    objective = cp.Minimize(stage_cost + expected_future_cost)
    problem = cp.Problem(objective)

    # Solve the problem to find the minimum value, which is V_t(x)
    problem.solve()

    # Return the optimal value
    return problem.value


def compute_saa_value_functions_backward(params, N):
    """Computes SAA value functions by solving a linear system at each step."""
    T, Sigma, Q = params["T"], params["Sigma"], params["Q"]

    rng = np.random.default_rng(12345)
    noise_samples = rng.normal(loc=0, scale=Sigma, size=(T, N))
    noise_samples = rng.uniform(low=-np.sqrt(3)*Sigma, high=np.sqrt(3)*Sigma, size=(T, N))

    P_coeffs = np.zeros(T + 1)
    k_coeffs = np.zeros(T + 1)
    q_coeffs = np.zeros(T + 1)

    P_coeffs[T] = Q
    k_coeffs[T] = 0.0
    q_coeffs[T] = 0.0

    x_points = np.array([-1, 0, 1])
    M = np.vstack([x_points**2, x_points, np.ones(3)]).T

    for t in range(T - 1, -1, -1):
        P_next, k_next, q_next = P_coeffs[t + 1], k_coeffs[t + 1], q_coeffs[t + 1]
        noise_t = noise_samples[t, :]

        # Evaluate V_t(x) at x = -1, 0, 1 using the numerical solver
        y_minus_1 = _compute_saa_value_at_x_numerically(-1, P_next, k_next, q_next, noise_t, params)
        y0 = _compute_saa_value_at_x_numerically(0, P_next, k_next, q_next, noise_t, params)
        y1 = _compute_saa_value_at_x_numerically(1, P_next, k_next, q_next, noise_t, params)
        y = np.array([y_minus_1, y0, y1])

        # Solve the linear system M*c = y for c = [P, k, q]
        coeffs = np.linalg.solve(M, y)

        P_coeffs[t] = coeffs[0]
        k_coeffs[t] = coeffs[1]
        q_coeffs[t] = coeffs[2]

    return P_coeffs, k_coeffs, q_coeffs

def compute_true_value_function(params):
    """Computes the true value function coefficients by solving the Riccati equation."""
    A, B, Q, R = params["A"], params["B"], params["Q"], params["R"]
    T, Sigma = params["T"], params["Sigma"]
    

    P_true = np.zeros(T + 1)
    q_true = np.zeros(T + 1)

    P_true[T] = Q
    q_true[T] = 0.0

    for t in range(T - 1, -1, -1):
        P_next = P_true[t + 1]
        P_true[t] = Q + A**2 * P_next - (A * B * P_next)**2 / (R + B**2 * P_next)
        q_true[t] = q_true[t + 1] +  P_next * Sigma**2 

    return P_true, q_true

def plot_value_functions(P_saa, k_saa, q_saa, P_true, q_true, time_points, N):
    """Plots the SAA and true value functions for selected time points."""
    x_grid = np.linspace(-2, 2, 40)
    

    for t in time_points:
        plt.figure()
        idx = t - 1

        # Plot SAA Value Function
        saa_vf = P_saa[idx] * x_grid**2 + k_saa[idx] * x_grid + q_saa[idx]
        line, = plt.plot(x_grid, saa_vf, label=f'$\\hat{{V}}_{{t,N}}(x)$ ($t={t}, N={N}$)')

        # Plot True Value Function
        true_vf = P_true[idx] * x_grid**2 + q_true[idx]
        plt.plot(x_grid, true_vf, color="tab:orange", linestyle='--',
                 label=f'$V_{{t}}(x)$ ($t={t}$)')
        
        plt.xlabel('State (x)')
        plt.ylabel('Value (Cost-to-Go)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(save_path + "/"+ "saa_vs_true_vf_numerical_t={}_N={}.pdf".format(t,N))

# --- Main Execution ---
if __name__ == '__main__':
    import problem_data as problem_data
    lqr_params = problem_data.system_parameters()
    Ns = [1, 10, 100, 1000]

    print("Computing SAA value functions using numerical optimization...")

    # 1. Compute the SAA coefficients
    for N in Ns:
        P_saa, k_saa, q_saa = compute_saa_value_functions_backward(lqr_params, N=N)

        # 2. Compute the True coefficients
        P_true, q_true = compute_true_value_function(lqr_params)

        # 3. Plot the results
        selected_times = [1, 5, 10, 20]
        plot_value_functions(P_saa, k_saa, q_saa, P_true, q_true, selected_times, N)