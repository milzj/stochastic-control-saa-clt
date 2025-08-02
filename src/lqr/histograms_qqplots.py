import numpy as np

def compute_lq_simulation_results(params):
    """
    Computes the normalized difference between SAA and true value functions.

    Args:
        params (dict): A dictionary containing all system and SAA parameters.

    Returns:
        dict: A dictionary where keys are time steps and values are lists of
              the computed normalized differences.
    """
    # Unpack parameters
    A = params.get('A')
    B = params.get('B')
    Q = params.get('Q')
    R = params.get('R')
    SIGMA = params.get('Sigma')
    T = params.get('T')
    X_EVAL = params.get('X_EVAL')
    N = params.get('N')
    REPLICATIONS = params.get('REPLICATIONS')
    HISTOGRAM_TIMES = params.get('HISTOGRAM_TIMES')

    parent_rng = np.random.default_rng(12345)
    streams = parent_rng.spawn(REPLICATIONS)


    print("--- Starting Computation ---")
    # 1. Compute True Value Function Coefficients (V_t(x) = P_t * x^2 + p_t)
    P_true = np.zeros(T + 2)
    p_true = np.zeros(T + 2)
    P_true[T + 1] = Q

    for t in range(T, 0, -1):
        P_next = P_true[t + 1]
        P_true[t] = Q + A**2 * P_next - (A * B * P_next)**2 / (R + B**2 * P_next)
        p_true[t] = p_true[t + 1] + P_next * SIGMA**2

    # 2. Run Replications
    results = {t: [] for t in HISTOGRAM_TIMES}

    for r in range(REPLICATIONS):
        if (r + 1) % 1000 == 0:
            print(f"Running Replication {r + 1}/{REPLICATIONS}...")

        # Generate noise for this replication
        noise = streams[r].normal(loc=0, scale=SIGMA, size=(T, N))

        xi_bar = np.mean(noise, axis=1)
        xi_sq_bar = np.mean(noise**2, axis=1)

        # Compute SAA Value Function Coefficients (V_hat(x) = P_t*x^2 + k_hat*x + q_hat)
        k_hat = np.zeros(T + 2)
        q_hat = np.zeros(T + 2)

        for t in range(T, 0, -1):
            P_next = P_true[t + 1]
            k_hat_next = k_hat[t + 1]
            q_hat_next = q_hat[t + 1]
            idx = t - 1  # 0-based index for noise arrays

            K = -(A * B * P_next) / (R + B**2 * P_next)
            M = A + B * K

            k_hat[t] = M * (k_hat_next + 2 * P_next * xi_bar[idx])
            inf_u_cost_part = q_hat_next + xi_sq_bar[idx] * P_next + k_hat_next * xi_bar[idx] \
                             - 0.25 * (2 * P_next * xi_bar[idx] + k_hat_next)**2 * B**2 / (R + B**2 * P_next)
            q_hat[t] = inf_u_cost_part

        # Calculate and store the normalized difference
        for t in HISTOGRAM_TIMES:
            V_hat = P_true[t] * X_EVAL**2 + k_hat[t] * X_EVAL + q_hat[t]
            V_true = P_true[t] * X_EVAL**2 + p_true[t]
            Z = np.sqrt(N) * (V_hat - V_true)
            results[t].append(Z)

    print("--- Computation Complete ---")
    return results


