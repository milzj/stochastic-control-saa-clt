def system_parameters():
    """Defines and returns all system and simulation parameters in a dictionary."""
    return {
        "A": 1.0,       # System dynamics
        "B": 1.0,       # Control influence
        "Q": 1.0,       # State cost
        "R": 0.5,       # Control cost
        "Sigma": 0.2,   # Noise standard deviation
        "T": 20,        # Time horizon
    }

def histogram_parameters():
    """Defines and returns history parameters for the simulation."""

    params = system_parameters()

    _params = {
        "X_EVAL": 1.0, # Using X_EVAL to match the simulation function
        "N": 1000,
        "HISTOGRAM_TIMES": [1, 10, 20],
        "REPLICATIONS": 10000,  # High number of replications for histograms
    }

    base_simulation_parameters = {**params, **_params}

    return base_simulation_parameters

def variance_parameters():
    """Defines and returns variance parameters for the simulation."""
    
    params = system_parameters()
    T = params["T"]

    _params = {
        "snapshots_times": [1, T // 4, T // 2, (3 * T) // 4, T],
        "x_test_values": [0.5, 1.0, 1.5]
    }

    base_simulation_parameters = {**params, **_params}

    return base_simulation_parameters