
import os
import sys

# Add the src directory to the Python path so we can import lqr modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from lqr.figure_style import *
from matplotlib.ticker import MaxNLocator
from typing import Dict, Any, Tuple, List
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting
from lqr import problem_data
from lqr import true_variance

import os



save_path = "../data/true_variance"
os.makedirs(save_path, exist_ok=True)

def plot_variance_components(params: Dict[str, Any], S_series: np.ndarray, v_series: np.ndarray):
    """
    Generates a plot decomposing the variance into its state-dependent
    and state-independent components over time.
    """
    T = params['T']
    time_axis = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(time_axis, v_series, marker='s', linestyle='--', color='tab:blue', label='$v_t$ (State-Independent)')
    ax.plot(time_axis, S_series, marker='^', linestyle=':', color='tab:orange', label='$S_t$ (State-Dependent Coeff.)')
    ax.set_xlabel('Time period $t$')
    ax.set_ylabel('Magnitude of Variance Components')
    ax.legend(frameon=True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(1, params["T"])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    filename = "variance_components.pdf"
    fig.savefig(save_path + "/" + filename, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()

def plot_variance_snapshots(params: Dict[str, Any], S_series: np.ndarray, v_series: np.ndarray):
    """
    Generates a plot showing the full variance function C_t(x_t) at different
    snapshot moments in time.
    """
    T = params['T']
    x_range = np.linspace(-10, 10, 200)
    snapshot_times = params['snapshots_times']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    linestyles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5))]
    fig, ax = plt.subplots(figsize=(5,5))
    for i, t in enumerate(snapshot_times):
        S_t, v_t = S_series[t-1], v_series[t-1]
        C_values = S_t * x_range**2 + v_t
        ax.plot(x_range, C_values, color=colors[i], lw=2.5, label=f'$t={t}$', linestyle=linestyles[i])
    ax.set_xlabel('State $x_t$')
    ax.set_ylabel(r'Asymptotic Variance $\sigma_{t,\mathrm{asymp}}^2(x_t)$')
    ax.legend( loc='upper center', frameon=True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    filename = "variance_snapshots.pdf"
    fig.savefig(save_path + "/" + filename, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()

def plot_decomposition_subplots(params: Dict[str, Any], P: np.ndarray, K: np.ndarray, S: np.ndarray, v: np.ndarray):
    """
    Generates individual figures for each x_test value, using a consistent
    Y-axis scale across all plots for comparability.
    """
    x_test_values = params['x_test_values']
    time_axis = np.arange(1, params["T"] + 1)

    # --- Pre-computation Step: Find the global maximum Y-value ---
    global_y_max = 0
    for x_val in x_test_values:
        _, _, total_var = true_variance.calculate_variance_decomposition(params, P, K, S, v, x_test=x_val)
        current_max = total_var.max()
        if current_max > global_y_max:
            global_y_max = current_max

    # Set the consistent Y-axis limits with a 5% padding at the top
    y_max_limit = global_y_max * 1.05
    y_min_limit = 0  # Variance is non-negative

    # --- Plotting Loop ---
    x_val_min = min(x_test_values)
    for x_val in x_test_values:
        fig, ax = plt.subplots(figsize=(5, 5))

        local_var, prop_var, total_var = true_variance.calculate_variance_decomposition(params, P, K, S, v, x_test=x_val)

        ax.plot([], [], ' ', label=f'$x_t = {x_val}$')
        ax.plot(time_axis, total_var, color='black', lw=2.5, linestyle='-', label=r'Total variance $\sigma_{t,\mathrm{asymp}}^2(x_t)$')
        ax.plot(time_axis, prop_var, color='tab:blue', lw=2, linestyle='--', label=r'Propagated variance $\sigma_{t,\mathrm{prop}}^2(x_t)$')
        ax.plot(time_axis, local_var, color='tab:orange', lw=2, linestyle=':', label=r'Current stage variance $\sigma_{t,\mathrm{curr}}^2(x_t)$')

        # 4. Manually order the legend handles
        handles, labels = ax.get_legend_handles_labels()
        # The desired order is [params, data, fit], which corresponds to indices [2, 0, 1]
        print(x_val, x_val_min)
        if x_val == x_val_min:
            order = [0, 1, 2, 3]  # Show all components for the first plot
        else:
            order = [0]
            
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper right', frameon=True)


        ax.set_ylabel('Asymptotic Variance')
        ax.set_xlabel('Time period $t$')
        ax.grid(True)

        # Apply the consistent Y-axis and X-axis limits
        ax.set_ylim(y_min_limit, y_max_limit)
        ax.set_xlim(1, params["T"])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        filename = f"decomposition_x{str(x_val).replace('.', '_')}.pdf"
        plt.savefig(save_path + "/" + filename, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path + filename}")

def plot_variance_surfaces(params: Dict[str, Any], P: np.ndarray, K: np.ndarray, S: np.ndarray, v: np.ndarray):
    """
    Generates separate 3D surface plots with a simplified and reusable structure.
    """
    # Define a helper function to avoid repeating plot setup code
    def _create_surface_plot(data_grid, z_label, filename):
        """Creates and saves a single 3D surface plot with consistent styling."""
        print(f"\nGenerating 3D surface for {z_label}...")
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface with consistent colormap and z-axis limits
        surf = ax.plot_surface(T_grid, X_grid, data_grid, cmap='magma', edgecolor='none', vmin=z_min, vmax=z_max)

        # Apply consistent labels, ticks, and limits
        ax.set_xlabel('Time $t$', labelpad=5)
        ax.set_ylabel('State $x_t$', labelpad=5)
        ax.set_zlabel(z_label, labelpad=5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_zlim(plot_z_min, plot_z_max)
        ax.tick_params(axis='x', which='major', pad=1)
        ax.tick_params(axis='y', which='major', pad=1)
        ax.tick_params(axis='z', which='major', pad=1)

        # Add colorbar and save the figure
        fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10, pad=0.2)
        plt.savefig(save_path + "/" + filename, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path + filename}")

    # --- Main function logic ---
    T, A, B, Sigma = (params[k] for k in ["T", "A", "B", "Sigma"])

    # Create the grid for t and x_t
    t_range = np.arange(1, T + 1)
    x_range = np.linspace(-2.5, 2.5, 100)
    T_grid, X_grid = np.meshgrid(t_range, x_range)

    # Calculate variance components on the grid
    C1_grid = np.zeros_like(T_grid, dtype=float)
    C2_grid = np.zeros_like(T_grid, dtype=float)
    for t_idx, t in enumerate(t_range):
        t_array_idx = t - 1
        P_next, S_next, v_next = P[t_array_idx + 1], S[t_array_idx + 1], v[t_array_idx + 1]
        M_t = A + B * K[t_array_idx]
        x_col = X_grid[:, t_idx]
        C1_grid[:, t_idx] = x_col**2 * M_t**2 * S_next + v_next
        C2_grid[:, t_idx] = (4 * x_col**2 * M_t**2 * P_next**2 * Sigma**2) + (2 * P_next**2 * Sigma**4)
    Total_C_grid = C1_grid + C2_grid

    # Determine common z-axis limits based on the total variance
    z_min = Total_C_grid.min()
    z_max = Total_C_grid.max()
    padding = (z_max - z_min) * 0.05
    plot_z_min = z_min - padding
    plot_z_max = z_max + padding

    # --- Call the helper function for each plot ---
    _create_surface_plot(C1_grid, r'$\sigma_{1,\mathrm{prop}}^2(x_t)$', "variance_surface_propagated.pdf")
    _create_surface_plot(C2_grid, r'$\sigma_{2,\mathrm{curr}}^2(x_t)$', "variance_surface_current.pdf")
    _create_surface_plot(Total_C_grid, r'$\sigma_{t,\mathrm{asymp}}^2(x_t)$', "variance_surface_total.pdf")


# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-talk')

    # 1. Define system and scenarios
    params = problem_data.variance_parameters()
 

    # 2. Run foundational computations once
    print("Computing coefficients...")
    P, K, S, v = true_variance.compute_coefficients(params)
    print("Computation complete.")

    # 3. Generate the 2D visualizations
    print("\nGenerating Plot 1: Variance Component Magnitudes...")
    plot_variance_components(params, S[:params['T']], v[:params['T']])
    print("\nGenerating Plot 2: Variance Function Snapshots...")
    plot_variance_snapshots(params, S[:params['T']], v[:params['T']])
    print("\nGenerating Plot 3: Decomposition Subplots for multiple scenarios...")
    plot_decomposition_subplots(params, P, K, S, v)

    # 4. Generate the new, separate 3D surface plots
    print("\nGenerating separate 3D Variance Surface Plots...")
    plot_variance_surfaces(params, P, K, S, v)

    print("\nDone.")

