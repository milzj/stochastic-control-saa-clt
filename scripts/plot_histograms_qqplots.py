import os
import sys

# Add the src directory to the Python path so we can import lqr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

save_path = "../data/histograms_qqplots"
os.makedirs(save_path, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from figure_style import *
from scipy import stats
import shutil

def plot_histograms(results, params, save_path):
    """
    Plots each histogram from the simulation results in a separate figure,
    with simulation parameters added to the legend.

    Args:
        results (dict): The output from compute_lq_simulation_results.
        params (dict): The dictionary of parameters used for the simulation.
    """
    # Unpack parameters needed for plotting labels
    N = params.get('N')
    REPLICATIONS = params.get('REPLICATIONS')
    X_EVAL = params.get('X_EVAL')

    global_min_x = np.inf
    global_max_x = -np.inf
    global_min_y = np.inf
    global_max_y = -np.inf

    # First pass: calculate global x and y limits
    for t, data in results.items():
        mu, std = np.mean(data), np.std(data)
        print(mu)
        print(std)

        # Determine min/max for x-axis using 3-sigma rule
        current_min_x = mu - 3 * std
        current_max_x = mu + 3 * std
        if current_min_x < global_min_x:
            global_min_x = current_min_x
        if current_max_x > global_max_x:
            global_max_x = current_max_x
            
        # Calculate histogram to determine y-limits
        counts, bin_edges = np.histogram(data, bins=100, density=True)
        max_density = np.max(counts)
        if max_density > global_max_y:
            global_max_y = max_density
        if 0 < global_min_y:  # Keep minimum y at 0 for histograms
            global_min_y = 0

    x_range = global_max_x - global_min_x
    x_limit = (global_min_x - 0.05 * x_range, global_max_x + 0.05 * x_range)
    
    # Add 5% buffer to y-axis
    y_range = global_max_y - global_min_y
    y_limit = (global_min_y, global_max_y + 0.05 * y_range)

    print("--- Generating Separate Plots with Custom Legend ---")

    min_t = min(results.keys())
    for t, data in results.items():
        # 1. Create a new figure
        plt.figure(figsize=(5, 5))

        # 2. Plot the main data
        label = r'$N^{1/2}(\hat{V}_{t,N}(x_t) - V_t(x_t))$'

        plt.hist(data, bins=100, color="tab:blue", density=True,  alpha=0.7, label=label)

        mu, std = np.mean(data), np.std(data)
        xmin, xmax = plt.xlim()
        x_fit = np.linspace(xmin, xmax, 100)
        y_fit = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-(x_fit - mu)**2 / (2 * std**2))
        plt.plot(x_fit, y_fit, color='tab:orange',  label='Normal fit')

        # 3. Create an invisible proxy artist for the parameters text
        params_label = f"$t={t}$,\n$x_t={X_EVAL}$,\n$N={N}$,\n replications=${REPLICATIONS}$"

        if shutil.which("latex"):
            params_label = (
            fr"$\begin{{array}}{{rl}}"
            fr"t & = {t} \\ "
            fr"x_t & = {X_EVAL} \\ "
            fr"N & = {N} \\"
            fr"\# \mathrm{{replications}}  & = {REPLICATIONS} \\"
            r"\end{array}$"
        )
        
        if t > min_t:
            params_label = f"$t={t}$"
            
        # The ' ' format string creates an invisible artist
        plt.plot([], [], ' ', label=params_label)

        # 4. Manually order the legend handles
        handles, labels = plt.gca().get_legend_handles_labels()
        # The desired order is [params, data, fit], which corresponds to indices [2, 0, 1]
        if t == min_t:
            order = [2, 0, 1]
        else:
            order = [2]
            
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')

        # 5. Set titles and labels
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.xlim(x_limit)
        plt.ylim(y_limit)
        plt.grid(True, linestyle='--', alpha=0.6)
        filename = f"histogram_t{t}_N{N}_reps{REPLICATIONS}.pdf"
        plt.savefig(save_path + "/"  +  filename, bbox_inches='tight')
        plt.close()


    print("--- Plotting Complete ---")

def plot_probplot_scipy_plots(results, params, save_path=''):
  """
  Plots each Q-Q plot from simulation results using scipy.stats.probplot.
  Uses a consistent axis range and shows parameters in a legend.
  """
  # Unpack parameters
  N = params.get('N')
  REPLICATIONS = params.get('REPLICATIONS')
  X_EVAL = params.get('X_EVAL')

  if save_path and not os.path.exists(save_path):
    os.makedirs(save_path)

  # --- Pre-calculate unified axis ranges for all Q-Q plots ---
  global_min_x, global_max_x = np.inf, -np.inf
  global_min_y, global_max_y = np.inf, -np.inf

  print("--- Pre-calculating Q-Q axis ranges for consistency (using SciPy) ---")
  for data_list in results.values():
    data = np.array(data_list)
    # Get quantiles from scipy.stats.probplot without plotting
    (sample_quantiles, theoretical_quantiles), _ = stats.probplot(data, dist="norm")
    global_min_x = min(global_min_x, np.min(theoretical_quantiles))
    global_max_x = max(global_max_x, np.max(theoretical_quantiles))
    global_min_y = min(global_min_y, np.min(sample_quantiles))
    global_max_y = max(global_max_y, np.max(sample_quantiles))
    # Add a 5% buffer to the ranges for better visualization
    x_range = global_max_x - global_min_x
    y_range = global_max_y - global_min_y
    x_limit_qq = (global_min_x - 0.05 * x_range, global_max_x + 0.05 * x_range)
    y_limit_qq = (global_min_y - 0.05 * y_range, global_max_y + 0.05 * y_range)

  print("--- Generating Separate Q-Q Plots (using SciPy) ---")
  for t, data_list in results.items():
    data = np.array(data_list)
    fig, ax = plt.subplots(figsize=(5, 5))

    # --- Use scipy.stats.probplot to generate and plot the points ---
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=ax)
    # --- Manually draw the best-fit line in orange ---
    # The 'osr' contains the x-coordinates (theoretical quantiles)
    ax.plot(osr, intercept + slope * osr, color='tab:orange')
    # Apply the unified axis limits
    ax.set_xlim(y_limit_qq)
    ax.set_ylim(x_limit_qq)

    ax.plot(osm, osr, 'o', color='tab:blue')
    ax.plot(osm, osm * slope + intercept, color='tab:orange')

    # Set title and labels
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title("")

    # Create an invisible artist for the parameters text to put in the legend
    params_label = f"$t={t}$,\n$x_t={X_EVAL}$,\n$N={N}$,\n replications=${REPLICATIONS}$"

    if shutil.which("latex"):
        params_label = (
        fr"$\begin{{array}}{{rl}}"
        fr"t & = {t} \\ "
        fr"x_t & = {X_EVAL} \\ "
        fr"N & = {N} \\"
        fr"\# \mathrm{{replications}}  & = {REPLICATIONS} \\"
        r"\end{array}$"
    )

    ax.plot([], [], ' ', label=params_label)
    # Get all handles, but only display the legend for our custom parameter box
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]], [labels[-1]], loc='best',
    handlelength=0, handletextpad=0)

    ax.grid(True, linestyle='--', alpha=0.6)
    filename = f"probplot_scipy_plot_t{t}_N{N}_reps{REPLICATIONS}.pdf"
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
    plt.close()
    print("--- Q-Q Plotting Complete ---")

if __name__ == '__main__':
    # Define base system and simulation parameters

    from lqr import histograms_qqplots
    from lqr import problem_data

    histogram_parameters = problem_data.histogram_parameters()


    # --- 1. HISTOGRAM GENERATION (High Replications) ---
    print(">>> STARTING HISTOGRAM GENERATION <<<\n")
    hist_params = histogram_parameters.copy()
    hist_results = histograms_qqplots.compute_lq_simulation_results(hist_params)
    plot_histograms(hist_results, hist_params, save_path=save_path)
    print("\n>>> HISTOGRAM GENERATION COMPLETE <<<\n")


    # --- 2. Q-Q PLOT GENERATION (Low Replications) ---
    print(">>> STARTING Q-Q PLOT GENERATION <<<\n")
    for reps in [50, 100, 500, 1000]:
      qq_params = histogram_parameters.copy()
      qq_params["REPLICATIONS"] = reps  # Using fewer replications for Q-Q plots


      qq_results = histograms_qqplots.compute_lq_simulation_results(qq_params)
      plot_probplot_scipy_plots(qq_results, qq_params, save_path=save_path)
      print("\n>>> Q-Q PLOT GENERATION COMPLETE <<<")
