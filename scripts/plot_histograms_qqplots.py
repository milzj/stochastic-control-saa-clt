from pathlib import Path
import sys

# Add the src directory to the Python path so we can import lqr
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

save_path = PROJECT_ROOT / 'data' / 'histograms_qqplots'

import numpy as np
import matplotlib.pyplot as plt
if __package__:
    from . import figure_style
else:
    import figure_style
from scipy import stats

def _padded_limits(low, high):
    width = high - low
    if width == 0:
        width = max(abs(low), 1.0)
    return low - 0.05 * width, high + 0.05 * width


def _samples(results, minimum_size=1):
    if not results:
        raise ValueError("At least one stage of simulation results is required.")
    samples = {t: np.asarray(values, dtype=float) for t, values in results.items()}
    for values in samples.values():
        if values.ndim != 1 or values.size < minimum_size or not np.isfinite(values).all():
            raise ValueError(f"Each stage must contain at least {minimum_size} finite samples.")
    return samples


def _save_figure(fig, directory, filename):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / filename, bbox_inches='tight')


def plot_histograms(results, params, save_path=save_path):
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

    results = _samples(results)

    global_min_x = np.inf
    global_max_x = -np.inf
    global_min_y = np.inf
    global_max_y = -np.inf

    # First pass: calculate global x and y limits
    for t, data in results.items():
        mu, std = np.mean(data), np.std(data)
        print("mu", mu)
        print("std", std)

        # Show all samples as well as the central range of the fitted normal.
        current_min_x = min(float(np.min(data)), mu - 3 * std)
        current_max_x = max(float(np.max(data)), mu + 3 * std)
        if current_min_x < global_min_x:
            global_min_x = current_min_x
        if current_max_x > global_max_x:
            global_max_x = current_max_x
            
        # Calculate histogram to determine y-limits
        counts, bin_edges = np.histogram(data, bins=100, density=True)
        max_density = np.max(counts)
        if std > 0:
            max_density = max(max_density, 1 / (std * np.sqrt(2 * np.pi)))
        if max_density > global_max_y:
            global_max_y = max_density
        if 0 < global_min_y:  # Keep minimum y at 0 for histograms
            global_min_y = 0

    x_limit = _padded_limits(global_min_x, global_max_x)
    
    # Add 5% buffer to y-axis
    y_range = global_max_y - global_min_y
    y_limit = (global_min_y, global_max_y + 0.05 * y_range)

    print("--- Generating Separate Plots with Custom Legend ---")

    min_t = min(results.keys())
    for t, data in results.items():
        # 1. Create a new figure
        fig = plt.figure()

        # 2. Plot the main data
        label = r'$N^{1/2}(\hat{V}_{t,N}(x_t) - V_t(x_t))$'

        plt.hist(data, bins=100, color="tab:blue", density=True,  alpha=0.7, label=label)

        mu, std = np.mean(data), np.std(data)
        if std > 0:
            x_fit = np.linspace(*x_limit, 100)
            y_fit = stats.norm.pdf(x_fit, loc=mu, scale=std)
            plt.plot(x_fit, y_fit, color='tab:orange', label='Normal fit')

        # 3. Create an invisible proxy artist for the parameters text
        params_label = f"$t={t}$,\n$x_t={X_EVAL}$,\n$N={N}$,\n replications=${REPLICATIONS}$"

        if plt.rcParams["text.usetex"]:
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
        order = [labels.index(params_label)]
        if t == min_t:
            order.append(labels.index(label))
            if std > 0:
                order.append(labels.index('Normal fit'))
            
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')

        # 5. Set titles and labels
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.xlim(x_limit)
        plt.ylim(y_limit)
        plt.grid(True, linestyle='--', alpha=0.6)
        filename = f"histogram_t{t}_N{N}_reps{REPLICATIONS}.pdf"
        _save_figure(fig, save_path, filename)
        plt.close(fig)


    print("--- Plotting Complete ---")

def plot_probplot_scipy_plots(results, params, save_path=save_path):
    """Plot sample versus standard-normal quantiles using common axis limits."""
    N = params.get('N')
    REPLICATIONS = params.get('REPLICATIONS')
    X_EVAL = params.get('X_EVAL')
    results = _samples(results, minimum_size=2)

    # SciPy returns (theoretical quantiles, ordered samples), then the fit.
    quantiles = {
        t: stats.probplot(data, dist="norm") for t, data in results.items()
    }
    global_min_x, global_max_x = np.inf, -np.inf
    global_min_y, global_max_y = np.inf, -np.inf
    for (osm, osr), (slope, intercept, _) in quantiles.values():
        fitted = intercept + slope * osm
        global_min_x = min(global_min_x, float(osm.min()))
        global_max_x = max(global_max_x, float(osm.max()))
        global_min_y = min(global_min_y, float(osr.min()), float(fitted.min()))
        global_max_y = max(global_max_y, float(osr.max()), float(fitted.max()))
    x_limit_qq = _padded_limits(global_min_x, global_max_x)
    y_limit_qq = _padded_limits(global_min_y, global_max_y)

    print("--- Generating Separate Q-Q Plots (using SciPy) ---")
    for t, ((osm, osr), (slope, intercept, _)) in quantiles.items():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(osm, osr, 'o', color='tab:blue')
        ax.plot(osm, intercept + slope * osm, color='tab:orange')
        ax.set_xlim(x_limit_qq)
        ax.set_ylim(y_limit_qq)
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')

        params_label = f"$t={t}$,\n$x_t={X_EVAL}$,\n$N={N}$,\n replications=${REPLICATIONS}$"
        if plt.rcParams["text.usetex"]:
            params_label = (
                fr"$\begin{{array}}{{rl}}"
                fr"t & = {t} \\ "
                fr"x_t & = {X_EVAL} \\ "
                fr"N & = {N} \\"
                fr"\# \mathrm{{replications}}  & = {REPLICATIONS} \\"
                r"\end{array}$"
            )
        ax.plot([], [], ' ', label=params_label)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[-1]], [labels[-1]], loc='best',
                  handlelength=0, handletextpad=0)
        ax.grid(True, linestyle='--', alpha=0.6)
        filename = f"probplot_scipy_plot_t{t}_N{N}_reps{REPLICATIONS}.pdf"
        _save_figure(fig, save_path, filename)
        plt.close(fig)
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
