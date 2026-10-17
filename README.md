# Supplementary code for the manuscript: Central Limit Theorems for Sample Average Approximations in Stochastic Optimal Control

This repository contains supplementary code for the manuscript
> Johannes Milz and Alexander Shapiro, "[Central Limit Theorems for Sample Average Approximations in Stochastic Optimal Control](https://arxiv.org/abs/2508.01942)", 2025.

[![arXiv](https://img.shields.io/badge/arXiv-2508.01942-b31b1b.svg)](https://arxiv.org/abs/2508.01942)

## Abstract

We establish central limit theorems for the sample average approximation (SAA) method in discrete-time, finite-horizon stochastic optimal control. Using the dynamic programming principle and backward induction, we characterize the limiting distribution of the SAA value functions. A key result is that the asymptotic variance at each stage decomposes into two components: a current-stage variance arising from immediate randomness, and a propagated future variance accumulated from subsequent stages. This decomposition clarifies how statistical uncertainty propagates backward through time. Our derivation relies on a stochastic equicontinuity condition, for which we provide sufficient conditions. We illustrate the variance decomposition using the classical linear-quadratic regulator (LQR) problem. Although its unbounded state and control spaces violate the compactness assumptions of our framework, the LQR setting enables explicit computation and visualization of both variance components.

## Installation

### Dependencies
- See requirements in [requirements.txt](requirements.txt) for specific package versions.
- Python 3.11 or higher is required by the pinned dependencies. The simulation also uses NumPy 1.25 or newer for independent random streams.

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/milzj/stochastic-control-saa-clt.git
   cd stochastic-control-saa-clt
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate 
   ```

3. **Install dependencies:**
   ```bash
   python3 -m pip install -r requirements.txt
   ```
   To install the local package as well (for example, to run the numerical
   value-function module), run:
   ```bash
   python3 -m pip install --no-deps -e .
   ```

## Replicating results

Run from the repository root:

```bash
python3 scripts/plot_true_variance.py
python3 scripts/plot_histograms_qqplots.py
```

The scripts also work from the `scripts` directory. All output paths are
resolved relative to the repository, so plots always go to
`data/true_variance/` and `data/histograms_qqplots/`.

The experiments use independent disturbances
`Uniform(-sqrt(3)*Sigma, sqrt(3)*Sigma)`, where `Sigma` is the noise standard
deviation (one by default). The analytic variance uses
`Var(xi**2) = (4/5)*Sigma**4`, consistent with these uniform samples.
`Sigma` is not the covariance matrix in the paper's multivariate notation;
this implementation is scalar.

The default simulation uses seed 12345, sample size 1000, 10,000 histogram
replications, and 50, 100, 500, and 1000 replications for the Q-Q plots.
Histograms show fitted normal densities; Q-Q plots compare the scaled
estimation errors to normal quantiles. The normal limit does not require
Gaussian disturbances.

After installing the package, the separate numerical value-function
comparison can be run with:

```bash
python3 -m lqr.value_functions
```

Its figures are saved in `data/value_functions/`.

## Checking the implementation

From the repository root, run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The regression checks compare the analytic variance with independent
quadrature and the SAA recursion with numerical minimization, and check
the plotted coordinates and script entry points.

The [GitHub Actions test workflow](.github/workflows/tests.yml) runs these
checks on Python 3.11 and 3.12 for pushes and pull requests. It can also be
started manually from the repository's Actions tab. CI uses the pinned
dependencies and a noninteractive plotting backend; no LaTeX installation is
required.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

[GitHub Copilot](https://github.com/features/copilot) (with Claude Sonnet 4 Gemini 2.5 Pro) and 
Gemini's 2.5 Pro model have been used to assist in code 
generation and documentation.

[OpenAI Codex](https://developers.openai.com/codex/) (desktop application
version 26.1007.71326, build 20398; used on October 17, 2026) assisted with
implementation and mathematical review, the uniform-noise variance correction,
regression tests, plotting fixes, GitHub Actions, and documentation. Additional
Codex agents reviewed the changes. Validation included independent quadrature,
scenario-tree optimization, and regression tests.

## Contact

For questions or issues, please open an issue on GitHub.
