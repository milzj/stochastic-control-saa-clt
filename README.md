# Supplementary code for the manuscript: Central Limit Theorems for Sample Average Approximations in Stochastic Optimal Control

This repository contains supplementary code for the manuscript
> Johannes Milz and Alexander Shapiro, "Central Limit Theorems for Sample Average Approximations in Stochastic Optimal Control", 2025.

## Abstract

We establish central limit theorems for the sample average approximation (SAA) method in discrete-time, finite-horizon stochastic optimal control. Using the dynamic programming principle and backward induction, we characterize the limiting distribution of the SAA value functions. A key result is that the asymptotic variance at each stage decomposes into two components: a current-stage variance arising from immediate randomness, and a propagated future variance accumulated from subsequent stages. This decomposition clarifies how statistical uncertainty propagates backward through time. Our derivation relies on a stochastic equicontinuity condition, for which we provide sufficient conditions. We illustrate the variance decomposition using the classical linear-quadratic regulator (LQR) problem. Although its unbounded state and control spaces violate the compactness assumptions of our framework, the LQR setting enables explicit computation and visualization of both variance components.

## Installation

### Dependencies
- See requirements in [requirements.txt](requirements.txt) for specific package versions.
- Python 3.8 or higher is recommended. We have tested the code with Python 3.11.

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
   Or 
   ```bash
   pip install -r requirements.txt
   ```

## Replicating results

Navigate to the `scripts` directory 

```bash
cd scripts
```

and run the scripts:

```bash
python3 plot_true_variance.py && python3 plot_histograms_qqplots.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

[GitHub Copilot](https://github.com/features/copilot) (with Claude Sonnet 4 Gemini 2.5 Pro) and 
Gemini's 2.5 Pro model have been used to assist in code 
generation and documentation.

## Contact

For questions or issues, please open an issue on GitHub.
