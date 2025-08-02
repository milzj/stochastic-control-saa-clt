# Stochastic Control Analysis: Sample Average Approximation and Central Limit Theorem for LQR Systems

This repository contains Python code for analyzing Linear Quadratic Regulator (LQR) systems using Sample Average Approximation (SAA) and Central Limit Theorem (CLT) analysis. The project focuses on understanding the statistical properties of value function approximations in stochastic optimal control.

## Overview

The project implements:
- **Linear Quadratic Regulator (LQR)** systems with stochastic disturbances
- **Sample Average Approximation (SAA)** for value function estimation
- **Statistical analysis** including histograms, Q-Q plots, and variance decomposition
- **Visualization tools** for understanding convergence properties and distribution behavior


## Installation

### Dependencies
- See requirements in `lqr/requirements.txt` for specific package versions.
- Python 3.8 or higher is recommended.

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/milzj/stochastic-control-saa-clt.git
   cd stochastic-control-saa-clt
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   cd lqr
   python3 -m pip install -r requirements.txt
   ```
    Or 
    ```bash
   cd lqr
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

Navigate to the `lqr` directory and run the analysis scripts:

```bash
cd lqr

# Generate histograms and Q-Q plots
python lqr_histograms_qqplots.py

# Perform variance analysis
python lqr_variance_revised.py
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{stochastic_control_saa_clt,
  title={Stochastic Control Analysis: SAA and CLT for LQR Systems},
  author={Your Name},
  year={2025},
  url={https://github.com/milzj/stochastic-control-saa-clt}
}
```

## Contact

For questions or issues, please open an issue on GitHub.