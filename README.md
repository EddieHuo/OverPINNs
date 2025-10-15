# Over-PINNs: Enhancing Physics-Informed Neural Networks via Higher-Order Partial Derivative Overdetermination of PDEs

![GitHub last commit](https://img.shields.io/github/last-commit/username/repo-name)
![License](https://img.shields.io/github/license/username/repo-name)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Overview

This repository implements the Over-PINNs (Overdetermined Physics-Informed Neural Networks) framework proposed in the paper ["Over-PINNs: Enhancing Physics-Informed Neural Networks via Higher-Order Partial Derivative Overdetermination of PDEs"](https://arxiv.org/html/2506.05918). The framework leverages automatic differentiation to generate higher-order auxiliary equations that impose additional physical constraints, enhancing the model's ability to capture physical information through an "overdetermined" approach.

## Features

- Implementation of Over-PINNs for solving various types of PDEs
- Support for multiple test cases: Allen-Cahn equation, Burgers' equation, and Poisson equation
- Based on JAX for efficient automatic differentiation
- Comprehensive configurations and utilities for training and evaluation

## Project Structure

```
OverPINNs/
├── Polynomial/
│   ├── Polynomial.py          # Implementation for polynomial-based tests
│   ├── pinn_poisson.py        # PINN implementation for Poisson equation
│   └── predict_with_saved_model.py
├── allen_cahn/
│   ├── main.py                # Main script for Allen-Cahn equation
│   ├── models.py              # Neural network models
│   ├── train.py               # Training logic
│   ├── utils.py               # Utility functions
│   ├── configs/               # Configuration files
│   └── data/                  # Data files
└── burgers/
    ├── main.py                # Main script for Burgers' equation
    ├── models.py              # Neural network models
    ├── train.py               # Training logic
    ├── utils.py               # Utility functions
    ├── eval.py                # Evaluation script
    ├── error_heatmaps.py      # Error visualization
    ├── configs/               # Configuration files
    └── data/                  # Data files
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- JAX and JAXPI
- NumPy, Matplotlib, SciPy

### Installation

1. Install JAX following the official instructions for your platform:
   ```bash
   # For CPU
   pip install --upgrade "jax[cpu]"
   
   # For GPU (CUDA)
   pip install --upgrade "jax[cuda]"
   ```

2. Install JAXPI:
   ```bash
   pip install jaxpi
   ```

3. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/OverPINNs.git
   cd OverPINNs
   ```

4. Install additional dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Models

To train an Over-PINN model for a specific PDE, navigate to the corresponding directory and run the main script:

```bash
# For Allen-Cahn equation
cd allen_cahn
python main.py

# For Burgers' equation
cd burgers
python main.py

# For Poisson equation
cd Polynomial
python pinn_poisson.py
```

### Configuration

Model configurations can be adjusted in the `configs/` directory for each test case. These configurations include:

- Network architecture parameters
- Training hyperparameters
- Problem-specific settings

### Evaluation

After training, you can evaluate the model performance using the evaluation scripts provided in each directory:

```bash
# For Burgers' equation error visualization
cd burgers
python eval.py
python error_heatmaps.py
```

## Methodology

The Over-PINNs framework enhances traditional PINNs by:

1. Using automatic differentiation to derive higher-order auxiliary equations from the original PDE
2. Incorporating these auxiliary equations as additional loss terms in the training process
3. Creating an overdetermined system that provides stronger physical constraints
4. Improving solution accuracy without significant additional computational cost

## Results

Numerical results demonstrate that Over-PINNs achieve significant improvements in solution accuracy across various types of PDEs, including:

- Time-dependent PDEs (Allen-Cahn equation, Burgers' equation)
- Elliptic PDEs (Poisson equation)

The framework exhibits strong versatility and can be applied to a wide range of scientific computing problems.

## Citation

If you use this code in your research, please cite the original paper:

```
@article{huo2025overpinns,
  title={Over-PINNs: Enhancing Physics-Informed Neural Networks via Higher-Order Partial Derivative Overdetermination of PDEs},
  author={Huo, Wenxuan and He, Qiang and Zhu, Gang and Huang, Weifeng},
  journal={arXiv preprint arXiv:2506.05918},
  year={2025}
}
```

## Acknowledgments

This work is based on research conducted at the State Key Laboratory of Tribology, Tsinghua University.

## License

This project is licensed under the MIT License - see the LICENSE file for details.