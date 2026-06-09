<div align="center">
  <img src="https://github.com/Samuel-Vangu/qmc-lib" alt="QMC Lib Logo" width="180"/>

# QMC Lib

**A Python library for Monte Carlo and Quasi-Monte Carlo numerical integration.**

</div>

---

## Overview

`qmc_lib` is a small Python library for experimenting with Monte Carlo and Quasi-Monte Carlo methods for numerical integration.

The goal of the library is to provide simple, readable and explicit implementations of several sampling methods, together with tools for integration, statistics and visualization.

The library is mainly designed for educational and experimental purposes: it makes it easy to generate point sets, approximate integrals, compare sampling methods and visualize how points fill the unit cube.

---

## Features

The library currently provides several sampling methods:

- Uniform Monte Carlo sampling
- Latin Hypercube Sampling
- Halton sequence
- Sobol sequence
- Lattice rules
- Kronecker sequence

It also includes tools for:

- numerical integration on the unit cube;
- statistical estimation;
- confidence intervals for random methods;
- visualization of point sets;
- comparison of approximation errors;
- benchmarking against existing QMC libraries.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Samuel-Vangu/qmc-lib.git
cd qmc-lib
````

Install the package locally:

```bash
pip install -e .
```

Install the required dependencies:

```bash
pip install numpy scipy matplotlib pandas
```

For the benchmark notebooks, you may also need:

```bash
pip install qmcpy pyperf
```

---

## Quick Start

The following example approximates an integral over the unit cube using a Sobol point set.

```python
import numpy as np

from qmc_lib.sampling.Sobol import SobolSampler
from qmc_lib.integration.Integrate import Integrator

# Function to integrate on [0,1]^d
def f(x):
    return np.prod(1.0 / (1.0 + x))

dimension = 4
n_samples = 4096
seed = 0

# Generate Sobol points
samples = SobolSampler(
    dimension=dimension,
    n_samples=n_samples,
    seed=seed,
).generate()

# Approximate the integral
estimate = Integrator(
    f=f,
    samples=samples,
).compute()

print(f"Estimate: {estimate}")
```

For this function,

[
f(x)=\prod_{i=1}^d \frac{1}{1+x_i},
]

the exact value of the integral is

[
\int_{[0,1]^d} f(x),dx = (\log 2)^d.
]

This makes it useful for testing and benchmarking numerical integration methods.

---

## Example: Expected Loss Estimation

One of the main examples of the library is the estimation of the expected loss of a regression model.

The goal is to approximate a quantity of the form

[
R(\theta)
=========

\mathbb{E}
\left[
\left(
y_{\mathrm{model}}(X)-Y
\right)^2
\right].
]

After transforming Gaussian random variables into uniform variables on the unit cube, the problem becomes an integral over ([0,1]^4). This allows all sampling methods implemented in the library to be applied in the same framework.

The corresponding notebook can be found in:

```text
examples/qmc_expected_loss_example.ipynb
```

---

## Visualization

The library provides simple visualization tools to compare how different sampling methods fill the unit square.

Example:

```python
from qmc_lib.visualization import Visualization

Visualization.compare_point_sets_2d(
    samples_dict={
        "Sobol": sobol_samples,
        "Halton": halton_samples,
        "Uniform": uniform_samples,
    }
)
```

These visualizations help illustrate the difference between purely random sampling and more structured low-discrepancy point sets.

---

## Benchmark

A benchmark was performed to compare the methods implemented in `qmc_lib` and to compare some of them with equivalent methods from **QMCPy**, an existing Python library for Monte Carlo and Quasi-Monte Carlo integration.

The benchmark studies two main aspects:

1. **Accuracy**
   The absolute error is compared against an exact reference value.

2. **Computational time**
   The generation time of point sets is measured using `pyperf`.

The common methods compared with QMCPy include:

* Uniform sampling
* Sobol sequence
* Halton sequence
* Lattice rules

Example benchmark figures:

```markdown
![Accuracy comparison](assets/benchmark_accuracy.png)

![Timing vs number of samples](assets/benchmark_time_vs_samples.png)

![Timing vs dimension](assets/benchmark_time_vs_dimension.png)
```

The benchmark shows that `qmc_lib` produces coherent numerical results and that its methods converge toward the expected values. QMCPy often achieves better accuracy for some QMC methods, which is expected since it is a mature and optimized library. However, `qmc_lib` remains useful as a clear and pedagogical implementation of the main ideas behind MC and QMC integration.

---

## Project Context

This library was developed as part of my **Stage d'Excellence** at **Université Grenoble Alpes**, in the **Laboratoire Jean Kuntzmann (LJK)**.

The project was carried out under the supervision of **Quoc-Tung Le**.

The main objective of the internship was to study Monte Carlo and Quasi-Monte Carlo methods for numerical integration, both from a theoretical and computational point of view.

---

## Repository Structure

```text
qmc_lib/
    sampling/          # Sampling methods
    integration/       # Numerical integration tools
    visualization/     # Visualization utilities

examples/
    notebooks/         # Example notebooks
    benchmarks/        # Benchmark scripts and results
```

---

## License

This project is currently intended for educational and research purposes.

---

## Author

**Samuel Vangu**
Université Grenoble Alpes

Laboratoire Jean Kuntzmann

