<div align="center">
  <br>
  <img src="https://raw.githubusercontent.com/Samuel-Vangu/qmc-lib/main/QMC_lib%20logo.png" alt="QMC Lib Logo" width="260"/>
  <br><br>

  <h1>QMC Lib</h1>

  <p>
    <strong>A Python library for Monte Carlo and Quasi-Monte Carlo numerical integration.</strong>
  </p>
</div>

---

## Overview

`qmc_lib` is a Python library for experimenting with Monte Carlo and Quasi-Monte Carlo methods for numerical integration.

The goal of the library is to provide simple, readable, and explicit implementations of several sampling methods. It is designed to make it easy to generate point sets, approximate integrals, compare sampling methods, visualize their behavior, and run basic benchmarks.

The library is mainly intended for educational and experimental purposes. It focuses on clarity and usability, while still providing practical tools for numerical integration on the unit cube.

---

## Features

The library currently provides the following sampling methods:

* Uniform Monte Carlo sampling
* Latin Hypercube Sampling
* Halton sequence
* Sobol sequence
* Lattice rules
* Kronecker sequence

It also includes tools for:

* numerical integration on the unit cube;
* statistical estimation;
* confidence intervals for random methods;
* visualization of point sets;
* comparison of approximation errors;
* benchmarking against existing QMC libraries.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Samuel-Vangu/qmc-lib.git
cd qmc-lib
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Install the library in editable mode:

```bash
pip install -e .
```

You can now use the library in your Python scripts or notebooks.

To check that the installation works:

```python
from qmc_lib.sampling.Sobol import SobolSampler

sampler = SobolSampler(dimension=2, n_samples=1024, seed=0)
samples = sampler.generate()

print(samples.shape)
```

For benchmark notebooks and development tools, you may also need optional dependencies such as `QMCPy` and `pyperf`. If a development requirements file is provided, you can install them with:

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

The library follows a simple workflow:

1. choose a sampling method;
2. generate points in the unit cube;
3. pass the points to the integrator;
4. compute the approximation.

The following example approximates an integral over $[0,1]^d$ using a Sobol point set.

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

$$
f(x)=\prod_{i=1}^d \frac{1}{1+x_i},
$$

the exact value of the integral is

$$
\int_{[0,1]^d} f(x),dx = (\log 2)^d.
$$

Therefore, the exact value can be computed with:

```python
exact_value = np.log(2.0) ** dimension
absolute_error = abs(estimate - exact_value)

print(f"Exact value: {exact_value}")
print(f"Absolute error: {absolute_error}")
```

---

## Using Other Sampling Methods

All sampling methods in the library follow the same general interface:

```python
sampler = SamplerClass(
    dimension=dimension,
    n_samples=n_samples,
    seed=seed,
)

samples = sampler.generate()
```

This means that once you know how to use one sampler, you can use the others in almost the same way.

For example, you can replace `SobolSampler` with another sampler:

```python
from qmc_lib.sampling.Halton import HaltonSampler
from qmc_lib.sampling.LatinHypercube import LatinHypercubeSampler
from qmc_lib.sampling.LatticeRule import LatticeSampler
from qmc_lib.sampling.Kronecker import KroneckerSampler
from qmc_lib.sampling.UniformSampler import UniformSampler
```

Example with Halton:

```python
samples = HaltonSampler(
    dimension=dimension,
    n_samples=n_samples,
    seed=seed,
).generate()

estimate = Integrator(
    f=f,
    samples=samples,
).compute()
```

Example with Uniform Monte Carlo:

```python
samples = UniformSampler(
    dimension=dimension,
    n_samples=n_samples,
    seed=seed,
).generate()

estimate = Integrator(
    f=f,
    samples=samples,
).compute()
```

Example with Latin Hypercube Sampling:

```python
samples = LatinHypercubeSampler(
    dimension=dimension,
    n_samples=n_samples,
    seed=seed,
).generate()

estimate = Integrator(
    f=f,
    samples=samples,
).compute()
```

This common structure makes it easy to compare different methods on the same integration problem.

---

## Visualization

The library also provides visualization tools to compare how different sampling methods fill the unit square.

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

These visualizations help illustrate the difference between purely random point sets and more structured low-discrepancy point sets.

<p align="center">
  <img src="https://github.com/Samuel-Vangu/qmc-lib/blob/main/qmc%20sequences%201.png" alt="Point set visualization" width="750"/>
</p>

<p align="center">
  <img src="https://github.com/Samuel-Vangu/qmc-lib/blob/main/qmc%20sequences%202.png" alt="Point set visualization" width="750"/>
</p>

<p align="center">
  <em>Example visualization of different sampling methods in two dimensions.</em>
</p>

---

## Example: Expected Loss Estimation

One of the main examples of the library is the estimation of the expected loss of a regression model.

The goal is to approximate a quantity of the form


$$
\mathbb{E}
\left[
\left(
y_{\mathrm{model}}(X)-Y
\right)^2
\right].
$$

After transforming Gaussian random variables into uniform variables on the unit cube, the problem becomes an integral over $[0,1]^4$. This allows all sampling methods implemented in the library to be applied in the same framework.

The corresponding notebook can be found in:


[examples/notebooks/qmc_expected_loss_example.ipynb](https://github.com/Samuel-Vangu/qmc-lib/blob/main/examples/notebooks/qmc_expected_loss_example.ipynb.ipynb)

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

The goal of this benchmark is not to claim that `qmc_lib` is faster or more accurate than a mature library such as QMCPy. Instead, the objective is to check that the implementations behave coherently and to identify possible directions for improvement.

<p align="center">
  <img src="assets/benchmark_accuracy.png" alt="Accuracy benchmark" width="750"/>
</p>

<p align="center">
  <em>Accuracy comparison between the methods implemented in the library.</em>
</p>

<p align="center">
  <img src="assets/benchmark_time_vs_samples.png" alt="Timing benchmark versus number of samples" width="750"/>
</p>

<p align="center">
  <em>Mean generation time as a function of the number of samples.</em>
</p>

<p align="center">
  <img src="assets/benchmark_time_vs_dimension.png" alt="Timing benchmark versus dimension" width="750"/>
</p>

<p align="center">
  <em>Mean generation time as a function of the dimension.</em>
</p>

The results show that `qmc_lib` produces coherent numerical approximations and that its estimators converge toward the expected values. QMCPy often achieves better accuracy for some QMC methods, which is expected since it is a mature and optimized library. However, `qmc_lib` remains useful as a clear and pedagogical implementation of the main ideas behind Monte Carlo and Quasi-Monte Carlo integration.

---

## Project Context

This library was developed as part of my [**Stage d'Excellence**](https://leo.univ-grenoble-alpes.fr/menu-principal/mon-projet-d-etudes-et-professionnel/faire-un-stage/les-stages-specifiques/les-stages-d-excellence-2026-153646.kjsp?RH=8631794101180600&ksession=7b607bcf-d0cc-4c95-bba7-b0a703dab8bd) at **Université Grenoble Alpes**.

The project was carried out in the [**DAO team**](https://dao-ljk.imag.fr/) of the [**LJK lab**](https://www-ljk.imag.fr/) — *Data, Learning and Optimization* — under the supervision of [**Quoc-Tung Le**](https://tung-qle.github.io/).

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
DAO team, LJK lab — Data, Learning and Optimization

