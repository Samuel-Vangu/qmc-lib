"""
Compare samplers.

This script compares several sampling methods provided by qmc-lib:
- Uniform random sampling
- Latin Hypercube Sampling
- Halton sequence
- Sobol sequence
- Lattice rule
- Kronecker sequence

The goal is to:
1. visualize the point sets in dimension 2;
2. compare their L2-star discrepancy visually;
3. test them on a simple integration problem with a known exact value.

Generation times are not studied here. They will be handled separately in
benchmark scripts, because runtime depends on several factors such as the
dimension, the number of samples, and the options used for each sampler.
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from qmc_lib.sampling.Halton import HaltonSampler
from qmc_lib.sampling.Sobol import SobolSampler
from qmc_lib.sampling.Kronecker import KroneckerSampler
from qmc_lib.sampling.LatinHypercube import LatinHypercubeSampler
from qmc_lib.sampling.LatticeRule import LatticeSampler
from qmc_lib.sampling.UniformSampler import UniformSampler

from qmc_lib.visualization import Visualization
from qmc_lib.integration.Integrate import Integrator


# ============================================================
# Parameters
# ============================================================

dimension = 2
n_samples = 2**10
seed = 0


# ============================================================
# Generate samples
# ============================================================

# We generate 2^10 points in dimension 2 from each sampler.
# All point sets lie in the unit square [0, 1]^2.

samples_dict = {
    "Halton": HaltonSampler(
        dimension=dimension,
        n_samples=n_samples,
        seed=seed,
    ).generate(),

    "Sobol": SobolSampler(
        dimension=dimension,
        n_samples=n_samples,
        seed=seed,
    ).generate(),

    "Kronecker": KroneckerSampler(
        dimension=dimension,
        n_samples=n_samples,
        seed=seed,
    ).generate(),

    "LHS": LatinHypercubeSampler(
        dimension=dimension,
        n_samples=n_samples,
        seed=seed,
    ).generate(randomized=False),

    "LatticeRule": LatticeSampler(
        dimension=dimension,
        n_samples=n_samples,
        seed=seed,
    ).generate(),

    "Uniform": UniformSampler(
        dimension=dimension,
        n_samples=n_samples,
        seed=seed,
    ).generate(),
}


# ============================================================
# Visual comparison of point sets
# ============================================================

# We compare the point sets in 2D.
# The goal is to see how each method fills the unit square.

Visualization.compare_point_sets_2d(
    samples_dict={
        "Halton": samples_dict["Halton"],
        "Sobol": samples_dict["Sobol"],
        "LHS": samples_dict["LHS"],
    }
)

Visualization.compare_point_sets_2d(
    samples_dict={
        "LatticeRule": samples_dict["LatticeRule"],
        "Kronecker": samples_dict["Kronecker"],
        "Uniform": samples_dict["Uniform"],
    }
)

# The L2-star discrepancy displayed on the plots is an indicator of uniformity.
# Smaller discrepancy usually means that the point set fills the space more evenly.


# ============================================================
# Test function for numerical integration
# ============================================================

# We now test the samplers on the function:
#
#     f(x) = prod_{j=1}^d 1 / (1 + x_j)
#
# over the unit cube [0, 1]^d.
#
# Since the function is separable, the exact integral is:
#
#     I = (log(2))^d
#
# Here, log means the natural logarithm.

def f(x: npt.NDArray) -> float:
    return np.prod(1.0 / (1.0 + x))


exact_value = np.log(2.0) ** dimension


# ============================================================
# Compute approximations and errors
# ============================================================

approximations = {}
absolute_errors = {}

for method_name, samples in samples_dict.items():
    approximation = Integrator(
        f=f,
        samples=samples,
    ).compute()

    approximations[method_name] = approximation
    absolute_errors[method_name] = abs(exact_value - approximation)


# Print numerical results.

print("\nIntegration test")
print("================")
print(f"Function: prod_j 1 / (1 + x_j)")
print(f"Dimension: {dimension}")
print(f"Number of samples: {n_samples}")
print(f"Exact value: {exact_value:.10f}\n")

for method_name in samples_dict:
    print(
        f"{method_name:12s} | "
        f"estimate = {approximations[method_name]:.10f} | "
        f"absolute error = {absolute_errors[method_name]:.3e}"
    )


# ============================================================
# Plot absolute errors
# ============================================================

methods = list(absolute_errors.keys())
errors = np.array([absolute_errors[name] for name in methods])

fig, ax = plt.subplots(figsize=(9, 5), dpi=130)

ax.bar(methods, errors)

ax.set_title("Absolute integration error by sampling method")
ax.set_xlabel("Sampling method")
ax.set_ylabel("Absolute error")

ax.set_yscale("log")
ax.grid(True, axis="y", alpha=0.25)

plt.xticks(rotation=25)
plt.tight_layout()
plt.show()


# In general, deterministic or quasi-random point sets such as Halton, Sobol,
# lattice rules, and Kronecker sequences are designed to cover the unit cube
# more regularly than purely random points. On smooth test functions, this can
# often lead to smaller integration errors than standard Monte Carlo sampling.
