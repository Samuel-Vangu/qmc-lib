"""
Randomized statistics demo.

This script shows how to obtain statistical information when using
randomized sampling methods.

We consider two types of methods:

1. Standard randomized methods:
   - Uniform Monte Carlo
   - randomized Latin Hypercube Sampling

   For these methods, the points are random samples in the unit cube.
   We can estimate the variance of f(X) directly from one run and then
   derive a standard error and a confidence interval.

2. Randomized quasi-Monte Carlo methods:
   - Sobol with random shifting or scrambling
   - Halton with scrambling
   - lattice rules with random shifting

   These methods are originally deterministic, but randomness can be
   introduced through randomization techniques. In that case, the uncertainty
   is estimated from several independent randomizations.
"""

import numpy as np
import numpy.typing as npt

from qmc_lib.integration.Statistics import Statistics


# ============================================================
# Test function
# ============================================================

# We approximate the integral of:
#
#     f(x) = exp(- sum_{j=1}^d x_j)
#
# over the unit cube [0, 1]^d.
#
# Since the function is separable, the exact value is:
#
#     I = (1 - exp(-1))^d
#
# Here we choose d = 10.

def f(x: npt.NDArray) -> float:
    return np.exp(-np.sum(x))


dimension = 10
n_samples = 10000
confidence_level = 0.95

exact_value = (1.0 - np.exp(-1.0)) ** dimension


# ============================================================
# Helper function for displaying statistics
# ============================================================

def print_statistics(name: str, stats: dict, exact_value: float | None = None) -> None:
    """
    Pretty-print the statistics returned by the library.

    Important quantities:

    estimate:
        Approximation of the integral.

    variance:
        For standard Monte Carlo and LHS, this is the empirical variance
        of f(X).

        For randomized QMC methods, this is the variance between independent
        randomized estimates.

    std:
        Standard deviation associated with the quantity whose variance is
        being estimated.

    standard_error:
        Estimated uncertainty on the final estimator.

        This is the quantity that should decrease when the number of samples
        or the number of independent randomizations increases.

    confidence_interval:
        Approximate confidence interval for the integral, when available.

    n_samples:
        Number of points used in each integration run.
    """

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    print(f"Estimate:        {stats['estimate']:.12e}")
    print(f"Variance:        {stats['variance']:.12e}")
    print(f"Std:             {stats['std']:.12e}")
    print(f"Standard error:  {stats['standard_error']:.12e}")
    print(f"n_samples:       {stats['n_samples']}")

    if stats.get("confidence_level") is not None:
        print(f"Confidence level: {stats['confidence_level']}")

    if stats.get("confidence_interval") is not None:
        low, high = stats["confidence_interval"]
        print(f"Confidence interval: [{low:.12e}, {high:.12e}]")
    else:
        print("Confidence interval: not returned")

    if exact_value is not None:
        absolute_error = abs(stats["estimate"] - exact_value)
        print(f"Exact value:     {exact_value:.12e}")
        print(f"Absolute error:  {absolute_error:.12e}")


# ============================================================
# 1. Standard Monte Carlo and randomized LHS
# ============================================================

# Uniform Monte Carlo and randomized Latin Hypercube Sampling are genuinely
# random methods.
#
# For these methods, one run produces random points X_1, ..., X_N.
# We can estimate:
#
#     Var(f(X))
#
# using the empirical variance of the values:
#
#     f(X_1), ..., f(X_N).
#
# Then the standard error of the Monte Carlo estimator is:
#
#     SE = std(f(X)) / sqrt(N).
#
# This standard error measures the statistical uncertainty of the estimate.
# The confidence interval is built using the normal approximation given by
# the central limit theorem.

uniform_stats = Statistics().compute_statistics(
    f=f,
    sampling_method="Uniform",
    dimension=dimension,
    n_samples=n_samples,
    confidence_level=confidence_level,
)

lhs_stats = Statistics().compute_statistics(
    f=f,
    sampling_method="LHS",
    dimension=dimension,
    n_samples=n_samples,
    confidence_level=confidence_level,
)

print_statistics(
    name="Uniform Monte Carlo",
    stats=uniform_stats,
    exact_value=exact_value,
)

print_statistics(
    name="Randomized Latin Hypercube Sampling",
    stats=lhs_stats,
    exact_value=exact_value,
)


# ============================================================
# 2. Randomized quasi-Monte Carlo methods
# ============================================================

# Quasi-Monte Carlo methods such as Sobol, Halton, lattice rules, or
# Kronecker sequences are usually deterministic.
#
# For deterministic QMC, a single run does not directly provide a classical
# statistical confidence interval, because the points are not independent
# random samples.
#
# However, we can introduce randomness using randomization techniques such as:
#
#     - scrambling;
#     - random shifting.
#
# The idea is:
#
#     one independent randomization = one independent estimate.
#
# If we perform M independent randomizations, we obtain:
#
#     I_1, I_2, ..., I_M.
#
# We can then estimate the uncertainty of the final estimator from the
# variability between these M estimates.
#
# In this case:
#
#     variance        = empirical variance between randomized estimates;
#     std             = standard deviation between randomized estimates;
#     standard_error  = std / sqrt(M).
#
# This is different from standard Monte Carlo, where the variance is computed
# from the values f(X_i) inside one simulation.

sobol_stats = Statistics().compute_statistics(
    f=f,
    sampling_method="Sobol",
    dimension=dimension,
    n_samples=n_samples,
    confidence_level=confidence_level,
)

halton_stats = Statistics().compute_statistics(
    f=f,
    sampling_method="Halton",
    dimension=dimension,
    n_samples=n_samples,
    confidence_level=confidence_level,
)

print_statistics(
    name="Randomized Sobol",
    stats=sobol_stats,
    exact_value=exact_value,
)

print_statistics(
    name="Randomized Halton",
    stats=halton_stats,
    exact_value=exact_value,
)


# ============================================================
# Interpretation
# ============================================================

# The estimate gives the numerical approximation of the integral.
#
# The absolute error can only be computed here because we know the exact value.
# In real applications, the exact integral is usually unknown.
#
# The standard error is an internal uncertainty estimate. It does not require
# knowing the exact value of the integral.
#
# For standard Monte Carlo and LHS:
#
#     standard_error = estimated std(f(X)) / sqrt(N).
#
# For randomized QMC:
#
#     standard_error = std between independent randomized estimates / sqrt(M).
#
# In both cases, a smaller standard error indicates a more stable estimator.


