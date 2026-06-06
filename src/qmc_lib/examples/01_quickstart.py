"""
Quickstart example.

This script shows how to use the basic tools provided by the library:
- generate quasi-Monte Carlo samples;
- approximate an integral over the unit cube;
- compare the result with the exact value.
"""

import numpy as np
import numpy.typing as npt

from qmc_lib.integration.Integrate import Integrator
from qmc_lib.sampling.Halton import HaltonSampler


# We want to compute the integral of:
#
#     f(x) = sum_{j=1}^d x_j^2
#
# over the unit cube [0, 1]^d.
def f(x: npt.NDArray) -> float:
    return np.sum(x**2)


# The exact value of this integral is:
#
#     ∫_[0,1]^d sum_{j=1}^d x_j^2 dx = d / 3
#
# Here we choose d = 3, so the exact value is 1.
dimension = 3
n_samples = 2**14

exact_value = dimension / 3


# We generate sample points using the Halton sequence.
# Halton is a quasi-Monte Carlo sequence designed to fill the unit cube
# more uniformly than purely random Monte Carlo points.
samples = HaltonSampler(
    dimension=dimension,
    n_samples=n_samples,
).generate(scramble=True)


# We approximate the integral using the generated points.
result = Integrator(
    f=f,
    samples=samples,
).compute()


print(f"Estimated value: {result}")
print(f"Exact value:     {exact_value}")
print(f"Absolute error:  {abs(result - exact_value)}")


