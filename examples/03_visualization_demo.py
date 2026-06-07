"""
Visualization demo.

This script presents some of the visualization tools provided by qmc-lib.

It shows how to:
1. visualize a 2D point set;
2. compare several point sets side by side;
3. plot the evolution of the discrepancy;
4. visualize 2D projections of higher-dimensional point sets.
"""

from qmc_lib.visualization import Visualization
from qmc_lib.sampling.Kronecker import KroneckerSampler
from qmc_lib.sampling.Sobol import SobolSampler


# ============================================================
# Parameters
# ============================================================

n_samples = 2**10
seed = 0


# ============================================================
# 1. plot_points_2d
# ============================================================

# plot_points_2d is the basic function used to visualize points
# in the unit square [0, 1]^2.

sobol_2d = SobolSampler(
    dimension=2,
    n_samples=n_samples,
    seed=seed,
).generate()

Visualization.plot_points_2d(
    samples=sobol_2d,
    title="Sobol point set in dimension 2",
)

# The L2-star discrepancy, denoted D*_2 on the plot, is an indicator
# of how uniformly the points cover the space.
#
# The smaller the discrepancy, the more evenly the points fill the unit square.


# ============================================================
# 2. compare_point_sets_2d
# ============================================================

# compare_point_sets_2d allows us to compare point sets generated
# by different sampling methods.

kronecker_2d = KroneckerSampler(
    dimension=2,
    n_samples=n_samples,
    seed=seed,
).generate()

Visualization.compare_point_sets_2d(
    samples_dict={
        "Sobol": sobol_2d,
        "Kronecker": kronecker_2d,
    }
)

# For readability, it is usually better not to compare too many
# point sets at once. Two or three methods are often enough for
# a clear visual comparison.


# ============================================================
# 3. plot_discrepancy_evolution
# ============================================================

# plot_discrepancy_evolution shows how the discrepancy decreases
# as the number of sample points increases.
#
# This is useful for observing how different methods fill the space
# progressively.

Visualization.plot_discrepancy_evolution(
    samples_dict={
        "Sobol": sobol_2d,
        "Kronecker": kronecker_2d,
    }
)


# ============================================================
# 4. plot_projection_grid
# ============================================================

# In dimension higher than 2, we cannot visualize the full point set directly.
# plot_projection_grid displays several 2D projections of the coordinates.
#
# This is useful for checking whether a high-dimensional point set has
# good low-dimensional projections.

sobol_4d = SobolSampler(
    dimension=4,
    n_samples=n_samples,
    seed=seed,
).generate()

Visualization.plot_projection_grid(
    samples=sobol_4d,
    max_dimensions=3,
    title="2D projections of a 4D Sobol point set",
)


