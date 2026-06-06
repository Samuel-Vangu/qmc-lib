"""
visualization.py

Visualization utilities for Monte Carlo and Quasi-Monte Carlo samples.

This module provides:
- 2D point cloud visualization
- 2D projections of high-dimensional samples
- side-by-side comparison of samplers
- L2-star discrepancy computation
- approximate star discrepancy computation
- discrepancy convergence plots

Expected sample shape:
    samples.shape == (n_samples, dimension)

All points are assumed to lie in [0, 1]^d.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt



Array = npt.NDArray[np.float64]


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class PlotConfig:
    """Global plotting configuration."""

    figsize: tuple[float, float] = (7.0, 7.0)
    dpi: int = 140
    point_size: float = 14.0
    alpha: float = 0.75
    grid_alpha: float = 0.18
    title_size: int = 14
    label_size: int = 11
    annotation_size: int = 10


DEFAULT_CONFIG = PlotConfig()


# ============================================================
# Validation utilities
# ============================================================

def _as_samples(samples: npt.ArrayLike) -> Array:
    """Convert input to a valid sample array of shape (N, d)."""
    arr = np.asarray(samples, dtype=float)

    if arr.ndim != 2:
        raise ValueError(
            f"samples must be a 2D array of shape (n_samples, dimension), got shape {arr.shape}"
        )

    if arr.shape[0] == 0:
        raise ValueError("samples must contain at least one point")

    if arr.shape[1] == 0:
        raise ValueError("samples must have positive dimension")

    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("all samples must lie in [0, 1]^d")

    return arr


def _prepare_save_path(save_path: str | Path | None) -> Path | None:
    """Create parent directory if a save path is provided."""
    if save_path is None:
        return None

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finalize_figure(
    fig: plt.Figure,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Save and/or show a figure."""
    path = _prepare_save_path(save_path)

    if path is not None:
        fig.savefig(path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _beautify_axis(ax: plt.Axes, grid_alpha: float = 0.18) -> None:
    """Apply consistent visual styling to axes."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=grid_alpha, linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_alpha(0.35)

    ax.tick_params(axis="both", labelsize=9)


# ============================================================
# Discrepancy indicators
# ============================================================

def l2_star_discrepancy(samples: npt.ArrayLike, batch_size: int = 2048) -> float:
    """
    Compute the L2-star discrepancy of a point set in [0, 1]^d.

    This is the exact L2-star discrepancy:

        D_{N,2}^* = ( ∫_{[0,1]^d} (F_N(t) - prod_j t_j)^2 dt )^{1/2}

    Formula:

        D^2 = (1/3)^d
              - (2/N) sum_i prod_j (1 - x_{ij}^2)/2
              + (1/N^2) sum_{i,k} prod_j (1 - max(x_{ij}, x_{kj}))

    Parameters
    ----------
    samples:
        Array of shape (N, d).
    batch_size:
        Batch size for the double sum. Useful for memory control.

    Returns
    -------
    float
        L2-star discrepancy.
    """
    x = _as_samples(samples)
    n, d = x.shape

    term1 = (1.0 / 3.0) ** d

    term2_values = np.prod((1.0 - x**2) / 2.0, axis=1)
    term2 = (2.0 / n) * np.sum(term2_values)

    term3_sum = 0.0

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        xb = x[start:stop]

        # Shape: (batch, N, d)
        max_values = np.maximum(xb[:, None, :], x[None, :, :])

        # Shape: (batch, N)
        products = np.prod(1.0 - max_values, axis=2)

        term3_sum += np.sum(products)

    term3 = term3_sum / (n**2)

    discrepancy_squared = term1 - term2 + term3

    # Small negative values may appear due to floating point roundoff.
    discrepancy_squared = max(float(discrepancy_squared), 0.0)

    return float(np.sqrt(discrepancy_squared))


def approx_star_discrepancy(
    samples: npt.ArrayLike,
    n_test_boxes: int = 20_000,
    seed: int | None = 0,
    include_sample_boxes: bool = True,
) -> float:
    """
    Approximate the star discrepancy.

    Star discrepancy is:

        D_N^* = sup_{t in [0,1]^d} | F_N(t) - prod_j t_j |

    where F_N(t) is the empirical measure of the anchored box [0,t].

    Exact computation is hard in moderate dimension. This function estimates it
    by testing many anchored boxes.

    Parameters
    ----------
    samples:
        Array of shape (N, d).
    n_test_boxes:
        Number of random boxes to test.
    seed:
        RNG seed.
    include_sample_boxes:
        If True, also test boxes whose upper corner is one of the sample points.

    Returns
    -------
    float
        Approximate star discrepancy.
    """
    x = _as_samples(samples)
    n, d = x.shape

    rng = np.random.default_rng(seed)
    random_boxes = rng.uniform(0.0, 1.0, size=(n_test_boxes, d))

    if include_sample_boxes:
        boxes = np.vstack([random_boxes, x])
    else:
        boxes = random_boxes

    max_disc = 0.0

    # Batch boxes to avoid huge memory use.
    batch_size = 1024

    for start in range(0, boxes.shape[0], batch_size):
        stop = min(start + batch_size, boxes.shape[0])
        b = boxes[start:stop]

        # inside[k, i] = whether sample i lies in anchored box [0, b_k]
        inside = np.all(x[None, :, :] <= b[:, None, :], axis=2)

        empirical = np.mean(inside, axis=1)
        volume = np.prod(b, axis=1)

        disc = np.max(np.abs(empirical - volume))
        max_disc = max(max_disc, float(disc))

    return max_disc


def discrepancy_summary(samples: npt.ArrayLike) -> dict[str, float]:
    """
    Compute a small discrepancy summary for a point set.

    Returns both:
    - exact L2-star discrepancy
    - approximate star discrepancy
    """
    x = _as_samples(samples)

    return {
        "n_samples": float(x.shape[0]),
        "dimension": float(x.shape[1]),
        "l2_star_discrepancy": l2_star_discrepancy(x),
        "approx_star_discrepancy": approx_star_discrepancy(x),
    }


# ============================================================
# Point cloud plots
# ============================================================

def plot_points_2d(
    samples: npt.ArrayLike,
    dims: tuple[int, int] = (0, 1),
    title: str | None = None,
    discrepancy: str = "l2",
    annotate: bool = True,
    config: PlotConfig = DEFAULT_CONFIG,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot a 2D projection of samples.

    Parameters
    ----------
    samples:
        Array of shape (N, d).
    dims:
        Pair of dimensions to plot.
    title:
        Plot title.
    discrepancy:
        "l2", "star", or "none".
    annotate:
        If True, display discrepancy indicator on the plot.
    """
    x = _as_samples(samples)
    n, d = x.shape

    i, j = dims

    if not (0 <= i < d and 0 <= j < d):
        raise ValueError(f"dims must be valid dimensions in [0, {d - 1}]")

    if i == j:
        raise ValueError("dims must contain two distinct dimensions")

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    ax.scatter(
        x[:, i],
        x[:, j],
        s=config.point_size,
        alpha=config.alpha,
        edgecolors="none",
    )

    _beautify_axis(ax, grid_alpha=config.grid_alpha)

    ax.set_xlabel(f"dimension {i}", fontsize=config.label_size)
    ax.set_ylabel(f"dimension {j}", fontsize=config.label_size)

    if title is None:
        title = f"2D projection: dimensions ({i}, {j})"

    ax.set_title(title, fontsize=config.title_size, pad=12)

    if annotate:
        text_lines = [f"N = {n}", f"d = {d}"]

        if discrepancy == "l2":
            disc = l2_star_discrepancy(x[:, [i, j]])
            text_lines.append(rf"$D_{{2}}^*$ ≈ {disc:.4e}")

        elif discrepancy == "star":
            disc = approx_star_discrepancy(x[:, [i, j]])
            text_lines.append(rf"$D^*$ approx ≈ {disc:.4e}")

        elif discrepancy == "none":
            pass

        else:
            raise ValueError("discrepancy must be one of: 'l2', 'star', 'none'")

        ax.text(
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=config.annotation_size,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.82,
                "edgecolor": "0.75",
            },
        )

    _finalize_figure(fig, save_path=save_path, show=show)


def compare_point_sets_2d(
    samples_dict: dict[str, npt.ArrayLike],
    dims: tuple[int, int] = (0, 1),
    discrepancy: str = "l2",
    config: PlotConfig = DEFAULT_CONFIG,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Compare several point sets side by side in 2D.

    Example
    -------
    compare_point_sets_2d({
        "Uniform": uniform_samples,
        "Halton": halton_samples,
        "Sobol": sobol_samples,
    })
    """
    if len(samples_dict) == 0:
        raise ValueError("samples_dict must contain at least one point set")

    names = list(samples_dict.keys())
    arrays = [_as_samples(samples_dict[name]) for name in names]

    n_methods = len(names)

    n_cols = min(3, n_methods)
    n_rows = int(np.ceil(n_methods / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 5.2 * n_rows),
        dpi=config.dpi,
        squeeze=False,
    )

    for idx, (name, x) in enumerate(zip(names, arrays)):
        ax = axes[idx // n_cols][idx % n_cols]

        n, d = x.shape
        i, j = dims

        if not (0 <= i < d and 0 <= j < d):
            raise ValueError(f"dims {dims} invalid for method {name} with dimension {d}")

        ax.scatter(
            x[:, i],
            x[:, j],
            s=config.point_size,
            alpha=config.alpha,
            edgecolors="none",
        )

        _beautify_axis(ax, grid_alpha=config.grid_alpha)

        if discrepancy == "l2":
            disc = l2_star_discrepancy(x[:, [i, j]])
            subtitle = rf"$D_2^*$ ≈ {disc:.3e}"

        elif discrepancy == "star":
            disc = approx_star_discrepancy(x[:, [i, j]])
            subtitle = rf"$D^*$ approx ≈ {disc:.3e}"

        elif discrepancy == "none":
            subtitle = ""

        else:
            raise ValueError("discrepancy must be one of: 'l2', 'star', 'none'")

        ax.set_title(f"{name}\n{subtitle}", fontsize=config.title_size)
        ax.set_xlabel(f"dim {i}", fontsize=config.label_size)
        ax.set_ylabel(f"dim {j}", fontsize=config.label_size)

    # Hide unused axes
    for idx in range(n_methods, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(
        "Comparison of point sets in 2D projection",
        fontsize=config.title_size + 2,
        y=1,
    )

    fig.tight_layout()

    _finalize_figure(fig, save_path=save_path, show=show)


# ============================================================
# Projection plots
# ============================================================

def plot_projection_grid(
    samples: npt.ArrayLike,
    dimensions: Sequence[int] | None = None,
    max_dimensions: int = 3,
    title: str | None = None,
    config: PlotConfig = DEFAULT_CONFIG,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot a grid of 2D projections for a high-dimensional point set.

    This is useful for detecting bad projections or correlations.

    Parameters
    ----------
    samples:
        Array of shape (N, d).
    dimensions:
        Dimensions to include. If None, use the first max_dimensions.
    max_dimensions:
        Maximum number of dimensions to plot if dimensions is None.
    """
    x = _as_samples(samples)
    n, d = x.shape

    if dimensions is None:
        dims = list(range(min(d, max_dimensions)))
    else:
        dims = list(dimensions)

    if len(dims) < 2:
        raise ValueError("at least two dimensions are required")

    for dim in dims:
        if not (0 <= dim < d):
            raise ValueError(f"invalid dimension {dim}")

    k = len(dims)

    fig, axes = plt.subplots(
        k - 1,
        k - 1,
        figsize=(3.2 * (k - 1), 3.2 * (k - 1)),
        dpi=config.dpi,
        squeeze=False,
    )

    for row in range(k - 1):
        for col in range(k - 1):
            ax = axes[row][col]

            if col > row:
                ax.axis("off")
                continue

            i = dims[col]
            j = dims[row + 1]

            ax.scatter(
                x[:, i],
                x[:, j],
                s=max(config.point_size * 0.55, 4),
                alpha=config.alpha,
                edgecolors="none",
            )

            _beautify_axis(ax, grid_alpha=config.grid_alpha)

            if row == k - 2:
                ax.set_xlabel(f"dim {i}", fontsize=9)

            if col == 0:
                ax.set_ylabel(f"dim {j}", fontsize=9)

            disc = l2_star_discrepancy(x[:, [i, j]])
            ax.set_title(rf"$D_2^*$={disc:.2e}", fontsize=9)

    if title is None:
        title = f"2D projection grid — N={n}, d={d}"

    fig.suptitle(title, fontsize=config.title_size + 2, y=1)
    fig.tight_layout()

    _finalize_figure(fig, save_path=save_path, show=show)


# ============================================================
# Discrepancy evolution
# ============================================================

def discrepancy_evolution(
    samples: npt.ArrayLike,
    n_values: Sequence[int] | None = None,
    discrepancy_type: str = "l2",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute discrepancy evolution for prefixes of a point set.

    Parameters
    ----------
    samples:
        Array of shape (N, d).
    n_values:
        Sequence of prefix sizes. If None, use powers of 2.
    discrepancy_type:
        "l2" or "star".

    Returns
    -------
    n_values:
        Array of sample sizes.
    discrepancies:
        Array of discrepancy values.
    """
    x = _as_samples(samples)
    n, _ = x.shape

    if n_values is None:
        max_power = int(np.floor(np.log2(n)))
        n_values_arr = 2 ** np.arange(3, max_power + 1)
        n_values_arr = n_values_arr[n_values_arr <= n]
    else:
        n_values_arr = np.asarray(n_values, dtype=int)

    if np.any(n_values_arr <= 0) or np.any(n_values_arr > n):
        raise ValueError("n_values must be between 1 and n_samples")

    discrepancies = []

    for n_prefix in n_values_arr:
        prefix = x[:n_prefix]

        if discrepancy_type == "l2":
            disc = l2_star_discrepancy(prefix)

        elif discrepancy_type == "star":
            disc = approx_star_discrepancy(prefix)

        else:
            raise ValueError("discrepancy_type must be 'l2' or 'star'")

        discrepancies.append(disc)

    return n_values_arr, np.asarray(discrepancies)


def plot_discrepancy_evolution(
    samples_dict: dict[str, npt.ArrayLike],
    n_values: Sequence[int] | None = None,
    discrepancy_type: str = "l2",
    loglog: bool = True,
    reference_mc_rate: bool = True,
    title: str | None = None,
    config: PlotConfig = DEFAULT_CONFIG,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot discrepancy evolution for several point sets.

    This is useful to compare random points and QMC sequences.

    Parameters
    ----------
    samples_dict:
        Dictionary mapping method names to sample arrays.
    n_values:
        Prefix sizes.
    discrepancy_type:
        "l2" or "star".
    loglog:
        If True, use log-log scale.
    reference_mc_rate:
        If True, draw a reference N^{-1/2} rate.
    """
    if len(samples_dict) == 0:
        raise ValueError("samples_dict must contain at least one point set")

    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=config.dpi)

    all_curves: list[tuple[np.ndarray, np.ndarray]] = []

    for name, samples in samples_dict.items():
        n_arr, disc_arr = discrepancy_evolution(
            samples,
            n_values=n_values,
            discrepancy_type=discrepancy_type,
        )

        all_curves.append((n_arr, disc_arr))

        ax.plot(
            n_arr,
            disc_arr,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=name,
        )

    if reference_mc_rate and all_curves:
        n_ref = all_curves[0][0]
        d_ref = all_curves[0][1]

        if len(n_ref) > 0:
            c = d_ref[0] * np.sqrt(n_ref[0])
            ref = c / np.sqrt(n_ref)

            ax.plot(
                n_ref,
                ref,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=r"reference $N^{-1/2}$",
            )

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.grid(True, which="both", alpha=config.grid_alpha)
    ax.set_xlabel("Number of points N", fontsize=config.label_size)

    if discrepancy_type == "l2":
        ax.set_ylabel(r"$L^2$-star discrepancy", fontsize=config.label_size)
    else:
        ax.set_ylabel("Approximate star discrepancy", fontsize=config.label_size)

    if title is None:
        title = "Discrepancy evolution"

    ax.set_title(title, fontsize=config.title_size, pad=12)

    ax.legend(frameon=True, fontsize=9)

    for spine in ax.spines.values():
        spine.set_alpha(0.35)

    fig.tight_layout()

    _finalize_figure(fig, save_path=save_path, show=show)


# ============================================================
# Distribution diagnostics
# ============================================================

def plot_coordinate_histograms(
    samples: npt.ArrayLike,
    dimensions: Sequence[int] | None = None,
    bins: int = 20,
    title: str | None = None,
    config: PlotConfig = DEFAULT_CONFIG,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot histograms of selected coordinates.

    Useful for checking marginal uniformity.
    """
    x = _as_samples(samples)
    _, d = x.shape

    if dimensions is None:
        dims = list(range(min(d, 4)))
    else:
        dims = list(dimensions)

    if len(dims) == 0:
        raise ValueError("at least one dimension is required")

    n_dims = len(dims)

    fig, axes = plt.subplots(
        1,
        n_dims,
        figsize=(4.2 * n_dims, 3.4),
        dpi=config.dpi,
        squeeze=False,
    )

    for idx, dim in enumerate(dims):
        if not (0 <= dim < d):
            raise ValueError(f"invalid dimension {dim}")

        ax = axes[0][idx]

        ax.hist(
            x[:, dim],
            bins=bins,
            range=(0.0, 1.0),
            density=True,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.6,
        )

        ax.axhline(1.0, linestyle="--", linewidth=1.4, alpha=0.85)

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(f"dim {dim}", fontsize=config.label_size)
        ax.set_ylabel("density", fontsize=config.label_size)
        ax.grid(True, alpha=config.grid_alpha)

        for spine in ax.spines.values():
            spine.set_alpha(0.35)

    if title is None:
        title = "Coordinate histograms"

    fig.suptitle(title, fontsize=config.title_size + 1, y=1.03)
    fig.tight_layout()

    _finalize_figure(fig, save_path=save_path, show=show)


# ============================================================
# High-level dashboard
# ============================================================

def plot_sampler_dashboard(
    samples: npt.ArrayLike,
    name: str = "Sampler",
    dims: tuple[int, int] = (0, 1),
    config: PlotConfig = DEFAULT_CONFIG,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Produce a small dashboard for one sampler:
    - 2D scatter plot
    - projection grid
    - coordinate histograms

    If save_dir is provided, figures are saved inside it.
    """
    x = _as_samples(samples)

    save_base = Path(save_dir) if save_dir is not None else None

    scatter_path = None
    grid_path = None
    hist_path = None

    if save_base is not None:
        save_base.mkdir(parents=True, exist_ok=True)
        safe_name = name.lower().replace(" ", "_")
        scatter_path = save_base / f"{safe_name}_scatter.png"
        grid_path = save_base / f"{safe_name}_projection_grid.png"
        hist_path = save_base / f"{safe_name}_histograms.png"

    plot_points_2d(
        x,
        dims=dims,
        title=f"{name} — 2D point set",
        discrepancy="l2",
        config=config,
        save_path=scatter_path,
        show=show,
    )

    if x.shape[1] >= 3:
        plot_projection_grid(
            x,
            title=f"{name} — projection grid",
            config=config,
            save_path=grid_path,
            show=show,
        )

    plot_coordinate_histograms(
        x,
        title=f"{name} — coordinate histograms",
        config=config,
        save_path=hist_path,
        show=show,
    )













