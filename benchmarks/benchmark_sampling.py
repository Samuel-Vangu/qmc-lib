

import qmcpy as qp

from qmc_lib.sampling.Sobol import SobolSampler
from qmc_lib.sampling.Halton import HaltonSampler
from qmc_lib.sampling.LatticeRule import LatticeSampler
from qmc_lib.sampling.UniformSampler import UniformSampler


# ============================================================
# Generic generation functions
# ============================================================

def generate_qmcpy_samples(sampler_class, dimension, n_samples, seed):
    """
    Generate samples with QMCPy.
    """
    sampler = sampler_class(
        dimension=dimension,
        seed=seed,
    )
    return sampler.gen_samples(n_samples)


def generate_qmc_lib_samples(sampler_class, dimension, n_samples, seed):
    """
    Generate samples with qmc_lib.
    """
    sampler = sampler_class(
        dimension=dimension,
        n_samples=n_samples,
        seed=seed,
    )
    return sampler.generate()


# ============================================================
# QMCPy benchmark functions
# ============================================================

def benchmark_qmcpy_uniform(dimension, n_samples, seed=0):
    return generate_qmcpy_samples(
        qp.IIDStdUniform,
        dimension,
        n_samples,
        seed,
    )


def benchmark_qmcpy_sobol(dimension, n_samples, seed=0):
    return generate_qmcpy_samples(
        qp.Sobol,
        dimension,
        n_samples,
        seed,
    )


def benchmark_qmcpy_halton(dimension, n_samples, seed=0):
    return generate_qmcpy_samples(
        qp.Halton,
        dimension,
        n_samples,
        seed,
    )


def benchmark_qmcpy_lattice(dimension, n_samples, seed=0):
    return generate_qmcpy_samples(
        qp.Lattice,
        dimension,
        n_samples,
        seed,
    )


# ============================================================
# qmc_lib benchmark functions
# ============================================================

def benchmark_qmc_lib_uniform(dimension, n_samples, seed=0):
    return generate_qmc_lib_samples(
        UniformSampler,
        dimension,
        n_samples,
        seed,
    )


def benchmark_qmc_lib_sobol(dimension, n_samples, seed=0):
    return generate_qmc_lib_samples(
        SobolSampler,
        dimension,
        n_samples,
        seed,
    )


def benchmark_qmc_lib_halton(dimension, n_samples, seed=0):
    return generate_qmc_lib_samples(
        HaltonSampler,
        dimension,
        n_samples,
        seed,
    )


def benchmark_qmc_lib_lattice(dimension, n_samples, seed=0):
    return generate_qmc_lib_samples(
        LatticeSampler,
        dimension,
        n_samples,
        seed,
    )