import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from qmc_lib.sampling.Halton import HaltonSampler
from qmc_lib.sampling.Kronecker import KroneckerSampler
from qmc_lib.sampling.LatinHypercube import LatinHypercubeSampler
from qmc_lib.sampling.UniformSampler import UniformSampler
from qmc_lib.sampling.Sobol import SobolSampler
from qmc_lib.sampling.LatticeRule import LatticeSampler
from qmc_lib.integration.Integrate import Integrator

def confidence_factor(c: float) -> float:
    return norm.ppf((1 + c) / 2)

class Statistics:
    """
    Statistical analysis and Monte Carlo / Quasi-Monte Carlo integration estimator.
    
    This class computes integral estimates along with variance, standard error,
    and confidence intervals for different sampling methods (Uniform, Latin Hypercube,
    Halton, Sobol, Lattice, Kronecker, etc.).
    
    It automatically adapts the statistical estimation strategy:
        - Crude Monte Carlo / LHS → single run with internal variance
        - Low-discrepancy sequences (QMC) → multiple independent replications
          to estimate the variance of the estimator.
    
    Attributes
    ----------
    estimate : float or None
        The estimated integral value.
    variance : float or None
        Estimated variance of the integrand (or of the estimator).
    std : float or None
        Standard deviation.
    standard_error : float or None
        Standard error of the mean.
    confidence_level : float or None
        Confidence level used (only for MC/LHS).
    confidence_interval : tuple[float, float] or None
        Confidence interval (only for MC/LHS).
    n_samples : int or None
        Number of samples used.
    """

    def __init__(self):
        self.estimate = None
        self.variance = None
        self.std = None
        self.standard_error = None
        self.confidence_level = None
        self.confidence_interval = None
        self.n_samples = None

    def compute_statistics(
        self,
        f,
        sampling_method: str,
        dimension: int,
        n_samples: int,
        seed: int = 0,
        confidence_level: float = 0.95,
        n_replications: int = 50
    ) :
        """
        Compute integral estimate and associated statistics for a given sampling method.
        
        Parameters
        ----------
        f : callable
            Function to integrate. It must accept a 1D array of length `dimension`
            and return a scalar.
        sampling_method : str
            Sampling method to use. Supported values:
                - "Uniform"
                - "LHS" (Latin Hypercube)
                - "Halton"
                - "Sobol"
                - "Lattice"
                - "Kronecker"
        dimension : int
            Dimensionality of the integration domain [0, 1)^dimension.
        n_samples : int
            Number of samples per integral estimate.
        seed : int, default=0
            Base random seed for reproducibility.
        confidence_level : float, default=0.95
            Confidence level for the interval (only used with Uniform and LHS).
        n_replications : int, default=50
            Number of independent replications (used for QMC methods to estimate
            the variance of the estimator).
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing the computed statistics:
                - "estimate": float
                - "variance": float
                - "std": float
                - "standard_error": float
                - "confidence_level": float or None
                - "confidence_interval": tuple[float, float] or None
                - "n_samples": int
        
        Notes
        -----
        For Uniform and LHS, a single run with analytical variance is used.
        For quasi-Monte Carlo methods (Halton, Sobol, etc.), multiple replications
        are performed to obtain a reliable estimate of the variance.
        """
        f_vec = np.vectorize(f, signature=f"({dimension}) -> ()")

        if sampling_method in ["Uniform", "LHS"]:
            if sampling_method == "Uniform":
                sampler = UniformSampler(
                    dimension=dimension,
                    n_samples=n_samples,
                    seed=seed
                )
            else:
                sampler = LatinHypercubeSampler(
                    dimension=dimension,
                    n_samples=n_samples,
                    seed=seed
                )

            samples = sampler.generate()
            integrator = Integrator(f_vec, samples=samples)
            estimate = integrator.compute()

            values = f_vec(samples)

            variance = float(np.var(values, ddof=1))
            std = float(np.sqrt(variance))
            standard_error = float(std / np.sqrt(samples.shape[0]))

            factor = confidence_factor(confidence_level)

            confidence_interval = (
                estimate - factor * standard_error,
                estimate + factor * standard_error
            )

            self.estimate = estimate
            self.variance = variance
            self.std = std
            self.standard_error = standard_error
            self.confidence_level = confidence_level
            self.confidence_interval = confidence_interval
            self.n_samples = n_samples

        else:
            estimates = []

            for i in range(n_replications):
                current_seed = seed + i

                if sampling_method == "Halton":
                    sampler = HaltonSampler(
                        dimension=dimension,
                        n_samples=n_samples,
                        seed=current_seed
                    )
                    samples = sampler.generate(scramble=True)

                elif sampling_method == "Sobol":
                    sampler = SobolSampler(
                        dimension=dimension,
                        n_samples=n_samples,
                        seed=current_seed
                    )
                    samples = sampler.generate(shifting=True)

                elif sampling_method == "Lattice":
                    sampler = LatticeSampler(
                        dimension=dimension,
                        n_samples=n_samples,
                        seed=current_seed
                    )
                    samples = sampler.generate(shifting=True)

                elif sampling_method == "Kronecker":
                    sampler = KroneckerSampler(
                        dimension=dimension,
                        n_samples=n_samples,
                        seed=current_seed
                    )
                    samples = sampler.generate(shifting=True)

                else:
                    raise ValueError(f"Unknown sampling method: {sampling_method}")

                integrator = Integrator(f_vec, samples=samples)
                estimates.append(integrator.compute())

            estimates = np.array(estimates)

            estimate = float(np.mean(estimates))
            variance = float(np.var(estimates, ddof=1))
            std = float(np.sqrt(variance))
            standard_error = float(std / np.sqrt(n_replications))

            self.estimate = estimate
            self.variance = variance
            self.std = std
            self.standard_error = standard_error
            self.confidence_level = None
            self.confidence_interval = None
            self.n_samples = n_samples

        return {
            "estimate": self.estimate,
            "variance": self.variance,
            "std": self.std,
            "standard_error": self.standard_error,
            "confidence_level": self.confidence_level,
            "confidence_interval": self.confidence_interval,
            "n_samples": self.n_samples,
        }
    
