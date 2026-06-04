from __future__ import annotations 
from qmc_lib.core.sampler_core import BaseSampler
import numpy as np
import numpy.typing as npt

class UniformSampler(BaseSampler):
    """
    Uniform random sampler (Monte Carlo sampler).
    
    Generates independent and identically distributed (i.i.d.) uniform 
    random samples in the unit hypercube [0, 1)^dimension.
    
    This is the classical Monte Carlo sampling method, used as a baseline
    for comparison with low-discrepancy sequences.
    
    Parameters
    ----------
    dimension : int
        Number of dimensions of the sample space.
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self,dimension:int,n_samples:int,seed:int):
        super().__init__(dimension,n_samples,seed) 
        self.rng =  np.random.default_rng(self.seed)
    def generate(self) -> npt.NDArray :
        """
        Generate uniform random samples in the unit hypercube [0, 1)^dimension.
        
        Returns
        -------
        npt.NDArray
            Array of shape (n_samples, dimension) containing uniformly 
            distributed random samples.
        """
        return self.rng.uniform(0,1,(self.n_samples,self.dim))




    
