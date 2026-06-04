from __future__ import annotations  
from qmc_lib.core.sampler_core import BaseSampler
import numpy as np
import numpy.typing as npt

class LatinHypercubeSampler(BaseSampler):
    """
    Latin Hypercube Sampler for quasi-Monte Carlo and experimental design.
    
    Generates Latin Hypercube Sampling (LHS) points, which provide excellent 
    space-filling properties and stratification in each dimension.
    
    Two variants are available:
        - Centered LHS (randomized=False)
        - Randomized (jittered) LHS (randomized=True)
    
    Parameters
    ----------
    dimension : int
        Number of dimensions of the sample space.
    n_samples : int
        Number of samples to generate.
    seed : int, default=0
        Random seed for reproducibility.
    """

    def __init__(self,dimension : int ,n_samples :int ,seed:int = 0) :
        super().__init__(dimension,n_samples,seed)

    def generate(self,randomized :bool = True) -> npt.NDArray :
        """
        Generate Latin Hypercube samples in the unit hypercube [0, 1)^dimension.
        
        Parameters
        ----------
        randomized : bool, default=True
            If True, generates a randomized (jittered) Latin Hypercube where
            each point is uniformly distributed inside its cell.
            If False, uses the centered version (points at the center of each cell).
        
        Returns
        -------
        npt.NDArray
            Array of shape (n_samples, dimension) containing the Latin Hypercube
            samples in [0, 1)^dimension.
        """
        tab = np.tile(np.arange(1,self.n_samples+1), (self.dim,1))
        rng = np.random.default_rng(self.seed)
        permute = np.vectorize(rng.permutation, signature="(n) -> (n)")
        permutations = permute(tab)
        if randomized :
            U = rng.uniform(0,1 , size = permutations.shape)
            result = ((permutations - U) / self.n_samples).T
            return result
        result = ((permutations - 0.5) / self.n_samples).T
        return result
    




