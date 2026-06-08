from __future__ import annotations  
from qmc_lib.core.sampler_core import BaseSampler
from sympy import sieve
import numpy as np
import numpy.typing as npt

class KroneckerSampler(BaseSampler):
    """
    Kronecker sequence sampler for quasi-Monte Carlo methods.
    
    Generates low-discrepancy sequences using the Kronecker method 
    (also known as Weyl sequence) based on the square roots of the first 
    `dimension` prime numbers.
    
    This method is particularly efficient and has good uniformity properties
    in moderate dimensions.
    
    Parameters
    ----------
    dimension : int
        Number of dimensions of the sample space.
    n_samples : int
        Number of samples to generate.
    seed : int, default=0
        Random seed used for the shifting procedure.
    """

    def __init__(self,dimension : int ,n_samples :int ,seed :int = 0):
        super().__init__(dimension,n_samples,seed)
        self.result = None
        self.shift = None
    def generate(self,shifting : bool = True,first : int = 0) -> npt.NDArray :
        """
        Generate Kronecker sequence samples.
        
        Parameters
        ----------
        shifting : bool, default=True
            If True, applies a random shift (modulo 1) to all dimensions.
            The shift is generated once and reused for subsequent calls.
        first : int, default=0
            Starting index for sample generation (used for incremental sampling).
        
        Returns
        -------
        npt.NDArray
            Array of shape (n_samples, dimension) containing the samples
            in the unit hypercube [0, 1)^dimension.
        """
        sieve.extend_to_no(self.dim)
        primes_sqrt = np.sqrt(np.array(sieve._list)[0:self.dim])
        tab = np.tile(np.arange(first,self.n_samples+first), (self.dim,1)).T
        result = np.mod(tab * primes_sqrt,1)
        if shifting :
            rng =  np.random.default_rng(self.seed)
            if self.shift is None:
                self.shift = rng.uniform(0,1,size= self.dim)
            result = np.mod(result+ self.shift,1)
        self.result = result
        return result
    def forward(self,n : int , shifting :bool = True) -> npt.NDArray:
        """
        Generate `n` new samples and append them to the existing ones.
        
        This method supports incremental generation of the Kronecker sequence.
        
        Parameters
        ----------
        n : int
            Number of new samples to generate and append.
        shifting : bool, default=True
            Whether to use random shifting. If True, the same shift is 
            applied consistently across all generated samples.
        
        Returns
        -------
        npt.NDArray
            Array of shape (total_samples, dimension) with all samples generated so far.
        """
        self.n_samples = n
        if self.result is None :
            return self.generate(shifting )
        old_result = self.result.copy()
        start = old_result.shape[0]
        result = self.generate(shifting ,first = start)
        self.result = np.concatenate((old_result,result), axis = 0)
        return self.result
    def reset(self):
        """
        Reset the sampler to its initial state.
        
        Clears all generated samples and the random shift (if any).
        The next call to `generate()` or `forward()` will start from scratch.
        """
        self.result = None
        self.shift = None
        





