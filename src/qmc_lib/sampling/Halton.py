from __future__ import annotations  
from qmc_lib.core.sampler_core import BaseSampler
import numpy as np
from sympy import sieve
from numba import njit
import numpy.typing as npt

@njit
def generate_halton(primes,tab):
        result = np.zeros_like(tab,dtype=np.float64)
        facteur = 1/primes 
        while np.any(tab != 0):
            result += (tab % primes) * facteur
            tab //= primes
            facteur /= primes
        return result.T

class HaltonSampler(BaseSampler):
    """
    Halton sequence sampler for quasi-Monte Carlo methods.
    
    Generates low-discrepancy sequences using the Halton algorithm,
    with optional scrambling (Owen scrambling variant).
    
    Parameters
    ----------
    dimension : int
        Number of dimensions of the sample space.
    n_samples : int
        Number of samples to generate.
    seed : int, default=0
        Random seed for reproducibility (used only when scrambling is enabled).
    """
    def __init__(self,dimension : int, n_samples : int, seed :int = 0):
        super().__init__(dimension,n_samples,seed)
        self.result = None
    
    def generate(self,backend : str = "numpy",scramble : bool = False,first : int = 0) -> npt.NDArray : 
        """
        Generate Halton samples.
        
        Parameters
        ----------
        backend : {'numpy', 'numba'}, default='numpy'
            Backend to use for computation. 'numba' can be significantly faster
            for large sample counts.
        scramble : bool, default=False
            Whether to apply Owen scrambling to improve uniformity.
        first : int, default=0
            Index of the first sample to generate (useful for incremental generation).
        
        Returns
        -------
        npt.NDArray
            Array of shape (n_samples, dimension) containing the Halton samples
            in the unit hypercube [0, 1)^dimension.
        """
        self.rng =  np.random.default_rng(self.seed)
        sieve.extend_to_no(self.dim )
        primes_list = np.array(sieve._list)[0:self.dim ]
        primes = np.tile((primes_list),(self.n_samples,1)).T
        tab = np.tile(np.arange(first,self.n_samples + first),(self.dim,1))
        if scramble :
            facteur = 1/primes
            m = 0
            digit = []
            while np.any(tab != 0):
                digit.append(tab % primes)
                tab //= primes
                facteur /= primes
                m +=1
            digit = np.stack(digit , axis = 2)
            Mat = np.zeros(shape = (self.dim,m,m) , dtype = np.int64)
            sigma = np.zeros(shape = (self.dim,1,m) , dtype = np.int64)
            for i in range(self.dim):
                Mat[i] = self.rng.integers(low= 0 , high = primes_list[i], size = (m,m))
                Mat[i] = np.tril(Mat[i])
                diag = self.rng.integers(low= 1 , high = primes_list[i], size = m)
                np.fill_diagonal(Mat[i],diag)
                sigma[i] = self.rng.integers(low= 0 , high = primes_list[i], size = (1,m))
            scrambled = ((digit @ np.transpose(Mat, (0,2,1))) + sigma ) % (primes_list[:,None,None])
            k = np.arange(1, m + 1)              # shape (m,)
            bases = primes_list[:, None].astype(np.float64)         # shape (dim, 1)
            fact= bases ** (-k)             # shape (dim, m)
            fact = fact[:, None, :]      # shape (dim, 1, m)
            result = np.sum(scrambled * fact, axis=2)
            self.result = result.T
            return result.T
        else :
            if backend == "numba":
                self.result = generate_halton(primes,tab)
                return self.result
            result = np.zeros_like(tab,dtype=np.float64)
            facteur = 1/primes
            while np.any(tab != 0):
                result += (tab % primes) * facteur
                tab //= primes
                facteur /= primes
            self.result = result.T
            return result.T
    def forward(self, n : int , backend : str ="numpy", scramble : bool = False) -> npt.NDArray : 
        """
        Generate `n` samples and append them to previously generated samples.
        
        This method allows incremental generation of Halton sequences while
        maintaining continuity.
        
        Parameters
        ----------
        n : int
            Number of new samples to generate.
        backend : {'numpy', 'numba'}, default='numpy'
            Computation backend.
        scramble : bool, default=False
            Whether to use scrambled Halton sequences.
        
        Returns
        -------
        npt.NDArray
            Array of shape (total_samples, dimension) containing all generated
            samples so far.
        """
        self.n_samples = n
        if self.result is None:
            return self.generate(backend=backend, scramble=scramble, first=0)
        old_result = self.result.copy()
        start = old_result.shape[0]
        self.n_samples = n
        new_points = self.generate(backend, scramble, first=start)
        self.result = np.concatenate((old_result, new_points), axis=0)
        return self.result
    
    def reset(self) :
        """
        Reset the sampler, clearing all previously generated samples.
        
        After calling reset, the next call to `generate()` or `forward()`
        will start from the beginning of the sequence.
        """
        self.result = None


        

        









