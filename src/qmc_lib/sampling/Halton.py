from qmc_lib.core.sampler_core import BaseSampler
import numpy as np
from sympy import sieve
from numba import njit


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
    def __init__(self,dimension, n_samples,seed = 0):
        super().__init__(dimension,n_samples,seed)
        self.result = None
    
    def generate(self,backend = "numpy",scramble = False,first = 0):
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
    def forward(self, n, backend="numpy", scramble=False):
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
        self.result = None
        

            










