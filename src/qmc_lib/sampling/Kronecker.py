from qmc_lib.core.sampler_core import BaseSampler
from sympy import sieve
import numpy as np

class KroneckerSampler(BaseSampler):

    def __init__(self,dimension,n_samples,seed = 0):
        super().__init__(dimension,n_samples,seed)
        self.result = None
        self.shift = None
    def generate(self,shifting = False,first = 0):
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
    def forward(self,n, shifting = False):
        self.n_samples = n
        if self.result is None :
            return self.generate(shifting )
        old_result = self.result.copy()
        start = old_result.shape[0]
        result = self.generate(shifting ,first = start)
        self.result = np.concatenate((old_result,result), axis = 0)
        return self.result
    def reset(self):
        self.result = None
        self.shift = None
        






