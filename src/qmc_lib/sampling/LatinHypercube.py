from qmc_lib.core.sampler_core import BaseSampler
import numpy as np


class LatinHypercubeSampler(BaseSampler):

    def __init__(self,dimension,n_samples,seed = 0):
        super().__init__(dimension,n_samples,seed)

    def generate(self,randomized = True):
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


