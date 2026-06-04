from qmc_lib.core.sampler_core import BaseSampler
import numpy as np

class RandomSampler(BaseSampler):

    def __init__(self,dimension,n_samples,seed):
        super().__init__(dimension,n_samples,seed) 
        self.rng =  np.random.default_rng(self.seed)
    def generate(self):
        return self.rng.uniform(0,1,(self.n_samples,self.dim))
    



    
