from __future__ import annotations 
from qmc_lib.core.sampler_core import BaseSampler
import numpy as np
import numpy.typing as npt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

def load_direction_numbers(filename="new-joe-kuo-6.21201"):
    path = DATA_DIR / filename
    data = {}

    with open(path, "r") as f:

        lines = f.readlines()

        # skip header
        for line in lines[1:]:

            parts = line.strip().split()

            d = int(parts[0])
            s = int(parts[1])
            a = int(parts[2])

            m = list(map(int, parts[3:]))

            data[d] = {
                "s": s,
                "a": a,
                "m": m
            }

    return data

Sobol_data = load_direction_numbers()



def coord_j(j,n_bits,n_samples,first):
    if j == 1 :
        liste_m = np.ones(n_bits, dtype=np.uint64)
    else :
        data = Sobol_data[j]
        s = data["s"]
        coef = list(np.binary_repr(data["a"], width=s-1)) + ["1"]
        coef = np.array(list(map(int, coef)), dtype=np.uint64)
        liste_m = np.zeros(n_bits, dtype=np.uint64)
        m_init = np.array(data["m"], dtype=np.uint64)
        init_len = min(len(m_init), n_bits)
        liste_m[0:init_len] = m_init[0:init_len]
        for k in range(init_len,n_bits):
            deux = (2 ** np.arange(1, s+1)).astype(np.uint64)
            resultat = (deux * np.flip(liste_m[k-s:k]) * coef) 
            resultat = np.bitwise_xor.reduce(resultat)
            liste_m[k] = np.bitwise_xor(resultat, liste_m[k-s])
    direction_int = liste_m.astype(np.uint64) << (n_bits - 1 - np.arange(n_bits, dtype=np.uint64))

    I = np.arange(first ,first + n_samples, dtype=np.uint64)

    bin_tab = ((I.reshape(-1, 1) >> np.arange(n_bits, dtype=np.uint64)) & 1)

    matrix = direction_int * bin_tab

    result = np.bitwise_xor.reduce(matrix, axis=1) / float(1 << n_bits)
    return result 



class SobolSampler(BaseSampler):
    """
    Sobol sequence sampler for quasi-Monte Carlo methods.
    
    Generates Sobol sequences, which are highly efficient low-discrepancy 
    sequences widely used in numerical integration, sensitivity analysis,
    and optimization. This implementation supports randomized Sobol 
    sequences via shifting.
    
    Supports dimensions up to 21,201 thanks to precomputed direction numbers.
    
    Parameters
    ----------
    dimension : int
        Number of dimensions of the sample space.
    n_samples : int
        Number of samples to generate.
    seed : int, default=0
        Random seed used for the shifting procedure.
    """

    def __init__(self,dimension : int , n_samples:int,seed : int = 0):
        if not( 1 <= dimension <= 21201):
            raise ValueError("dimension must be an integer between 1 and 21201")
        super().__init__(dimension,n_samples,seed)
        self.result = None
        self.shift = None
    
    def generate(self,shifting : bool = False,first:int = 0) -> npt.NDArray:
        """
        Generate Sobol sequence samples in the unit hypercube [0, 1)^dimension.
        
        Parameters
        ----------
        shifting : bool, default=True
            If True, applies a random digital shift (modulo 1) to improve 
            uniformity. The shift is generated once and reused for consistency.
        first : int, default=0
            Starting index for sample generation (used for incremental sampling).
        
        Returns
        -------
        npt.NDArray
            Array of shape (n_samples, dimension) containing the Sobol
            samples in [0, 1)^dimension.
        """
        n_bits = max(1,(first + self.n_samples).bit_length()) # Nombre de direction numbers necessaire 
        vectorized_coord = np.vectorize(coord_j, signature=f" () -> ({self.n_samples})", excluded={'n_bits', 'n_samples','first'})
        dims = np.arange(1, self.dim+1)
        result  = vectorized_coord(dims, n_bits=n_bits,n_samples=self.n_samples,first = first).T
        if shifting :
            rng = np.random.default_rng(self.seed)
            if self.shift is None:
                self.shift = rng.uniform(0,1,size=self.dim)
            result = np.mod(result + self.shift, 1)
        self.result = result
        return result
    def forward(self,n : int,shifting : bool = False) -> npt.NDArray:
        """
        Generate `n` new samples and append them to the previously generated ones.
        
        This method enables incremental generation of the Sobol sequence while
        maintaining continuity.
        
        Parameters
        ----------
        n : int
            Number of new samples to generate and append.
        shifting : bool, default=False
            Whether to apply (and reuse) a random shift.
        
        Returns
        -------
        npt.NDArray
            Array of shape (total_samples, dimension) containing all samples
            generated so far.
        """
        self.n_samples = n
        if self.result is None :
            return self.generate(shifting = shifting,first = 0)
        old_result = self.result.copy()
        start = self.result.shape[0]
        new_result = self.generate(shifting = shifting ,first = start)
        self.result = np.concatenate((old_result,new_result), axis = 0)
        return self.result

    def reset(self):
        """
        Reset the sampler to its initial state.
        
        Clears all generated samples and the stored shift (if any).
        The next call to `generate()` or `forward()` will start from the beginning.
        """
        self.result = None
        self.shift = None



    



