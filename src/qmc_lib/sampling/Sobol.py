from qmc_lib.core.sampler_core import BaseSampler
import numpy as np


def load_direction_numbers(path):

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

Sobol_data = load_direction_numbers("new-joe-kuo-6.21201")



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

    def __init__(self,dimension, n_samples:int,seed = 0):
        if not( 1 <= dimension <= 21201):
            raise ValueError("dimension must be an integer between 1 and 21201")
        super().__init__(dimension,n_samples,seed)
        self.result = None
        self.shift = None
    
    def generate(self,shift = False,first = 0):
        n_bits = max(1,(first + self.n_samples).bit_length()) # Nombre de direction numbers necessaire 
        vectorized_coord = np.vectorize(coord_j, signature=f" () -> ({self.n_samples})", excluded={'n_bits', 'n_samples','first'})
        dims = np.arange(1, self.dim+1)
        result  = vectorized_coord(dims, n_bits=n_bits,n_samples=self.n_samples,first = first).T
        if shift :
            rng = np.random.default_rng(self.seed)
            if self.shift is None:
                self.shift = rng.uniform(0,1,size=self.dim)
            result = np.mod(result + self.shift, 1)
        self.result = result
        return result
    def forward(self,n,shift = False):
        self.n_samples = n
        if self.result is None :
            return self.generate(shift = shift,first = 0)
        old_result = self.result.copy()
        start = self.result.shape[0]
        new_result = self.generate(shift = shift ,first = start)
        self.result = np.concatenate((old_result,new_result), axis = 0)
        return self.result

    def reset(self):
        self.result = None
        self.shift = None

    



