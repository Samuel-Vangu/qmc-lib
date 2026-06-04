import numpy as np
from qmc_lib.core.sampler_core import BaseSampler

def load_lattice_vector(d, filename="lattice-33002-1024-1048576.9125"):
    """
    Charge les d premières composantes du vecteur générateur z
    depuis un fichier de generating vectors pour rank-1 lattice rules.

    Le fichier doit contenir deux colonnes :
        dimension   z_j

    Paramètres
    ----------
    d : int
        Dimension souhaitée.
    filename : str
        Chemin vers le fichier de vecteurs générateurs.

    Retour
    ------
    z : np.ndarray
        Tableau numpy [z1, z2, ..., zd] de dtype uint64.
    """

    if d <= 0:
        raise ValueError("La dimension d doit être strictement positive.")

    data = np.loadtxt(filename, dtype=np.uint64)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Le fichier doit contenir au moins deux colonnes : dimension et z_j.")

    max_d = data.shape[0]

    if d > max_d:
        raise ValueError(f"Dimension demandée d={d}, mais le fichier ne contient que {max_d} composantes.")

    z = data[:d, 1]

    return z

class LatticeSampler(BaseSampler):

    def __init__(self,dimension,n_samples,seed=0):
        if not( 1 <= dimension <= 9125):
            raise ValueError("dimension must be an integer between 1 and 9125")
        super().__init__(dimension,n_samples,seed)
    
    def generate(self,shift = True):
        z = load_lattice_vector(d=self.dim)
        index = np.tile(np.arange(0 ,self.n_samples ),(self.dim,1)).T
        result = np.mod((index * z)/self.n_samples,1)
        if shift :
            rng = np.random.default_rng(self.seed)
            self.shift = rng.uniform(0,1,size = self.dim)
            result = np.mod(result + self.shift, 1)
        return result
    
    




