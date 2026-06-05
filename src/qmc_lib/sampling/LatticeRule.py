from __future__ import annotations 
import numpy as np
from qmc_lib.core.sampler_core import BaseSampler
import numpy.typing as npt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

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
    path = DATA_DIR / filename
    if d <= 0:
        raise ValueError("La dimension d doit être strictement positive.")

    data = np.loadtxt(path, dtype=np.uint64)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Le fichier doit contenir au moins deux colonnes : dimension et z_j.")

    max_d = data.shape[0]

    if d > max_d:
        raise ValueError(f"Dimension demandée d={d}, mais le fichier ne contient que {max_d} composantes.")

    z = data[:d, 1]

    return z

class LatticeSampler(BaseSampler):
    """
    Lattice rule sampler for quasi-Monte Carlo integration.
    
    Generates samples using a lattice rule (also known as rank-1 lattice rule),
    which are highly efficient low-discrepancy sequences, especially in low to 
    moderate dimensions.
    
    This implementation relies on precomputed lattice vectors (good lattice points)
    for dimensions up to 9125.
    
    Parameters
    ----------
    dimension : int
        Number of dimensions (must be between 1 and 9125).
    n_samples : int
        Number of samples to generate.
    seed : int, default=0
        Random seed used for the shifting procedure.
    """

    def __init__(self,dimension : int,n_samples : int ,seed : int =0):
        if not( 1 <= dimension <= 9125):
            raise ValueError("dimension must be an integer between 1 and 9125")
        super().__init__(dimension,n_samples,seed)
    
    def generate(self,shifting : bool = True) -> npt.NDArray :
        """
        Generate lattice rule samples in the unit hypercube [0, 1)^dimension.
        
        Parameters
        ----------
        shifting : bool, default=True
            If True, applies a random shift modulo 1 to improve uniformity
            (recommended for most use cases).
        
        Returns
        -------
        npt.NDArray
            Array of shape (n_samples, dimension) containing the lattice
            samples in [0, 1)^dimension.
        """
        z = load_lattice_vector(d=self.dim)
        index = np.tile(np.arange(0 ,self.n_samples ),(self.dim,1)).T
        result = np.mod((index * z)/self.n_samples,1)
        if shifting :
            rng = np.random.default_rng(self.seed)
            self.shift = rng.uniform(0,1,size = self.dim)
            result = np.mod(result + self.shift, 1)
        return result
    



    




