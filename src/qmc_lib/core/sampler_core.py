class BaseSampler :
    # Informations communes à tous les Samplers

    def __init__(self, dimension,n_samples,seed =  None):
        self.dim = dimension 
        self.seed = seed
        self.n_samples = n_samples