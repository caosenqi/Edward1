class Posterior:
    """
    Base class for approximate variational posteriors, r(lambda | z, phi).
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = None

    def log_prob(self, lamda):
        """log r(lambda | phi)"""
        pass
