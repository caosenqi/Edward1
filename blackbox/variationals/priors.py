class Prior:
    """
    Base class for variational priors, q(lambda | theta).
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = None
        self.lambda_samples = None

    def print_params(self, sess):
        raise NotImplementedError()

    def sample_noise(self, size):
        """
        eps = sample_noise() ~ s(eps)
        s.t. lambda = reparam(eps; theta) ~ q(lambda | theta)

        Returns
        -------
        np.ndarray
            n_minibatch x dim(lambda) array of type np.float32, where each
            row is a sample from q.

        Notes
        -----
        Unlike the other methods, this return object is a realization
        of a TensorFlow array. This is required as we rely on
        NumPy/SciPy for sampling from distributions.
        """
        raise NotImplementedError()

    def reparam(self, eps):
        """
        eps = sample_noise() ~ s(eps)
        s.t. lambda = reparam(eps; theta) ~ q(lambda | theta)
        """
        raise NotImplementedError()

    def sample(self, size, sess):
        """
        lambda ~ q(lambda | theta)

        Returns
        -------
        np.ndarray
            n_minibatch x dim(lambda) array of type np.float32, where each
            row is a sample from q.

        Notes
        -----
        Unlike the other methods, this return object is a realization
        of a TensorFlow array. This is required as we rely on
        NumPy/SciPy for sampling from distributions.

        The method defaults to sampling noise and reparameterizing it
        (which will error out if this is not possible).

        TODO
        The method stores the samples for usage in log_prob. This is
        temporary as I don't know a good way for lambda_samples to be
        carried around in the Inference class rather than the prior
        class.
        """
        self.lambda_samples = self.reparam(self.sample_noise(size))
        return sess.run(self.lambda_samples)

    def log_prob(self):
        """sum_{b=1}^B log q(lambda_samples[b,:] | theta)"""
        raise NotImplementedError()
