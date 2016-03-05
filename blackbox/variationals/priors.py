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
        tf.tensor
            n_minibatch x dim(lambda) array of type np.float32, where each
            row is a sample from q.

        Notes
        -----
        Unlike the sample method for variational likelihoods, this
        return object is a TensorFlow array. This is all computations
        with respect to lambda are written internally, so we don't
        require an object like a placeholder so that computation can
        be compatible with probability models not written in
        TensorFlow.

        The method defaults to sampling noise and reparameterizing it
        (which will error out if it is not possible).

        TODO
        The method stores the samples for usage in log_prob. This is
        temporary as I don't know a good way for lambda_samples to be
        carried around in the Inference class rather than the prior
        class.
        """
        self.lambda_samples = self.reparam(self.sample_noise(size))
        return self.lambda_samples

    def log_prob(self):
        """sum_{b=1}^B log q(lambda_samples[b,:] | theta)"""
        raise NotImplementedError()
