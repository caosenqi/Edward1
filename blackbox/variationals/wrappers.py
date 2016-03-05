import tensorflow as tf

class HVM:
    """
    Hierarchical variational model.

    Arguments
    ----------
    q_lik: Likelihood
        variational likelihood q(z | lambda)
    q_prior: Prior
        variational prior q(lambda | theta)
    r_post: Posterior
        approximate variational posterior r(lambda | z, phi)
    """
    def __init__(self, q_lik, q_prior, r_post):
        self.q_lik = q_lik
        # TODO don't make q_mf's parameters tf Variables anymore but
        # empty arrays
        #self.q_mf.set_params()
        self.q_prior = q_prior
        self.r_post = r_post

    def print_params(self, sess):
        self.q_prior.print_params(sess)

    def sample(self, size, sess):
        """
        lambda ~ q(lambda | theta)
        z ~ q(z | lambda)
        """
        eps = self.q_prior.sample_noise(size, sess)
        lamda = self.q_prior.reparam(size, sess)
        self.q_mf.set_params(lamda) # TODO
        return self.q_mf.sample(size, sess)
