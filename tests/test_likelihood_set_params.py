import blackbox as bb

sess = tf.InteractiveSession()

q_lik = bb.MFBernoulli(2)
q_lik.set_params(tf.constant([0.5, 0.5]))
q_lik.print_params(sess)
