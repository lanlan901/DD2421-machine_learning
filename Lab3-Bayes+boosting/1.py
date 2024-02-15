# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts, 1)) / Npts
    else:
        assert (W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses, 1))

    # TODO: compute the values of prior for each class!
    # ==========================
    for i, k in enumerate(classes):
        prior[i] = np.sum(W[labels == k]) / np.sum(W)
    # ==========================

    return prior


# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert (X.shape[0] == labels.shape[0])
    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts, 1)) / float(Npts)

    mu = np.zeros((Nclasses, Ndims))
    sigma = np.zeros((Nclasses, Ndims, Ndims))

    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    for k in classes:
        index = np.where(labels == k)[0]
        Xk = X[index]
        Wk = W[index]

        weighted_sum = np.sum(Xk * Wk, axis=0)
        sum_of_weights = np.sum(Wk)
        mu_k = weighted_sum / sum_of_weights
        mu[k, :] = mu_k

        Xk_centered = Xk - mu_k
        for i in range(Ndims):
            sigma[k, i, i] = np.sum(Wk * (Xk_centered[:, i] ** 2)) / sum_of_weights
    # ==========================

    return mu, sigma


# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):
    Npts = X.shape[0]
    Nclasses, Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================
    for i in range(Npts):
        for k in range(Nclasses):
            diff = X[i] - mu[k]
            inv_sigma = np.linalg.inv(sigma[k])
            sign, logdet = np.linalg.slogdet(sigma[k])
            log_likelihood = -0.5 * np.dot(diff.T, np.dot(inv_sigma, diff))
            log_prior = np.log(prior[k])
            log_norm_constant = -0.5 * (Ndims * np.log(2 * np.pi) + logdet)
            logProb[k, i] = log_likelihood + log_prior + log_norm_constant

    # ==========================

    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb, axis=0)
    return h