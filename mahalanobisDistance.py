import imp

from scipy.stats import chi2
import numpy as np

def mahalanobis(x=None, data=None, cov=None):

    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)

    return mahal.diagonal()

def calculatePValue(mahal):
    pvalue = 1 - chi2.cdf(mahal, (len(mahal)- 1))
    return pvalue


