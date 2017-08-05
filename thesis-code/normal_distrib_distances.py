import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from scipy.spatial.distance import euclidean

def D_kl_gauss(mean1, logstd1, mean2, logstd2):
    """
    Computes KL divergence of two gaussian distributions according to this thread :
    http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians

    :param mean1: Mean of the first distribution
    :param mean2: Mean of the second distribution
    :param logstd1: Logarithm of standard deviation of the first distribution
    :param logstd2: Logarithm of standard deviation of the second distribution
    :return: KL divergence of two gaussian distributions
    """

    # e ^ (2 log(x)) = x ^ 2
    var1 = np.exp(2 * logstd1)
    var2 = np.exp(2 * logstd2)

    kl = np.sum(logstd2 - logstd1 + (var1 + np.square(mean1 - mean2)) / (2 * var2) - 0.5)
    return kl



x = np.linspace(-20, 20, 1000)


mu1 = -2
sigma1 = 1
pdf1 = mlab.normpdf(x, mu1, sigma1)

plt.plot(x,pdf1, color='b')
plt.fill_between(x, pdf1, 0, facecolor='blue', alpha=0.3)

mu2 = 2
sigma2 = 1
pdf2 = mlab.normpdf(x, mu2, sigma2)

plt.plot(x,pdf2, color='r')
plt.fill_between(x, pdf2, 0, facecolor='red', alpha=0.3)
plt.title("Euclid distance = {}, KL div = {}".format(euclidean((mu1, sigma1), (mu2, sigma2)), D_kl_gauss(mu1, np.log(sigma1), mu2, np.log(sigma2))))
plt.show()


mu1 = -5
sigma1 = 6
pdf1 = mlab.normpdf(x, mu1, sigma1)

plt.plot(x,pdf1, color='b')
plt.fill_between(x, pdf1, 0, facecolor='blue', alpha=0.3)

mu2 = 5
sigma2 = 6
pdf2 = mlab.normpdf(x, mu2, sigma2)

plt.plot(x,pdf2, color='r')
plt.fill_between(x, pdf2, 0, facecolor='red', alpha=0.3)
plt.title("Euclid distance = {}, KL div = {}".format(euclidean((mu1, sigma1), (mu2, sigma2)), D_kl_gauss(mu1, np.log(sigma1), mu2, np.log(sigma2))))
plt.show()
