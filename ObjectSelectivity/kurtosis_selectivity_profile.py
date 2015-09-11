# -*- coding: utf-8 -*-
__author__ = 'bptripp'

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma


class LehkySparseness:

    def __init__(self, scale_samples):
        """
        A statistical model of max spike rate distribution based on:

        Lehky, S. R., Kiani, R., Esteky, H., & Tanaka, K. (2011). Statistics of
            visual responses in primate inferotemporal cortex to object stimuli.
            Journal of Neurophysiology, 106(3), 1097–117.

        This paper shows (with a large passive-viewing IT dataset) that selectivity
        (kurtosis of neuron responses to different images) is lower than sparseness
        (kurtosis of population responses to each image). They propose a simple
        model (see pg. 1112) that explains this difference in terms of heterogeneity
        of spike rate distributions.

        They model these spike rate distributions as gamma functions, whereas we need
        a distribution of maximum rates. To relate this to the Lehky et al. model
        we take the point at which the CDF = 0.99 as the maximum.

        We don't directly use Lehky et al.'s distribution of gamma PDF parameters, since
        probably some of their variability was due to stimulus parameters such as size,
        position, etc. being non-optimal for many neurons. We do use their shape-
        parameter distribution, but we use a different scale parameter distribution that
        approximates that of Lehky et al. after being scaled by a realistic distribution
        of scale factors for non-optimal size, position, etc.

        The distribution of scale factors must come from the rest of the IT model. It is
        needed here as a list of samples.

        :param scale_samples: Random samples of scale factors that would be expected in
            the Lehky et al. experiment. The number of samples isn't critical as they are
            just used to estimate a distribution. Samples should lie between [0,1].
        """

        # Lehky et al. parameters ...
        ala = 4 # shape parameter (a) of Lehky (l) for PDF of shape parameters (a) for rate PDFs
        alb = 2
        bla = 0.5
        blb = 0.5

        # empirical expectation and variance of scales ...
        Es = np.mean(scale_samples)
        Vs = np.var(scale_samples)

        # expectation and variance of Lehky et al. distribution of mean rates ...
        El = ala*bla*alb*blb
        Vl = ala*bla**2*alb*blb**2 + ala*bla**2*(alb*blb)**2 + alb*blb**2*(ala*bla)**2

        # expectation and variance of "full" (unscaled) distribution of mean rates
        #   that will approximate Lehky after scaling ...
        Ef = El / Es
        Vf = (Vl - Vs*(El/Es)**2) / (Vs + Es**2)

        # shape and scale parameters for full distribution (keeping shape same as scaled one) ...
        self._afa = ala
        self._bfa = bla
        self._bfb = (Vf - Ef**2/ala) / (Ef*bla*(1+ala))
        self._afb = Ef / (self._afa*self._bfa*self._bfb)

    def sample_max_rates(self, n):
        """
        :param n: Number of maximum spike rates needed
        :return: n random spike-rate samples
        """
        a = np.random.gamma(self._afa, scale=self._bfa, size=n)  # Samples of shape parameter
        b = np.random.gamma(self._afb, scale=self._bfb, size=n)  # Samples of scale parameter

        a = np.maximum(1.01, a)  #avoid making PDF go to infinity at zero spike rate

        return gamma.ppf(.99, a, loc=0, scale=b)

if __name__ == '__main__':
    scale_factor_samples = np.random.rand(500)
    ls = LehkySparseness(scale_factor_samples)

    n = 1000
    full_max_rates = ls.sample_max_rates(n)
    scale_factors = np.random.rand(n)
    scaled_max_rates = scale_factors * full_max_rates

    plt.subplot(211)
    plt.hist(full_max_rates)
    plt.title('histogram of full (unscaled) max spike rates')
    plt.subplot(212)
    plt.hist(scaled_max_rates)
    plt.title('histogram of scaled max spike rates')
    plt.show()
