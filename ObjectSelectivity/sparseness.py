# -*- coding: utf-8 -*-
__author__ = 'bptripp'

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from kurtosis_selectivity_profile import calculate_kurtosis

import sys
sys.path.append("..")

from PositionTolerance import gaussian_position_profile as gpp
from SizeTolerance import log_normal_size_profile as lnsp

import population_utils as utils
# Force reload (compile) IT cortex modules to pick changes not included in cached version.
reload(gpp)
reload(lnsp)


def get_average_selectivity_and_sparseness(r_mat):
    """
    :param r_mat: n_neurons x n_objects
    :return: avg neuron selectivity, avg population sparseness
    """
    # Single Neuron selectivities
    n_neurons = r_mat.shape[0]
    n_objs = r_mat.shape[1]

    selectivities = np.zeros(n_neurons)
    sparsenesses = np.zeros(n_objs)

    for n_idx in np.arange(n_neurons):
        rates = r_mat[n_idx, :]
        selectivities[n_idx] = calculate_kurtosis(rates)

    # Population sparseness
    for o_idx in np.arange(n_objs):
        rates = r_mat[:, o_idx]
        sparsenesses[o_idx] = calculate_kurtosis(rates)

    return np.mean(selectivities), np.mean(sparsenesses)


def get_scale_factors_from_model(n_samples):
    """
    Get scale parameters from gaussian position and size profiles

    :param n_samples: "
    :return:

    """
    factors = np.zeros(shape=n_samples)

    for ii in np.arange(n_samples):

        # Use a random distribution for position tolerance. In the IT cortex model
        # position tolerance is derived from the activity fraction selectivity. This is a
        # calculated value and takes a long time to generate. It was found to lie between 0-0.6
        # to speed up just use a uniform distribution over this range.
        position = gpp.GaussianPositionProfile(np.random.uniform(0, 0.6))
        size = lnsp.LogNormalSizeProfile(position.position_tolerance)

        # Both
        factors[ii] = position.firing_rate_modifier(0, 0) * \
            size.firing_rate_modifier(7 * np.pi / 180.0)

        # Position Only
        # factors[ii] = position.firing_rate_modifier(0, 0)

        # Size only
        # factors[ii] = size.firing_rate_modifier(7 * np.pi / 180.0)

    # plt.plot(factors)
    # print("Factors average %0.4f, var %0.4f" % (np.mean(factors), np.var(factors)))
    # raw_input("Continue?")

    return factors


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
            just used to estimate a distribution.
        """

        # Lehky et al. parameters ...
        ala = 4  # shape parameter (a) of Lehky (l) for PDF of shape parameters (a) for rate PDFs
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

        #print("Scale distribution Gamma parameters: a=%0.4f, b=%0.4f" %(self._afb, self._bfb))


    def sample_rates_rates(self, n):
        """
        :param n: Number of maximum spike rates needed
        :return: n random spike-rate samples
        """

        a = np.random.gamma(self._afa, scale=self._bfa, size=n) #samples of shape parameter
        b = np.random.gamma(self._afb, scale=self._bfb, size=n) #samples of scale parameter

        a = np.maximum(1.01, a) #avoid making PDF go to infinity at zero spike rate

        return gamma.ppf(.99, a, loc=0, scale=b)

    def get_sample_profile_and_rates_mat(self, n_neurons, n_objs):

        # Create sample profiles
        a_arr = np.random.gamma(self._afa, scale=self._bfa, size=n_neurons)
        b_arr = np.random.gamma(self._afb, scale=self._bfb, size=n_neurons)

        r_mat = np.zeros(shape=(n_neurons, n_objs))

        # Create sample rates to all objects for each neuron
        for n_idx in np.arange(n_neurons):
            r_mat[n_idx, :] = \
                gamma.rvs(a=a_arr[n_idx], loc=0, scale=b_arr[n_idx], size=n_objs)

        return r_mat

    @property
    def bfb(self):
        return self._bfb

    @property
    def afb(self):
        return self._afb


if __name__ == '__main__':
    plt.ion()
    # scale_factor_samples = np.random.rand(500)
    # ls = LehkySparseness(scale_factor_samples)
    #
    # n = 1000
    # full_max_rates = ls.sample_max_rates(n)
    # scale_factors = np.random.rand(n)
    # scaled_max_rates = scale_factors * full_max_rates
    #
    # plt.subplot(211)
    # plt.hist(full_max_rates)
    # plt.title('histogram of full (unscaled) max spike rates')
    # plt.subplot(212)
    # plt.hist(scaled_max_rates)
    # plt.title('histogram of scaled max spike rates')
    # plt.show()

    # -----------------------------------------------------
    n_runs = 100
    neurons = 674
    objects = 806

    f_avg_selectivity_arr = np.zeros(n_runs)
    f_avg_sparseness_arr = np.zeros(n_runs)

    s_avg_selectivities_arr = np.zeros(n_runs)
    s_avg_sparseness_arr = np.zeros(n_runs)

    scale_a_arr = np.zeros(n_runs)
    scale_b_arr = np.zeros(n_runs)

    for r_idx in np.arange(n_runs):

        # print ("Run %i" % r_idx)

        # Scale Factors
        # scale_factor_samples = np.random.rand(10000)
        scale_factor_samples = get_scale_factors_from_model(1000)

        # ------------------------------------------------------------------------
        ls = LehkySparseness(scale_factor_samples)

        # Store modified scale values these will be used in the larger model
        scale_a_arr[r_idx] = ls.afb
        scale_b_arr[r_idx] = ls.bfb

        # Generate sample rates
        rates_mat = ls.get_sample_profile_and_rates_mat(neurons, objects)

        # Calculate average neuron selectivities and population sparseness
        f_avg_selectivity_arr[r_idx], f_avg_sparseness_arr[r_idx] = \
            get_average_selectivity_and_sparseness(rates_mat)

        # Generate random scale factors and multiply them with rates and
        # recalculate selectivity and sparseness
        # scale_factors = np.random.rand(neurons)
        scale_factors = get_scale_factors_from_model(neurons)

        # Scale the rates - multiple a scale factor with all the firing rates of a neuron
        scaled_rates = np.multiply(scale_factors, rates_mat.T)
        scaled_rates = scaled_rates.T

        # Calculate average neuron selectivities and population sparseness
        s_avg_selectivities_arr[r_idx], s_avg_sparseness_arr[r_idx] = \
            get_average_selectivity_and_sparseness(scaled_rates)

    plt.figure()
    plt.plot(np.arange(n_runs), f_avg_selectivity_arr, label="Neuron Selectivity")
    plt.plot(np.arange(n_runs), f_avg_sparseness_arr, label="Population Sparseness")

    plt.plot(np.arange(n_runs), s_avg_selectivities_arr, label="Scaled Neuron Selectivity")
    plt.plot(np.arange(n_runs), s_avg_sparseness_arr, label="Scaled Population Sparseness")

    plt.legend()
    plt.xlabel("Run")

    print("Full: average neuron selectivity %0.4f, population sparseness %0.4f"
          % (np.mean(f_avg_selectivity_arr), np.mean(f_avg_sparseness_arr)))

    print("Scaled: average neuron selectivity %0.4f, population sparseness %0.4f"
          % (np.mean(s_avg_selectivities_arr), np.mean(s_avg_sparseness_arr)))

    print("Average scale parameters a=%0.4f, b=%0.4f"
          % (np.mean(scale_a_arr), np.mean(scale_b_arr)))
