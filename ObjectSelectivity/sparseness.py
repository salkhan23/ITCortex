# -*- coding: utf-8 -*-
""" --------------------------------------------------------------------------------------------
In Lehky 2011, stimuli were presented at fixed position and sizes for the IT neurons. As a result
the parameters that were used to generate mean firing rate distribution for various objects was
non-ideal. Here we use our position and size tuning profiles to generate scale factors that model
the amount of distortion expected under the testing conditions. This is used to modify the
parameters of the rate generating distribution of model IT neurons.

REF: Lehky, S. R., Kiani, R., Esteky, H., & Tanaka, K. (2011).
Statistics of visual responses in primate inferotemporal cortex to object stimuli.
Journal of Neurophysiology, 106(3), 1097–117.

Random samples of scale factors that would be expected in the Lehky et al. experiment.
The number of samples isn't critical as they are just used to estimate a distribution.
Samples should lie between [0, 1].

POSITION:
All stimuli were presented foveally at (0, 0) radians. However the receptive field centers
of IT neurons and their preferred positions are distributed around the fovea. To account for this
we create a large sample of position profiles and determine the amount of deviation expected if
stimuli are presented at (0, 0) to all neurons.

SIZE:
The largest dimension of all stimuli extended 7 degrees. Similarly to account for this non-optimal
size we generate a large sample of size tuning profiles of IT neurons and determine the amount of
deviation expected if stimuli are presented with a size of 7 degrees.

@author: bptripp, s362khan
-------------------------------------------------------------------------------------------- """
__author__ = 'bptripp'


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from kurtosis_selectivity_profile import calculate_kurtosis

import sys
sys.path.append("..")

from PositionTolerance import gaussian_position_profile as gpp
from SizeTolerance import log_normal_size_profile as lnsp
import it_neuron_vrep as it

# Force reload (compile) IT cortex modules to pick changes not included in cached version.
reload(gpp)
reload(lnsp)
reload(it)


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

    return np.nanmean(selectivities), np.mean(sparsenesses)


def plot_selectivity_and_sparseness(r_mat, font_size=50):

    f, ax_arr = plt.subplots(2, 1, sharex=True)

    # Single Neuron selectivities
    n_neurons = r_mat.shape[0]
    n_objs = r_mat.shape[1]

    selectivities = np.zeros(n_neurons)
    sparsenesses = np.zeros(n_objs)

    for n_idx in np.arange(n_neurons):
        rates = r_mat[n_idx, :]
        selectivities[n_idx] = calculate_kurtosis(rates)

    for o_idx in np.arange(n_objs):
        rates = r_mat[:, o_idx]
        sparsenesses[o_idx] = calculate_kurtosis(rates)

    # Plot selectivities ------------------------------------------------
    ax_arr[0].hist(selectivities, bins=np.arange(0, 100, step=1))

    ax_arr[0].set_ylabel('Frequency', fontsize=font_size)
    # ax_arr[0].set_xlabel('Kurtosis', fontsize=font_size)
    ax_arr[0].tick_params(axis='x', labelsize=font_size)
    ax_arr[0].tick_params(axis='y', labelsize=font_size)
    # ax_arr[0].set_xlim([0, 80])

    ax_arr[0].annotate('Mean=%0.2f' % np.mean(selectivities),
                       xy=(0.95, 0.9),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')

    ax_arr[0].annotate('N=%d' % len(selectivities),
                       xy=(0.95, 0.75),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')

    ax_arr[0].annotate('Single Neuron Selectivity',
                       xy=(0.7, 0.95),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')

    # Plot sparsenesses ------------------------------------------------
    ax_arr[1].hist(sparsenesses, bins=np.arange(0, 100, step=1))

    ax_arr[1].set_ylabel('Frequency', fontsize=font_size)
    ax_arr[1].set_xlabel('Kurtosis', fontsize=font_size)
    ax_arr[1].tick_params(axis='x', labelsize=font_size)
    ax_arr[1].tick_params(axis='y', labelsize=font_size)

    ax_arr[1].annotate('Mean=%0.2f' % np.mean(sparsenesses),
                       xy=(0.95, 0.9),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')

    ax_arr[1].annotate('N=%d' % len(sparsenesses),
                       xy=(0.95, 0.75),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')

    ax_arr[1].annotate('Population Sparseness',
                       xy=(0.7, 0.95),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')

    ax_arr[1].set_xlim([0, 70])


def get_scale_factors_from_model(n_samples):
    """
    Get scale parameters from gaussian position and size profiles

    :param n_samples: "
    :return:

    """
    factors = np.zeros(shape=n_samples)

    for ii in np.arange(n_samples):

        # The gaussian position profile expects activity fraction as input. This is used to
        # determine it position tolerance.
        #
        # [Zoccolan et al. 2007] found a wide range of activity fraction selectivity
        # (mean +-SD = 0.4+-0.22). and activity fraction selectivity ranged between 0.05 and 0.95.
        # Figure 5b plots a histogram of the observed selectivity. The histogram is essentially 
        # flat(similar frequencies within the range 0.1 and 0.7). Values above/below this range 
        # rare. We use a uniform distribution over this high frequency range to model
        #  activity fraction selectivity

        position = gpp.GaussianPositionProfile(np.random.uniform(0.1, 0.7))
        size = lnsp.LogNormalSizeProfile(position.position_tolerance)

        # Both
        factors[ii] = position.firing_rate_modifier(0, 0) * \
            size.firing_rate_modifier(7 * np.pi / 180.0)

        # Position Only
        # factors[ii] = position.firing_rate_modifier(0, 0)

        # Size only
        # factors[ii] = size.firing_rate_modifier(7 * np.pi / 180.0)

    # plt.hist(factors)
    # print("Factors average %0.4f, var %0.4f" % (np.mean(factors), np.var(factors)))
    # raw_input("Continue?")

    return factors


def get_lehky_sample_rates(n_neurons, n_objs):
    """
    Same as  LehkySparseness.get_sample_profile_and_rates_mat but using original Lehky variables

    :param n_neurons:
    :param n_objs:
    :return:
    """

    ala = 4  # shape parameter (a) of Lehky (l) for PDF of shape parameters (a) for rate PDFs
    alb = 2
    bla = 0.5
    blb = 0.5

    # Create sample profiles
    a_arr = np.random.gamma(ala, scale=bla, size=n_neurons)
    b_arr = np.random.gamma(alb, scale=blb, size=n_neurons)

    r_mat = np.zeros(shape=(n_neurons, n_objs))

    # Create sample rates to all objects for each neuron
    for n_idx in np.arange(n_neurons):
        r_mat[n_idx, :] = \
            gamma.rvs(a=a_arr[n_idx], loc=0, scale=b_arr[n_idx], size=n_objs)

    return r_mat


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
        mu_s = np.mean(scale_samples)
        var_s = np.var(scale_samples)

        # expectation and variance of Lehky et al. distribution of mean rates ...
        mu_l = ala*bla*alb*blb
        var_l = ala*bla**2*alb*blb**2 + ala*bla**2*(alb*blb)**2 + alb*blb**2*(ala*bla)**2

        # expectation and variance of "full" (unscaled) distribution of mean rates
        #   that will approximate Lehky after scaling ...
        mu_f = mu_l / mu_s
        var_f = (var_l - var_s*(mu_l/mu_s)**2) / (var_s + mu_s**2)

        # Prevent negative scale parameters and large values by preventing
        # bfb calculations from going negative or too small
        # ------------------------------------------------------------------------------
        epsilon = 0.1
        var_s_max = (var_l - (mu_f**2/ala + epsilon)*mu_s**2) / \
            (mu_f**2/ala + epsilon + (mu_l/mu_s)**2)

        if var_s > var_s_max:
            print('********FIXING*********')
            print("V scaled %0.4f, Solution %0.4f" % (var_s, var_s_max))
            var_s = var_s_max

            # Recalculate the full variance again as this is the parameter that changes
            # with the new var_s in the calculation of bfb and bfa
            var_f = (var_l - var_s*(mu_l/mu_s)**2) / (var_s + mu_s**2)
        # ----------------------------------------------------------------------------

        # # Debug Prints
        # print("sample mean %0.4f, var %0.4f" %(mu_s, var_s))
        # print("Lehky mean %0.4f, var %0.4f" %(mu_l, var_l))
        # print("Full mean %0.4f, var %0.4f" %(mu_f, var_f))

        # shape and scale parameters for full distribution (keeping shape same as scaled one) ...
        self._afa = ala
        self._bfa = bla
        self._bfb = (var_f - mu_f**2/ala) / (mu_f*bla*(1+ala))
        self._afb = mu_f / (self._afa*self._bfa*self._bfb)

        # print("Scale distribution Gamma parameters: a=%0.4f, b=%0.4f" %(self._afb, self._bfb))

    def sample_rates_rates(self, n):
        """
        :param n: Number of maximum spike rates needed
        :return: n random spike-rate samples
        """

        a = np.random.gamma(self._afa, scale=self._bfa, size=n)  # Samples of shape parameter
        b = np.random.gamma(self._afb, scale=self._bfb, size=n)  # Samples of scale parameter

        #a = np.maximum(1.01, a)  # Avoid making PDF go to infinity at zero spike rate

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

    # --------------------------------------------------------------
    #  Original Code
    # # --------------------------------------------------------------
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

    # # --------------------------------------------------------------
    # # Reproduce Lehky Results
    # # --------------------------------------------------------------
    # n_runs = 1000
    # neurons = 674
    # objects = 806
    #
    # rates_mat = get_lehky_sample_rates(neurons, objects)
    #
    # l_avg_selectivity_arr = np.zeros(n_runs)  # Lehky
    # l_avg_sparseness_arr = np.zeros(n_runs)
    #
    # for r_idx in np.arange(n_runs):
    #
    #     l_avg_selectivity_arr[r_idx], l_avg_sparseness_arr[r_idx] = \
    #         get_average_selectivity_and_sparseness(rates_mat)
    #
    # print("Lehky: average neuron selectivity %0.4f, population sparseness %0.4f"
    #       % (np.mean(l_avg_selectivity_arr), np.mean(l_avg_sparseness_arr)))

    # --------------------------------------------------------------
    # Find scale distribution parameter
    # --------------------------------------------------------------
    n_runs = 10   # change to multiple runs to get average scale parameters
    neurons = 674
    objects = 806

    f_avg_selectivity_arr = np.zeros(n_runs)
    f_avg_sparseness_arr = np.zeros(n_runs)

    s_avg_selectivity_arr = np.zeros(n_runs)
    s_avg_sparseness_arr = np.zeros(n_runs)

    scale_a_arr = np.zeros(n_runs)
    scale_b_arr = np.zeros(n_runs)

    for r_idx in np.arange(n_runs):

        # Scale Factors
        # scale_factor_samples = np.random.rand(10000)
        scale_factor_samples = get_scale_factors_from_model(5000)

        # ------------------------------------------------------------------------
        ls = LehkySparseness(scale_factor_samples)

        # Store modified scale values. These will be used in the larger model
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

        # Scale the rates - multiply a scale factor with all the firing rates of a neuron
        scaled_rates = np.multiply(scale_factors, rates_mat.T)
        scaled_rates = scaled_rates.T

        # Calculate average neuron selectivities and population sparseness
        s_avg_selectivity_arr[r_idx], s_avg_sparseness_arr[r_idx] = \
            get_average_selectivity_and_sparseness(scaled_rates)

        # Plot the selectivities and sparseness distributions of the last run
        if n_runs > 1:
            print ("Run %i" % r_idx)
        else:
            plot_selectivity_and_sparseness(scaled_rates)

    print("Full  : average neuronal selectivity %0.4f+%0.4f, population sparseness %0.4f+0.4%f"
          % (np.mean(f_avg_selectivity_arr), np.std(f_avg_selectivity_arr),
             np.mean(f_avg_sparseness_arr), np.std(f_avg_sparseness_arr)))

    print("Scaled: average neuronal selectivity %0.4f+%0.4f, population sparseness %0.4f+%0.4f"
          % (np.mean(s_avg_selectivity_arr), np.std(s_avg_selectivity_arr),
             np.mean(s_avg_sparseness_arr), np.std(s_avg_sparseness_arr)))

    print("Average scale parameters a=%0.4f+%0.4f, b=%0.4f+%0.4f"
          % (np.mean(scale_a_arr), np.std(scale_a_arr),
             np.mean(scale_b_arr), np.std(scale_b_arr)))

    # # -------------------------------------------------------------------------
    # # Validate larger model can reproduce selectivity and sparseness values
    # # -------------------------------------------------------------------------
    # neurons = 674
    # objects = 806
    # n_runs = 100
    #
    # # Stimulus Set
    # list_of_objects = []
    # for idx_n in np.arange(objects):
    #     list_of_objects.append('random_' + str(idx_n))
    #
    # f_avg_selectivity_arr = np.zeros(n_runs)
    # f_avg_sparseness_arr = np.zeros(n_runs)
    #
    # s_avg_selectivity_arr = np.zeros(n_runs)
    # s_avg_sparseness_arr = np.zeros(n_runs)
    #
    # for r_idx in np.arange(n_runs):
    #
    #     print ("Run %i" % r_idx)
    #
    #     # Create Population ------------------------------------------------------------------
    #     it_cortex = []
    #     for idx_n in np.arange(neurons):
    #
    #         # print ("Creating neuron %i" % idx_n)
    #         neuron = it.Neuron(
    #             list_of_objects,
    #             selectivity_profile='Kurtosis',
    #             position_profile='Gaussian',
    #             size_profile='Lognormal',
    #             # rotation_profile='Gaussian',
    #             # dynamic_profile='Tamura',
    #             # occlusion_profile='TwoInputSigmoid'
    #         )
    #
    #         it_cortex.append(neuron)
    #
    #     # Initialization
    #     scaled_fire_rates = np.zeros(shape=(neurons, objects))
    #     full_fire_rates = np.zeros(shape=(neurons, objects))
    #
    #     # Calculate full selectivity and sparseness ----------------------------------------
    #     for idx_n in np.arange(neurons):
    #         full_fire_rates[idx_n, :] = \
    #             np.array(it_cortex[idx_n].selectivity.objects.values()) * \
    #             it_cortex[idx_n].max_fire_rate
    #
    #     f_avg_selectivity_arr[r_idx], f_avg_sparseness_arr[r_idx] = \
    #         get_average_selectivity_and_sparseness(full_fire_rates)
    #
    #     # Get scaled selectivity and sparseness --------------------------------------------
    #     for idx_o in np.arange(objects):
    #
    #         ground_truth = [
    #             list_of_objects[idx_o],
    #             0, 0,  # presented foveally
    #             7 * np.pi / 180,  # spanning 7 degrees
    #             1, 1, 1,
    #             1, 1, 1,
    #             1, 1, 1,
    #             1, 1
    #         ]
    #
    #         for idx_n in np.arange(neurons):
    #             scaled_fire_rates[idx_n, idx_o] = it_cortex[idx_n].firing_rate([ground_truth])
    #
    #     # Calculate average neuron selectivities and population sparseness
    #     s_avg_selectivity_arr[r_idx], s_avg_sparseness_arr[r_idx] = \
    #         get_average_selectivity_and_sparseness(scaled_fire_rates)
    #
    # print("Full: average neuron selectivity %0.4f, and population sparseness %0.4f"
    #       % (np.mean(f_avg_selectivity_arr), np.mean(f_avg_sparseness_arr)))
    #
    # print("Scaled: average neuron selectivity %0.4f, population sparseness %0.4f"
    #       % (np.mean(s_avg_selectivity_arr), np.mean(s_avg_sparseness_arr)))
    #
    # # Plot the selectivities and sparseness distributions of the last run
    # plot_selectivity_and_sparseness(scaled_fire_rates)
