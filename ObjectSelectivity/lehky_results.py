# -*- coding: utf-8 -*-
"""
Reproduce sparseness mean values from Lehky paper.

REF:  Lehky, S. R., Kiani, R., Esteky, H., & Tanaka, K. (2011). Statistics of
            visual responses in primate inferotemporal cortex to object stimuli.
            Journal of Neurophysiology, 106(3), 1097–117.

@author: s362khan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from kurtosis_selectivity_profile import calculate_kurtosis


def get_average_selectivity_and_sparseness(ala, alb, bla, blb, n_neurons, n_objs):
    # Create sample Gamma profiles using these parameters
    a_arr = np.random.gamma(ala, scale=bla, size=n_neurons)  # samples of shape parameter
    b_arr = np.random.gamma(alb, scale=blb, size=n_neurons)  # samples of scale parameter

    rates_mat = np.zeros(shape=(n_neurons, n_objs))
    r_max_arr = np.zeros(shape=n_neurons)

    # Generate Random Firing Rates
    for profile_idx in np.arange(a_arr.shape[0]):
        rates_mat[profile_idx, :] = \
            ss.gamma.rvs(a=a_arr[profile_idx], loc=0, scale=b_arr[profile_idx], size=n_objs)

        r_max_arr[profile_idx] = \
            ss.gamma.ppf(0.99, a=a_arr[profile_idx], loc=0, scale=b_arr[profile_idx])

        # Normalize the firing rates as done in the IT cortex model
        # This step is not necessary (raw rates can be used), but this is what is done
        # in the IT cortex model.
        rates_mat[profile_idx, :] = rates_mat[profile_idx, :] / r_max_arr[profile_idx]

    selectivities = np.zeros(shape=n_neurons)
    sparsenesses = np.zeros(shape=n_objs)

    # Multiply the normalized rates with the max firing rate before calculating kurtosis.
    # The impact this should have (according to Lehky) is population sparseness increases,
    # but individual neuron selectivity should stay the same.
    raw_rates = np.multiply(r_max_arr, rates_mat.T)
    raw_rates = raw_rates.T  # just a column multiply

    # Single neuron selectivities
    for n_idx in np.arange(n_neurons):
        rates = raw_rates[n_idx, :]
        selectivities[n_idx] = calculate_kurtosis(rates)

    # Population sparseness
    for o_idx in np.arange(n_objs):
        rates = raw_rates[:, o_idx]
        sparsenesses[o_idx] = calculate_kurtosis(rates)

    return np.mean(selectivities), np.mean(sparsenesses),


if __name__ == "__main__":
    plt.ion()

    n_runs = 100

    n_selectivities_arr = np.zeros(shape=n_runs)
    p_sparseness_arr = np.zeros(shape=n_runs)

    # Validate Lehky mean neuron selectivity and population sparseness
    for r_idx in np.arange(n_runs):
        n_selectivities_arr[r_idx], p_sparseness_arr[r_idx] = \
            get_average_selectivity_and_sparseness(
                ala=4,
                alb=2,
                bla=0.5,
                blb=0.5,
                n_neurons=674,
                n_objs=806)

    print("Lehky Avg neuron selectivity %0.4f, Avg population_sparseness %0.4f"
          % (np.mean(n_selectivities_arr), np.mean(p_sparseness_arr)))

    plt.figure()
    plt.title("Lehky Parameters")
    plt.plot(np.arange(n_runs), n_selectivities_arr, label='Average neuron selectivities')
    plt.plot(np.arange(n_runs), p_sparseness_arr, label='Average population sparseness')
    plt.xlabel("run")
