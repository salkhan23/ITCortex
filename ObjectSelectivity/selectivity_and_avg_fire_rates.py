# -*- coding: utf-8 -*-
""" --------------------------------------------------------------------------------------------
Find the correlation between mean firing rate and selectivity (Kurtosis).

[Lehky, 2011] found that that the mean firing rate of neurons decreased  as their selectivity
increased. Here fnd the correlation between the mean firing rates of neurons and their
selectivities and show that the same trend is preserved even with our modified firing rate
distribution of model neurons.

A plot of selectivity and mean firing rates is also generated. (Figure 5 in the paper)
-------------------------------------------------------------------------------------------- """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import kurtosis_selectivity_profile as ksp

import sys
sys.path.append("..")

import it_neuron_vrep as it
# Force reload (compile) IT cortex modules to pick changes not included in cached version.
reload(it)
reload(ksp)


def plot_selectivity_vs_mean_rates(selectivity, avg_rates, font_size=50):

    f, axis = plt.subplots()
    axis.scatter(avg_rates, selectivity + 1, color='g', s=60)
    axis.set_yscale('log')
    axis.set_xscale('log')

    axis.set_xlabel("Average Fire Rate (Spikes/s)", fontsize=font_size)
    axis.set_ylabel(r"$log(\sigma_{KI} + 1)$", fontsize=font_size)
    # axis.grid()
    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)
    axis.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
    axis.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))

    axis.annotate(
        'r=%0.4f' % np.corrcoef(avg_rates, selectivity)[0, 1],
        xy=(0.90, 0.95),
        xycoords='axes fraction',
        fontsize=font_size,
        horizontalalignment='right',
        verticalalignment='top'
    )


if __name__ == '__main__':
    plt.ion()

    neurons = 674
    objects = 806
    runs = 1000

    # Stimulus Set
    list_of_objects = []

    for idx_n in np.arange(objects):
        list_of_objects.append('random_' + str(idx_n))

    # -----------------------------------------------------------------------------------------
    # Using Selectivity Profile Directly
    # -----------------------------------------------------------------------------------------
    correlation_arr = np.zeros(shape=runs)

    for r_idx in np.arange(runs):

        print("Run %d" % r_idx)

        # Initialization
        selectivities = np.zeros(shape=neurons)
        mean_rates = np.zeros(shape=neurons)

        for n_idx in np.arange(neurons):
            profile = ksp.KurtosisSparseness(list_of_objects)

            selectivities[n_idx] = profile.kurtosis_measured

            rates = np.array(profile.objects.values())
            mean_rates[n_idx] = np.mean(rates) * profile.get_max_firing_rate()

        correlation_arr[r_idx] = np.corrcoef(mean_rates, selectivities)[0, 1]

        # Plot neuronal selectivities versus average fire rate for last run
        if r_idx == (runs - 1):
            plot_selectivity_vs_mean_rates(mean_rates, selectivities)

    print("Average correlation between selectivity and mean_rates %0.4f+%0.4f"
          % (np.mean(correlation_arr), np.std(correlation_arr)))
