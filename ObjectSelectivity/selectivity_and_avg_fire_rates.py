# -*- coding: utf-8 -*-
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
    runs = 100

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

    print("Average correlation between selectivity and mean_rates %0.4f+%0.4f"
          % (np.mean(correlation_arr), np.std(correlation_arr)))

    # Plot neuronal selectivities versus average fire rate for last run
    plot_selectivity_vs_mean_rates(mean_rates, selectivities)

    # # -----------------------------------------------------------------------------------------
    # # Using Model Initialization
    # # -----------------------------------------------------------------------------------------
    # correlation_arr = np.zeros(shape=runs)
    #
    # for r_idx in np.arange(runs):
    #
    #     print("Run %d" % r_idx)
    #
    #     # Create Population ------------------------------------------------------------------
    #     selectivities = np.zeros(shape=neurons)
    #     mean_rates = np.zeros(shape=neurons)
    #
    #     for n_idx in np.arange(neurons):
    #
    #         # print ("Creating neuron %i" % idx_n)
    #         neuron = it.Neuron(
    #             list_of_objects,
    #             selectivity_profile='Kurtosis',
    #             # position_profile='Gaussian',
    #             # size_profile='Lognormal',
    #             # rotation_profile='Gaussian',
    #             # dynamic_profile='Tamura',
    #             # occlusion_profile='TwoInputSigmoid'
    #         )
    #
    #         selectivities[n_idx] = neuron.selectivity.kurtosis_measured
    #
    #         rates = np.array(neuron.selectivity.objects.values())
    #         mean_rates[n_idx] = np.mean(rates) * neuron.max_fire_rate
    #
    #     correlation_arr[r_idx] = np.corrcoef(mean_rates, np.log(selectivities + 1))[0, 1]
    #
    # print("Average correlation between selectivity and mean_rates %0.4f+%0.4f"
    #       % (np.mean(correlation_arr), np.std(correlation_arr)))
    #
    # # Plot neuronal selectivities versus average fire rate for last run
    # plot_selectivity_vs_mean_rates(mean_rates, selectivities)
