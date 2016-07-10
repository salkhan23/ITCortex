# -*- coding: utf-8 -*-
""" --------------------------------------------------------------------------------------------
This file defines functions that can be run on populations of neurons created from neuron in
it_neuron_vrep.py

Created on Mon Aug  22 17:30:01 2015

@author: s362khan
----------------------------------------------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from mpl_toolkits.mplot3d import proj3d
import it_neuron_vrep as it


# Do relative import of the main folder to get files in sibling directories
top_level_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if top_level_dir_path not in sys.path:
    sys.path.append(top_level_dir_path)

from ObjectSelectivity.kurtosis_selectivity_profile import calculate_kurtosis


def population_max_firing_rate(it_population):
    """
    Return the highest firing rate of any neuron in the population

    :param it_population: list of neurons
    :return:  max firing rate of Population
    """
    rates = [n.max_fire_rate for n in it_population]

    return np.max(rates)


def plot_max_fire_distribution(it_population, axis=None):
    rates = [n.max_fire_rate for n in it_population]

    if axis is None:
        f, axis = plt.subplots()

    axis.plot(np.arange(len(it_population)), rates)


def plot_population_selectivity_distribution(it_population, axis=None):
    """Plot selectivity distribution (activity fractions) of the population
    :param axis:
    :param it_population:
    """
    if axis is None:
        f, axis = plt.subplots()

    selectivity_arr_abs = [n.selectivity.activity_fraction_absolute for n in it_population]
    selectivity_arr_meas = [n.selectivity.activity_fraction_measured for n in it_population]

    axis.hist(selectivity_arr_abs, bins=np.arange(1, step=0.05), label='absolute')
    axis.hist(selectivity_arr_meas, bins=np.arange(1, step=0.05), label='measured')
    axis.set_ylabel('Frequency')
    axis.set_xlabel('Selectivity (Activity Fraction)')
    axis.set_title('Population Selectivity Distribution')
    axis.legend()


def plot_single_neuron_selectivities(it_population, axis=None,  font_size=40):
    """ Plot histogram of single neuron selectivity (kurtosis) across population
    :param font_size:
    :param axis:
    :param it_population:
    """

    if axis is None:
        f, axis = plt.subplots()

    objects = it_population[0].selectivity.objects.keys()

    single_neuron_selectivity = []
    for neuron in it_population:

        firing_rates = []

        for obj in objects:
            firing_rates.append(neuron.selectivity.objects[obj] * neuron.max_fire_rate)

        single_neuron_selectivity.append(calculate_kurtosis(np.array(firing_rates)))

    axis.hist(single_neuron_selectivity, bins=np.arange(0, 100, step=1))

    axis.set_ylabel('Frequency', fontsize=font_size)
    # axis.set_xlabel('Kurtosis', fontsize=font_size)
    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)

    axis.annotate('Mean=%0.2f' % np.mean(single_neuron_selectivity),
                  xy=(0.95, 0.9),
                  xycoords='axes fraction',
                  fontsize=font_size,
                  horizontalalignment='right',
                  verticalalignment='top')

    axis.annotate('N=%d' % len(single_neuron_selectivity),
                  xy=(0.95, 0.82),
                  xycoords='axes fraction',
                  fontsize=font_size,
                  horizontalalignment='right',
                  verticalalignment='top')

    axis.annotate('Single Neuron Selectivity',
                  xy=(0.5, 0.95),
                  xycoords='axes fraction',
                  fontsize=font_size,
                  horizontalalignment='right',
                  verticalalignment='top')

    axis.set_xlim([0, 80])


def plot_population_sparseness(it_population, axis=None, font_size=40):
    """ Plot histogram of population sparseness (kurtosis) of each object in the population
    :param font_size:
    :param axis:
    :param it_population:
    """

    if axis is None:
        f, axis = plt.subplots()

    objects = it_population[0].selectivity.objects.keys()

    population_sparseness = []

    for obj in objects:

        firing_rates = []
        for neuron in it_population:
            firing_rates.append(neuron.selectivity.objects[obj] * neuron.max_fire_rate)

        population_sparseness.append(calculate_kurtosis(np.array(firing_rates)))

    axis.hist(population_sparseness, bins=np.arange(0, 100, step=1))

    axis.set_ylabel('Frequency', fontsize=font_size)
    axis.set_xlabel('Kurtosis', fontsize=font_size)
    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)

    axis.annotate('Mean=%0.2f' % np.mean(population_sparseness),
                  xy=(0.95, 0.9),
                  xycoords='axes fraction',
                  fontsize=font_size,
                  horizontalalignment='right',
                  verticalalignment='top')

    axis.annotate('N=%d' % len(population_sparseness),
                  xy=(0.95, 0.82),
                  xycoords='axes fraction',
                  fontsize=font_size,
                  horizontalalignment='right',
                  verticalalignment='top')

    axis.annotate('Population Sparseness',
                  xy=(0.5, 0.95),
                  xycoords='axes fraction',
                  fontsize=font_size,
                  horizontalalignment='right',
                  verticalalignment='top')

    axis.set_xlim([0, 80])


def plot_population_obj_preferences(it_population, axis=None):
    """ Plot Selectivity Profiles of entire Population
    :param axis:
    :param it_population:
    """
    if axis is None:
        f, axis = plt.subplots()

    font_size = 34

    for n in it_population:
        lst = n.selectivity.get_ranked_object_list()
        objs, rate = zip(*lst)
        x = np.arange(len(rate))
        axis.plot(x, rate)

    axis.set_ylabel('Normalized Firing Rate (spikes/s)', fontsize=font_size)
    axis.set_xlabel('Ranked Objects', fontsize=font_size)
    axis.set_title('Population Object Preferences', fontsize=font_size + 10)

    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)

    axis.set_ylim([0, 1])
    axis.grid()

    axis.annotate('N=%d' % len(it_population),
                  xy=(0.95, 0.9),
                  xycoords='axes fraction',
                  fontsize=font_size,
                  horizontalalignment='right',
                  verticalalignment='top')


def plot_selectivity_vs_position_tolerance(it_population, axis=None, f_size=40):
    """  Plot population selectivity, activity fraction, verses  position tolerance
    :param f_size:
    :param axis:
    :param it_population:
    """

    if axis is None:
        f, axis = plt.subplots()

    position_tolerance = [neuron.position.position_tolerance for neuron in it_population]
    position_tolerance = np.array(position_tolerance)

    selectivity = [neuron.selectivity.activity_fraction_measured for neuron in it_population]
    selectivity = np.array(selectivity)

    axis.scatter(selectivity, position_tolerance, marker='o', s=60, color='green',
                 label='Generated Data')

    # Plot the linear regression fit as well.
    fit = np.polyfit(selectivity, position_tolerance, 1)

    plt.plot(selectivity, fit[0] * selectivity + fit[1], ':', color='green')

    axis.set_ylabel('Position Tolerance (Radians)', fontsize=f_size)
    axis.set_xlabel('Selectivity (Activity Fraction)', fontsize=f_size)
    # axis.set_title('Position Tolerance vs Selectivity', fontsize=f_size + 10)
    axis.tick_params(axis='x', labelsize=f_size)
    axis.tick_params(axis='y', labelsize=f_size)

    plt.legend(loc='best', fontsize=f_size)

    # axis.annotate(
    #     'r=%0.4f' % np.corrcoef(selectivity, position_tolerance)[0, 1],
    #     xy=(0.90, 0.85),
    #     xycoords='axes fraction',
    #     fontsize=f_size,
    #     horizontalalignment='right',
    #     verticalalignment='top')


def plot_selectivity_vs_mean_response(it_population, axis=None, font_size=40):

    if axis is None:
        f, axis = plt.subplots()

    # selectivities_abs = []
    selectivities_meas = []
    mean_rates = []

    population_size = len(it_population)

    for neuron_idx in np.arange(population_size):

        obj_pref = it_population[neuron_idx].selectivity.objects.values()
        max_fire = it_population[neuron_idx].max_fire_rate

        mean_rates.append(np.mean(obj_pref) * max_fire)
        selectivities_meas.append(it_population[neuron_idx].selectivity.kurtosis_measured)

    mean_rates = np.array(mean_rates)
    # selectivities_abs = np.array(selectivities_abs)
    selectivities_meas = np.array(selectivities_meas)

    axis.scatter(mean_rates, np.log(selectivities_meas + 1), color='g', s=60)
    # axis.loglog(mean_rates, selectivities_meas + 1, 'go')
    axis.set_xlabel("Average Fire Rate (Spikes/s)", fontsize=font_size)
    axis.set_ylabel(r"$log(\sigma_{KI} + 1)$", fontsize=font_size)
    # axis.grid()
    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)
    axis.set_xlim([0, np.max(mean_rates) * 1.1])
    axis.set_ylim([0, np.max(np.log(selectivities_meas + 1)) * 1.1])

    axis.annotate(
        'r=%0.4f' % np.corrcoef(mean_rates, np.log(selectivities_meas + 1))[0, 1],
        xy=(0.90, 0.95),
        xycoords='axes fraction',
        fontsize=font_size,
        horizontalalignment='right',
        verticalalignment='top')


def plot_neuron_tuning_profiles(it_neuron, dt=0.005, net_fire_rates=None, font_size=30):
    """
    Once the figure is generated:
        [1] Get tick labels for selectivity and abbreviate
        [2] Zoom in on the figure B.
        [3] Remove overlap of X,Y ticks in figure B and C.

    :param it_neuron:
    :param dt:
    :param net_fire_rates:
    :param font_size:
    :return:
    """

    f = plt.figure()

    ax1 = f.add_subplot(3, 2, 1)
    it_neuron.selectivity.plot_object_preferences(
        axis=ax1, font_size=font_size, print_parameters=False)
    ax1.grid()  # Turn the default grid off
    ax1.tick_params(axis='y', labelsize=font_size)
    ax1.set_xlabel('Ranked Objects', fontsize=font_size)
    ax1.set_ylabel('FR (Hz)', fontsize=font_size)
    ax1.xaxis.set_label_position('top')
    ax1.annotate(
        'A',
        xy=(0.10, 0.95),
        xycoords='axes fraction',
        fontsize=30,
        horizontalalignment='right',
        verticalalignment='top')
    labels = [item.get_text() for item in f.axes[0].get_xticklabels()]
    ax1.set_xticklabels(labels, rotation=45)

    ax2 = f.add_subplot(3, 2, 2)
    it_neuron.position.plot_position_tolerance_contours(
        axis=ax2, font_size=font_size, print_parameters=False)
    ax2.yaxis.set_label_position('right')
    ax2.xaxis.set_label_position('top')
    ax2.set_xlabel('X', fontsize=font_size)
    ax2.set_ylabel('Y', fontsize=font_size)
    ax2.annotate(
        'B',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=30,
        horizontalalignment='right',
        verticalalignment='top')
    # legend below figure
    ax2.legend(
        bbox_to_anchor=[0.5, -0.35], fontsize=font_size - 5, ncol=2, loc='center', scatterpoints=1)

    ax3 = f.add_subplot(3, 2, 3)
    it_neuron.size.plot_size_tolerance(axis=ax3, font_size=font_size, print_parameters=False)
    ax3.grid()  # Turn the default grid off
    ax3.set_ylabel('FR (Hz)', fontsize=font_size)
    ax3.annotate(
        'C',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=30,
        horizontalalignment='right',
        verticalalignment='top')

    ax4 = f.add_subplot(3, 2, 4)
    it_neuron.rotation.plot_tuning_profile(
        axis=ax4,
        rotation_symmetry_period=1,
        mirror_symmetric=False,
        font_size=font_size,
        print_parameters=False,
    )
    ax4.set_ylabel('FR (Hz)', fontsize=font_size)
    ax4.yaxis.set_label_position('right')
    ax4.annotate(
        'D',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=30,
        horizontalalignment='right',
        verticalalignment='top')

    ax5 = f.add_subplot(3, 2, 5, projection='3d')
    it_neuron.occlusion.plot_complete_profile(
        axis=ax5, font_size=font_size, print_parameters=False)
    ax5.set_zlabel('\nFR (Hz)', fontsize=font_size)
    ax5.grid()  # Turn the default grid off
    ax5.annotate(
        'E',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=30,
        horizontalalignment='right',
        verticalalignment='top')

    ax6 = f.add_subplot(3, 2, 6)
    it_neuron.clutter.plot_clutter_profile(axis=ax6, font_size=font_size, print_parameters=False)
    ax6.grid()
    ax6.yaxis.set_label_position('right')
    ax6.legend(bbox_to_anchor=[0.3, 0.95], fontsize=font_size - 5)

    ax6.annotate(
        'F',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=30,
        horizontalalignment='right',
        verticalalignment='top')
    ax6.set_xticks(np.arange(0.5, 2.1, step=0.5))
    ax6.set_xlabel("Sum Isolated Responses (Hz)")
    ax6.set_ylabel("Joint FR (Hz)")

    #
    # ax7 = f.add_subplot(4, 2, 7)
    # it.plot_neuron_dynamic_profile(it_neuron, axis=ax7, font_size=font_size)
    # ax7.grid()  # Turn the default grid off
    # ax7.set_ylabel('FR (Hz)', fontsize=font_size)
    # ax7.legend(loc='best')
    # ax7.annotate(
    #     'G',
    #     xy=(0.05, 0.95),
    #     xycoords='axes fraction',
    #     fontsize=30,
    #     horizontalalignment='right',
    #     verticalalignment='top')

    # if net_fire_rates is not None:
    #     ax8 = f.add_subplot(4, 2, 8)
    #     ax8.plot(np.arange(0, net_fire_rates.shape[0] * dt, step=dt), net_fire_rates)
    #     ax8.set_xlabel("Time(s)", fontsize=font_size)
    #     ax8.set_ylabel("FR (Hz)", fontsize=font_size)
    #     ax8.yaxis.set_label_position('right')
    #     ax8.tick_params(axis='x', labelsize=font_size)
    #     ax8.tick_params(axis='y', labelsize=font_size)
    #
    #     ax8.annotate(
    #         'H',
    #         xy=(0.05, 0.95),
    #         xycoords='axes fraction',
    #         fontsize=30,
    #         horizontalalignment='right',
    #         verticalalignment='top')

    return f


def plot_population_fire_rates(rates_array, dt=0.005, font_size=20):

    markers = ['+', '.', '*', '^', 'o', '8', 'd', 's']

    # Get maximum_fire_rate
    max_fire_rate = np.max(rates_array)

    population_size = rates_array.shape[1]

    quotient, remainder = divmod(population_size, len(markers))

    t_stop = rates_array.shape[0] * dt
    time = np.arange(0, t_stop, dt)

    n_subplots = quotient

    if 0 != remainder:
        n_subplots += 1

    fig_rates_vs_time, ax_array = plt.subplots(n_subplots, sharex=True)
    fig_rates_vs_time.subplots_adjust(hspace=0.0)

    for neuron_idx in np.arange(population_size):
        marker_idx = neuron_idx % len(markers)
        subplot_idx = neuron_idx / len(markers)

        ax_array[subplot_idx].plot(time, rates_array[:, neuron_idx],
                                   marker=markers[marker_idx], label='N%i' % neuron_idx)

        # Set the limits for all subplots
        for ax in ax_array:
                ax.legend(fontsize='5')
                ax.set_ylim(0, max_fire_rate)
                ax.yaxis.set_ticks(np.arange(20, max_fire_rate, step=20))

    # Set the limits for the last subplot
    ax_array[-1].set_xlabel("Time (s)", fontsize=font_size)
    ax_array[-1].tick_params(axis='x', labelsize=font_size)

    # Set the y label on the middle subplot
    ax_array[n_subplots / 2].set_ylabel("Fire Rates", fontsize=font_size)

    fig_rates_vs_time.suptitle("Population (N=%d) Firing Rates " % population_size,
                               fontsize=font_size + 10)


def plot_receptive_field_centers(it_population, axis=None, font_size=40):

    if axis is None:
        f, axis = plt.subplots()

    population_size = len(it_population)
    centers = []

    for neuron_idx in np.arange(population_size):
        centers.append(it_population[neuron_idx].position.rf_center)

    centers = np.array(centers)

    axis.scatter(centers[:, 0], centers[:, 1],
                 label='Generated Data', color='green', marker='o', s=60)

    # axis.scatter(0, 0, color='red', marker='+', linewidth=10, label='Gaze Center')

    axis.set_xlabel('Horizontal position (Radians)', fontsize=font_size)
    axis.set_ylabel('Vertical position (Radians)',  fontsize=font_size)

    plt.axvline(linewidth=1, color='k')
    plt.axhline(linewidth=1, color='k')

    plt.tick_params(axis='x', labelsize=font_size)
    plt.tick_params(axis='y', labelsize=font_size)

    plt.legend(loc='best', fontsize=font_size)

    plt.annotate("Ipsilateral",
                 xy=(0.15, 0.1),
                 xycoords='axes fraction',
                 fontsize=font_size,
                 horizontalalignment='right',
                 verticalalignment='top')

    plt.annotate("Contralateral",
                 xy=(0.95, 0.1),
                 xycoords='axes fraction',
                 fontsize=font_size,
                 horizontalalignment='right',
                 verticalalignment='top')


def plot_neuron_scales_factors(n_idx, neurons_arr, scales_arr, dt, t_stop, obj_idx=0,
                               axis=None, font_size=30):
    """
    Plot the scale factors of a specified neuron and object
    :param n_idx:
    :param neurons_arr:
    :param scales_arr:
    :param dt:
    :param t_stop:
    :param obj_idx:
    :param axis:
    :param font_size:
    :return:
    """
    if axis is None:
        f, axis = plt.subplots()

    # Extract the rates/ scale factors for the specified neuron
    neuron_scales_factors = []
    for scales_per_run in scales_arr:
        neuron_scales_factors.append(scales_per_run[n_idx])

    # From it_neuron_vrep.py
    # scales[:, 0] = isolated_rates
    # scales[:, 1] = obj_pref_list
    # scales[:, 2] = position_weights
    # scales[:, 3] = size_fr
    # scales[:, 4] = rot_fr
    # scales[:, 5] = occ_fr
    # scales[:, 6] = size_arr

    isolated_rates = []
    position_scales = []
    size_scales = []
    rot_scales = []
    occ_scales = []

    for scales_per_run in neuron_scales_factors:
        isolated_rates.append(scales_per_run[obj_idx][0])
        position_scales.append(scales_per_run[obj_idx][2])
        size_scales.append(scales_per_run[obj_idx][3])
        rot_scales.append(scales_per_run[obj_idx][4])
        occ_scales.append(scales_per_run[obj_idx][5])

    t_arr = np.arange(0, t_stop, step=dt)

    axis.plot(t_arr, np.array(isolated_rates),
              label=r'$r_{iso}/r_{obj}$', linewidth=3, color='black')

    axis.plot(t_arr, position_scales, label=r'$r_{pos}$', linewidth=2, color='blue')
    axis.plot(t_arr, size_scales, label=r'$r_{size}$', linewidth=2, color='green')
    axis.plot(t_arr, rot_scales, label=r'$r_{rot}$', linewidth=2, color='red', )
    axis.plot(t_arr, occ_scales, label=r'$r_{occ}$', linewidth=2, color='magenta')

    axis.legend(fontsize=font_size)
    axis.set_xlabel('Time(s)', fontsize=font_size)
    axis.set_xticks(np.arange(1, t_stop + 1))
    axis.set_ylabel("Fire Rate (spikes/s)", fontsize=font_size)

    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)

    neuron_object_list = neurons_arr[n_idx].selectivity.get_ranked_object_list()
    axis.set_title("Object %s, Rank=%d" % (neuron_object_list[obj_idx][0], obj_idx))
