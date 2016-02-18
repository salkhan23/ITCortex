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


def plot_single_neuron_selectivities(it_population, axis=None):
    """ Plot histogram of single neuron selectivity (kurtosis) across population
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

    font_size = 34
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


def plot_population_sparseness(it_population, axis=None):
    """ Plot histogram of population sparseness (kurtosis) of each object in the population
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

    font_size = 34
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


def plot_selectivity_vs_position_tolerance(it_population, axis=None):
    """  Plot population selectivity, activity fraction, verses  position tolerance
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

    f_size = 34
    axis.set_ylabel('Position Tolerance (Radians)', fontsize=f_size)
    axis.set_xlabel('Selectivity (Activity Fraction)', fontsize=f_size)
    axis.set_title('Position Tolerance vs Selectivity', fontsize=f_size + 10)
    axis.tick_params(axis='x', labelsize=f_size)
    axis.tick_params(axis='y', labelsize=f_size)

    plt.legend(loc='best', fontsize=f_size)


def plot_selectivity_vs_mean_response(it_population, axis=None):

    if axis is None:
        f, axis = plt.subplots()

    mean_responses = []
    selectivities = []
    for neuron in it_population:
        mean_responses.append(
            np.mean(neuron.selectivity.objects.values()) * neuron.max_fire_rate * 10)
        selectivities.append(neuron.selectivity.kurtosis_measured)

    mean_responses = np.array(mean_responses)
    selectivities = np.array(selectivities)

    axis.scatter(selectivities, mean_responses)


def plot_neuron_tuning_profiles(it_neuron, dt=0.005, net_fire_rates=None):

    f = plt.figure()
    font_size = 16

    ax1 = f.add_subplot(4, 2, 1)
    it_neuron.selectivity.plot_object_preferences(axis=ax1, font_size=font_size)

    ax2 = f.add_subplot(4, 2, 2)
    it_neuron.position.plot_position_tolerance_contours(axis=ax2, font_size=font_size)

    ax3 = f.add_subplot(4, 2, 3)
    it_neuron.size.plot_size_tolerance(axis=ax3, font_size=font_size)

    ax4 = f.add_subplot(4, 2, 4)
    it_neuron.rotation.plot_tuning_profile(axis=ax4, mirror_symmetric=True, font_size=font_size)

    ax5 = f.add_subplot(4, 2, 5, projection='3d')
    it_neuron.occlusion.plot_complete_profile(axis=ax5, font_size=font_size)

    ax4 = f.add_subplot(4, 2, 6)
    it_neuron.clutter.plot_clutter_profile(axis=ax4, font_size=font_size)

    ax7 = f.add_subplot(4, 2, 7)
    it.plot_neuron_dynamic_profile(it_neuron, axis=ax7, font_size=font_size)

    if net_fire_rates is not None:
        ax8 = f.add_subplot(4, 2, 8)
        ax8.plot(np.arange(0, net_fire_rates.shape[0] * dt, step=dt), net_fire_rates)
        ax8.set_xlabel("time(s)", fontsize=font_size)
        ax8.set_ylabel("FR (Spikes/s)", fontsize=font_size)
        ax8.tick_params(axis='x', labelsize=font_size)
        ax8.tick_params(axis='y', labelsize=font_size)

    return f
