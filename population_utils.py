# -*- coding: utf-8 -*-
""" --------------------------------------------------------------------------------------------
This file defines functions that can be run on populations of neurons created from neuron in
it_neuron_vrep.py

Created on Mon Aug  22 17:30:01 2015

@author: s362khan
----------------------------------------------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt


def population_max_firing_rate(it_population):
    """
    Return the highest firing rate of any neuron in the population

    :param it_population: list of neurons
    :return:  max firing rate of Population
    """
    rates = [n.max_fire_rate for n in it_population]

    return np.max(rates)


def plot_population_selectivity_distribution(it_population, axis=None):
    """Plot selectivity distribution of the population """
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


def plot_population_obj_preferences(it_population, axis=None):
    """ Plot Selectivity Profiles of entire Population """
    if axis is None:
        f, axis = plt.subplots()

    for n in it_population:
        lst = n.selectivity.get_ranked_object_list()
        objs, rate = zip(*lst)
        x = np.arange(len(rate))
        axis.plot(x, rate)

    axis.set_ylabel('Normalized Firing Rate')
    axis.set_xlabel('Ranked Object Preferences')
    axis.set_title('Population Object Preferences')


def plot_selectivity_vs_position_tolerance(it_population, axis=None):
    """  Plot population selectivity, activity fraction, verses  position tolerance """

    if axis is None:
        f, axis = plt.subplots()

    position_tolerance = [neuron.position.position_tolerance for neuron in it_population]
    position_tolerance = np.array(position_tolerance)

    selectivity = [neuron.selectivity.activity_fraction_measured for neuron in it_population]
    selectivity = np.array(selectivity)

    axis.scatter(selectivity, position_tolerance, marker='+', s=60, color='green',
                 label='Generated Data')

    # Plot the linear regression fit as well.
    m, b = np.polyfit(selectivity, position_tolerance, 1)

    plt.plot(selectivity, m*selectivity + b, ':', color='green')

    f_size = 34
    axis.set_ylabel('Position Tolerance (Radians)', fontsize=f_size)
    axis.set_xlabel('Selectivity (Activity Fraction)', fontsize=f_size)
    axis.set_title('Position Tolerance vs Selectivity',fontsize=f_size+10)
    axis.tick_params(axis='x', labelsize=f_size)
    axis.tick_params(axis='y', labelsize=f_size)

    plt.legend(loc='best', fontsize=f_size)