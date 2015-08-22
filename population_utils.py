# -*- coding: utf-8 -*-
""" --------------------------------------------------------------------------------------------
This file defines functions that can be run on populations of neurons created from neuron in
it_neuron_vrep.py

Created on Mon Aug  22 17:30:01 2015

@author: s362khan
----------------------------------------------------------------------------------------------"""
import numpy as np


def population_max_firing_rate(pop):
    """
    Return the highest firing rate of any neuron in the population

    :param pop: list of neurons
    :return:  max firing rate of Population
    """
    rates = [neuron.max_fire_rate for neuron in pop]

    return np.max(rates)
