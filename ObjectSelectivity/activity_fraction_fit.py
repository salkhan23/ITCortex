# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:44:45 2015

@author: s362khan
"""

import numpy as np


def get_activity_fraction_sparseness(n=1):
    """
    Generate a population level distribution of object selectivity.

    REF: Zoccolan et. al. 2007 - Trade-Off between Object Selectivity and Tolerance in Monkey
    Inferotemporal Cortex.

    The selectivity (sparseness) of each neuron was quantified by the activity fraction of its
    response (Rolls and Tovee, 1995a; Vinje and Gallant, 2000; Olshausen and Field, 2004):

        S =  {1 - [(sum(Ri/n))^2 / sum(Ri^2/n)] / [1 - (1/n)]

    where Ri is the neuron response to the ith stimulus and n is the number of stimuli in the set.
    S ranges from 0 (no object selectivity) to 1 (maximal object selectivity).

    Broad range of sparseness was observed. Even though supplementary figure 5 gives a
    scatter plot of selectivity for a population, No curve fitting was done. From Figure 2 A
    selectivity as defined above is uniformly distributed over [0, 1] and we choose to model
    this as a simple uniform distribution.

    :param n: size of population. Default=1.
    :rtype : sparseness (activity fraction) metrics of size n.
    """
    return np.random.uniform(size=n)
