# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:44:45 2015

Generate a population level distribition of Neuronal selectivity

REF: Zoccolan et. al. 2007 - Trade-Off between Object Selectivity and Tolerence in Monkey
     Inferotemporal Cortex

The selectivity of each neuron was quantified by the sparseness of its response
(Rolls and Tovee, 1995a; Vinje and Gallant, 2000; Olshausen and Field, 2004):
    S =  {1 - [(sum(Ri/n))^2 / sum(Ri^2/n)] / [1 - (1/n)]

where Ri is the neuron response to the ith stimulus and n is the number of stimuli in the set.
S ranges from 0 (no object selectivity) to 1 (maximal object selectivity).

Broad Spectrum of Sensitivity was observed. Even though supplmentary figure 5 gives a scatter plot
of selectivity for a population,

 nocurve fitting was observed. From Figure 2 A selectivity as defined above seams
uniformly distributed over [0, 1] and we choose to model this as a simple uniform distribution.

@author: s362khan
"""

import numpy as np


def GenerateSelectivityDistribution(n=1):
    return (np.random.uniform(size=n))
