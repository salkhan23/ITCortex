# -*- coding: utf-8 -*-
"""
Created on Fri September 11 9:26:45 2015

@author: s362khan
"""

import activity_fraction_fit as selectivity_fit
import random
import numpy as np
import matplotlib.pyplot as plt


def get_activity_fraction(rates_per_object):
    """
    Given an array of firing rates of the neuron to objects, return the sparseness metric
    activity fraction of the neuron as defined in

    [1] REF: Zoccolan et. al. 2007 - Trade-Off between Object Selectivity and Tolerance in Monkey
    Inferotemporal Cortex.

    [2] SI_R in REF:  Lehky, S. R., Kiani, R., Esteky, H., & Tanaka, K. (2011). Statistics of
    visual responses in primate inferotemporal cortex to object stimuli.
    Journal of Neurophysiology, 106(3), 1097–117.

        activity_fraction  =  {1 - [(sum(Ri/n))^2 / sum(Ri^2/n)] / [1 - (1/n)]

    Originally defined by Rolls & Tovee in Rolls ET, Tovée MJ. Sparseness of the neuronal
    representation of stimuli in the primate temporal visual cortex.
    J Neurophysiology 73: 713–726, 1995.

    :param rates_per_object: array of firing rates of the neuron to multiple objects.
    :return: activity fraction sparseness. Ranges from 0 (low selectivity) to 1(high selectivity).

    This is defined outside the class as it is used by other selectivity profiles.
    """
    n = rates_per_object.shape[0]

    rates_square = rates_per_object ** 2

    activity_fraction = n / (n - 1) * \
                        (1 - ((rates_per_object.sum() / n) ** 2 / (rates_square.sum() / n)))

    return activity_fraction


class PowerLawSparseness:
    def __init__(self, list_of_objects):
        """
        Models object selectivity as a power law over sparseness defined as the
        activity fraction.

        The selectivity (sparseness) of each neuron was quantified by the activity fraction of its
        response (Rolls and Tovee, 1995a; Vinje and Gallant, 2000; Olshausen and Field, 2004):

            S =  {1 - [(sum(Ri/n))^2 / sum(Ri^2/n)] / [1 - (1/n)]

        REF: Zoccolan et. al. - 2007 - Trade-Off between Object Selectivity and Tolerance in Monkey
        Inferotemporal Cortex.
        """

        self.type = 'power_law'

        # Absolute activity fraction if all rates for all objects the neuron responds
        # are included.
        self.activity_fraction_absolute = \
            np.float(selectivity_fit.get_activity_fraction_sparseness(1))

        random.shuffle(list_of_objects)  # Randomize (in place) objects the neuron responds to.

        self.objects = self.__power_law_selectivity(list_of_objects)

        # Activity fraction for the list of included objects. As the number of stimuli increases
        # measured activity fraction should approach the absolute value.
        self.activity_fraction_measured = \
            get_activity_fraction(np.array(self.objects.values()))

    def __power_law_selectivity(self, ranked_obj_list):
        """
        Object preference normalized rate (rate modifier) modeled as a power law distribution.
            Rate Modifier = objectIdx^(-selectivity)

        REF: Zoccolan et.al. 2007 - Fig2.

        :param ranked_obj_list: Ranked list of neurons preferred objects.

        :rtype : Dictionary of {object: rate modification factor}.

        Note: Implementation assumes the first object is the neurons most preferred object.
        """
        return ({item: np.power(np.float(idx), -self.activity_fraction_absolute)
                 for idx, item in enumerate(ranked_obj_list, start=1)})

    def get_ranked_object_list(self):
        """ Return neurons rank list of objects and rate modification factors """
        return sorted(self.objects.items(), key=lambda item: item[1], reverse=True)

    def print_parameters(self):
        """ Print parameters of the selectivity profile """
        print("Profile                                   = %s" % self.type)
        print("Sparseness(absolute activity fraction)    = %0.4f"
              % self.activity_fraction_absolute)
        print("Sparseness(measured activity fraction)    = %0.4f"
              % self.activity_fraction_measured)

        print("Object Preferences                        = ")

        max_name_length = np.max([len(name) for name in self.objects.keys()])

        lst = self.get_ranked_object_list()
        for obj, rate in lst:
            print ("\t%s : %0.4f" % (obj.ljust(max_name_length), rate))


if __name__ == "__main__":
    plt.ion()

    obj_list = ['car',
                'van',
                'Truck',
                'bus',
                'pedestrian',
                'cyclist',
                'tram',
                'person sitting']

    profile1 = PowerLawSparseness(obj_list)
    profile1.print_parameters()

    firing_rates = np.array(profile1.objects.values())

    print "Sparseness (Activity Fraction) of neuron: %f " % get_activity_fraction(firing_rates)
