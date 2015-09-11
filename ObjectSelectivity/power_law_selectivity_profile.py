# -*- coding: utf-8 -*-
"""
Created on Fri September 11 9:26:45 2015

@author: s362khan
"""

import selectivity_fit as selectivity
import random
import numpy as np
import matplotlib.pyplot as plt


class PowerLawSelectivity:

    def __init__(self, list_of_objects):

        self.type = 'power_law'

        self.sparseness_activity_fraction = \
            np.float(selectivity.get_activity_fraction_sparseness(1))

        list_of_objects = [obj.lower() for obj in list_of_objects]
        random.shuffle(list_of_objects)  # Randomize (in place) objects the neuron responds to.

        self.objects = self.__power_law_selectivity(list_of_objects)

    def __power_law_selectivity(self, ranked_obj_list):
        """
        Object preference normalized rate (rate modifier) modeled as a power law distribution.
            Rate Modifier = objectIdx^(-selectivity)

        REF: Zoccolan et.al. 2007 - Fig2.

        :param ranked_obj_list: Ranked list of neurons preferred objects.

        :rtype : Dictionary of {object: rate modification factor}.

        Note: Implementation assumes the first object is the neurons most preferred object.
        """
        return({item: np.power(np.float(idx), -self.sparseness_activity_fraction)
               for idx, item in enumerate(ranked_obj_list, start=1)})

    def get_ranked_object_list(self):
        """ Return neurons rank list of objects and rate modification factors """
        return sorted(self.objects.items(), key=lambda item: item[1], reverse=True)

    def print_parameters(self):
        """ Print parameters of the selectivity profile """
        print("Profile                          = %s" % self.type)
        print("Sparseness(activity fraction)    = %0.4f " % self.sparseness_activity_fraction)
        print("Object Preferences               = ")

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

    profile1 = PowerLawSelectivity(obj_list)
    profile1.print_parameters()

    profile2 = PowerLawSelectivity
