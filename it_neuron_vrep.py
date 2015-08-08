# -*- coding: utf-8 -*-
"""


@author: s362khan
"""
import random
import numpy as np
import matplotlib.pyplot as plt

import ObjectSelectivity.selectivity_fit as selectivity


class Neuron:
    def __init__(
            self,
            selectivity_idx,
            ranked_object_list,
            max_fire_rate=100,
            position_profile=None,
            size_profile=None):
        """
        Create an Inferior Temporal Cortex  neuron instance.

        :param selectivity_idx          : Activity fraction.
            Number of objects neuron responds to divided by total number of objects.
            Defined in [Zoccolan et. al. 2007]. {1 - [sum(Ri/n)^2 / sum(Ri^2/n)] } / (1-1/n).

        :param ranked_object_list   : Ranked list of preferred objects.

        :param max_fire_rate        : Maximum firing Rate (Spikes/second). (default = 100)

        :param position_profile     : Type of position tuning.
            Allowed types = {None (default), gaussian}

        :param size_profile         : Type of size tuning.
            Allowed types = {None(default, 'Lognormal)}

        :rtype : It neuron instance.
    """
        # Selectivity Index
        if not (0 < selectivity_idx <= 1):
            raise Exception("Selectivity %0.2f not within [0, 1]" % self.selectivity)
        self.selectivity = selectivity_idx

        # Objects
        ranked_object_list = [item.lower() for item in ranked_object_list]
        self.objects = self.__power_law_selectivity(ranked_object_list)
        self.selectivity_type = 'power_law'

        # Max Firing Rate
        self.max_fire_rate = max_fire_rate

        # Position Profile
        if position_profile.lower() == 'gaussian':
            import PositionTolerance.gaussian_position_profile as gpt
            self.position = gpt.GaussianPositionProfile(self.selectivity)

    def __power_law_selectivity(self, ranked_obj_list):
        """
        Object preference normalized rate (rate modifier) modeled as a power law distribution.
            Rate Modifier = objectIdx^(-selectivity)

        REF: Zoccolan et.al. 2007 - Fig2

        :param ranked_obj_list: Ranked list of neurons preferred objects.

        :rtype : Dictionary of {object: rate modification factor}.
        """
        return({item: np.power(idx, -self.selectivity)
               for idx, item in enumerate(ranked_obj_list, start=1)})

    def get_ranked_object_list(self):
        """ Return neurons rank list of objects and rate modification factors """
        return sorted(self.objects.items(), key=lambda item: item[1], reverse=True)

    def print_object_list(self):
        """ Print a ranked list of neurons object preferences """
        print("Object Preferences           :")

        max_name_length = np.max([len(name) for name in self.objects.keys()])

        lst = self.get_ranked_object_list()
        for obj, rate in lst:
            print ("\t%s : %0.4f" % (obj.ljust(max_name_length), rate))

    def print_properties(self):
        """ Print all parameters of neuron """
        print ("*"*20 + " Neuron Properties " + "*"*20)
        print("Neuron Selectivity           : %0.2f" % self.selectivity)
        print("Selectivity Profile          : %s" % self.selectivity_type)
        print("Max Firing Rate (spikes/s)   : %i" % self.max_fire_rate)
        self.print_object_list()

        print("POSITION TOLERANCE %s" % ('-'*30))
        self.position.print_parameters()

    def firing_rate(self, ground_truth_list):
        """
        Get Neurons overall firing rate to specified input.

        :param ground_truth_list: list of (object_name, x, y) entries for all objects in the
        screen. Add more elements to this  list/tuple and update the zip function.

        :rtype :  Return the average firing rate of the neuron given for the specified input(s)
        """
        if not isinstance(ground_truth_list, list):
            ground_truth_list = [ground_truth_list]

        objects, x_arr, y_arr = zip(*ground_truth_list)

        objects = list(objects)
        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)

        obj_pref_list = np.array([self.objects.get(obj.lower(), 0) for obj in objects])


        # Get position rate modifiers they will by used to weight isolated responses to get the
        # single clutter responses.
        position_weights = self.position.firing_rate_modifier(x_arr, y_arr)
        sum_position_weights = np.sum(position_weights, axis=0)

        rate = self.max_fire_rate * obj_pref_list

        # Single response to multiple objects
        # TODO: Add noise to the averaged response based on Zoccolan-2005
        rate = rate * position_weights / sum_position_weights

        return np.sum(rate, axis=0)


def main(population_size, list_of_objects):

    """

    :rtype : Population of IT neurons of specifed size and that respond to the list of objects.
    """
    population = []

    for _ in np.arange(population_size):
        sel_idx = selectivity.get_selectivity_distribution(1)
        random.shuffle(list_of_objects)

        neuron = Neuron(sel_idx, list_of_objects, position_profile='gaussian')

        population.append(neuron)

    return np.array(population)


if __name__ == "__main__":
    plt.ion()

    n = 100
    obj_list = ['car',
                'van',
                'Truck',
                'bus',
                'pedestrian',
                'cyclist',
                'tram',
                'person sitting']

    it_cortex = main(n, obj_list)

    # Print RFs of all Neurons
    # TODO: Move/Use function into population.py
    # f, axis = plt.subplots()
    # for it_neuron in it_cortex:
    #     it_neuron.position.plot_position_tolerance_contours(axis=axis, n_contours=1)

    # TODO: Temporary. Validation code.
    it_cortex[0].print_properties()
    
    most_pref_object = it_cortex[0].get_ranked_object_list()[0][0]
    rf_center = it_cortex[0].position.rf_center

    print ("most preferred object %s " % most_pref_object)
    print ("RF center %s" % rf_center)

    ground_truth = (most_pref_object, rf_center[0], rf_center[1])
    print it_cortex[0].firing_rate(ground_truth)

    ground_truth = [
        [most_pref_object, rf_center[0], rf_center[1]],
        ['monkey',         rf_center[0], rf_center[1]]]

    print it_cortex[0].firing_rate(ground_truth)
