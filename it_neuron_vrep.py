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


def main(population_size, list_of_objects):

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

    f, axis = plt.subplots()
    for it_neuron in it_cortex:
        it_neuron.position.plot_position_tolerance_contours(axis=axis, n_contours=1)
