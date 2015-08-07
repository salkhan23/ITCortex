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
            selectivity,
            ranked_object_list,
            max_fire_rate=100,
            position_profile=None,
            size_profile=None):
        """
        Create an Inferior Temporal Cortex  neuron instance.

        :param selectivity          : Activity fraction.
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
        pass


def main(population_size, list_of_objects):

    population = []

    for _ in np.arange(population_size):
        sel_idx = selectivity.get_selectivity_distribution(1)
        random.shuffle(list_of_objects)

        neuron = Neuron(sel_idx, list_of_objects)

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
