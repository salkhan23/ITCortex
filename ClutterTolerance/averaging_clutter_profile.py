# -*- coding: utf-8 -*-
"""


@author: s362khan
"""
import numpy as np
import matplotlib.pyplot as plt


class AveragingClutterProfile:

    def __init__(self):
        """
        Model clutter tolerance as a function of the position weighted average of all objects
        in the receptive field of the neuron. The normalized position average response is used to
        determine whether the object lies within the receptive field of the neuron.
        Derived from the results in Zoccolan, Cox and Dicarlo - 2005 - Multiple Response
        normalization in Monkey InferoTemporal Cortex.

        Properties modeled:
        1. Response is closer to the average of isolated responses rather then their linear sum.

        2. Even objects that do not elicit responses on their own can impact the responses of
           responsive objects if they lie within the receptive field. We implement this by
           weighting each isolated each response with its normalized position firing rate.

        Properties not modeled:
        1. Rolls et. al (2003) - showed that receptive field sizes or responses of IT neurons were
           reduced in the presence of background noise. We do not include the effect on background

        2. Responses of neurons to multiple objects have been well studied in other areas,
           particularly in V1. Range of responses varied between the average and winner take all
           responses and depended on the contrast of the individual stimuli. It is likely that
           a similar mechanism is present in IT. We do not model winner take all behavior.
        """

        self.type = 'position weighted average'
        self.d = max(0, self._deviation_from_average())

    def print_parameters(self):
        print("Profile: %s" % self.type)

    @staticmethod
    def _deviation_from_average():
        """
        Generate deviation from averaging rule. This is modeled as a gaussian distribution with
        mean = 0 and sigma determined from data fitting. See ClutterModelFit.py for details.
        """
        return np.float(np.random.normal(loc=0, scale=0.17))

    def firing_rate_modifier(self, isolated_fire_rates, weights):

        if not type(isolated_fire_rates) == np.ndarray:
            isolated_fire_rates = np.array(isolated_fire_rates)

        if not type(weights) == np.array:
            weights = np.array(weights)

        sum_weights = np.sum(weights)

        if 0 != sum_weights:
            clutter_rate = np.sum(isolated_fire_rates * weights) / np.float(sum_weights)
        else:
            clutter_rate = 0

        return clutter_rate + self.d

    def plot_clutter_profile(self, axis=None, font_size=34, num_objs=2, num_samples=100):
        """
        Plots the clutter profile of the neuron if it were seeing two and three objects at random
        places within its receptive field

        :param num_samples: number of sample points to consider. Default=100
        :param num_objs: number of objects in the neurons receptive field.Default=2.
        :param font_size: graph font size. Default=34
        :param axis: axis to plot. Default is None, which means create a new figure.
        """

        if axis is None:
            fig, axis = plt.subplots()

        # Generate some random responses and random positions of objects
        isolated_responses = np.random.uniform(low=0, high=1, size=(num_samples, num_objs))
        position_weights = np.random.uniform(low=0, high=1, size=(num_samples, num_objs))

        clutter_rates = []
        for sample in np.arange(num_samples):
            clutter_rates.append(
                self.firing_rate_modifier(
                    isolated_responses[sample, :],
                    position_weights[sample, :])
            )

        sum_isolated_responses = np.sum(isolated_responses, axis=1)

        axis.scatter(sum_isolated_responses, clutter_rates, s=60)

        x = np.linspace(0, np.max(sum_isolated_responses), 100)
        axis.plot(x, x / num_objs, label="Average", linewidth=2)
        axis.plot(x, x, label="Sum", linewidth=2)
        axis.legend(fontsize=font_size - 5)

        axis.set_xlabel('Sum Isolated Responses (spikes/s)', fontsize=font_size)
        axis.set_ylabel('Joint FR', fontsize=font_size)
        axis.tick_params(axis='x', labelsize=font_size)
        axis.tick_params(axis='y', labelsize=font_size)
        # axis.set_title("%d Objects" % num_objs, fontsize=font_size)
        axis.grid()
        axis.set_xlim([0, num_objs])
        axis.set_ylim([0, num_objs])

        axis.annotate('%d objects' % num_objs,
                      xy=(0.25, 0.9),
                      xycoords='axes fraction',
                      fontsize=font_size,
                      horizontalalignment='right',
                      verticalalignment='top')


if __name__ == "__main__":
    plt.ion()

    profile1 = AveragingClutterProfile()

    f, ax_arr = plt.subplots(1, 2, sharey=True)
    profile1.plot_clutter_profile(num_objs=2, axis=ax_arr[0])
    profile1.plot_clutter_profile(num_objs=3, axis=ax_arr[1])
