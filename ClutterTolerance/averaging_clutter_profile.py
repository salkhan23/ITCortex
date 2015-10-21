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

        return np.max([0, clutter_rate + self._deviation_from_average()])


if __name__ == "__main__":
    plt.ion()

    profile1 = AveragingClutterProfile()

    # Plot the clutter profile of the neuron
    num_obj_array = [2, 3]
    num_samples = 100

    f, ax_arr = plt.subplots(1, len(num_obj_array))

    for idx, num_obj in enumerate(num_obj_array):

        clutter_rates = []
        isolated_responses = np.random.uniform(low=0, high=1, size=(num_samples, num_obj))
        position_weights = np.random.uniform(low=0, high=1, size=(num_samples, num_obj))

        for sample in np.arange(num_samples):
            clutter_rates.append(
                profile1.firing_rate_modifier(
                    isolated_responses[sample, :],
                    position_weights[sample, :])
            )

        sum_isolated_responses = np.sum(isolated_responses, axis=1)

        ax_arr[idx].scatter(sum_isolated_responses, clutter_rates)

        x = np.linspace(0, np.max(sum_isolated_responses), 100)
        ax_arr[idx].plot(x, x / num_obj, label="Average")
        ax_arr[idx].plot(x, x, label="Sum")
        ax_arr[idx].legend()
