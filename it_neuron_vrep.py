# -*- coding: utf-8 -*-
"""


@author: s362khan
"""
import numpy as np
import matplotlib.pyplot as plt

import population_utils as utils
# Force reload (compile) IT cortex modules to pick changes not included in cached version.
reload(utils)


class CompleteTolerance:
    def __init__(self):
        """
        Default Tuning Profile. Complete Tolerance, No rate modification. This is overloaded for
        each property IT neuron expects but for which the parameters are not provided.
        It makes getting the firing_rate_modifier easier instead of adding multiple conditions
        that check whether a profile exist or not before getting its firing rate modifier.

        :rtype : object
        """
        self.type = 'none'

    @staticmethod
    def firing_rate_modifier(*args, **kwargs):
        """Return 1 no matter what inputs are provided"""
        del args
        del kwargs
        return 1

    def print_parameters(self):
        print("Profile: %s" % self.type)


class Neuron:
    def __init__(
            self,
            object_list,
            sim_time_step_s=0.005,
            selectivity_profile='power_law',
            max_fire_rate=100,
            position_profile=None,
            dynamic_profile=None,
            size_profile=None):
        """
        Create an Inferior Temporal Cortex  neuron instance.

        :param sim_time_step_s      : simulation time step in seconds(dt).(Default=0.005ms)

        :param selectivity_profile  : Type of object selectivity tuning.
                                      Allowed types: {power_law(Default), 'kurtosis'}

        :param object_list          : List of objects in scene.

        :param max_fire_rate        : Maximum firing Rate (Spikes/second). (Default = 100, unless
                                      selectivity_profile = kurtosis, in which case it is
                                      internally generated.

        :param position_profile     : Type of position tuning.
                                      Allowed types = {None (Default), gaussian}

        :param size_profile         : Type of size tuning.
                                      Allowed types = {None(Default), 'Lognormal)}

        :param dynamic_profile      : Type of dynamic profile.
                                      Allowed types = {None(Default), tamura}

        :rtype : It neuron instance.
        """

        # Selectivity Tuning
        if selectivity_profile.lower() == 'power_law':
            from ObjectSelectivity import power_law_selectivity_profile as pls
            reload(pls)  # Force recompile to pick up any new changes not in cached module.

            self.selectivity = pls.PowerLawSparseness(object_list)

        elif selectivity_profile.lower() == 'kurtosis':
            from ObjectSelectivity import kurtosis_selectivity_profile as ks
            reload(ks)

            self.selectivity = ks.KurtosisSparseness(object_list)

        else:
            raise Exception("Invalid selectivity profile: %s" % selectivity_profile)

        # Max Firing Rate Distribution
        if self.selectivity.type == 'kurtosis':
            self.max_fire_rate = self.selectivity.get_max_firing_rate()
        else:
            self.max_fire_rate = max_fire_rate

        # Dynamic Profile
        if dynamic_profile is None:
            self.dynamics = None
        elif dynamic_profile.lower() == 'tamura':
            from Dynamics import tamura_dynamic_profile as td
            reload(td)

            self.dynamics = td.TamuraDynamics(sim_time_step_s, self.selectivity.objects)
        else:
            raise Exception("Invalid dynamic profile %s", dynamic_profile)

        # Position Tuning
        if position_profile is None:
            self.position = CompleteTolerance()

        elif position_profile.lower() == 'gaussian':
            import PositionTolerance.gaussian_position_profile as gpt
            reload(gpt)

            self.position = gpt.GaussianPositionProfile(
                self.selectivity.activity_fraction_absolute)
        else:
            raise Exception("Invalid position profile: %s" % position_profile)

        # Size Tuning
        if size_profile is None:
            self.size = CompleteTolerance()
        elif size_profile.lower() == 'lognormal':

            import SizeTolerance.log_normal_size_profile as lst
            reload(lst)

            # Lognormal size tolerance expects a position tolerance parameter.
            try:
                pos_tol = self.position.position_tolerance
            except:
                raise Exception("Position tolerance needed to create log normal size tuning")

            self.size = lst.LogNormalSizeProfile(pos_tol)
        else:
            raise Exception("Invalid size profile: %s" % size_profile)

    def print_properties(self):
        """ Print all parameters of neuron """
        print ("*"*20 + " Neuron Properties " + "*"*20)

        print("SELECTIVITY TOLERANCE %s" % ('-'*27))
        self.selectivity.print_parameters()

        if self.dynamics is not None:
            print("DYNAMIC FIRING RATE PROFILE: %s" % ('-'*33))
            self.dynamics.print_parameters()

        print("POSITION TOLERANCE %s" % ('-'*30))
        self.position.print_parameters()

        print("SIZE TOLERANCE: %s" % ('-'*33))
        self.size.print_parameters()

        print ("*"*60)

    def firing_rate(self, ground_truth_list):
        """
        """
        rate = 0
        default_rate = 0
        early_rate = 0

        if self.dynamics is not None and self.dynamics.type == 'tamura':

            if ground_truth_list:
                default_rate = self._get_static_firing_rate(
                    self.selectivity.objects,
                    ground_truth_list)

                early_rate = self._get_static_firing_rate(
                    self.dynamics.early_obj_pref,
                    ground_truth_list)

            rate = self.dynamics.get_dynamic_rates(early_rate, default_rate)

        else:

            if ground_truth_list:
                rate = self._get_static_firing_rate(
                    self.selectivity.objects,
                    ground_truth_list)

        return np.float(rate)

    def _get_static_firing_rate(
            self,
            object_dict,
            ground_truth_list):
        """
        Get Neurons static overall firing rate to specified input.

        :param ground_truth_list: list of:
            [object_name,
             x,
             y,
             size,
             rot_x,
             rot_y,
             rot_z,
             vis_nondiag,
             vis_diag]
        entries for all objects in the
        screen. Add more elements to this list and update the zip function.

        :rtype : Return the net average (multi object response) firing rate of the neuron
                 for the specified input(s)
        """
        if not isinstance(ground_truth_list, list):
            ground_truth_list = [ground_truth_list]

        # In the VREP scene, rotations are specified with respect to the world reference
        # frame. Rotations in the extracted ground truth are with respect to the vision sensor
        # coordinate system and include the rotations of the vision sensor. Currently the vision
        # sensor is rotated by +90 degrees around the y (beta) & z (gamma) axes.
        # The IT cortex rotation tuning profile is around the vertical axis which is defined
        # as the y-axis of the vision sensor. Rotating the object (in real world coordinates)
        # around the x-axis results in rotations around the y axis of the vision sensor.
        objects, x_arr, y_arr, size_arr, _, _, _, _, _ = zip(*ground_truth_list)

        objects = list(objects)
        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        size_arr = np.array(size_arr)

        obj_pref_list = np.array([object_dict.get(obj, 0) for obj in objects])

        # Get position rate modifiers they will by used to weight isolated responses to get a
        # single clutter response.
        position_weights = self.position.firing_rate_modifier(x_arr, y_arr)
        sum_position_weights = np.sum(position_weights, axis=0)

        rate = self.max_fire_rate * \
            obj_pref_list * \
            position_weights * \
            self.size.firing_rate_modifier(size_arr)

        # Clutter Response
        # TODO: Add noise to the averaged response based on Zoccolan-2005
        if 0 != sum_position_weights:
            rate2 = rate * position_weights / sum_position_weights
        else:
            rate2 = 0

        # # Debug Code
        # for ii in np.arrange(len(objects)):
        #     print("Object %s, pref %0.2f,pos_weight %0.2f, isolated FR %0.2f, weighted FR %0.2f"
        #           % (objects[ii], obj_pref_list[ii], position_weights[ii], rate[ii], rate2[ii]))
        #
        # print ("firing rate sum %0.2f" % np.sum(rate2, axis=0))
        # raw_input('Continue?')

        return np.sum(rate2, axis=0)


def main(it_cortex):
    # Print RFs of all Neurons
    # TODO: Move/Use function into population.py
    # f, axis = plt.subplots()
    # for it_neuron in it_cortex:
    #     it_neuron.position.plot_position_tolerance_contours(axis=axis, n_contours=1)

    # TODO: Temporary. Validation code.
    it_cortex[0].print_properties()

    most_pref_object = it_cortex[0].selectivity.get_ranked_object_list()[0][0]
    rf_center = it_cortex[0].position.rf_center
    pref_size = it_cortex[0].size.pref_size
    # print("Most preferred object %s " % most_pref_object)
    # print("RF center %s" % rf_center)
    # print("Preferred Size %0.4f Radians" % pref_size)

    # Example 1
    title = "Test 1 - Response to single Item"
    print title
    ground_truth = (most_pref_object, rf_center[0], rf_center[1], 0.0, 0.0, 0.0, 0.0)
    print ("Ground Truth: ", ground_truth)

    print ("Neurons response %0.4f" % it_cortex[0].firing_rate(ground_truth))

    # Example 2
    title = "Test 2 -  Response to multiple items"
    print title
    ground_truth = [
        # object,           x,            y,            size,     rot_x, rot_y, rot_z
        [most_pref_object, rf_center[0], rf_center[1], pref_size, 0.0,   0.0,   0.0],
        ['monkey',         rf_center[0], rf_center[1], pref_size, 0.0,   0.0,   0.0]]

    print ("Ground Truth:")
    for entry in ground_truth:
        print entry
    print ("Neurons response %0.4f" % it_cortex[0].firing_rate(ground_truth))

    # Example 3 dynamic firing rates only.
    title = "Test 3 - Test Dynamic Response"
    print title

    if it_cortex[0].dynamics is not None:

        time_step = it_cortex[0].dynamics.dt
        steps = 500
        rates_arr = np.zeros(shape=steps)
        ground_truth_present = np.zeros_like(rates_arr)

        ground_truth = [
            # object,           x,            y,            size,     rot_x, rot_y, rot_z
            [most_pref_object, rf_center[0], rf_center[1], pref_size, 0.0,   0.0,   0.0],
            ['monkey',         rf_center[0], rf_center[1], pref_size, 0.0,   0.0,   0.0]]

        for ii in np.arange(steps):

            if ii > (steps/2.0):
                ground_truth = []

            if ground_truth:
                ground_truth_present[ii] = 1

            rates_arr[ii] = it_cortex[0].firing_rate(ground_truth)

        plt.figure("Dynamic Firing Rate of Neuron 0")
        plt.plot(np.arange(steps*time_step, step=time_step), rates_arr)
        plt.plot(np.arange(steps*time_step, step=time_step), ground_truth_present,
                 label='Input stimulus', color='k', linewidth=2)
        plt.legend()

    # Population Plots
    # Plot the selectivity distribution of the population
    utils.plot_population_selectivity_distribution(it_cortex)

    # Plot Object preferences of population
    utils.plot_population_obj_preferences(it_cortex)


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

    # Example 1: Neuron Population with power_law selectivity distribution ----------------------
    it_population = []
    for _ in np.arange(n):
        neuron = Neuron(obj_list,
                        selectivity_profile='power_law',
                        position_profile='Gaussian',
                        size_profile='Lognormal')

        it_population.append(neuron)

    main(it_population)

    # Example 2: Neuron Population with kurtosis based selectivity distribution -----------------
    it_population = []
    for _ in np.arange(n):
        neuron = Neuron(obj_list,
                        dynamic_profile='Tamura',
                        selectivity_profile='Kurtosis',
                        position_profile='Gaussian',
                        size_profile='Lognormal')

        it_population.append(neuron)

    main(it_population)
