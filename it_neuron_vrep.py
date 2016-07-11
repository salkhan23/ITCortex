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
        """
        Return 1 no matter what inputs are provided. Ones with dimensions = first dimension of
        first  non keyword argument

        :param  kwargs  : Any arguments specified as a keyword.
        :param  args    : Any arguments specified without a keyword.
        """
        del kwargs
        return np.ones_like(args[0])

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
            size_profile=None,
            rotation_profile=None,
            occlusion_profile=None,
            clutter_profile='average',
            dynamic_profile=None,):
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

        :param rotation_profile     : Type of Rotation tuning.
                                      Allowed types = {None(Default), gaussian}

        :param occlusion_profile    : Type of occlusion profile.
                                      Allowed types = {None(Default), 'TwoInputSigmoid'}

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

        # Rotation Tuning
        if rotation_profile is None:
            self.rotation = CompleteTolerance()

        elif rotation_profile.lower() == 'gaussian':
            import RotationalTolerance.gaussian_rotation_profile as grt
            reload(grt)

            self.rotation = grt.GaussianRotationProfile()

        else:
            raise Exception("Invalid rotation profile: %s" % size_profile)

        # Occlusion Profile
        if occlusion_profile is None:
            self.occlusion = CompleteTolerance()

        elif occlusion_profile.lower() == 'twoinputsigmoid':
            import OcclusionTolerance.two_input_sigmoid_occlusion_profile as sot
            reload(sot)

            self.occlusion = sot.TwoInputSigmoidOcclusionProfile()

        else:
            raise Exception("Invalid occlusion profile: %s" % occlusion_profile)

        # Clutter Profile
        if clutter_profile.lower() == 'average':
            import ClutterTolerance.averaging_clutter_profile as act
            reload(act)
            self.clutter = act.AveragingClutterProfile()
        else:
            raise Exception("Invalid Clutter Profile %s" % clutter_profile)

        # Dynamic Profile
        if dynamic_profile is None:
            self.dynamics = None
        elif dynamic_profile.lower() == 'tamura':
            from Dynamics import tamura_dynamic_profile as td
            reload(td)

            self.dynamics = td.TamuraDynamics(sim_time_step_s, self.selectivity.objects)
        else:
            raise Exception("Invalid dynamic profile %s", dynamic_profile)

    def print_properties(self):
        """ Print all parameters of neuron  """
        print (("*" * 20) + " Neuron Properties " + ("*" * 20))

        print ("Max fire rate %0.2f" % self.max_fire_rate)

        print("SELECTIVITY TOLERANCE %s" % ('-' * 27))
        self.selectivity.print_parameters()

        print("POSITION TOLERANCE %s" % ('-' * 30))
        self.position.print_parameters()

        print("SIZE TOLERANCE: %s" % ('-' * 33))
        self.size.print_parameters()

        print("ROTATION TOLERANCE: %s" % ('-' * 33))
        self.rotation.print_parameters()

        if self.dynamics is not None:
            print("DYNAMIC FIRING RATE PROFILE: %s" % ('-' * 33))
            self.dynamics.print_parameters()

        print("CLUTTER TOLERANCE: %s" % ('-' * 33))
        self.clutter.print_parameters()

        print("OCCLUSION TOLERANCE: %s" % ('-' * 33))
        self.occlusion.print_parameters()

        print ("*" * 60)

    def firing_rate(self, ground_truth_list):
        """
        Get Neuron firing rate for the current time step.

        :param ground_truth_list: see method _get_static_firing_rate for format.
        """
        rate = 0
        scales = 0
        default_rate = 0
        early_rate = 0

        if self.dynamics is not None and self.dynamics.type == 'tamura':

            if ground_truth_list:
                default_rate, scales = self._get_static_firing_rate(
                    self.selectivity.objects,
                    ground_truth_list)

                early_rate, early_scales = self._get_static_firing_rate(
                    self.dynamics.early_obj_pref,
                    ground_truth_list)

            rate = self.dynamics.get_dynamic_rates(early_rate, default_rate)

        else:

            if ground_truth_list:
                rate, scales = self._get_static_firing_rate(
                    self.selectivity.objects,
                    ground_truth_list)

        return np.float(rate), scales

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
        objects, \
        x_arr, y_arr, \
        size_arr, \
        _, _, _, \
        rot_y, rot_y_period, rot_y_m, \
        _, _, _,\
        vis_nd, vis_d = zip(*ground_truth_list)

        objects = list(objects)
        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        size_arr = np.array(size_arr)

        obj_pref_list = np.array([object_dict.get(obj, 0) for obj in objects])

        # Get position rate modifiers they will by used to weight isolated responses to get a
        # single clutter response.
        position_weights = self.position.firing_rate_modifier(x_arr, y_arr)
        size_fr = self.size.firing_rate_modifier(size_arr)
        occ_fr = self.occlusion.firing_rate_modifier(np.array(vis_nd), np.array(vis_d))
        rot_fr = self.rotation.firing_rate_modifier(np.array(rot_y),
                                                    np.array(rot_y_period),
                                                    np.array(rot_y_m))

        isolated_rates = self.max_fire_rate * \
            obj_pref_list * \
            position_weights * \
            size_fr * \
            occ_fr * \
            rot_fr

        if len(isolated_rates) > 1:
            joint_rate = self.clutter.firing_rate_modifier(isolated_rates, position_weights)
        else:
            joint_rate = isolated_rates

        # # Debug Code - print all Isolated fire rates
        # print("Static Isolated Fire Rates:")
        # for ii in np.arange(len(objects)):
        #     print ("%s: FR=%0.2f: pref=%0.2f, pos=%0.2f, size=%0.2f, rot=%0.2f, occ=%0.2f"
        #            % (objects[ii],
        #               isolated_rates[ii],
        #               obj_pref_list[ii],
        #               position_weights[ii],
        #               size_fr[ii],
        #               rot_fr[ii],
        #               occ_fr[ii]))
        #
        # print joint_rate
        # print ("static clutter rate %0.2f" % np.sum(joint_rate, axis=0))
        # raw_input('Continue?')

        scales = np.zeros((len(objects), 7))
        scales[:, 0] = isolated_rates
        scales[:, 1] = obj_pref_list
        scales[:, 2] = position_weights
        scales[:, 3] = size_fr
        scales[:, 4] = rot_fr
        scales[:, 5] = occ_fr
        scales[:, 6] = joint_rate

        return joint_rate, scales


def plot_neuron_dynamic_profile(
        it_neuron,
        t_stop_ms=1000,
        time_step_ms=5,
        axis=None,
        font_size=34):
    """
    Plot the dynamic firing rate of the neuron, if it was seeing its ideal stimulus for half the
    specified interval.

    :param font_size    : graph font size. Default=34
    :param  axis        : axis object to use for plotting.
    :param  time_step_ms: time step in milliseconds.
    :param  t_stop_ms   : Time period to plot dynamic response until.
    :param  it_neuron   : IT neuron instance.
    """

    # Get optimum stimulus for neuron
    # TODO: Add checks to see if they exist before taking their values
    pref_obj = it_neuron.selectivity.get_ranked_object_list()[0][0]
    pref_pos = it_neuron.position.rf_center
    pref_size = it_neuron.size.pref_size
    pref_rot = it_neuron.rotation.preferred_angle

    # Create ideal stimulus
    # TODO: Add optimum values for other ground truths when enabled in the larger model.
    ground_truth = [pref_obj,
                    pref_pos[0], pref_pos[1],
                    pref_size,
                    0, 1, 0,
                    pref_rot, 1, 0,
                    0, 1, 0,
                    1, 1]

    time_arr = np.arange(t_stop_ms, step=time_step_ms)
    rates = np.zeros(shape=time_arr.shape[0])
    stimulus = np.zeros(shape=time_arr.shape[0])

    for ii, time in enumerate(time_arr):

        if ii < time_arr.shape[0] / 2:
            rates[ii], scales = it_neuron.firing_rate([ground_truth])
            stimulus[ii] = 10
        else:
            rates[ii], scales = it_neuron.firing_rate([])
            stimulus[ii] = 0

    if axis is None:
        f, axis = plt.subplots()

    axis.plot(time_arr, rates, linewidth=2)
    axis.plot(time_arr, stimulus, linewidth=2, color='black', label="Input")

    # axis.set_title("Dynamic Firing Rate Profile", fontsize=font_size)
    axis.set_ylabel('FR (Spikes/s)', fontsize=font_size)
    axis.set_xlabel('Time (ms)', fontsize=font_size)

    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)

    axis.grid()
    axis.legend(fontsize=font_size, loc='best')

    axis.annotate(r'$SI_{K\_early}=%0.2f,\ SI_{K\_late}=%0.2f$' %
                  (it_neuron.dynamics.early_kurtosis_measured,
                   it_neuron.selectivity.kurtosis_measured),
                  xy=(0.7, 0.90),
                  xycoords='axes fraction',
                  fontsize=font_size,
                  horizontalalignment='right',
                  verticalalignment='top')

    # axis.annotate(r'$SI_{K/_late}=%0.2f$' % it_neuron.selectivity.kurtosis_measured,
    #               xy=(0.7, 0.75),
    #               xycoords='axes fraction',
    #               fontsize=font_size,
    #               horizontalalignment='right',
    #               verticalalignment='top')

    axis.annotate(r'$L_{early}=%0.2f,\ L_{late}=%0.2f$'
                  % (it_neuron.dynamics.early_tau * 1000 * time_step_ms,
                     it_neuron.dynamics.late_tau * 1000 * time_step_ms),
                  xy=(0.7, 0.75),
                  xycoords='axes fraction',
                  fontsize=font_size,
                  horizontalalignment='right',
                  verticalalignment='top')

    # axis.annotate(r'$Latency_{late}=%0.2f$'
    #               % (it_neuron.dynamics.late_tau * 1000 * time_step_ms),
    #               xy=(0.7, 0.55),
    #               xycoords='axes fraction',
    #               fontsize=font_size,
    #               horizontalalignment='right',
    #               verticalalignment='top')


def main(it_cortex):
    # Print RFs of all Neurons
    # TODO: Move/Use function into population.py
    # f, axis = plt.subplots()
    # for it_neuron in it_cortex:
    #     it_neuron.position.plot_position_tolerance_contours(axis=axis, n_contours=1)

    # TODO: Temporary. Validation code.
    it_cortex[0].print_properties()

    most_pref_obj = it_cortex[0].selectivity.get_ranked_object_list()[0][0]
    rf_center = it_cortex[0].position.rf_center
    pref_size = it_cortex[0].size.pref_size
    # print("Most preferred object %s " % most_pref_obj)
    # print("RF center %s" % rf_center)
    # print("Preferred Size %0.4f Radians" % pref_size)

    # Example 1
    title = "Test 1 - Response to single Item"
    print title

    ground_truth = (most_pref_obj,
                    rf_center[0], rf_center[1],
                    pref_size,
                    0.0, 1, False,
                    0.0, 1, False,
                    0.0, 1, False,
                    1.0, 1.0)
    print ("Ground Truth: ", ground_truth)

    print ("Neurons response %0.4f" % it_cortex[0].firing_rate(ground_truth)[0])

    # Example 2
    title = "Test 2 -  Response to multiple items"
    print title
    ground_truth = [
        [most_pref_obj,                   # object,
         rf_center[0], rf_center[1],      # x, y,
         pref_size,                       # size
         0.0, 1, False,                   # x Rotation angle, period of symmetry, mirror symmetry
         0.0, 1, False,                   # y Rotation angle, period of symmetry, mirror symmetry
         0.0, 1, False,                   # z Rotation angle, period of symmetry, mirror symmetry
         1.0, 1.0],                       # nondiagnostic, diagnostic visibilities

        ['monkey',
         rf_center[0], rf_center[1],
         pref_size,  0.0, 1, False,
         0.0, 1, False,
         0.0, 1, False,
         1.0, 1.0]
    ]

    print ("Ground Truth:")
    for entry in ground_truth:
        print entry
    print ("Neurons response %0.4f" % it_cortex[0].firing_rate(ground_truth)[0])

    # Example 3 dynamic firing rates only.
    title = "Test 3 - Test Dynamic Response"
    print title

    if it_cortex[0].dynamics is not None:

        time_step = it_cortex[0].dynamics.dt
        steps = 500
        rates_arr = np.zeros(shape=steps)
        ground_truth_present = np.zeros_like(rates_arr)

        ground_truth = [
            [most_pref_obj,                 # object,
             rf_center[0], rf_center[1],    # x, y,
             pref_size,                     # size
             0.0, 1, False,                 # x Rotation angle, period of symmetry, mirror symmetry
             0.0, 1, False,                 # y Rotation angle, period of symmetry, mirror symmetry
             0.0, 1, False,                 # z Rotation angle, period of symmetry, mirror symmetry
             1.0, 1.0],                     # nondiagnostic, diagnostic visibilities

            ['monkey',
             rf_center[0], rf_center[1],
             pref_size,  0.0, 1, False,
             0.0, 1, False,
             0.0, 1, False,
             1.0, 1.0]
        ]

        for ii in np.arange(steps):

            if ii > (steps / 2.0):
                ground_truth = []

            if ground_truth:
                ground_truth_present[ii] = 1

            rates_arr[ii], scales = it_cortex[0].firing_rate(ground_truth)

        plt.figure("Dynamic Firing Rate of Neuron 0")
        plt.plot(np.arange(steps * time_step, step=time_step), rates_arr)
        plt.plot(np.arange(steps * time_step, step=time_step), ground_truth_present,
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
                        size_profile='Lognormal',
                        occlusion_profile='TwoInputSigmoid')

        it_population.append(neuron)

    main(it_population)
