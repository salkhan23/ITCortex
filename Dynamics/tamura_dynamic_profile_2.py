# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Do relative import of the main folder to get files in sibling directories
top_level_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if top_level_dir_path not in sys.path:
    sys.path.append(top_level_dir_path)

from ObjectSelectivity.power_law_selectivity_profile import calculate_activity_fraction
from ObjectSelectivity.kurtosis_selectivity_profile import calculate_kurtosis


def get_poisson_spikes(dt, rates):
    """
    Poisson approximation via Bernoulli process.

    :param dt: Size of time step (s; should be ~1ms)
    :param rates: List of neuron spike rates (spikes / s)
    :return: List of random spike events (0 = no spike; 1 = spike)
    """
    assert len(rates.shape) == 1
    # noinspection PyArgumentList
    return np.random.rand(len(rates)) < (dt * rates)


def integrate(dt, a, b, c, x, u):
    """
    Euler integration of state-space equations.

    :param dt: Time step (s; should be ~5ms)
    :param a: The standard feedback matrix from linear-systems theory
    :param b: Input matrix
    :param c: Output matrix
    :param x: State vector
    :param u: Input vector
    :return: (x, y), i.e. the state and the output
    """

    dxdt = np.dot(a, x) + np.dot(b, u)
    x = x + dxdt * dt
    y = np.dot(c, x)

    return x, y


# noinspection PyArgumentList
class TamuraDynamics:
    """
    Simple model of IT spike-rate tamura_dynamic_profile.py based on:

    Tamura, H., & Tanaka, K. (2001). Visual response properties of cells in the ventral
        and dorsal parts of the macaque inferotemporal cortex. Cerebral Cortex, 11(5), 384–399.

    Some key points from this paper are:
      1) There are usually two response phases, early (transient) and late (sustained)
      2) Selectivity is higher in late phase than early phase
      3) There is heterogeneity in the ratio of early to late magnitude, but peak
        early responses are typically higher (note: this probably arises at least
        in part from selectivity)
      4) Response latency is negatively correlated with peak response magnitude

    There are many other interesting dynamic phenomena that we don't model here, including from:

    Brincat, S. L., & Connor, C. E. (2006). Dynamic shape synthesis in posterior inferotemporal
        cortex. Neuron, 49(1), 17–24.
    Kiani, R., Esteky, H., & Tanaka, K. (2005). Differences in onset latency of macaque
        inferotemporal neural responses to primate and non-primate faces. Journal of
        Neurophysiology, 94(2), 1587–1596.
    Matsumoto, N., Okada, M., Sugase-Miyamoto, Y., Yamane, S., & Kawano, K. (2005). Population
        dynamics of face-responsive neurons in the inferior temporal cortex. Cerebral Cortex,
        15(8), 1103–1112.
    Ringo, J. L. (1996). Stimulus specific adaptation in inferior temporal and medial temporal
        cortex of the monkey, 76, 191–197.
    """
    def __init__(self, dt, obj_dict, max_fire_rate, max_latency=0.25):
        """
        :param dt           : Simulation time step in seconds.
        :param obj_dict     : Dictionary of {object: selectivity} for the neuron.
                              Selectivity ranges between (0, 1) and is the normalized firing rate
                              of the neuron to the specified object.
        :param max_latency  : maximum response latency of the neuron. Default = 0.25s
        """
        self.dt = dt

        self.n = 1  # do not change this. Code will not work if not =1
        assert self.n == 1

        self.type = 'tamura'

        self.transience = self._get_transience()

        # Lehky2011 average rate were measured over the (71 - 210)ms window. This corresponds to
        # the early response stage of Tamura2001. Use the transience value to get the late object
        # selectivities
        self.late_obj_dict = self._get_late_obj_selectivities(obj_dict)

        # Two LTE systems are used to describe the dynamics profile: (1) Early LTE system and
        # (2) Late LTE system. These are the list of parameters of these systems that are constant
        # these are templates that must be multiplied by 1/tau for each neuron ...
        self.early_A = np.array([[-1, 0], [1, -1]])
        self.early_B = np.array([1, 0])

        self.late_A = np.array([[-1, 0], [1, -1]])
        self.late_B = np.array([1, 0])

        # state of LTI (linear time-invariant) dynamical systems
        self.early_x = np.zeros((2, self.n))
        self.late_x = np.zeros((2, self.n))

        # Early_tau
        self.early_tau = self._get_early_tau()

        # Early Gain
        ranked_objs_list = self.get_ranked_object_list(obj_dict)
        self.early_gain = self._get_early_gain(ranked_objs_list[0][1], max_fire_rate)

        # Late Tau
        self.late_tau = self._get_late_tau()

        # Late transient gain and late sustained gain
        # TODO

        # # Print the two dictionaries
        # for k, v in obj_dict.items():
        #     print k, v
        #
        # print("-" * 20)
        # for k, v in self.late_obj_dict.items():
        #     print k, v

    @staticmethod
    def _get_transience():
        """
        Tamura 2001 defined transience as  (init_resp - late_resp) / (init_resp + late_resp)

        A positive value indicates initial response is greater that late response, while
        a negative value indicates later response is larger.

        Transience ranges between [-1, 1]. Tamura provides a distribution of transience over their
        population in figure 5b. It had a mean of 0.4 and a sd of 0.26. We use a normal
        distribution to get the transience value.

        :return: transience of the neuron
        """
        t = np.random.normal(loc=0.4, scale=0.26)

        # clip between -1 and 1
        if t < -1:
            t = -1.0
        elif t > 1:
            t = 1.0

        return t

    def _get_late_obj_selectivities(self, early_obj_dict):

        late_obj_dict = {k: v * (1 - self.transience) / (1 + self.transience)
                         for k, v in early_obj_dict.items()}

        return late_obj_dict

    @staticmethod
    def _get_early_tau():
        """
        The rise/fall of the early LTI system. Its distribution is found by fitting multiple
        dynamic profiles  in Figure 3 and 4 of Tamura 2001.

        See dynamics_profile_fits.txt and tamura_profile_fits.py
        mean = 0.0198, std = 0.0053

        :return:
        """
        t = 0
        while t <= 0:
            t = np.random.normal(loc=0.0198, scale=0.0053)

        return t

    def _get_early_gain(self, r_obj, max_fr):
        """

        :return:
        """
        # According to Lehky, our object_pref is equal to the average fire rate in the first
        # 0-140 ms window [Assuming no latency]. To get the raw fire rate we do r_obj*max_fr
        #
        # To ensure the two measures match up get the area under the curve of the LTI system
        # and make sure it is the same as the average fire rate.

        avg_rate = r_obj * max_fr

        time_arr = np.arange(0, 0.140, step=self.dt)
        input_arr = np.ones_like(time_arr) * avg_rate

        early_a = 1 / self.early_tau * self.early_A
        early_b = 1 / self.early_tau * self.early_B

        # We have to determine this, for now just use a default value
        default_gain = 1.0
        early_c = np.array([default_gain, -default_gain])

        lti_out_arr = np.zeros_like(input_arr)
        for u_idx, u in enumerate(input_arr):

            self.early_x[:, 0], y = integrate(
                self.dt,
                early_a,
                early_b,
                early_c,
                self.early_x[:, 0],
                u
            )

            lti_out_arr[u_idx] = np.maximum(0, y)

        # clear_out the system
        self.early_x[:, 0] = 0
        plt.plot(time_arr, lti_out_arr, linestyle='--')

        # Integrate the output of the LTI system to get early_gain
        area_lti = np.trapz(lti_out_arr, dx=self.dt)
        area_desired = np.trapz(input_arr, dx=self.dt)

        early_gain = area_desired / area_lti
        print("Early_gain", early_gain)

        # # Debug make sure everything is making sense
        # lti_out_arr = np.zeros_like(input_arr)
        #
        # early_c = np.array([early_gain, -early_gain])
        #
        # for u_idx, u in enumerate(input_arr):
        #     self.early_x[:, 0], y = integrate(
        #         self.dt,
        #         early_a,
        #         early_b,
        #         early_c,
        #         self.early_x[:, 0],
        #         u
        #     )
        #
        #     lti_out_arr[u_idx] = np.maximum(0, y)
        #
        # # clear_out the system
        # self.early_x[:, 0] = 0
        # plt.plot(time_arr, lti_out_arr)
        #
        # print("Area Under LTI system %0.4f" % np.trapz(lti_out_arr, dx=self.dt))
        # print("Desired area %0.4f" % area_desired)

        return early_gain

    @staticmethod
    def _get_late_tau():
        """
        The rise/fall of the late LTI system. Its distribution is found by fitting multiple
        dynamic profiles  in Figure 3 and 4 of Tamura 2001.

        See dynamics_profile_fits.txt and tamura_profile_fits.py
        mean = 0.3525, std = 0.67

        :return:
        """
        t = 0
        while t <= 0:
            t = np.random.normal(loc=0.3525, scale=0.67)

        return t

    @staticmethod
    def get_ranked_object_list(obj_dict):
        """ Return neurons rank list of objects and rate modification factors """
        return sorted(obj_dict.items(), key=lambda item: item[1], reverse=True)

    def print_parameters(self):
        """ Print parameters of the profile """
        print("Profile                              = %s" % self.type)
        print("transience                           = %0.2f" % self.transience)


if __name__ == '__main__':
    plt.ion()

    time_step = .005

    # Input Object preferences dictionary
    default_obj_pref = {'Truck'          : 0.0902061634469222,
                        'bus'            : 0.60042052408207613,
                        'car'            : 0.41523454488601136,
                        'cyclist'        : 0.36497039000201714,
                        'pedestrian'     : 0.18954386060352452,
                        'person sitting' : 0.18705214386720551,
                        'tram'           : 0.24122725774540257,
                        'van'            : 0.42843038621161039}

    # print("Default(Late) object preferences:")
    # name_len_max = np.max([len(name) for name in default_obj_pref.keys()])
    #
    # for obj_type, obj_pref in default_obj_pref.items():
    #     print ("\t%s : %0.4f" % (obj_type.ljust(name_len_max), obj_pref))
    #
    # print("Default (Late) Selectivity Measures:")
    # print ("\tActivity Fraction %0.4f"
    #        % calculate_activity_fraction(np.array(default_obj_pref.values())))
    # print ("\tKurtosis %0.4f"
    #        % calculate_kurtosis(np.array(default_obj_pref.values())))
    #
    # print ("-" * 80)

    d = TamuraDynamics(time_step, default_obj_pref, max_fire_rate=20)

    d.print_parameters()

    # plot latency vs. static rate for each neuron ...
    # d.plot_latencies_verses_rate_profile()

    # # run some neurons with square-pulse input ...
    # steps = 200
    # early_fire_rates = np.zeros((1, steps))
    # early_fire_rates[:, 20:100] = 50
    # late_fire_rates = early_fire_rates / 2
    # early_fire_rates = early_fire_rates * np.random.rand(1, 1)
    # late_fire_rates = late_fire_rates * np.random.rand(1, 1)
    #
    # time = time_step * np.array(range(steps))
    # dynamic_rates = np.zeros_like(early_fire_rates)
    # for i in range(steps):
    #     dynamic_rates[:, i] = d.get_dynamic_rates(early_fire_rates[:, i], late_fire_rates[:, i])
    #
    # font_size = 34
    #
    # plt.figure("Dynamic fire rates")
    # plt.plot(time, dynamic_rates.T, linewidth=2)
    # plt.xlabel('Time (s)', fontsize=font_size)
    # plt.ylabel('Spike Rate (spikes / s)', fontsize=font_size)
    # #
    # # plt.plot(time, early_fire_rates.T, label='Early fire rate', linewidth=2)
    # # plt.plot(time, late_fire_rates.T, label='Late fire rate', linewidth=2)
    #
    # plt.tick_params(axis='x', labelsize=font_size)
    # plt.tick_params(axis='y', labelsize=font_size)
    # plt.legend()
