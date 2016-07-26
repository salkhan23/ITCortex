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
        ranked_objs_list = self.get_ranked_object_list(self.late_obj_dict)

        self.late_transient_gain, self.late_sustained_gain = self._get_late_gains(
            ranked_objs_list[0][1], max_fire_rate)

        # setup the C parameters
        self.early_C = np.array([self.early_gain, -self.early_gain])
        # self.late_C = late_gain
        self.late_C = np.array(
            [self.late_sustained_gain + self.late_transient_gain, - self.late_transient_gain])

        # Latencies of the model
        # parameters of exponential functions to map static rate to latency for each neuron
        self.min_latencies = .09 + .01 * np.random.rand(self.n)
        self.max_latencies = np.minimum(max_latency,
                                        self.min_latencies + np.random.gamma(5, .02, self.n))

        self.tau_latencies = np.random.gamma(2, 20, self.n)

        # matrices for storing recent input history, to allow variable-latency responses
        self.late_additional_latency = 30
        latency_steps = max_latency / dt + 1 + self.late_additional_latency
        self.early_memory = np.zeros((self.n, latency_steps))
        self.late_memory = np.zeros((self.n, latency_steps))
        self.memory_index = 0

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
        # plt.plot(time_arr, lti_out_arr, linestyle='--')

        # Integrate the output of the LTI system to get early_gain
        area_lti = np.trapz(lti_out_arr, dx=self.dt)
        area_desired = np.trapz(input_arr, dx=self.dt)

        early_gain = area_desired / area_lti
        # print("Early_gain", early_gain)

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

    def _get_late_gains(self, r_obj, max_fr):
        """

        :param r_obj: late object preference
        :param max_fr:
        :return:
        """

        # First get the Ratio of late gains
        ratio = self._get_late_gains_ratio()

        # Use the same technique as done to find early gain

        # Recall that the late LTE system uses two gains. These are needed in the calculation of
        # late_c. However, we do not have actual values for them for now just set transient gain
        # to  1 and the sustained gain to 1 / ratio
        transient_gain = 1.0
        sustained_gain = 1.0 / ratio

        avg_rate = r_obj * max_fr

        time_arr = np.arange(0, 0.140, step=self.dt)
        input_arr = np.ones_like(time_arr) * avg_rate

        late_a = 1 / self.late_tau * self.late_A
        late_b = 1 / self.late_tau * self.late_B

        # We have to determine this, for now just use a default value
        late_c = np.array([sustained_gain + transient_gain, - transient_gain])

        lti_out_arr = np.zeros_like(input_arr)
        for u_idx, u in enumerate(input_arr):
            self.late_x[:, 0], y = integrate(
                self.dt,
                late_a,
                late_b,
                late_c,
                self.late_x[:, 0],
                u
            )

            lti_out_arr[u_idx] = np.maximum(0, y)

        # clear_out the system
        self.late_x[:, 0] = 0
        # plt.plot(time_arr, lti_out_arr, linestyle='--')

        # Integrate the output of the LTI system to get early_gain
        area_lti = np.trapz(lti_out_arr, dx=self.dt)
        area_desired = np.trapz(input_arr, dx=self.dt)

        transient_gain = area_desired / area_lti
        sustained_gain = transient_gain / ratio
        # print("transient_gain %0.4f, sustain gain %0.4f"  %(transient_gain, sustained_gain))

        # # Debug make sure everything is making sense
        # lti_out_arr = np.zeros_like(input_arr)
        #
        # late_c = np.array([sustained_gain + transient_gain, - transient_gain])
        #
        # for u_idx, u in enumerate(input_arr):
        #     self.late_x[:, 0], y = integrate(
        #         self.dt,
        #         late_a,
        #         late_b,
        #         late_c,
        #         self.late_x[:, 0],
        #         u
        #     )
        #
        #     lti_out_arr[u_idx] = np.maximum(0, y)
        #
        # # clear_out the system
        # self.late_x[:, 0] = 0
        # plt.plot(time_arr, lti_out_arr)
        #
        # print("Area Under LTI system %0.4f" % np.trapz(lti_out_arr, dx=self.dt))
        # print("Desired area %0.4f" % area_desired)

        return transient_gain, sustained_gain

    @staticmethod
    def _get_late_gains_ratio():
        """
        This is the ratio of transient late gain to sustained late gain. It is extracted from
        fitting various dynamic profiles in the Tamura paper

        We found the mean and standard deviation from the data and use a normal distribution
        to model the ratio

        mean = 1.675
        standard deviation = 5.723

        :return:
        """
        return np.random.normal(loc=1.675, scale=5.723)

    def _get_lagged_rates(self, latencies):
        # get appropriately lagged rates for input to LTI dynamics
        latency_steps = np.rint(latencies / self.dt).astype('int')

        early_indices = self._get_index(latency_steps)
        late_indices = self._get_index(latency_steps + self.late_additional_latency)

        # early_u = self.early_memory[range(self.n), early_indices][0]
        # late_u = self.late_memory[range(self.n), late_indices][0]
        early_u = self.early_memory[range(self.n), early_indices]
        late_u = self.late_memory[range(self.n), late_indices]

        return early_u, late_u

    def _get_index(self, latency_steps):
        index = self.memory_index - np.minimum(latency_steps, self.early_memory.shape[1])
        index[index < 0] = index[index < 0] + self.early_memory.shape[1]
        return index

    def _get_latencies(self, static_rates):
        # latency varies with response strength
        return self.min_latencies + \
            (self.max_latencies - self.min_latencies) * np.exp(
                -static_rates / self.tau_latencies)

    def _step_dynamics(self, early_u, late_u):
        # run a single step of the LTI dynamics for each neuron
        y = np.zeros(self.n)

        for ii in range(self.n):
            early_a = 1 / self.early_tau * self.early_A
            early_b = 1 / self.early_tau * self.early_B
            self.early_x[:, ii], early_y = integrate(
                self.dt,
                early_a,
                early_b,
                self.early_C,
                self.early_x[:, ii],
                early_u[ii])

            late_a = 1 / self.late_tau * self.late_A
            late_b = 1 / self.late_tau * self.late_B

            self.late_x[:, ii], late_y = integrate(
                self.dt,
                late_a,
                late_b,
                self.late_C,
                self.late_x[:, ii],
                late_u[ii])

            y[ii] = np.maximum(0, early_y) + np.maximum(0, late_y)

        return y

    @staticmethod
    def get_ranked_object_list(obj_dict):
        """ Return neurons rank list of objects and rate modification factors """
        return sorted(obj_dict.items(), key=lambda item: item[1], reverse=True)

    def print_parameters(self):
        """ Print parameters of the profile """
        print("Profile                              = %s" % self.type)
        print("transience                           = %0.2f" % self.transience)

        print("Early tau                            = %0.2f" % self.early_tau)
        print("Early gain                           = %0.2f" % self.early_gain)

        print("Late tau                             = %0.2f" % self.late_tau)
        print("Late transient gain                  = %0.2f" % self.late_transient_gain)
        print("Late sustained gain                  = %0.2f" % self.late_sustained_gain)

    def get_dynamic_rates(self, early_rates, late_rates):
        """
        :param early_rates: Static rates for versions of the neurons with early selectivity
        :param late_rates: Static rates for versions of the neurons with late selectivity
        :return: Spike rates with latency and early and late dynamics
        """

        self.early_memory[:, self.memory_index] = early_rates
        self.late_memory[:, self.memory_index] = late_rates

        latencies = self._get_latencies(early_rates)

        early_u, late_u = self._get_lagged_rates(latencies)

        self.memory_index += 1
        if self.memory_index == self.early_memory.shape[1]:
            self.memory_index = 0

        return self._step_dynamics(early_u, late_u)


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

    # run some neurons with square-pulse input ...
    steps = 200
    early_fire_rates = np.zeros((1, steps))
    early_fire_rates[:, 20:100] = 50
    late_fire_rates = early_fire_rates / 2
    early_fire_rates = early_fire_rates * np.random.rand(1, 1)
    late_fire_rates = late_fire_rates * np.random.rand(1, 1)

    time = time_step * np.array(range(steps))
    dynamic_rates = np.zeros_like(early_fire_rates)
    for i in range(steps):
        dynamic_rates[:, i] = d.get_dynamic_rates(early_fire_rates[:, i], late_fire_rates[:, i])

    font_size = 34

    plt.figure("Dynamic fire rates")
    plt.plot(time, dynamic_rates.T, linewidth=2)
    plt.xlabel('Time (s)', fontsize=font_size)
    plt.ylabel('Spike Rate (spikes / s)', fontsize=font_size)
    #
    # plt.plot(time, early_fire_rates.T, label='Early fire rate', linewidth=2)
    # plt.plot(time, late_fire_rates.T, label='Late fire rate', linewidth=2)

    plt.tick_params(axis='x', labelsize=font_size)
    plt.tick_params(axis='y', labelsize=font_size)
    plt.legend()
