# -*- coding: utf-8 -*-
"""
Ref:
[1] Neilson, Logothesis & Rainer - 2006 - Dissociation between Local Field Potentials & spiking
activity in Macaque Inferior Temporal Cortex reveals diagnosticity based encoding of complex
objects.

[2] Neilson, Phd Thesis - The Influences of Occlusion on Macaque Inferior Temporal Neurons -2005

[3] Kovacs, Vogels & Orban -1995 - Selectivity of Macaque Inferior Temporal Neurons for Partially
Occluded Shapes.

[4] Oreilly et. al. - 2013 - Recurrent processing during object recognition.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import warnings
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


def sigmoid(x, w, b):
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))


class TwoInputSigmoidOcclusionProfile:
    def __init__(self):
        """
        A two input sigmoid is used to model occlusion tolerances of IT Neurons.
        Parameters for this distribution are: x1=nondiagnostic visibility, x2=diagnostic
        visibility and a bias term.

        It is derived from [1, 2] where diagnostic and non-diagnostic parts visibilities are
        considered separately. Typically, in other papers diagnostic and nondiagnostic visibilities
        are not separated. However as noted by Neilson in [1] neurons respond preferentially to
        diagnostic parts.

        The level of preference for diagnostic parts was quantified by the ratio of variance
        between groups to the total variance across all trials when trials were grouped by whether
        diagnostic or nondiagnostic parts were visible, the variance explained by grouping
        diagnosticity. A diagnostic group variance of 1 means that the neuron was highly selective
        for diagnostic parts and did not respond to nondiagnostic parts (high between group
        variance). While a diagnostic group variance of 0 means that the neuron did not show any
        preference for diagnostic parts and weights along the diagnostic and nondiagnostic
        visibility axis were equal.

        Diagnostic Group Variance to Total Variance ratio

            d_to_t_ratio = Diagnostic Group Variance (V_group)
                           -----------------------------------
                           Total Variance (V_total)

            V_group = 1/2 [ (fdiag - f)^2 + (fnondiag - f)^2 ]


                where fdiag      = mean firing rate across all diagnostic trials
                      fnondiag   = mean firing rate across all non-diagnostic trails
                      f          = mean firing rate across all trials

            V_total = variance across all trials.

            See [1, 2] for details. Also think ANOVA.

        First we find a distribution for d_to_t_ratio. Distribution is from Figure 4 of [1].
        See diagnostic_group_variance_fit.py for details. At class instantiation, a d_to_t_ratio
        is picked from this distribution.

        Most occlusion tuning curves [3, 4] do not differentiate between diagnostic and
        nondiagnostic visibilities. We fit these tuning curves to a single input sigmoid and assume
        equal visibility levels for diagnostic and nondiagnostic parts. The weight (combined
        weight) and bias parameter of the best fit sigmoid are then found using LSE fitting.
        We find the mean and variance for combined weight and bias across all tuning curves found
        in the literature [1,3, 4]. We then assume a normal distribution for the combined weight
        and the bias term. See two_input_sigmoid_fit.py for details. We chose this method as there
        is insufficient data to generate typical diagnostic and nondiagnostic tuning curves in [1].

        Given a combined weight and bias term, we use non-linear optimization
        (scipy.optimize.fsolve) to determine separate diagnostic and non-diagnostic weights
        to generate the full 2D occlusion tuning curve.

        :return: 2 input occlusion tuning profile instance.
        """

        self.type = 'two_input_sigmoid'

        self.d_to_t_ratio = self._get_diagnostic_group_to_total_variance_ratio()

        if self.d_to_t_ratio > 1:
            warnings.warn("Invalid d_to_t ratio! %0.4f", self.d_to_t_ratio)

        self.w_combine, self.bias = self._get_combined_weight_and_bias()
        self.w_combine = np.float(self.w_combine)

        # Check if the chosen w_combine and bias terms can generate the desired diagnostic
        # group to total variances ratio. The maximum d_to_t var ratio for a chosen w_combine
        # and bias set may not be = 1, because a sigmoid is used to model the occlusion profile.
        # For details see two_input_sigmoid_fit.py - the bias term may always return a non zero
        # sigmoid output. maximum var ration when one term accounts for all of w_combine
        max_d_to_t_ratio = self._get_d_to_t_var_ratio(
            np.sqrt(2) * self.w_combine,
            self.w_combine,
            self.bias)

        if self.d_to_t_ratio > max_d_to_t_ratio:
            # modify w_c and bias such that it is possible to generate the desired d_to_t_ratio.
            # we are not to concerned about preserving the distribution of w_c and bias as there
            # wasn't any real distribution to model. However we are interested in preserving the
            # d_to_t_ratio distribution as seen in the population. Adjust w_combined and/or bias
            # to get the desired d_to_t_ratio.
            self.w_combine = self._adjust_w_combined_to_generate_d_to_t_ratio(
                self.w_combine,
                self.bias,
                self.d_to_t_ratio)

        # use nonlinear numerical optimization to solve for diagnostic weight that can generate
        # the desired d_to_t ratio
        w_diagnostic = so.fsolve(
            self._diff_between_desired_and_generated_d_to_t_ratio,
            (np.sqrt(2) * self.w_combine / 2),  # Initial guess half the combined weight
            args=(self.w_combine, self.bias, self.d_to_t_ratio),
            factor=0.5,  # w_d increases rapidly without this factor adjustment
        )
        w_diagnostic = np.float(w_diagnostic)

        # w_diagnostic, w_nondiagnostic and w_combined are related by wc = np.sqrt(2)*w_d - w_n
        # TODO: where is this relationship from.
        # w_diagnostic is always greater w_nondiagnostic
        w_nondiagnostic = (np.sqrt(2) * self.w_combine) - w_diagnostic

        if w_nondiagnostic > w_diagnostic:
            temp = w_nondiagnostic
            w_nondiagnostic = w_diagnostic
            w_diagnostic = temp

        # Store the diagnostic and non-diagnostic weights as a vector for easier computation
        # of fire rates.
        self.w_vector = np.array([[w_nondiagnostic], [w_diagnostic]])

        # Normalize the firing rate such that a fully visible object always returns a normalized
        # firing rate of 1 - at the maximum combined value
        self.scale = sigmoid(np.sqrt(2), self.w_combine, self.bias)

    @staticmethod
    def _get_diagnostic_group_to_total_variance_ratio():
        """
        Get a sample diagnostic to total ratio based on the distribution of these ratios found in
        Figure 4 of Neilson - 2006. For details on the fit please see file
        diagnostic_group_variance_fit.py.(exponentially distributed with a = 6.84)

        :return: diagnostic group to total group_ratio
        """

        # Use the inverse cdf method to get a sample ratio that follows the
        # distribution.
        y = np.random.uniform()

        # Inverse CDF of exponential.
        # Ref:
        # [1] http://www.ece.virginia.edu/mv/edu/prob/stat/random-number-generation.pdf
        # [2] SYSD 750 - Lecture Notes
        #
        # CDF = Fx(x) = y = 1-np.exp(-a*x)
        # ln(y-1) = -a*x, and y in uniformly distributed over 0, 1
        x = -np.log(1 - y) / 6.84

        return x

    @staticmethod
    def _get_combined_weight_and_bias():
        """
        Return a w_combined and bias pair. In file two_input_sigmoid_fit.py we take the average of
        all fitted weight_combined and bias terms for the fitted occlusion profiles and calculate a
        mean and a spread term  for these parameters. We then use a normal distribution to generate
        sample data. Fitting tuning profiles are from Kovacs - 1995, Neilson - 2006 & Oreilly 2013.
        See two_input_sigmoid_fit.py for more details on the chosen values.

        Here we use a normal distribution to generate typical combined weight and bias.

        :return: weight_combined, bias
        """
        w_c = np.float(np.random.normal(loc=5.5758, scale=1.7840))
        b = np.float(np.random.normal(loc=-3.2128, scale=1.3297))

        return w_c, b

    # noinspection PyTypeChecker
    @staticmethod
    def _get_d_to_t_var_ratio(w_d, w_c, b):
        """
        Given a weight for the combined axis and the diagnostic axis, find the
        diagnostic group variance to total variance ratio.

        :param w_c: weight_combined
        :param w_d: weight_diagnostic
        :param b: bias
        :return: diagnostic group to total variance ratio
        """
        w_n = (np.sqrt(2) * w_c) - w_d

        vis_arr = np.arange(1, step=0.05)
        vis_arr = np.reshape(vis_arr, (vis_arr.shape[0], 1))

        rates_n = sigmoid(vis_arr, w_n, b)
        rates_d = sigmoid(vis_arr, w_d, b)

        mean_n = np.mean(rates_n)
        mean_d = np.mean(rates_d)

        mean_overall = np.mean(np.append(rates_n, rates_d))
        var_overall = np.var(np.append(rates_n, rates_d))

        var_diagnostic_group = ((mean_n - mean_overall) ** 2 + (mean_d - mean_overall) ** 2) / 2

        ratio = var_diagnostic_group / var_overall

        # print("_get_d_to_t_var_ratio: w_d %0.4f, ratio %0.4f" % (w_d, ratio))

        return ratio

    # noinspection PyStringFormat
    def _adjust_w_combined_to_generate_d_to_t_ratio(self, w_c, b, desired_d_to_t_ratio):
        """
        Modify w_combined and bias terms to ensure the desired diagnostic group to total variance
        ratio is possible. As we do not have any good distribution for w_c or bias, okay to
        modify these parameters. However we should be able to generate all the required
        d_to_t ratios as we have good population level distribution.
        :param w_c: weight_combined
        :param b:   bias
        :param desired_d_to_t_ratio: target diagnostic group to total variance ratio

        :return: updated w_combined
        """
        generatable = False
        max_attempts = 100

        attempt = 0
        w_c_new = w_c
        print("Combined weight adjustment Start: Desired d_to_t ratio %0.4f, w_combined=%0.4f"
              % (desired_d_to_t_ratio, w_c))

        while not generatable:
            w_c_new = w_c_new + 0.5

            max_d_to_t_ratio = self._get_d_to_t_var_ratio(
                np.sqrt(2) * w_c_new,    # diagnostic weight
                w_c_new,                 # combine weight
                b)

            if max_d_to_t_ratio >= desired_d_to_t_ratio:
                generatable = True

            print("Attempt:%d: w_c %0.2f, b %0.2f, max ratio %0.2f, generatable %d"
                  % (attempt, w_c_new, b, max_d_to_t_ratio, generatable))

            attempt += 1

            if attempt == max_attempts:
                raise Exception("Unable to generate desired diagnostic group variance" +
                                "to total group variance ratio. Desired=%0.4f"
                                % desired_d_to_t_ratio)

        return w_c_new

    def _diff_between_desired_and_generated_d_to_t_ratio(self, w_d, w_c, b, desired_d_to_t_ratio):
        """
        Function to solve using nonlinear numerical optimization.
        :param w_d: diagnostic weight
        :param w_c: combined weight
        :param b: bias
        :param desired_d_to_t_ratio: desired diagnostic group to total variance ratio
        :return: difference between the generated d to to var ratio and desired.
        """
        return self._get_d_to_t_var_ratio(w_d, w_c, b) - desired_d_to_t_ratio

    def print_parameters(self):
        print("Profile                                      = %s" % self.type)
        print("diagnostic group to total variance ratio     = %0.4f" % self.d_to_t_ratio)
        print("weight combined                              = %0.4f" % self.w_combine)
        print("weight nondiagnostic                         = %0.4f" % self.w_vector[0])
        print("weight diagnostic                            = %0.4f" % self.w_vector[1])
        print("bias                                         = %0.4f" % self.bias)
        print("Scale Factor                                 = %0.4f" % self.scale)

    # noinspection PyTypeChecker
    def firing_rate_modifier(self, vis_nd, vis_d):
        """
         Get the normalized fire rate of the neuron based on the visibility levels provided.
         If vis_d = -1, fire rates along the combined axis is returned, where
         w_diagnostic = w_nondiagnostic.  Otherwise both visibilities are used to get the
         firing rate from the two input sigmoid model.

         Visibilities must either be a float or an ndarray.

        :param vis_nd   : visibility level(s) of nondiagnostic parts
        :param vis_d    : visibility level(s) of diagnostic parts

        :return         : normalized fire rate for each set of diagnostic and nondiagnostic
                          visibilities provided.
        """
        use_combined = False

        if isinstance(vis_d, np.ndarray):
            if vis_d[0] == -1:
                use_combined = True

        else:
            if vis_d == -1:
                use_combined = True

        if use_combined:
            # separate diagnostic and nondiagnostic visibilities are not available, use
            # the combined visibilities axis to get the firing rates.
            vis_nd = np.sqrt(2) * vis_nd  # On compressed scale, scale up to get true rate
            fire_rates = sigmoid(vis_nd, self.w_combine, self.bias) / self.scale

        else:
            x = np.array([vis_nd, vis_d])
            fire_rates = sigmoid(x.T, self.w_vector, self.bias) / self.scale
            fire_rates = np.reshape(fire_rates, fire_rates.shape[0])

        return fire_rates

    def plot_complete_profile(self, axis=None):
        font_size = 20

        if axis is None:
            f = plt.figure()
            axis = f.add_subplot(111, projection='3d')

        vis_arr = np.arange(0, 1, step=0.1)
        vis_arr = np.reshape(vis_arr, (vis_arr.shape[0], 1))

        fire_rates = np.zeros(shape=(vis_arr.shape[0], vis_arr.shape[0]))

        for r_idx in np.arange(vis_arr.shape[0]):
            for c_idx in np.arange(vis_arr.shape[0]):

                fire_rates[r_idx][c_idx] = \
                    self.firing_rate_modifier(vis_arr[r_idx], vis_arr[c_idx])

        yy, xx = np.meshgrid(vis_arr, vis_arr)
        axis.plot_wireframe(xx, yy, fire_rates)

        axis.set_xlabel("Nondiagnostic visibility", fontsize=font_size)
        axis.set_ylabel("Diagnostic Visibility", fontsize=font_size)
        axis.set_zlabel("Normalize fire rate (spikes/s)", fontsize=font_size)

        axis.set_xlim([1, 0])
        axis.set_ylim([0, 1])
        axis.set_zlim([0, 1])

        axis.tick_params(axis='x', labelsize=font_size)
        axis.tick_params(axis='y', labelsize=font_size)
        axis.tick_params(axis='z', labelsize=font_size)
        axis.set_title("Complete Tuning Curve", fontsize=font_size + 10)

        x2, y2, _ = proj3d.proj_transform(1.25, 0.15, 1.0, axis.get_proj())
        axis.annotate("w_n=%0.2f, w_d=%0.2f" % (self.w_vector[0], self.w_vector[1]),
                      xy=(x2, y2),
                      xytext=(-20, 20),
                      fontsize=font_size,
                      textcoords='offset points')

        x = np.arange(0, 1, step=0.1)
        z = np.zeros_like(x)
        axis.plot(x, x, z, label="Combined visibility axis", color='red', linewidth=2)

    def plot_combined_axis_tuning(self, axis=None):
        font_size = 20

        if axis is None:
            f, axis = plt.subplots()

        vis_levels = np.linspace(0, 1, num=100)
        axis.plot(vis_levels,
                  self.firing_rate_modifier(vis_levels, np.ones_like(vis_levels) * -1),
                  linewidth=2, label='Best fit sigmoid')

        axis.set_xlim([0, 1.1])
        axis.set_ylim([0, 1.1])
        axis.tick_params(axis='x', labelsize=font_size)
        axis.tick_params(axis='y', labelsize=font_size)
        axis.grid()

        axis.set_xlabel("Visibility Combined", fontsize=font_size)
        axis.set_ylabel("Normalized fire rate (spikes/s)", fontsize=font_size)
        axis.set_title("Tuning along equal visibilities axis",
                       fontsize=font_size + 10)

        axis.annotate('w_c=%0.2f, bias=%0.2f' % (self.w_combine, self.bias),
                      xy=(0.40, 0.95),
                      xycoords='axes fraction',
                      fontsize=font_size,
                      horizontalalignment='right',
                      verticalalignment='top')

if __name__ == "__main__":
    plt.ion()

    profile = TwoInputSigmoidOcclusionProfile()
    profile.print_parameters()
    profile.plot_complete_profile()
    profile.plot_combined_axis_tuning()

    # Test fire rates
    # (1) single only combined visibilities available
    visibility_nd = 0.5
    visibility_d = -1
    rates = profile.firing_rate_modifier(visibility_nd, visibility_d)

    print("Visibility levels (n, d) = (%0.2f,%0.2f). Fire Rate=%0.2f"
          % (visibility_nd, visibility_d, rates))

    # (2) Vector only combined visibilities
    visibility_nd = np.array([0.3, 0.7])
    visibility_d = np.array([-1, -1])
    rates = profile.firing_rate_modifier(visibility_nd, visibility_d)
    print("Visibility levels (n, d) = (%s, %s). Fire Rate=%s"
          % (visibility_nd, visibility_d, rates))

    # (3) single separate visibilities available
    visibility_nd = 0.1
    visibility_d = 0.4
    rates = profile.firing_rate_modifier(visibility_nd, visibility_d)

    print("Visibility levels (n, d) = (%0.2f,%0.2f). Fire Rate=%0.2f"
          % (visibility_nd, visibility_d, rates))

    # (4) Vector separate visibilities available
    visibility_nd = np.array([0.3, 0.7])
    visibility_d = np.array([0.4, 0.3])
    rates = profile.firing_rate_modifier(visibility_nd, visibility_d)
    print("Visibility levels (n, d) = (%s, %s). Fire Rate=%s"
          % (visibility_nd, visibility_d, rates))
