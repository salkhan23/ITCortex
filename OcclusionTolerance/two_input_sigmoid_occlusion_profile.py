
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm


def sigmoid(x, w, b):
    """
    :param b:
    :param w:
    :param x:
    :rtype: object
    """
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))


class TwoInputSigmoidOcclusionProfile:

    def __init__(self, d_to_t_ratio=None, w_c=None, b=None):
        """
        A two input sigmoid is used to model occlusion tolerances of IT Neurons.

        Parameters of this model are wd, w_nd, and b. Weights associated with diagnostic parts,
        nondiagnostic parts and a biasing term, respectively. These are chosen randomly based on
        fits to data found in the literature.

        It is derived from [1, 2] where diagnostic and non-diagnostic parts visibilities are
        considered separately. Typically, in other papers diagnostic and nondiagnostic visibilities
        are not separated. However as noted by Neilson in [1] neurons respond preferentially to
        diagnostic parts. Distributions of wd and wn are hard to find and we derive these from
        distributions of occlusion tuning profiles where a single measure of visibilities is used.
        We assume in these tuning curves, diagnostic and nondiagnostic visibilities are equal. We
        then use scipy.optimize.fsolve (Non linear optimization) to determined a set of w_d and
        w_nd that gives the desired d_to_t ratio.

        The level of preference for diagnostic parts is quantified by the ratio of variance
        between groups to the total variance across all trials. Here, trials are grouped according
        to which parts are visible, the variance explained by grouping by diagnosticity.

        A diagnostic group variance of 1 means that the neuron is highly selective for diagnostic
        parts and did not respond to nondiagnostic parts (high between group variance). While a
        diagnostic group variance of 0 indicates that the neuron does not show any preference for
        diagnostic parts and responds to all parts equally.

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
        is picked from this distribution, unless a d_to_t_ratio is specified.

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

        :param d_to_t_ratio:
        :return:
        """

        self.type = 'two_input_sigmoid'

        # Desired diagnostic to total variance ratio (R)
        if d_to_t_ratio is None:
            self.ratio = self._get_diagnostic_group_to_total_variance_ratio()
        elif 0 <= d_to_t_ratio <= 1:
            self.ratio = d_to_t_ratio
        else:
            raise Exception("Invalid diagnostic to total variance ratio specified %0.4f"
                            % d_to_t_ratio)

        generatable = False
        iteration = 0
        max_iterations = 100
        w_d = 0
        w_nd = 0

        while not generatable:

            # Get weight along combined visibilities axis
            if w_c is None and b is None:
                self.w_combine, self.bias = self._get_combined_weight_and_bias()
            else :
                self.w_combine = w_c
                self.bias = b
                # only one iteration for the case when w_combined and bias are defined
                iteration = max_iterations

            # Use nonlinear optimization to find w_diagnostic and w_nondiagnostic that can generate
            # the desired diagnostic to total variance ratio.
            w_d, w_nd = so.fsolve(
                self.optimization_equations,
                (self.w_combine / 2, self.w_combine / 2),
                args=(self.w_combine, self.bias, self.ratio),
                factor=0.5,  # w_d increases rapidly without this factor adjustment)
            )

            # w_d should always be greater than w_nd
            if w_d < w_nd:
                temp = w_d
                w_d = w_nd
                w_nd = temp

            if w_d < 0 or w_nd < 0:

                iteration = iteration + 1

                # print("Desired R %0.4f, Curr R %0.4f, w_c %0.4f, b %0.4f, w_d %0.4f, w_n %0.4d"
                #       % (self.ratio,
                #          self.calculate_ratio(w_d, w_nd, self.bias),
                #          self.w_combine,
                #          self.bias,
                #          w_d,
                #          w_nd,
                #          ))

                if iteration >= max_iterations:
                    raise Exception("Unable to generate desired diagnostic group variance" +
                                    "to total group variance ratio. Desired=%0.4f" % self.ratio)

            else:
                generatable = True

        # Store the diagnostic and non-diagnostic weights as a vector for easier computation
        # of fire rates.
        self.w_vector = np.array([[w_nd], [w_d]])

        # Normalize the firing rate such that a fully visible object always returns a normalized
        # firing rate of 1 - at the maximum combined value
        self.scale = sigmoid(1, self.w_combine, self.bias)

    @staticmethod
    def _get_diagnostic_group_to_total_variance_ratio():
        """
        Get a sample diagnostic to total ratio (R). Distribution of R was extracted from figure
        4 of [Neilson 2006]. These were then fit to an exponential distribution. For details see
        file diagnostic_group_variance_fit.py.

        Randomly pick a value from this distribution and use the inverse CDF to find R.

        Inverse CDF of exponential.
            CDF = y = 1-np.exp(-a*x)
            ln(y-1) = -a*x, and y in uniformly distributed over 0, 1

        References:
        [1] http://www.ece.virginia.edu/mv/edu/prob/stat/random-number-generation.pdf
        [2] SYSD 750 - Lecture Notes
        :return: diagnostic group to total group_ratio
        """
        y = np.float(np.random.uniform())
        x = -np.log(1 - y) / 6.84

        # If y is close to 1 (0.999), this results in x>1, since this is a probability
        # restrict to max value of 1.
        return min(x, 1.0)

    @staticmethod
    def _get_combined_weight_and_bias():
        """
        Return a w_combined and bias pair. Here w_combined is the weight on the combined
        visibilities axis where equal parts diagnostic and nondiagnostic visibilities are assumed.

        In file two_input_sigmoid_fit.py occlusion tuning  profiles from several references
        were fit to a single input sigma with parameters w_combined and bias. After finding
        mean and sigma of these, we use independent normal distributions to generate sample values.

        See two_input_sigmoid_fit.py for more details.

        :return: weight_combined, bias
        """
        w_c = np.float(np.random.normal(loc=5.5758, scale=1.7840, size=1))
        b = np.float(np.random.normal(loc=-3.2128, scale=1.3297, size=1))

        return w_c, b

    @staticmethod
    def calculate_ratio(w_d, w_nd, b, step_size=0.05):
        """
        Calculate the ratio of how much of the total variance is explained by the variance between
        diagnostic/non-diagnostic grouping

        :param w_d:
        :param w_nd:
        :param b:
        :param step_size:
        :return:
        """

        vis_arr = np.arange(1, step=step_size)
        vis_arr = vis_arr.reshape((vis_arr.shape[0], 1))

        rates_n = sigmoid(vis_arr, w_nd, b)
        rates_d = sigmoid(vis_arr, w_d, b)

        mean_n = np.mean(rates_n)
        mean_d = np.mean(rates_d)

        mean_t = np.mean(np.append(rates_n, rates_d))
        sigma_t = np.var(np.append(rates_n, rates_d))

        sigma_b = ((mean_n - mean_t) ** 2 + (mean_d - mean_t) ** 2) / 2

        ratio = sigma_b / sigma_t

        return ratio

    def optimization_equations(self, w, w_c, b, desired_ratio):
        """
        Function(s) to solve using nonlinear numerical optimization.

        (1) Desired ratio - Actual ratio = 0
        (2) the sum of the weights cannot be more than weight on the combined axis.

        :param w            : tuple of (w_d, w_nd). What we find optimum values for.
        :param w_c          : weight on combined axis
        :param b            : bias
        :param desired_ratio: desired diagnostic to total variance ratio

        :return: tuple (Desired ratio - Actual ratio, w_c - (w_d + w_nd)
        """
        w_d, w_nd = w
        return desired_ratio - self.calculate_ratio(w_d, w_nd, b), w_c - w_d - w_nd

    def print_parameters(self):
        print("Profile                                      = %s" % self.type)
        print("diagnostic group to total variance ratio     = %0.4f" % self.ratio)
        print("weight combined                              = %0.4f" % self.w_combine)
        print("weight nondiagnostic                         = %0.4f" % self.w_vector[0])
        print("weight diagnostic                            = %0.4f" % self.w_vector[1])
        print("bias                                         = %0.4f" % self.bias)
        print("Scale Factor                                 = %0.4f" % self.scale)

    def firing_rate_modifier(self, vis_nd, vis_d):
        """
         Get the normalized fire rate of the neuron based on the visibility levels provided.
         If vis_d = -1, fire rates along the combined axis is returned. Otherwise both
         visibilities are used to get the firing rate from the two input sigmoid model.

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
            fire_rates = sigmoid(vis_nd, self.w_combine, self.bias) / self.scale

        else:
            x = np.array([vis_nd, vis_d])
            fire_rates = sigmoid(x.T, self.w_vector, self.bias) / self.scale
            fire_rates = np.reshape(fire_rates, fire_rates.shape[0])

        return fire_rates

    def plot_complete_profile(self, axis=None, font_size=20):
        """

        :param font_size:   Font size on graph. Default = 20
        :param axis     :   Plotting axis. Default = None, will create a new figure.
                            Axis must be an axis with a projection of type 3d
        """

        if axis is None:
            f = plt.figure()
            axis = f.add_subplot(111, projection='3d')

        vis_arr = np.arange(0, 1.1, step=0.1)
        vis_arr = np.reshape(vis_arr, (vis_arr.shape[0], 1))

        fire_rates = np.zeros(shape=(vis_arr.shape[0], vis_arr.shape[0]))

        for r_idx in np.arange(vis_arr.shape[0]):
            for c_idx in np.arange(vis_arr.shape[0]):

                fire_rates[r_idx][c_idx] = \
                    self.firing_rate_modifier(vis_arr[r_idx], vis_arr[c_idx])

        yy, xx = np.meshgrid(vis_arr, vis_arr)
        axis.plot_surface(xx, yy, fire_rates, cmap=cm.coolwarm, rstride=1, cstride=1)

        axis.set_xlim([1, 0])
        axis.set_ylim([0, 1])
        axis.set_zlim([0, 1])

        axis.tick_params(axis='x', labelsize=font_size)
        axis.tick_params(axis='y', labelsize=font_size)
        axis.tick_params(axis='z', labelsize=font_size)

        axis.set_xlabel("\n" + r"$v_{nd}$", fontsize=font_size + 10, fontweight='bold', color='b')
        axis.set_ylabel("\n" + r"$v_d$", fontsize=font_size + 10, fontweight='bold', color='b')
        axis.set_zlabel("FR (spikes/s)", fontsize=font_size)
        # axis.set_title("Complete Tuning Curve", fontsize=font_size + 10)

        label = r"$w_n=%0.2f,$" % self.w_vector[0] + "\n" \
                + r'$w_{nd}=%0.2f$' % self.w_vector[1] + '\n' \
                + r'$R=%0.2f$' % self.ratio

        axis.text(1.3, 0, 0.5, label, fontsize=font_size)

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

    # Make several profiles and check that optimization is working. Doesnt break the model.
    # for i in np.arange(1000):
    #     profile = TwoInputSigmoidOcclusionProfile()
    #     print profile.ratio - \
    #           profile.calculate_ratio(profile.w_vector[1], profile.w_vector[0],profile.bias)

    # Check plots are working fine ----------------------------------------------------------
    profile = TwoInputSigmoidOcclusionProfile()
    profile.print_parameters()

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    profile.plot_complete_profile(axis=ax1)

    ax2 = fig.add_subplot(1, 2, 2)
    profile.plot_combined_axis_tuning(axis=ax2)

    # Test fire rates -----------------------------------------------------------------------
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

    # Check desired ratio can be input -----------------------------------------------------
    profile = TwoInputSigmoidOcclusionProfile(1)
    profile.print_parameters()
    profile.plot_complete_profile()
    print profile.firing_rate_modifier(1, -1)
