import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


class LogNormalSizeProfile:
    def __init__(self, pol_tol):
        """
        Generate a lognormal size tuning profile for a neuron. Reference Ito-95.

        REQUIRED PARAMETERS:
        :param pol_tol: 2x the standard deviation of the RF spatial extent of the neuron
                        in radians of eccentricity. It comes from the position profile.
        """
        self.type = 'lognormal'

        # pol_tol is defined as 2x the standard deviation of the RF spatial extent of the
        # neuron. It comes from the position profile. Specifically from gaussianPositionProfile.
        self.max_pref_stim_size = 2 * pol_tol

        # Minimum stimulus size = 0.08 degrees, from Ito-95. Below this size, the object is
        # too small to allow proper recognition. Converted to radians.
        self.min_stim_size = 0.08 * np.pi / 180.0

        # Get parameters for the Lognormal distribution. Preferred size is limited to the max
        # and min supported size inside the function
        self.pref_size = self.__get_preferred_size()

        self.size_bw = self.__get_size_bandwidth()

        # Internal parameters - Needed for calculating the firing rate
        self.set_params(self.pref_size, self.size_bw)

    def set_params(self, pref_size, size_bw):
        """
        Updated the neuron to have specified preferred size and bandwidth. Because some internal
        parameters are derived from this parameters, do not change preferred size and bandwidth
        directly.

        :param pref_size:  new preferred size in radians.
        :param size_bw  :  new size bandwidth in octaves.
        """
        self.pref_size = pref_size
        self.size_bw = size_bw

        self.__log2_mu = np.log2(self.pref_size)
        self.__log2_sigma = (self.size_bw) / (2 * np.sqrt(2 * np.log(2)))
        # For conversion factor see https://en.wikipedia.org/wiki/Full_width_at_half_maximum

    def __get_preferred_size(self):
        """
        Generate a preferred (optimum) stimulus size for the neuron based on figure 6+7 of Ito 95.
        The preferred size follows distribution of figure 6 with a max value of max_stimulus_size
        and min value of min_stimulus size.

        Best fit for the preferred size distribution presented in the paper was found to be a
        lognormal distribution with shape=0.80 and scale = 5.40 and loc = 0 (In degrees).
        For details see logNormalFit.py

        :rtype : preferred stimulus size in radians of eccentricity.

        # THE LOG NORMAL FIT:
        Maximum stimulus size tested was 31x29 degrees. The distribution of optimum size (Fig. 6)
        states two peaks, at 3.4 and 27 degrees. We ignore the peak at 27 degrees as it may be an
        artifact of the testing. All neurons with RFs > 27 may potentially have RF sizes greater
        then the maximum stimulus size tested and their results may have been grouped in the
        27 degree bin.

        We assume a peak at 3.4 and a long tail ending at the maximum stimulus size.
        """
        preferred_size = np.float(ss.lognorm.rvs(s=0.80, scale=5.40, size=1)) * np.pi / 180.0

        preferred_size = np.max((preferred_size, self.min_stim_size))
        preferred_size = np.min((preferred_size, self.max_pref_stim_size))

        return preferred_size

    @staticmethod
    def __get_size_bandwidth():
        """
        Generate a size tolerance bandwidth for the neuron based on the distribution given in
        figure 2+7 of Ito 95.

        Best fit for the size bandwidth was found to be a log normal distribution with
        shape=0.30, scale=1.90 and loc=0. For details see logNormalFit.py.

        :rtype : size tolerance bandwidth of the neuron in octaves.

        For bandwidths > 4 octaves the upper cutoff was not determined and it is likely that
        the results of all larger bandwidths are grouped into the 4 bw bin. We ignore the data
        in the > 4 bandwidth bin and assume it follows the trend as in the lower bandwidths
        continues.
        """
        return np.float(ss.lognorm.rvs(s=0.30, scale=1.90, loc=0, size=1))

    def firing_rate_modifier(self, stimulus_size):
        """
        Returns the normalized firing rate of the neuron to stimulus of given size.
        Note that data fitting (hence the parameters) has been done on the log2 of the data.
        Therefore we need to take the log2 of the stimulus size and use a normal distribution,
        with mu = log2(preferred size) and sigma =  (self.size_bw / 2) / np.sqrt(2 * np.log(2)).

        :param stimulus_size: The distance in Radians between the outer edges along the longest
                              axis of the stimulus.
        :rtype             : Normalized firing rate.
        """

        zero_safe_guard = 0.0000001
        stimulus_size = np.maximum(stimulus_size, zero_safe_guard)

        fire_rate = np.exp(
            -(np.log2(stimulus_size) - self.__log2_mu)**2 / (2 * self.__log2_sigma ** 2))

        return fire_rate

    def print_parameters(self):
        print("Profile                      = %s" % self.type)
        print("Preferred Stimulus Size      = %0.4f (Radians)" % self.pref_size)
        print("Size bandwidth               = %0.4f octaves" % self.size_bw)

    def plot_size_tolerance(self, axis=None, font_size=34, print_parameters=True):

        x = np.linspace(0, self.max_pref_stim_size * 1.2, num=100)

        if axis is None:
            f, axis = plt.subplots()

        axis.plot(x, self.firing_rate_modifier(x), linewidth=2)
        axis.set_xlabel('Size', fontsize=font_size)
        axis.set_ylabel('FR (Spikes/s)', fontsize=font_size)
        # axis.set_title("Size Tolerance", fontsize=font_size + 10)

        if print_parameters:
            axis.annotate(r'$\sigma_{ST}=%0.2f$' % self.size_bw,
                          xy=(0.95, 0.85),
                          xycoords='axes fraction',
                          fontsize=font_size,
                          horizontalalignment='right',
                          verticalalignment='top')

            axis.annotate(r'$\mu_s=%0.2f$' % self.pref_size,
                          xy=(0.95, 0.75),
                          xycoords='axes fraction',
                          fontsize=font_size,
                          horizontalalignment='right',
                          verticalalignment='top')

        axis.tick_params(axis='x', labelsize=font_size)
        axis.tick_params(axis='y', labelsize=font_size)

        axis.grid()

if __name__ == "__main__":
    import pickle

    plt.ion()

    with open("Ito95Data.pkl", 'rb') as fid:
        data = pickle.load(fid)

    # Fit the neuron in Figure 3 of Ito et. al. -1995 - Size and Position Invariances of Neuronal
    # Responses in Monkey Inferotemporal Cortex. For chosen parameters see LogNormalFit.py
    n1 = LogNormalSizeProfile(pol_tol=22 / 180.0 * np.pi)
    n1.set_params(pref_size=5.2 * np.pi / 180, size_bw=1.35)

    fig, ax = plt.subplots(1, 1)
    n1.plot_size_tolerance(axis=ax)

    ax.scatter(data['n1Size'] * np.pi / 180,
               data['n1FiringRate'],
               marker='o', s=60, color='green', label='Original Data')

    ax.legend(fontsize=34)
