import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


class LogNormalSizeProfile:
    def __init__(self, pos_tol_deg, deg2pixel=1):
        self.params = {}
        self.type = 'Lognormal'

        # TODO: Check required parameters

        # pos_tol_deg is defined as 2x the standard deviation of the RF spatial extent of the
        # neuron. it comes from the position profile. Specifically from gaussianPositionProfile

        # Convert from pos_tol_deg to rfSize as defined in Ito 95 paper: Square root of the area
        # extent of the neuron.
        #   pos_tol_deg = 2 * r
        #   rf_size     = np.sqrt(pi) * r = pos_tol_deg / 2 * np.sqrt(pi)
        self.params['rf_size'] = pos_tol_deg / 2 * np.sqrt(np.pi)

        # Given a receptive field size, get max. stimulus size supported.
        #   rf_size       = area extent of RF, if assume circular = np.sqrt(np.pi) * r
        #   stim_size     = distance between the the outer edges along the longest axis
        #                   of the stimulus. If we assume a circular stimulus = 2r
        self.params['max_stim_size'] = self.params['rf_size'] / np.sqrt(np.pi) * 2

        # Minimum stimulus size = 0.08 degrees, from Ito-95. Below this size, the object is
        # too small to allow proper recognition.
        self.params['min_stim_size'] = 0.08

        # Get parameters for the Lognormal distribution.
        self.params['pref_size'] = self.__get_preferred_size()
        self.params['size_bw'] = self.__get_size_bandwidth()

    def __get_preferred_size(self):
        """
        Generate a preferred (optimum) stimulus size for the neuron based on figure 6+7 of Ito 95.
        The preferred size follows distribution of figure 6 with a max value of max_stimulus_size
        and min value of min_stimulus size.

        Best fit for the preferred size distribution presented in the paper was found to be a
        lognormal distribution with shape=0.80 and scale = 5.40 and loc = 0. For details see
        logNormalFit.py.

        :rtype : preferred stimulus size in degrees.

        # THE LOG NORMAL FIT:
        Maximum stimulus size tested was 31x29 degrees. The distribution of optimum size (Fig. 6)
        states two peaks, at 3.4 and 27 degrees. We ignore the peak at 27 degrees as it may be an
        artifact of the testing. All neurons with RFs > 27 may potentially have RF sizes greater
        then the maximum stimulus size tested and their results may have been grouped in the
        27 degree bin.

        We assume a peak at 3.4 and a long tail ending at the maximum stimulus size.
        """
        preferred_size = ss.lognorm.rvs(s=0.80, scale=5.40, size=1)

        preferred_size = np.max((preferred_size, self.params['min_stim_size']))
        preferred_size = np.min((preferred_size, self.params['max_stim_size']))

        return preferred_size

    def __get_size_bandwidth(self):
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
        return ss.lognorm.rvs(s=0.30, scale=1.90, loc=0, size=1)

    def firing_rate_modifier(self, stimulus_size, deg2pixel= 1):
        """
        Returns the normalized firing rate of the neuron to stimulus of given size.

        :param stimulus_size: The distance in pixels between the outer edges along the longest
                              axis of the stimulus
        :param deg2pixel    : Degree to pixel value conversion factor
        :return             : Normalized firing rate
        """
        log_mu = np.log2(self.params['pref_size'])
        log_sigma = (self.params['size_bw'] / 2) / np.sqrt(2 * np.log(2))

        # Convert to stimulus size to degrees from pixels
        stimulus_size_deg = stimulus_size / float(deg2pixel)

        # Neuron does not respond to stimuli outsize its maximum supported size. This allows
        # the different type of tuning profiles as described in the paper.(1) <2 octaves log normal
        # (2) >5 octaves bandwidth and (3) responds only to the largest stimulus size.
        mask_upper = stimulus_size_deg <= self.params['max_stim_size']
        mask_lower = stimulus_size_deg > self.params['min_stim_size']

        fire_rate = np.exp(-(np.log2(stimulus_size_deg) - log_mu) ** 2 / (2 * log_sigma ** 2))

        return fire_rate * mask_upper * mask_lower

    def print_parameters(self):
        print("Profile: %s" % self.type)
        print("Valid stimulus size range: (%0.2f, %0.2f) degrees"
              % (self.params['min_stim_size'], self.params['max_stim_size']))
        print("Preferred Stimulus Size: %0.2f degrees" % self.params['min_stim_size'])
        print("Size bandwidth: %0.2f octaves" % self.params['size_bw'])
        print ("RF Size (area extent of receptive field): %0.2f" % self.params['rf_size'])


if __name__ == "__main__":
    plt.ion()
    n1 = LogNormalSizeProfile(pos_tol_deg=15)
    n1.print_parameters()

    size_arr = np.arange(20, step=0.1)
    plt.plot(size_arr, n1.firing_rate_modifier(size_arr))

    print n1.firing_rate_modifier(3)
