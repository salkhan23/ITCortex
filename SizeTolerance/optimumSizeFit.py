import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as ss


def get_lognormal_fit(size_arr, saturation_size, include_saturation_points=True):
    """
    Finds the ML estimates of a lognormal fit to input size_arr.

    To calculate ML estimates, by default (include_saturation_points=True) function uses the
    exact probability of points below cutoff (pdf)  and the probability of generating any point
    above cutoff for points above the cutoff P(x > cutoff)

    If include_saturation_points = True, it only considers points below cutoff.

    Parametrization of the lognormal distribution is different between scipy and general
    literature, General Literature uses parameters mu and sigma. In scipy, three parameters are
    used (1) s = sigma, (2) loc, (3) scale = np.exp(mu).

    This function returns the ML best fit s and scale parameters. It sets the loc parameter to
    zero.To convert back to regular lognormal parameters  use mu = np.log(scale) and sigma=s.

    :param include_saturation_points: Whether to include saturation points in ML estimate
                                      (Default=True)
    :param size_arr: Measured sizes to fit
    :param saturation_size: sizes in size_arr above cutoff will not be directly used, but will be
    accounted for as described above.

    :return: ML  fit of s (shape) and scale parameters
    """
    s_arr = np.arange(0.2, 20, step=0.1)
    scale_arr = np.arange(0.2, 20, step=0.1)

    regular_pnts = size_arr[size_arr < saturation_size]
    llrs = np.ones(shape=(s_arr.shape[0], scale_arr.shape[0])) * -10000

    for s_idx, s in enumerate(s_arr):
        for scale_idx, scale in enumerate(scale_arr):

            # pdf fo all regular points
            prob = ss.lognorm.pdf(regular_pnts, s, scale=scale, loc=0)
            llrs[s_idx][scale_idx] = np.log(prob).sum()

            # Calculate the probability of getting a point above the saturation. This is the
            # sum of probabilities of getting any point above saturation, and is equivalent to
            #  1 - cdf of the saturation point. scipy has a survival function which calculates
            # 1-cdf(x) which we use
            if include_saturation_points:
                above_sat_pnts = size_arr[size_arr >= saturation_size]
                prob_above_sat = ss.lognorm.sf(saturation_size, s=s, loc=0, scale=scale)
                prob = np.ones(shape=(len(above_sat_pnts))) * prob_above_sat

                llrs[s_idx][scale_idx] += np.log(prob).sum()

    # Find the best fit s and scale parameters
    max_s_idx, max_scale_idx = np.unravel_index(llrs.argmax(), llrs.shape)

    return s_arr[max_s_idx], scale_arr[max_scale_idx], llrs[max_s_idx][max_scale_idx]


def get_gamma_fit(size_arr, saturation_size, include_saturation_points=True):
    """

    :param size_arr:
    :param saturation_size:
    :param include_saturation_points:
    :return:
    """
    alpha_arr = np.arange(start=0.1, stop=10, step=0.1)
    scale_arr = np.arange(start=0.1, stop=20, step=0.1)

    llrs = np.ones(shape=(alpha_arr.shape[0], scale_arr.shape[0])) * -10000
    regular_pnts = size_arr[size_arr < saturation_size]

    for alpha_idx, alpha in enumerate(alpha_arr):
        for scale_idx, scale in enumerate(scale_arr):

            # pdf fo all regular points
            prob = ss.gamma.pdf(regular_pnts, a=alpha, scale=scale)
            llrs[alpha_idx][scale_idx] = np.log(prob).sum()

            # Calculate the probability of getting a point above the saturation. This is the
            # sum of probabilities of getting a point above saturation, this is the 1 - cdf of
            # the saturation point. scipy has a survival function which calculates 1-cdf(x) which
            # we use
            if include_saturation_points:
                above_sat_pnts = size_arr[size_arr >= saturation_size]
                prob_above_sat = ss.gamma.sf(saturation_size, a=alpha, scale=scale)
                prob = np.ones(shape=(len(above_sat_pnts))) * prob_above_sat

                llrs[alpha_idx][scale_idx] += np.log(prob).sum()

    # Find the best fit alpha and scale parameters
    max_alpha_idx, max_scale_idx = np.unravel_index(llrs.argmax(), llrs.shape)

    return alpha_arr[max_alpha_idx], scale_arr[max_scale_idx], llrs[max_alpha_idx][max_scale_idx]


def get_levy_fit(size_arr, saturation_size, include_saturation_points=True):
    """

    :param size_arr:
    :param saturation_size:
    :param include_saturation_points:
    :return:
    """
    loc_arr = np.arange(start=0.1, stop=10, step=0.1)
    scale_arr = np.arange(start=0.1, stop=20, step=0.1)

    llrs = np.ones(shape=(loc_arr.shape[0], scale_arr.shape[0])) * -10000
    regular_pnts = size_arr[size_arr < saturation_size]

    for loc_idx, loc in enumerate(loc_arr):
        for scale_idx, scale in enumerate(scale_arr):

            # pdf fo all regular points
            prob = ss.levy.pdf(regular_pnts, loc=loc, scale=scale)
            llrs[loc_idx][scale_idx] = np.log(prob).sum()

            # Calculate the probability of getting a point above the saturation. This is the
            # sum of probabilities of getting a point above saturation, this is the 1 - cdf of
            # the saturation point. scipy has a survival function which calculates 1-cdf(x) which
            # we use
            if include_saturation_points:
                above_sat_pnts = size_arr[size_arr >= saturation_size]
                prob_above_sat = ss.levy.sf(saturation_size, loc=loc, scale=scale)
                prob = np.ones(shape=(len(above_sat_pnts))) * prob_above_sat

                llrs[loc_idx][scale_idx] += np.log(prob).sum()

    # Find the best fit alpha and scale parameters
    max_loc_idx, max_scale_idx = np.unravel_index(llrs.argmax(), llrs.shape)

    return loc_arr[max_loc_idx], scale_arr[max_scale_idx], llrs[max_loc_idx][max_scale_idx]


def plot_histogram(size_arr, cut_off, axis=None):

    if axis is None:
        f, axis = plt.subplots()

    reg_pnts = size_arr[size_arr < cut_off]
    above_cutoff_points = size_arr[size_arr >= cut_off]

    result = axis.hist(
        [reg_pnts, above_cutoff_points],
        bins=np.arange(0, 31, step=2),
        normed=True,
        stacked=True,
        rwidth=1,
    )

    axis.set_xlim([0, 31])
    axis.set_xticks(np.arange(0, 31, step=2))

    # Hatch the above saturation bars - this is done manually. The last two bars correspond to
    # above saturation points
    # result[2] = patches, [1] index's the patches in the second histogram (not that this is the
    # complete histogram). Its integral (height*width) sums to 1. [-2:] extracts the last two bins
    above_sat_bins = result[2][1][-2:]
    for b in above_sat_bins:
        b.set_hatch('/')


if __name__ == "__main__":

    plt.ion()

    # Import the data
    with open("Ito95Data.pkl", 'rb') as fid:
        data = pickle.load(fid)

    opt_sizes = data["optSize"]
    opt_size_rf_size = data["optSizeRfSize"]

    # Ito's histogram of preferred sizes (Figure 6), identifies two types of 'incomplete'
    # measurements. We use a cutoff preferred size to separate preferred sizes we consider
    # precise from those that may not be as accurate.
    #
    # First, hashed neurons responded to the maximum size tested. Receptive field sizes of
    # these neurons were larger then the maximum size tested. It is likely that if larger stimulus
    # sizes had been used, preferred sizes of these neurons would be larger. We consider the
    # preferred sizes of these neurons as being not as precise and above the cutoff.
    #
    # Second, the larger half magnitude response points of cross hashed neurons was not
    # determined (it was larger than max size tested). These neurons were considered
    # large size bandwidth neurons. For these neurons size bandwidth was determined by doubling
    # the distance between the lower half magnitude point and the peak response.3 of these cells
    # are in the largest histogram bin. For these neurons, the difference between
    # preferred sizes and the max stimulus size is small. It is likely that the drop in the
    # neurons firing rate between the preferred size and the maximum stimulus size tested is small
    # and within measurement error. We consider these points to be above saturation as well.
    # Cross hashed neurons that were more than 1 bin away from the maximum stimulus size tested
    # were considered regular points. These 5 neurons are considered above saturation
    #
    # We manually determined a cutoff preferred size that separated these eight largest preferred
    # sizes. This was found to be 26 degrees.
    #
    # Ito states that two peak preferred sizes were found, at 3.4 and 27. Using our cutoff the
    # the peak at 27 is no longer visible.
    #
    # Note that we fit the raw preferred sizes in Figure 7B rather than the histogram of preferred
    # sizes in Figure 6 as they provide higher granularity.

    # When finding the best fit distribution we consider all measured preferred sizes. However,
    # we treat points below the cutoff and points above cutoff separately. We use maximum
    # log likelihood to find best fit distributions. For points below saturation we consider
    # the probability of generating the point exactly (pdf) while for points above saturation we
    # use the probability of generating any point above the cutoff (1 - cdf(cutoff))
    cutoff = 26.0

    plt.figure()
    x_arr = np.arange(30, step=0.5)

    # ---------------------------------------------------------------
    # Plot the histogram we try to fit
    # # ---------------------------------------------------------------
    plot_histogram(opt_sizes, cutoff)

    # ---------------------------------------------------------------
    # Best Fit Log normal Distribution
    # ---------------------------------------------------------------
    lognorm_s, lognorm_scale, lognorm_llr = get_lognormal_fit(opt_sizes, cutoff)

    label = 'Lognormal Fit (above saturation pnts included) s=%0.2f, scale=%0.2f, LLR=%0.2f' \
            % (lognorm_s, lognorm_scale, lognorm_llr)
    print label
    # TODO: Convert to regular log normal parameters mu and sigma, mu=log(scale), sigma=shape

    plt.plot(x_arr,
             ss.lognorm.pdf(x_arr, s=lognorm_s, loc=0, scale=lognorm_scale),
             linewidth=3,
             label=label,
             color='red',
             marker='+',
             markersize=10,
             markeredgewidth=2)

    #  Without saturation points --------------
    lognorm_s, lognorm_scale, lognorm_llr = \
        get_lognormal_fit(opt_sizes, cutoff, include_saturation_points=False)

    label = 'Lognormal Fit s=%0.2f, scale=%0.2f, LLR=%0.2f' \
            % (lognorm_s, lognorm_scale, lognorm_llr)
    print label
    # TODO: Convert to regular log normal parameters mu and sigma

    plt.plot(x_arr,
             ss.lognorm.pdf(x_arr, s=lognorm_s, loc=0, scale=lognorm_scale),
             linewidth=3,
             label=label,
             color='red',
             linestyle='--',
             marker='+',
             markersize=10,
             markeredgewidth=2)

    # ---------------------------------------------------------------
    # Best Fit Gamma Distribution
    # ---------------------------------------------------------------
    gamma_alpha, gamma_scale, gamma_llr = get_gamma_fit(opt_sizes, cutoff)

    label = 'Gamma Fit (above saturation pnts included) alpha=%0.2f, scale=%0.2f, LLR=%0.2f' \
            % (gamma_alpha, gamma_scale, gamma_llr)
    print label

    plt.plot(x_arr,
             ss.gamma.pdf(x_arr, a=gamma_alpha, scale=gamma_scale),
             linewidth=3,
             label=label,
             color='green',
             marker='o',
             markersize=10,

             markeredgewidth=2
             )

    gamma_alpha, gamma_scale, gamma_llr = \
        get_gamma_fit(opt_sizes, cutoff, include_saturation_points=False)

    label = 'Gamma Fit alpha=%0.2f, scale=%0.2f, LLR=%0.2f' \
            % (gamma_alpha, gamma_scale, gamma_llr)
    print label

    plt.plot(x_arr,
             ss.gamma.pdf(x_arr, a=gamma_alpha, scale=gamma_scale),
             linewidth=3,
             label=label,
             color='green',
             linestyle='--',
             marker='o',
             markersize=10,
             markeredgecolor='green',
             markerfacecolor='none',
             markeredgewidth=2,
             )

    # ---------------------------------------------------------------
    # Best Levy Fit Distribution
    # ---------------------------------------------------------------
    levy_loc, levy_scale, levy_llr = get_levy_fit(opt_sizes, cutoff)

    label = 'Levy Fit (above saturation pnts included) loc=%0.2f, scale=%0.2f, LLR=%0.2f' \
            % (levy_loc, levy_scale, levy_llr)
    print label

    plt.plot(x_arr,
             ss.levy.pdf(x_arr, loc=levy_loc, scale=levy_scale),
             linewidth=3,
             label=label,
             color='magenta',
             marker='^',
             markersize=10,
             markeredgecolor='magenta',
             markerfacecolor='none',
             markeredgewidth=2
             )

    levy_loc, levy_scale, levy_llr = \
        get_levy_fit(opt_sizes, cutoff, include_saturation_points=False)

    label = 'Levy Fit loc=%0.2f, scale=%0.2f, LLR=%0.2f' \
            % (levy_loc, levy_scale, levy_llr)
    print label

    plt.plot(x_arr,
             ss.levy.pdf(x_arr, loc=levy_loc, scale=levy_scale),
             linestyle='--',
             linewidth=3,
             label=label,
             color='magenta',
             marker='^',
             markersize=10,
             markeredgecolor='magenta',
             markerfacecolor='none',
             markeredgewidth=2,
             )

    # ----------------------------------------------------------------
    plt.legend()
    plt.title("Optimize Size Distributions", fontsize=20)
