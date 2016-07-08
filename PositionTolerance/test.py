import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pickle


def best_gamma_fit_mean_from_regression_fit(s_arr, pt_arr, lfit_coeff):
    """

    Find the best fit gamma alpha parameter, subject to the constraint that the mean of the
    distribution is given by the linear regression fit linear_fit_coeff[0]x + linear_fit_coeff[0]

    :param s_arr: array of measured selectivity (activity fraction)
    :param pt_arr:  array of measured position tolerances at the corresponding selectivity.
    :param lfit_coeff: list/array of linear regression fit. Highest order first.

    :return:
    """
    alpha_arr = np.arange(0.1, 20, step=0.01)
    llv = []  # Log likelihood values

    for alpha in alpha_arr:
        prob = [ss.gamma.pdf(pt, a=alpha,
                             scale=(lfit_coeff[0] * s_arr[pt_idx] + lfit_coeff[1]) / alpha)
                for pt_idx, pt in enumerate(pt_arr)]
        llv.append(np.log(prob).sum())

    llv = np.array(llv)

    # print alpha_arr[llv.argmax()], llv[llv.argmax()]
    # print max(llv)

    return alpha_arr[llv.argmax()]


def separate_points_into_bins(x_arr, y_arr, bin_arr):

    separated_points = []

    # Create a separate list for each bin
    for _ in bin_arr:
        separated_points.append([])

    # Find bin indices
    pos_arr = np.digitize(x_arr, bin_arr)

    # Assign x and y to the correct bins
    for x_idx, pos in enumerate(pos_arr):
        separated_points[pos].append((x_arr[x_idx], y_arr[x_idx]))

    return separated_points[1:]


if __name__ == '__main__':
    plt.ion()

    # Load the data
    with open("Zoccolan2007.pkl") as handle:
        data = pickle.load(handle)

    # ----------------------------------------------------------
    # Full Data Processing
    # ----------------------------------------------------------

    # Find the best fit alpha, subject to the constraint that mean of the distribution is given by
    # the linear regression fit
    ml_gamma_full = best_gamma_fit_mean_from_regression_fit(
        data["xScatterRaw"], data["yScatterRaw"], data["yLineFit"])
    print("Full: Best Gamma Fit, alpha=%0.2f, scale=mean_position_tolerance / alpha"
          % ml_gamma_full)

    # ----------------------------------------------------------
    # Partition Data into subsets
    # ----------------------------------------------------------
    bins = np.linspace(0, 1, num=4)
    partitioned_lists = separate_points_into_bins(data["xScatterRaw"], data["yScatterRaw"], bins)

    f_raw, ax_arr_raw = plt.subplots()
    f_diff, ax_arr_diff = plt.subplots()

    for bin_idx, bin_list in enumerate(partitioned_lists):

        x_val = np.array([x[0] for x in bin_list])
        y_val = np.array([x[1] for x in bin_list])

        # Get the difference between raw position tolerance and the mean position tolerance at
        # that selectivity.
        mean_y_val = data['yLineFit'][0] * x_val + data['yLineFit'][1]
        y_diff = y_val - mean_y_val

        ax_arr_raw.plot(x_val, y_val, marker='o', markersize=10, linestyle='')
        ax_arr_diff.plot(x_val, y_diff, marker='o', markersize=10, linestyle='')

        # Find the best fit alpha, subject to the constraint that mean of the distribution is
        # given by the linear regression fit
        ml_gamma_bin = best_gamma_fit_mean_from_regression_fit(x_val, y_val, data["yLineFit"])
        print("Bin < %0.3f: Best Gamma Fit, alpha=%0.2f, scale=mean_position_tolerance / alpha"
              % (bins[bin_idx + 1], ml_gamma_bin))

    ax_arr_raw.set_xlim([0, 1])
    ax_arr_raw.set_xlabel("Selectivity")
    ax_arr_raw.set_ylabel("Position Tolerance")

    ax_arr_diff.set_xlim([0, 1])
    ax_arr_diff.set_xlabel("Selectivity")
    ax_arr_diff.set_ylabel("Position Tolerance - mean position tolerance")

    # -----------------------------------------------------------
    #   PLot of the effect we are trying to model.
    # -----------------------------------------------------------
    a = ml_gamma_full
    selectivity_arr = np.arange(1.1, step=0.1)
    scale_arr = (data["yLineFit"][0] * selectivity_arr + data["yLineFit"][1]) / a

    plt.figure()
    plt.title("Distribution of position tolerances about mean values")

    ppf_05 = []
    ppf_95 = []

    for scale in scale_arr:
        ppf_05.append(ss.gamma.ppf(0.05, a=a, scale=scale))
        ppf_95.append(ss.gamma.ppf(0.95, a=a, scale=scale))

    plt.plot(selectivity_arr, ppf_05, label='cdf=0.05')
    plt.plot(selectivity_arr, ppf_95, label='cdf=0.95')
    plt.legend()

    plt.scatter(data["xScatterRaw"], data["yScatterRaw"], marker='o', s=100)
    plt.plot(data["xScatterRaw"], data["yLineFit"][0] * data["xScatterRaw"] + data["yLineFit"][1],
             label='Mean Position Tolerance')

    plt.legend()
    plt.xlim([0, 1])

    plt.xlabel("Activity Fraction")
    plt.ylabel("Position Tolerance (deg)")
