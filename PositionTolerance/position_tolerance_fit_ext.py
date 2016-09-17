# -*- coding: utf-8 -*-
""" ---------------------------------------------------------------------------------------------

In position_tolerance_fit.py it was found that the distribution of actual position tolerance about
the mean position is best described by a gamma distribution with a fixed gamma and
scale = mean /alpha. Here we expand on this result by separating the data into bins and find the
best fit alpha for each bin separately.

If the individual bins have similar alphas compared to the
previously found best fit alpha over the whole distribution, this would mean this distribution
model is a good fit.

Bins are divided such that they have an equal number of points per bin

Created on Thu Nov 13 16:53:46 2014

@author: s362khan
----------------------------------------------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pickle


def best_gamma_fit_mean_from_regression_fit(s_arr, pt_arr, lfit_coeffs):
    """
    Find the best fit gamma alpha parameter, subject to the constraint that the mean of the
    distribution is given by the linear regression fit linear_fit_coeff[0]x + linear_fit_coeff[0]

    :param s_arr: array of measured selectivity (activity fraction)
    :param pt_arr:  array of measured position tolerances at the corresponding selectivity.
    :param lfit_coeffs: list/array of linear regression fit. Highest order first.

    :return:
    """
    alpha_arr = np.arange(0.1, 20, step=0.01)
    llv = []  # Log likelihood values

    for alpha in alpha_arr:

        # alpha constant
        prob = [ss.gamma.pdf(pt, a=alpha,
                             scale=(lfit_coeffs[0] * s_arr[pt_idx] + lfit_coeffs[1]) / alpha)
                for pt_idx, pt in enumerate(pt_arr)]

        # # # Scale constant
        # prob = [ss.gamma.pdf(pt, a=(lfit_coeffs[0] * s_arr[pt_idx] + lfit_coeffs[1]) / alpha,
        #                      scale=alpha)
        #         for pt_idx, pt in enumerate(pt_arr)]

        llv.append(np.log(prob).sum())

    llv = np.array(llv)

    # print alpha_arr[llv.argmax()], llv[llv.argmax()]
    # print max(llv)

    return alpha_arr[llv.argmax()]


def separate_points_into_bins(x_arr, y_arr, n_bins):
    """
    Separate points into bins based on equal number of points per bin

    :param x_arr:
    :param y_arr:
    :param n_bins:
    :return:
    """
    points_per_bin = len(data['xScatterRaw']) / n_bins

    p_idx_lists = []
    for i in np.arange(n_bins):
        p_idx_lists.append(
            np.arange(0, points_per_bin) + i * points_per_bin
        )

    # Add remaining points
    p_idx_lists[-1] = np.append(
        p_idx_lists[-1],
        np.arange(p_idx_lists[-1][-1] + 1, len(data['xScatterRaw']))
    )

    separated_points = []
    for _ in np.arange(n_bins):
        separated_points.append([])

    for l_idx, index_list in enumerate(p_idx_lists):
        for index in index_list:
            separated_points[l_idx].append((x_arr[index], y_arr[index]))

    return separated_points


if __name__ == '__main__':
    plt.ion()

    # Load the data
    with open("Zoccolan2007.pkl") as handle:
        data = pickle.load(handle)

    # ----------------------------------------------------------
    # Full Data Processing - Sanity Checking
    # ----------------------------------------------------------
    # Find the best fit alpha, subject to the constraint that mean of the distribution is
    # given by the linear regression fit
    ml_gamma_full = best_gamma_fit_mean_from_regression_fit(
        data["xScatterRaw"], data["yScatterRaw"], data["yLineFit"])
    print("Full: Best Gamma Fit, alpha=%0.4f, scale=mean_position_tolerance / alpha"
          % ml_gamma_full)

    # Plot the effect we are trying to model
    a = ml_gamma_full
    selectivity_arr = np.arange(1.1, step=0.1)
    scale_arr = (data["yLineFit"][0] * selectivity_arr + data["yLineFit"][1]) / a

    plt.figure()

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

    # ----------------------------------------------------------
    # Partition Data into subsets
    # ----------------------------------------------------------
    bins = np.linspace(0, 1, num=4)
    partitioned_lists = separate_points_into_bins(data["xScatterRaw"], data["yScatterRaw"], 3)

    f_raw, ax_arr_raw = plt.subplots()

    for bin_idx, bin_list in enumerate(partitioned_lists):

        x_val = np.array([x[0] for x in bin_list])
        y_val = np.array([x[1] for x in bin_list])

        ax_arr_raw.plot(x_val, y_val, marker='o', markersize=10, linestyle='')

        # Find the best fit alpha, subject to the constraint that mean of the distribution is
        # given by the linear regression fit
        ml_gamma_bin = best_gamma_fit_mean_from_regression_fit(x_val, y_val, data["yLineFit"])
        print("Bin %i: Best Gamma Fit, alpha=%0.2f, scale=mean_position_tolerance / alpha"
              % (bin_idx, ml_gamma_bin))

    ax_arr_raw.set_xlim([0, 1])
    ax_arr_raw.set_xlabel("Selectivity")
    ax_arr_raw.set_ylabel("Position Tolerance")
