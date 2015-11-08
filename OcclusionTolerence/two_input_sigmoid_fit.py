# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 08:59:25 2015

A two input (x1=nondiagnostic visibility, x2=diagnostic visibility) sigmoid is used to model
occlusion tolerances of IT Neurons. It is derived from [1] where diagnostic and
non-diagnostic parts visibilities are specified separately. Typically, in other papers diagnostic
and nondiagnostic visibilities are not separated. However as noted by Neilson in [2] neurons
respond preferentially to diagnostic parts. The level of preference for diagnostic parts is given
by the diagnostic group variance. This is percentage explained variance by the diagnostic grouping
of the total variances. Here diagnostic group variances refers to the variance explained by
grouping according to diagnosticity one group is diagnostic and the other group is nondiagnostic.

TODO: Explain more.

Ref:
[1] Neilson, Logothesis & Rainer - 2006 - Dissociation between Local Field Potentials & spiking
activity in Macaque Inferior Temporal Cortex reveals diagnosticity based encoding of complex
objects.

[2] Kovacs, Vogels & Orban -1995 - Selectivity of Macaque Inferior Temporal Neurons for Partially
Occluded Shapes.

[3] Oreilly et. al. - 2013 - Recurrent processing during object recognition.

@author: s362khan
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


def sigmoid(x, w, b):
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))


# noinspection PyTypeChecker
def get_diagnostic_group_to_total_variance_ratio_from_diagnostic_weight(w_d, w_c, b):
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

    return ratio


# noinspection PyTypeChecker
def get_diagnostic_group_to_total_variance_ratio_from_combined_weight(w_c, w_d, b):
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

    return ratio


def diff_between_meas_and_tgt_d_to_total_var_ratio(wd, wc, b, d_to_t_var_ratio):
    return get_diagnostic_group_to_total_variance_ratio_from_diagnostic_weight(wd, wc, b)\
        - d_to_t_var_ratio


def diff_between_meas_and_tgt_d_to_total_var_ratio_2(wc, wd, b, d_to_t_var_ratio):
    return get_diagnostic_group_to_total_variance_ratio_from_combined_weight(wc, wd, b)\
        - d_to_t_var_ratio


def plot_tuning_curve_along_combined_axis(w_c, b, axis=None, font_size=20):

    if axis is None:
        f, axis = plt.subplots(projection='3d')

    vis_levels = np.linspace(0, 1 * np.sqrt(2), num=100)
    axis.plot(vis_levels,
              sigmoid(vis_levels, w_c, b), linewidth=2, label='Best fit sigmoid')

    axis.set_xlim([0, 1.5])
    axis.set_ylim([0, 1.1])
    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)
    axis.grid()

    axis.set_xlabel("Visibility Combined", fontsize=font_size)
    axis.set_ylabel("Normalized fire rate (spikes/s)", fontsize=font_size)
    axis.set_title("Tuning along equal visibilities axis",
                   fontsize=font_size + 10)

    axis.legend(fontsize=font_size, loc=4)

    axis.annotate('w_c=%0.2f, bias=%0.2f' % (w_c, b),
                  xy=(0.40, 0.95),
                  xycoords='axes fraction',
                  fontsize=font_size,
                  horizontalalignment='right',
                  verticalalignment='top')


def plot_full_tuning_curve(w_n, w_d, b, axis=None, font_size=20):

    if axis is None:
        f, axis = plt.subplots(projection='3d')

    vis_arr = np.arange(0, 1, step=0.1)
    vis_arr = np.reshape(vis_arr, (vis_arr.shape[0], 1))

    fire_rates = np.zeros(shape=(vis_arr.shape[0], vis_arr.shape[0]))

    for r_idx in np.arange(vis_arr.shape[0]):
        for c_idx in np.arange(vis_arr.shape[0]):

            x = np.array([vis_arr[r_idx], vis_arr[c_idx]])
            w = np.array([[w_n], [w_d]])

            fire_rates[r_idx][c_idx] = sigmoid(x.T, w, b)

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
    axis.annotate("w_n=%0.2f, w_d=%0.2f" % (w_n, w_d),
                  xy=(x2, y2),
                  xytext=(-20, 20),
                  fontsize=font_size,
                  textcoords='offset points')

    x = np.arange(0, 1, step=0.1)
    z = np.zeros_like(x)
    axis.plot(x, x, z, label="Combined visibility axis", color='red', linewidth=2)


def main(visibilities, fire_rates, d_to_t_var_ratio, title=''):
    # Scale up the visibilities to range from [0, np.sqrt(2)]. We assume on the combined scale
    # equal parts diagnostic and nondiagnostic visibilities. Along this axis
    #           visibility = np.sqrt(vis_d + vis_nd) = np.sqrt(2)*vis_c.
    # In measured data, visibility ranges between [0, 1]. Hence we scale up to get the correct
    # combined weight and bias.
    visibilities = np.sqrt(2) * visibilities

    # Find optimal w_c and bias that fit the data -------------------------------------------
    p_opt, p_cov = so.curve_fit(sigmoid, visibilities, fire_rates)
    w_combined = p_opt[0]
    bias = p_opt[1]
    # print("Data fit parameters: w_c=%0.2f, b=%0.2f: standard error of parameter fits %s"
    #       % (w_combined, bias, np.sqrt(np.diag(p_cov))))

    # Plot the data and the fitted sigmoid
    fig = plt.figure()
    font_size = 20

    if title:
        fig.suptitle(title + ". [Diagnostic group to total variance ratio=%0.2f]"
                     % d_to_t_var_ratio,
                     fontsize=font_size + 10)

    ax = fig.add_subplot(121)

    ax.scatter(visibilities, fire_rates,
               marker='+', linewidth=2, s=60, color='red', label="Original data")

    plot_tuning_curve_along_combined_axis(w_combined, bias, ax, font_size)

    # Find the highest possible diagnostic group variances to total variance ratio that is
    # possible for the fitted  w_combined and bias.
    # -------------------------------------------------------------------------------------
    # Diagnostic weight is equal to the scaled combined weight, which means the nondiagnostic
    # weight is zero. Depending on the bias, the firing rate sigmoid for the nondiagnostic
    # visibility may not go to zero. This means the variances between the diagnostic and
    # nondiagnostic groups may nut equal to the full variances of the data, when calculated from
    # the means. Therefore we find the maximum ratio possible for this sigmoid which happens when
    # one weight accounts for all the variance.
    w_diagnostic = np.sqrt(2) * w_combined
    max_diagnostic_group_to_total_var_ratio = \
        get_diagnostic_group_to_total_variance_ratio_from_diagnostic_weight(
            w_diagnostic,
            w_combined,
            bias)

    if d_to_t_var_ratio > max_diagnostic_group_to_total_var_ratio:
        # noinspection PyStringFormat
        raise Exception("Specified diagnostic group variance to total variance Ratio" +
                        " greater than maximum possible (Max=%0.2f, specified=%0.2f)"
                        % (max_diagnostic_group_to_total_var_ratio, d_to_t_var_ratio))

    #  Given w_combined, bias and the d_to_t_var_ratio  find a distribution w_diagnostic
    #  w_nondiagnostic pair that will generate the target diagnostic group variance to total
    #  variance ratio
    # ----------------------------------------------------------------------------------------
    w_diagnostic = so.fsolve(
        diff_between_meas_and_tgt_d_to_total_var_ratio,
        (np.sqrt(2) * w_combined / 2),  # Initial guess half the combined weight
        args=(w_combined, bias, d_to_t_var_ratio),
        factor=0.5)  # Reduce the step size for the non-linear optimization.
    # TODO: Add some error checks on the return parameters

    # w_diagnostic ranges between np.sqrt(2) * w_c and 0, and is always > weight_nondiagnostic.
    w_nondiagnostic = (np.sqrt(2) * w_combined) - w_diagnostic

    if w_nondiagnostic > w_diagnostic:
        temp = w_nondiagnostic
        w_nondiagnostic = w_diagnostic
        w_diagnostic = temp

    print("w_diagnostic = %0.2f, w_nondiagnostic = %0.2f" % (w_diagnostic, w_nondiagnostic))

    # Plot the 3D Tuning Curve
    ax2 = fig.add_subplot(122, projection='3d')
    plot_full_tuning_curve(
        np.float(w_nondiagnostic),
        np.float(w_diagnostic),
        bias,
        ax2,
        font_size)

    # # Plot the diagnostic and nondiagnostic sigmoid separately
    # plt.figure()
    # vis_levels = np.arange(1, step=0.01)
    # vis_levels = np.reshape(vis_levels, (vis_levels.shape[0], 1))

    # plt.plot(vis_levels, sigmoid(vis_levels, w_nondiagnostic, bias), label='nondiag')
    # plt.plot(vis_levels, sigmoid(vis_levels, w_diagnostic, bias), label='diag')
    # plt.legend()


def main2(visibilities, r_nondiagnostic, r_diagnostic, d_to_t_var_ratio, title=''):

    # Fit the original diagnostic and nondiagnostic data -----------------------------------
    # fit the diagnostic tuning curve
    p_opt, p_cov = so.curve_fit(sigmoid, visibilities, r_diagnostic)
    w_diagnostic = p_opt[0]
    bias = p_opt[1]
    # print("weight diagnostic %0.2f, bias diagnostic %0.2f" % (w_diagnostic, bias))

    # NOTE: too few points to do a meaningful fit (only one above 0)
    # p_opt, p_cov = so.curve_fit(sigmoid, visibilities, r_nondiagnostic)
    # w_nondiagnostic = p_opt[0]
    # b_nondiagnostic = p_opt[1]
    # print("weight nondiagnostic %0.2f, bias nondiagnostic %0.2f"
    #       % (w_nondiagnostic, b_nondiagnostic))

    # Plot the original data
    font_size = 20
    fig = plt.figure()
    if title:
        fig.suptitle(title + ". [Diagnostic group to total variance ratio=%0.2f]"
                     % d_to_t_var_ratio,
                     fontsize=font_size + 10)

    ax = fig.add_subplot(122, projection='3d')

    ax.scatter(visibilities, np.zeros_like(r_nondiagnostic), r_nondiagnostic,
               marker='+', linewidth=2, s=60, color='red', label="Original Nondiagnostic Data")

    ax.scatter(np.zeros_like(r_diagnostic), visibilities, r_diagnostic,
               marker='+', linewidth=2, s=60, color='blue', label="Original Diagnostic Data")

    # Given the diagnostic weight, bias and the diagnostic group to total variance ratio,
    # determine wn, and w_c

    w_combined = so.fsolve(
        diff_between_meas_and_tgt_d_to_total_var_ratio_2,
        (w_diagnostic / 2 / np.sqrt(2)),
        args=(w_diagnostic, bias, d_to_t_var_ratio),
        factor=0.5)
    # TODO: Add some error checks on the return parameters

    # w_diagnostic ranges between np.sqrt(2) * w_c and 0, and is always > weight_nondiagnostic.
    w_nondiagnostic = (np.sqrt(2) * w_combined) - w_diagnostic

    if w_nondiagnostic > w_diagnostic:
        temp = w_nondiagnostic
        w_nondiagnostic = w_diagnostic
        w_diagnostic = temp

    print("w_diagnostic = %0.2f, w_nondiagnostic = %0.2f" % (w_diagnostic, w_nondiagnostic))
    print ("w_combined=%0.2f, bias=%0.2f" % (w_combined, bias))

    plot_full_tuning_curve(
        np.float(w_nondiagnostic),
        np.float(w_diagnostic),
        bias,
        axis=ax,
        font_size=font_size)

    # Plot the tuning curve along the combined axis
    ax2 = fig.add_subplot(121)

    plot_tuning_curve_along_combined_axis(
        np.float(w_combined),
        bias,
        ax2,
        font_size=font_size)


if __name__ == "__main__":
    plt.ion()

    # Fit Kovacs 1995 Tuning Curves  -----------------------------------------------------
    with open('Kovacs1995.pkl', 'rb') as fid:
        kovacsData = pickle.load(fid)

    # Convert occlusion to visibility
    visibility_arr = np.array([(1 - (occlusion / 100.0))
                               for occlusion in kovacsData['occlusion']])

    for obj, perObjRates in enumerate(kovacsData['rates']):
        rates = perObjRates
        rates /= np.max(rates)

        main(visibility_arr, rates, d_to_t_var_ratio=0.3, title='Kovacs 1995 - Object %d' % obj)

    # Fit Oreilly 2013 Data -------------------------------------------------------------
    with open('Oreilly2013.pkl', 'rb') as fid:
        oreillyData = pickle.load(fid)

    # Convert occlusion to visibility
    visibility_arr = np.array([(1 - (occlusion / 100.0))
                               for occlusion in oreillyData['Occ']])

    rates = oreillyData['Rates']
    rates /= np.max(rates)

    main(visibility_arr, rates, d_to_t_var_ratio=0.1, title='Oreilly 2013')

    # Fit Neilson Tuning Curve  -------------------------------------------------------
    with open('Neilson2006.pkl', 'rb') as fid:
        NeilsonData = pickle.load(fid)

    # With Neilson Data, we have the diagnostic and non-diagnostic tuning curves,
    # we fit these and calculate the combined tuning curve
    visibility = [(1 - (occlusion / 100.0)) for occlusion in NeilsonData['singleOcc']]

    nondiag_rates = NeilsonData['singleNonDiagRate']
    diag_rates = NeilsonData['singleDiagRate']

    # First element of both the non_diag and diag rates is the full rate, so remove them
    r_max = nondiag_rates[0]

    nondiagnostic_rates = np.array(nondiag_rates[1:]) / r_max
    diagnostic_rates = np.array(diag_rates[1:]) / r_max
    visibility = np.array(visibility[1:])

    # Remove all negative rates
    nondiag_rates[nondiag_rates < 0] = 0
    diag_rates[diag_rates < 0] = 0

    ratio_diag_var_to_total_var = 0.437

    main2(visibility,
          nondiagnostic_rates,
          diagnostic_rates,
          ratio_diag_var_to_total_var,
          title='Neilson 2006 - Single Neuron')

    # TODO: Add tuning curves from Neilson 2005 PhD Thesis.

