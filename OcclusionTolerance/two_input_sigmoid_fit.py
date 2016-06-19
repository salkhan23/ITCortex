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
from mpl_toolkits.mplot3d import proj3d

import two_input_sigmoid_occlusion_profile as occlusion_profile
# Force reload (compile) IT cortex modules to pick changes not included in cached version.
reload(occlusion_profile)


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


def optimization_equations(w, w_d, b, desired_ratio):
    """
    Function(s) to solve using nonlinear numerical optimization.

    (1) Desired ratio - Actual ratio = 0
    (2) the sum of the weights cannot be more than weight on the combined axis.

    :param w            : tuple of (w_d, w_nd). What we find optimum values for.
    :param w_d          : weight on combined axis
    :param b            : bias
    :param desired_ratio: desired diagnostic to total variance ratio

    :return: tuple (Desired ratio - Actual ratio, w_c - (w_d + w_nd)
    """
    w_c, w_nd = w
    print ("w_c %0.4f, w_nd %0.4f, R=%0.2f" % (w_c, w_nd, calculate_ratio(w_d, w_nd, b)))
    return desired_ratio - calculate_ratio(w_d, w_nd, b), w_c - w_d - w_nd


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


def main(visibilities, fire_rates, ratio, title=''):

    # Find optimal w_c and bias that fit the data -------------------------------------------
    p_opt, p_cov = so.curve_fit(sigmoid, visibilities, fire_rates)
    w_combined, bias = p_opt
    # print("Data fit parameters: w_c=%0.2f, b=%0.2f: standard error of parameter fits %s"
    #        % (w_combined, bias, np.sqrt(np.diag(p_cov))))

    # Find w_d and w_nd
    neuron = occlusion_profile.TwoInputSigmoidOcclusionProfile(ratio, w_combined, bias)

    # Plot the figures -----------------------------------------
    f = plt.figure()
    font_size = 50

    if title:
        f.suptitle(title + ". [R=%0.2f]" % ratio, fontsize=font_size + 10)

    ax1 = f.add_subplot(1, 2, 1)
    neuron.plot_combined_axis_profile(axis=ax1, font_size=font_size)

    ax1.scatter(
        visibilities,
        fire_rates,
        s=60,
        color='blue',
        # label="Original data"
    )

    ax1.legend(fontsize=font_size, loc='best')

    ax2 = f.add_subplot(1, 2, 2, projection='3d')
    neuron.plot_complete_profile(axis=ax2, font_size=font_size)

    # To plot the combined visibility axis on the 3D plot, we assume combined visibility and
    # diagnostic and nondiagnostic visibility levels are related
    # by v_c = np.sqrt(vnd**2 + vd**2) / sqrt(2). When vnd=1 and vd=1, vc=1 and this function
    # monotonously increases with visibility as required by our logistic function.
    # Additionally along the combined visibility axis we assume v_nd = v_nd
    vis_d = np.linspace(0, 1, num=100)
    vis_c = np.sqrt(vis_d**2 + vis_d**2) / np.sqrt(2)

    ax2.plot(
        vis_d,
        vis_d,
        neuron.firing_rate_modifier(vis_c, np.ones_like(vis_c) * -1),
        color='green',
        linewidth=3,
    )

    return w_combined, bias


def main2(visibilities, r_nondiagnostic, r_diagnostic, d_to_t_var_ratio, title=''):
    # TODO: Fix me

    # Fit the original diagnostic and nondiagnostic data -----------------------------------
    # fit the diagnostic tuning curve
    p_opt, p_cov = so.curve_fit(sigmoid, visibilities, r_diagnostic)
    w_diagnostic = p_opt[0]
    bias = p_opt[1]
    print("weight diagnostic %0.2f, bias diagnostic %0.2f" % (w_diagnostic, bias))

    # NOTE: too few points to do a meaningful fit (only one above 0)
    p_opt, p_cov = so.curve_fit(sigmoid, visibilities, r_nondiagnostic)
    w_nondiagnostic = p_opt[0]
    b_nondiagnostic = p_opt[1]
    print("weight nondiagnostic %0.2f, bias nondiagnostic %0.2f"
          % (w_nondiagnostic, b_nondiagnostic))

    # Plot the original data
    font_size = 34
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

    # Given w_d, b and R, determine wn, and w_c

    # Use nonlinear optimization to find w_diagnostic and w_nondiagnostic that can generate
    # the desired diagnostic to total variance ratio.
    w_combined, w_nondiagnostic = so.fsolve(
        optimization_equations,
        (w_diagnostic , 0),
        args=(w_diagnostic, bias, d_to_t_var_ratio),
        factor=0.5,  # w_d increases rapidly without this factor adjustment)
    )

    # plot_full_tuning_curve(w_nondiagnostic, w_diagnostic, bias, axis=ax)

    return w_combined, bias


if __name__ == "__main__":
    plt.ion()

    # Store the w_combined and bias parameters seen across the fitted  data. Mean and
    # standard deviation will be used to generate sample w_combined and bias in the tuning
    # profile file
    w_combined_arr = []
    bias_arr = []

    # Fit Kovacs 1995 Tuning Curves  -----------------------------------------------------
    with open('Kovacs1995.pkl', 'rb') as fid:
        kovacsData = pickle.load(fid)

    # Convert occlusion to visibility
    visibility_arr = np.array([(1 - (occlusion / 100.0))
                               for occlusion in kovacsData['occlusion']])

    for obj, perObjRates in enumerate(kovacsData['rates']):
        rates = perObjRates
        rates /= np.max(rates)

        weight_c, bias_c = main(visibility_arr,
                                rates,
                                ratio=0.3,
                                # title='Kovacs 1995 - Object %d' % obj
                                )
        raw_input("Continue with next Figure?")

        w_combined_arr.append(weight_c)
        bias_arr.append(bias_c)

    # Fit Oreilly 2013 Data -------------------------------------------------------------
    with open('Oreilly2013.pkl', 'rb') as fid:
        oreillyData = pickle.load(fid)

    # Convert occlusion to visibility
    visibility_arr = np.array([(1 - (occlusion / 100.0))
                               for occlusion in oreillyData['Occ']])

    rates = oreillyData['Rates']
    rates /= np.max(rates)

    weight_c, bias_c = main(visibility_arr,
                            rates,
                            ratio=0.1,
                            title='Oreilly 2013')
    w_combined_arr.append(weight_c)
    bias_arr.append(bias_c)

    # Fit Neilson Tuning Curve  -------------------------------------------------------
    # Here we try the reverse of our method, given w_d and w_nd (found by fitting the diagnostic
    # and nondiagnostic tuning curves) we use fsolve to find a w_c. Unfortunately, the
    # nondiagnostic data does not have sufficient nonzero points to get a good logistic function
    # fit. Neither is there sufficient information to find the max firing rate to normalize the
    # the two input logistic complete tuning profile. Recall that as noted by Neilson, both the
    # diagnostic and nondiagnostic parts are needed to get the max firing rate of the neuron.
    # Neurons are sensitive to both size and diagnosticity and even if all the diagnostic parts
    # of the neuron are fully visible, depending on the neurons diagnosticity, it will not fire at
    # its max firing rate.
    # For now we assume the reverse process does not work and disable the following  code

    # with open('Neilson2006.pkl', 'rb') as fid:
    #     NeilsonData = pickle.load(fid)
    #
    # # With Neilson Data, we have the diagnostic and non-diagnostic tuning curves,
    # # we fit these and calculate the combined tuning curve
    # visibility_arr = [(1 - (occlusion / 100.0)) for occlusion in NeilsonData['singleOcc']]
    #
    # nondiag_rates = NeilsonData['singleNonDiagRate']
    # diag_rates = NeilsonData['singleDiagRate']
    #
    # # First element of both the non_diag and diag rates is the full rate, so remove them
    # r_max = nondiag_rates[0]
    #
    # nondiagnostic_rates = np.array(nondiag_rates[0:]) / r_max
    # diagnostic_rates = np.array(diag_rates[0:]) / r_max
    # visibility_arr = np.array(visibility_arr[0:])
    #
    # # Remove all negative rates
    # nondiag_rates[nondiag_rates < 0] = 0
    # diag_rates[diag_rates < 0] = 0
    #
    # ratio_diag_var_to_total_var = 0.437
    #
    # weight_c, bias_c = main2(visibility_arr,
    #                          nondiagnostic_rates,
    #                          diagnostic_rates,
    #                          ratio_diag_var_to_total_var,
    #                          title='Neilson 2006 - Single Neuron')

    w_combined_arr.append(weight_c)
    bias_arr.append(bias_c)

    # TODO: Add tuning curves from Neilson 2005 PhD Thesis.

    # Find the average and variances in the collected weight_combined and bias terms
    w_combined_arr = np.array(w_combined_arr)
    bias_c = np.array(bias_arr)

    w_c_mean = np.mean(w_combined_arr)
    w_c_std = np.std(w_combined_arr)

    b_mean = np.mean(bias_arr)
    b_std = np.std(bias_arr)

    # noinspection PyStringFormat
    print("Combined Weight. Mean= %0.4f, sigma=%0.4f" % (w_c_mean, w_c_std))
    # noinspection PyStringFormat
    print("Bias. Mean=%0.4f, sigma=%0.4f" % (b_mean, b_std))
