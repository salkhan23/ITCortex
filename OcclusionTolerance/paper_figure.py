import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from scipy.optimize import curve_fit
import pickle

import two_input_sigmoid_occlusion_profile as occlusion_profile
import two_input_sigmoid_fit as fit
import diagnostic_group_variance_fit as ratio_fit


if __name__ == "__main__":
    plt.ion()

    # Fit Kovacs 1995 Tuning Curves  -----------------------------------------------------

    # Import the data
    with open('Kovacs1995.pkl', 'rb') as fid:
        kovacsData = pickle.load(fid)

    # Convert occlusion to visibility
    visibility_arr = np.array([(1 - (occlusion / 100.0)) for occlusion in kovacsData['occlusion']])

    # Extract out the relevant data
    rates = kovacsData['rates'][0]  # shape at rank 1
    rates = rates / max(rates)

    # Fit the 1D logistic function to get w_c and bias
    p_opt, p_cov = so.curve_fit(fit.sigmoid, visibility_arr, rates)
    w_combined, bias = p_opt

    # Extend to a 2D Model
    ratio = 0.3
    neuron = occlusion_profile.TwoInputSigmoidOcclusionProfile(ratio, w_combined, bias)

    # Plot the figure
    font_size = 30
    f = plt.figure()

    # 1D logistic fit ---------------------------------------------------------------
    ax1 = f.add_subplot(1, 3, 1)
    neuron.plot_combined_axis_profile(
        ax1,
        font_size=font_size,
        print_parameters=False
    )

    ax1.annotate(
        r'$w_c=%0.2f,$' % w_combined + "\n" + r'$b=%0.2f$' % bias,
        xy=(0.40, 0.95),
        xycoords='axes fraction',
        fontsize=font_size,
        horizontalalignment='right',
        verticalalignment='top')

    ax1.scatter(visibility_arr, rates, s=60, color='blue')
    ax1.legend(loc='best', fontsize=font_size)

    # Diagnostic Ratio Distribution -------------------------------------------------
    ax2 = f.add_subplot(1, 3, 2)
    ratio_fit.fit_and_plot_diagnostic_variance_ratio_distribution(axis=ax2, font_size=font_size)

    # 2D Logistic Fit
    ax3 = f.add_subplot(133, projection='3d')
    neuron.plot_complete_profile(axis=ax3, font_size=font_size, print_parameters=True)
    # To plot the combined visibility axis on the 3D plot
    # we assume combined visibility and diagnostic and nondiagnostic visibility levels are related
    # by v_c = np.sqrt(vnd**2 + vd**2) / sqrt(2). We vnd=1 and vd=1, vc=1 and this function
    # monotonously increases with visibility as required by our logistic function
    vis_combined = np.linspace(0, 1, num=100)

    # To plot the combined visibility axis on the 3D plot, we assume combined visibility and
    # diagnostic and nondiagnostic visibility levels are related
    # by v_c = np.sqrt(vnd**2 + vd**2) / sqrt(2). When vnd=1 and vd=1, vc=1 and this function
    # monotonously increases with visibility as required by our logistic function.
    # Additionally along the combined visibility axis we assume v_nd = v_nd
    vis_d = np.linspace(0, 1, num=100)
    vis_c = np.sqrt(vis_d ** 2 + vis_d ** 2) / np.sqrt(2)

    ax3.plot(
        vis_d,
        vis_d,
        neuron.firing_rate_modifier(vis_c, np.ones_like(vis_c) * -1),
        color='green',
        linewidth=3,
    )

