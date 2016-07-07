import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as ss

import log_normal_size_profile as size_profile
import optimumSizeFit as sizeFit


if __name__ == "__main__":

    plt.ion()

    # Construct the custom subplots
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2, sharey=ax1)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2, sharey=ax1)

    ax4 = plt.subplot2grid((2, 6), (1, 0), colspan=3)
    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=3)

    f_size = 35

    # Get the Data
    with open("Ito95Data.pkl", 'rb') as fid:
        data = pickle.load(fid)

    # -----------------------------------------------------------------------------------
    # Neuron 1 - Figure 3D of Ito et. al - 1995
    # This was parametrized in the paper as pref_size = 5.2, and size_bw = 1.35 octaves
    # Just plot using the parameters from LogNormalFit.py

    # Plot the Original Data
    n1_stim_size = data['n1Size']
    n1_firing_rate = data['n1FiringRate']

    # ax1.scatter(np.log2(n1_stim_size), n1_firing_rate, marker='o', s=100, label='Original Data')

    # Create the size tuning profile
    neuron = size_profile.LogNormalSizeProfile(
        pol_tol=10 * np.pi / 180.0,  # This is arbitrarily choose to cover the who profile.
        pref_size=5.2 * np.pi / 180.0,
        size_bw=1.35
    )

    x_rad = np.linspace(0.001, neuron.max_pref_stim_size, num=100)
    x_deg = x_rad * 180 / np.pi

    ax1.plot(
        np.log2(x_deg),
        neuron.firing_rate_modifier(x_rad),
        linewidth=2)

    ax1.vlines(
        np.log2(neuron.max_pref_stim_size * 180 / np.pi),
        0,
        1,
        linewidth=3,
        linestyle='--',
        label='Max. size')

    ax1.set_xlim([0, np.log2(max(x_deg))])
    ax1.set_xticks(np.arange(1, 6))

    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel("Normalized Firing" + "\n" + "Rate (Spikes/s)", fontsize=f_size)

    ax1.tick_params(axis='x', labelsize=f_size)
    ax1.tick_params(axis='y', labelsize=f_size)

    ax1.annotate(
        'A',
        xy=(0.1, 0.95),
        xycoords='axes fraction',
        fontsize=30,
        horizontalalignment='right',
        verticalalignment='top'
    )

    # -----------------------------------------------------------------------------------
    # Neuron 2 - Figure 4D of Ito et. al - 1995
    # pref_size = 26.1, size_bwBw > 5octaves
    n2_stim_size = data['n2Size']
    n2_firing_rate = data['n2FiringRate']

    # ax2.scatter(np.log2(n2_stim_size), n2_firing_rate, marker='o', s=100, label='Original Data')

    # Create the size tuning profile
    neuron2 = size_profile.LogNormalSizeProfile(
        pol_tol=30 * np.pi / 180.0,  # This is arbitrarily choose to cover the who profile.
        pref_size=13.1 * np.pi / 180.0,
        size_bw=8
    )
    # These provide a better fit to the data, the original values from the paper are prefSize=26.1
    # and  sizeBw > 5 octaves. The difference in response at 26.1 and 13.1 is very small.

    x_rad = np.linspace(0.0001, neuron2.max_pref_stim_size, num=100)
    x_deg = x_rad * 180 / np.pi

    ax2.plot(
        np.log2(x_deg),
        neuron2.firing_rate_modifier(x_rad),
        linewidth=2)

    ax2.set_xlim([0, np.log2(max(x_deg))])
    ax2.set_xticks(np.arange(1, 7))
    ax2.set_xlabel(r'$log_2(Stimulus\ Size)$' + ' (Deg)', fontsize=f_size)

    ax2.set_ylim([0, 1.1])
    ax2.yaxis.set_visible(False)

    ax2.tick_params(axis='x', labelsize=f_size)
    ax2.tick_params(axis='y', labelsize=f_size)

    ax2.vlines(
        np.log2(neuron2.max_pref_stim_size * 180 / np.pi),
        0,
        1,
        linewidth=3,
        linestyle='--',
        label='Max. size')

    ax2.annotate(
        'B',
        xy=(0.1, 0.95),
        xycoords='axes fraction',
        fontsize=30,
        horizontalalignment='right',
        verticalalignment='top'
    )

    # -----------------------------------------------------------------------------------
    # Neuron 3 - Figure 5D of Ito et. al - 1995
    # pref_size = 27.1, size_bw ?  - Respond only to largest size tested.

    n3_stim_size = data['n3Size']
    n3_firing_rate = data['n3FiringRate']

    pref_size_n3 = 27.0
    size_bw_n3 = 3.3
    rf_size_n3 = 28.

    # ax3.scatter(np.log2(n3_stim_size), n3_firing_rate, marker='o', s=100, label='Original Data')

    # Create the size tuning profile
    neuron3 = size_profile.LogNormalSizeProfile(
        pol_tol=14 * np.pi / 180.0,  # This is arbitrarily choose to cover the who profile.
        pref_size=27.0 * np.pi / 180.0,
        size_bw=2.5
    )

    x_rad = np.linspace(0.0001, neuron3.max_pref_stim_size, num=100)
    x_deg = x_rad * 180 / np.pi

    ax3.plot(
        np.log2(x_deg),
        neuron3.firing_rate_modifier(x_rad),
        linewidth=2)

    ax3.vlines(
        np.log2(neuron3.max_pref_stim_size * 180 / np.pi),
        0,
        1,
        linewidth=3,
        linestyle='--',
        label='Max. size')

    ax3.set_xlim([0, np.log2(neuron3.max_pref_stim_size * 180 / np.pi)])
    ax3.set_xticks(np.arange(1, 6))

    ax3.set_ylim([0, 1.1])
    ax3.yaxis.set_visible(False)

    ax3.tick_params(axis='x', labelsize=f_size)
    ax3.tick_params(axis='y', labelsize=f_size)

    ax3.annotate(
        'C',
        xy=(0.1, 0.95),
        xycoords='axes fraction',
        fontsize=30,
        horizontalalignment='right',
        verticalalignment='top'
    )

    # -----------------------------------------------------------------------------------
    # Optimum Size Distribution
    # -----------------------------------------------------------------------------------
    print("Optimum Size Fits " + "*" * 20)

    opt_sizes = data["optSize"]
    cutoff = 26.0  # See optimum Size Fit for an explanation.

    sizeFit.plot_histogram(opt_sizes, cutoff, bins=np.arange(0, 31, step=2), axis=ax4)

    # Lognormal Fit
    lognorm_s, lognorm_scale, lognorm_llv = sizeFit.get_lognormal_fit(opt_sizes, cutoff)
    label = 'Lognorm'  # + r"$\mu=%0.2f,\sigma=%0.2f$" % (np.log(lognorm_scale), lognorm_s)
    print label + ": mu=%0.2f, sigma=%0.2f, LLV=%0.2f" \
                  % (np.log(lognorm_scale), lognorm_s, lognorm_llv)

    x_arr = np.arange(30, step=1)
    ax4.plot(
        x_arr,
        ss.lognorm.pdf(x_arr, s=lognorm_s, loc=0, scale=lognorm_scale),
        linewidth=3,
        label=label,
        color='red',
        marker='+',
        markersize=10,
        markeredgewidth=2
    )

    # Gamma Fit
    gamma_alpha, gamma_scale, gamma_llv = sizeFit.get_gamma_fit(opt_sizes, cutoff)
    label = 'Gamma '  # + r'$\alpha=%0.2f,\theta=%0.2f$' % (gamma_alpha, gamma_scale)
    print label + ": alpha=%0.2f, theta=%0.2f, LLV=%0.2f" % (gamma_alpha, gamma_scale, gamma_llv)

    ax4.plot(
        x_arr,
        ss.gamma.pdf(x_arr, a=gamma_alpha, scale=gamma_scale),
        linewidth=3,
        label=label,
        color='black',
        marker='o',
        markersize=10,
        markeredgecolor='black',
        markerfacecolor='none',
        markeredgewidth=2,
    )

    # Levey Fit
    levy_loc, levy_scale, levy_llv = sizeFit.get_levy_fit(opt_sizes, cutoff)

    label = 'Levy '  # + r'$\mu=%0.2f, c=%0.2f$' % (levy_loc, levy_scale)
    print label + ': mu=%0.2f, c=%0.2f, LLV=%0.2f' % (levy_loc, levy_scale, levy_llv)

    ax4.plot(
        x_arr,
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

    ax4.set_ylabel("Frequency", fontsize=f_size)
    ax4.set_xlabel("Preferred Size (Deg)", fontsize=f_size)

    ax4.set_xticks(np.arange(4, 30, step=4))

    ax4.tick_params(axis='x', labelsize=f_size)
    ax4.tick_params(axis='y', labelsize=f_size)

    ax4.legend(
        fontsize=f_size - 5,
        ncol=3,
        bbox_to_anchor=[1, -0.35],
        loc='center')

    ax4.annotate(
        'D',
        xy=(0.1, 0.95),
        xycoords='axes fraction',
        fontsize=30,
        horizontalalignment='right',
        verticalalignment='top'
    )

    # -----------------------------------------------------------------------------------
    # Size Bandwidth Distribution
    # -----------------------------------------------------------------------------------
    print("Size Bandwidth Fits " + "*" * 20)

    size_bw_dist = data["sizeDist"]
    size_bw_rf_size = data["sizeDistRfSize"]

    cutoff = 4
    sizeFit.plot_histogram(size_bw_dist, cutoff, bins=np.arange(0, 5, step=0.5), axis=ax5)

    # Lognormal Fit
    lognorm_s, lognorm_scale, lognorm_llv = sizeFit.get_lognormal_fit(size_bw_dist, cutoff)
    label = 'Lognorm'  # + r"$\mu=%0.2f,\sigma=%0.2f$" % (np.log(lognorm_scale), lognorm_s)
    print label + ": mu=%0.2f, sigma=%0.2f, LLV=%0.2f" \
                  % (np.log(lognorm_scale), lognorm_s, lognorm_llv)

    x_arr = np.arange(5, step=0.25)
    ax5.plot(
        x_arr,
        ss.lognorm.pdf(x_arr, s=lognorm_s, loc=0, scale=lognorm_scale),
        linewidth=3,
        label=label,
        color='red',
        marker='+',
        markersize=10,
        markeredgewidth=2
    )

    # Gamma Fit
    gamma_alpha, gamma_scale, gamma_llv = sizeFit.get_gamma_fit(size_bw_dist, cutoff)
    label = 'Gamma:'  # + r'$\alpha=%0.2f,\theta=%0.2f$' % (gamma_alpha, gamma_scale)
    print label + ": alpha=%0.2f, theta=%0.2f, LLV=%0.2f" % (gamma_alpha, gamma_scale, gamma_llv)

    ax5.plot(
        x_arr,
        ss.gamma.pdf(x_arr, a=gamma_alpha, scale=gamma_scale),
        linewidth=3,
        label=label,
        color='black',
        marker='o',
        markersize=10,
        markeredgecolor='black',
        markerfacecolor='none',
        markeredgewidth=2,
    )

    # Levey Fit
    levy_loc, levy_scale, levy_llv = sizeFit.get_levy_fit(size_bw_dist, cutoff)

    label = 'Levy'  # + r'$\mu=%0.2f, c=%0.2f$' % (levy_loc, levy_scale)
    print label + ': mu=%0.2f, c=%0.2f, LLV=%0.2f' % (levy_loc, levy_scale, levy_llv)

    ax5.plot(
        x_arr,
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

    ax5.set_xticks(np.arange(1, 5, step=1))
    ax5.set_xlim([0.5, 5])

    ax5.set_xlabel("Size BW (Octaves)", fontsize=f_size)

    ax5.tick_params(axis='x', labelsize=f_size)
    ax5.tick_params(axis='y', labelsize=f_size)

    ax5.annotate(
        'E',
        xy=(0.1, 0.95),
        xycoords='axes fraction',
        fontsize=30,
        horizontalalignment='right',
        verticalalignment='top'
    )
