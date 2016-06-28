import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit

import log_normal_size_profile as size_profile


if __name__ == "__main__":

    plt.ion()

    # Construct the custom subplots
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2, sharey=ax1)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2, sharey=ax1)

    ax4 = plt.subplot2grid((2, 6), (1, 0), colspan=3)
    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=3)

    f_size = 40

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

    ax1.scatter(np.log2(n1_stim_size), n1_firing_rate, marker='o', s=100,
                label='Original Data')

    # Create the size tuning profile
    neuron = size_profile.LogNormalSizeProfile(
        pol_tol=10 * np.pi / 180.0,  # This is arbitrarily choose to cover the who profile.
        pref_size=5.2 * np.pi / 180.0,
        size_bw=1.35
    )

    x_rad = np.linspace(0, neuron.max_pref_stim_size * 2, num=100)
    x_deg = x_rad * 180 / np.pi

    ax1.plot(
        np.log2(x_deg),
        neuron.firing_rate_modifier(x_rad),
        linewidth=2)

    ax1.set_xlim([0, np.log2(max(x_deg))])
    ax1.set_xticks(np.arange(1, 6))

    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel("Normalized Firing Rate" + "\n" + "(Spikes/s)", fontsize=f_size)

    ax1.tick_params(axis='x', labelsize=f_size)
    ax1.tick_params(axis='y', labelsize=f_size)

    # -----------------------------------------------------------------------------------
    # Neuron 2 - Figure 4D of Ito et. al - 199

    ax2.set_xlim([0, np.log2(max(x_deg))])
    ax2.set_xticks(np.arange(1, 7))
    ax2.set_xlabel(r'$log_2(Stimulus\ Size)\ [Degrees]$', fontsize=f_size)

    ax2.set_ylim([0, 1.1])
    ax2.yaxis.set_visible(False)


    ax2.tick_params(axis='x', labelsize=f_size)
    ax2.tick_params(axis='y', labelsize=f_size)