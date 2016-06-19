# -*- coding: utf-8 -*-
"""
Fit of the diagnostic group variance to total variance ratio as seen in [1].
In supplementary material of [1] diagnostic group variance was compared to the difference
between the net firing rates to diagnostic and non-diagnostic parts. For all cases there was a
positive correlation. Indicating neurons with high diagnostic group variance fired more to
diagnostic parts than to nondiagnostic parts.

[1] Neilson, Logothesis & Rainer - 2006 - Dissociation between Local Field Potentials & spiking
activity in Macaque Inferior Temporal Cortex reveals diagnosticity based encoding of complex
objects.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle


def exponential(x, a):
    return a * np.exp(-a * x)


def fit_and_plot_diagnostic_variance_ratio_distribution(axis=None, font_size=40):

    with open('Neilson2006.pkl', 'rb') as fid:
        data = pickle.load(fid)

    # PDF of Diagnostic variance
    ratio = data['diagVariance']
    ratio /= 100

    hist, bins = np.histogram(ratio, bins=np.arange(1, step=0.01), density=True)
    # Clean up zeros
    idxs = np.nonzero(hist)
    hist = hist[idxs]
    bins = bins[idxs]

    if axis is None:
        f, axis = plt.subplots(1)

    axis.scatter(bins, hist, label='Original Data', s=60)

    # Fit the pdf
    p_opt, p_cov = curve_fit(exponential, bins, hist)

    axis.plot(bins,
              exponential(bins, *p_opt),
              label=r'$p(x) = %0.2f \exp(-%0.2fx) $' % (p_opt[0], p_opt[0]),
              color='g',
              linewidth=2)

    axis.set_xlabel("R", fontsize=font_size)
    axis.set_ylabel("Frequency", fontsize=font_size)

    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)

    axis.set_xlim([0, 0.8])
    axis.set_ylim([0, np.max(hist) * 1.1])

    axis.set_yticks(np.arange(2, 10, step=2))

    axis.legend(loc='best', fontsize=font_size)
    # axis.grid()


if __name__ == "__main__":
    plt.ion()
    fit_and_plot_diagnostic_variance_ratio_distribution()
