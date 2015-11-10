# -*- coding: utf-8 -*-
"""
Fit of the diagnostic Group variance to total variance ratio as seen in

[1] Neilson, Logothesis & Rainer - 2006 - Dissociation between Local Field Potentials & spiking
activity in Macaque Inferior Temporal Cortex reveals diagnosticity based encoding of complex
objects.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle

with open('Neilson2006.pkl', 'rb') as fid:
    data = pickle.load(fid)


def exponential(x, a):
    return a * np.exp(-a * x)


if __name__ == "__main__":
    plt.ion()

    ''' PDF of Diagnostic variance '''
    ratio = data['diagVariance']
    ratio /= 100

    hist, bins = np.histogram(ratio, bins=np.arange(1, step=0.01), density=True)

    # Clean up zeros
    idxs = np.nonzero(hist)
    hist = hist[idxs]
    bins = bins[idxs]

    plt.figure('Diagnostic Variance Distribution')
    axis = plt.gca()
    axis.scatter(bins, hist, label='Original Data', s=60)

    # Fit the pdf
    p_opt, p_cov = curve_fit(exponential, bins, hist)
    axis.plot(bins, exponential(bins, *p_opt),
              label='Exponential %0.2f*exp(-%0.2fx)' % (p_opt[0], p_opt[0]))

    font_size = 30
    axis.set_xlabel("Ratio", fontsize=font_size)
    axis.set_ylabel("Frequency", fontsize=font_size)
    axis.set_title("Distribution of Diagnostic to Total Variance Ratio across IT",
                   fontsize=font_size)
    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)

    axis.set_xlim([0, 1])
    axis.set_ylim([0, np.max(hist) * 1.1])

    axis.legend(loc='best', fontsize=font_size)
    axis.grid()
