# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:29:46 2015

This file creates scale samples to account for non optimal position, size, rotation, etc
presentation of stimuli in the Lehky sparseness model.

REF:  Lehky, S. R., Kiani, R., Esteky, H., & Tanaka, K. (2011). Statistics of
            visual responses in primate inferotemporal cortex to object stimuli.
            Journal of Neurophysiology, 106(3), 1097–117.

Random samples of scale factors that would be expected in the Lehky et al. experiment.
The number of samples isn't critical as they are just used to estimate a distribution.
Samples should lie between [0,1].

POSITION:
All stimuli were presented foveally at (0, 0) radians. However the receptive field centers
of IT neurons and their preferred positions are distributed around the fovea. To account for this
we create a large sample of position profiles and determine the amount of deviation expected if
stimuli are presented at (0, 0) to all neurons.

SIZE:
The largest dimension of all stimuli extended 7 degrees. Similarly to account for this non-optimal
size we generate a large sample of size tuning profiles of IT neurons and determine the amount of
deviation expected if stimuli are presented with a size of 7 degrees.

ROTATION:
Individual IT neurons are tuned to different views of the object. If the object displays some
symmetry there be multiple peaks in the tuning profile. The choice of which angle to consider as
orientation/rotation 0 is arbitrary. How ever it is likely the view presented may not  be the
preferred orientation.
TODO: Do not know yet how to generate a population of rotation tuning profiles for neurons.
Calculate Adjustment samples for non optimal orientations once available.

@author: bptripp, s362khan
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.stats import gamma

# Do relative import of the main folder to get files in sibling directories
top_level_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if top_level_dir_path not in sys.path:
    sys.path.append(top_level_dir_path)

from PositionTolerance import gaussian_position_profile as gpt
from ObjectSelectivity import activity_fraction_fit as selectivity_distribution
from SizeTolerance import log_normal_size_profile as lst

reload(lst)
reload(selectivity_distribution)
reload(gpt)


sample_size = 10000

# Get Adjustment Samples ------------------------------------------------------------------------

# The gaussian position profile expects a selectivity metric (activity fraction) to determine its
# receptive field size. Generate a sample set of  selectivity measures.
selectivity_arr = selectivity_distribution.get_activity_fraction_sparseness(sample_size)
# plt.hist(selectivity_arr, bins = np.arange(1, step=0.1))

position_profiles = []
size_profiles = []

for index in np.arange(sample_size):

    current_position_profile = gpt.GaussianPositionProfile(selectivity_arr[index])

    position_profiles.append(current_position_profile)

    size_profiles.append(lst.LogNormalSizeProfile(current_position_profile.position_tolerance))


# Stimuli at position (0, 0)
position_samples = [profile.firing_rate_modifier(0, 0) for profile in position_profiles]
plt.figure('Position Samples')
plt.title("Distribution of position scale parameters")
plt.hist(position_samples, bins=np.arange(1, step=0.1))

# Stimuli of size 7
size_samples = [profile.firing_rate_modifier(7 * np.pi / 180.0) for profile in size_profiles]
plt.figure('Size Samples')
plt.title("Distribution of size scale parameters")
plt.hist(size_samples, bins=np.arange(1, step=0.1))

# TODO: Account for non-optimal rotation/orientation angles.

#  Adjust the shape and scale parameters for non-optimal stimuli set ----------------------------
scale_samples = [position_samples, size_samples]

mu_s = np.mean(scale_samples)
var_s = np.var(scale_samples)

print ("Scale samples: mean%f, var=%f" % (np.float(mu_s), np.float(var_s)))

# Original Lehky et al. parameters ...
ala = 4    # shape parameter (a) of Lehky (l) for PDF of shape parameters a for rate PDFs
bla = 0.5  # shape parameter (b) of Lehky (l) for PDF of shape parameters a for rate PDFs

alb = 2    # shape parameter (a) of Lehky (l) for PDF of shape parameters b for rate PDFs
blb = 0.5  # shape parameter (a) of Lehky (l) for PDF of shape parameters b for rate PDFs


# expectation and variance of Lehky et al. distribution of mean rates.
# Product of two independent gamma rvs
mu_l = ala*bla*alb*blb
var_l = ala*bla**2*alb*blb**2 + ala*bla**2*(alb*blb)**2 + alb*blb**2*(ala*bla)**2

# expectation and variance of "full" (unscaled) distribution of mean rates
# That will approximate Lehky after scaling ...
mu_f = mu_l / mu_s
var_f = (var_l - var_s*(mu_l/mu_s)**2) / (var_s + mu_s**2)

# shape and scale parameters for full distribution (keeping shape same as scaled one) ...
afa = ala
bfa = bla
bfb = (var_f - mu_f**2/ala) / (mu_f*bla*(1 + ala))
afb = mu_f / (afa*bfa*bfb)

print ("Shape parameter (a) gamma distribution (Full, unscaled) a=%f, b=%f" % (afa, bfa))
print ("Scale Parameter (b) gamma distribution (Full, unscaled) a=%f, b=%f"
       % (np.float(afb), np.float(bfb)))

# Plot Max firing rates based on Lehky (non-optimal stimuli set) & the full (optimal stimuli set)
n = 1000
shape_param_dist_f = gamma.rvs(afa, scale=bfa, loc=0, size=n)
scale_param_dist_f = gamma.rvs(afb, scale=bfb, loc=0, size=n)

shape_param_dist_l = gamma.rvs(ala, scale=bla, loc=0, size=n)
scale_param_dist_l = gamma.rvs(alb, scale=blb, loc=0, size=n)

max_rates_f = []
max_rates_l = []

for index in np.arange(n):
    max_rates_f.append(
        gamma.ppf(0.99, shape_param_dist_f[index], loc=0, scale=scale_param_dist_f[index]))

    max_rates_l.append(
        gamma.ppf(0.99, shape_param_dist_l[index], loc=0, scale=scale_param_dist_l[index]))

plt.figure("Max Fire Rate Distributions")
plt.subplot(211)
plt.hist(max_rates_f)
plt.title('Histogram of full (unscaled) max spike rates')
plt.subplot(212)
plt.hist(max_rates_l, label='method1')
plt.title('Histogram of scaled (Lehky) max spike rates')

# Method 2 of getting Lehky distribution from full spike rates
# noinspection PyArgumentList
scale_factors = np.random.rand(n)
scaled_max_rates = max_rates_f*scale_factors
plt.subplot(212)
plt.hist(scaled_max_rates, label='method2')
plt.legend()


if __name__ == '__main__':
    plt.ion()
