# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:29:46 2015

This file creates scale samples to account for non optimal position, size, rotation, etc
presentation of stimuli in the Lehky sparseness model.

REF:  Lehky, S. R., Kiani, R., Esteky, H., & Tanaka, K. (2011). Statistics of
            visual responses in primate inferotemporal cortex to object stimuli.
            Journal of Neurophysiology, 106(3), 1097–117.

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
plt.hist(position_samples, bins=np.arange(1, step=0.1))

# Stimuli of size 7
size_samples = [profile.firing_rate_modifier(7 * np.pi / 180.0) for profile in size_profiles]
plt.figure('Size Samples')
plt.hist(size_samples, bins=np.arange(1, step=0.1))

# TODO: Account for non-optimal rotation/orientation angles.

#  Calculate mean and variance of collected samples --------------------------------------------
scale_samples = []
scale_samples.append(position_samples)
scale_samples.append(size_samples)

mu_s = np.mean(scale_samples)
var_s = np.var(scale_samples)

print ("Scale samples: mean%0.4f, var=%0.4f" % (mu_s, var_s))

if __name__ == '__main__':
    plt.ion()
