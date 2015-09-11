# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:29:46 2015

The objective of this file is to create adjustment samples to account for non optimal position size,
rotation, etc tuning of neurons in the kurtois selectivity

@author: s362khan
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Do relative imports of the main folder to get files in sibling directories
top_level_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if top_level_dir_path not in sys.path:
    sys.path.append(top_level_dir_path)

from PositionTolerance import gaussian_position_profile as gpt
from ObjectSelectivity import activity_fraction_fit as selectivity_distribution


n = 1000
selectivity_arr = selectivity_distribution.get_activity_fraction_sparseness(n)
plt.hist(selectivity_arr, bins= np.arange(1, step=0.1))

# Account for non-optimal position tuning. All objects were presented foveally at (0,0).
# However RF centers of IT neurons and their preferred positions are scattered around the fovea.
position_profiles = []

for index in np.arange(n):
    position_profiles.append(gpt.GaussianPositionProfile(selectivity_arr[index]))


position_samples = [ profile.firing_rate_modifier(0,0) for profile in position_profiles]

# Account for non optimal size




if __name__ == '__main__':
    plt.ion()
