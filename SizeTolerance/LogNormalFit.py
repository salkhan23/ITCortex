# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:27:31 2015

Model Size Tuning of IT Neurons as described in Ito et. al. -1995 -
Size and Position Invariances of Neuronal Responses in Monkey Inferotemporal Cortex

@author: s362khan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle
import scipy.stats as ss


def get_best_fit_gamma(input_data):
    """
    Returns the best fit gamma parameters that fit the data using maximum likelihood data fitting.
    Returns: (1) alpha parameter of best fit gamma distribution
             (2) scale parameters of best fit gamma distribution. (Location parameter = 0)
             (3) Log likelihood ratio of best fit
    """
    alpha_arr = np.arange(start=0.1, stop=5, step=0.1)
    scale_arr = np.arange(start=0.1, stop=10, step=0.1)

    llrs = np.ones(shape=(alpha_arr.shape[0], scale_arr.shape[0])) * -10000.0

    for ii, alpha in enumerate(alpha_arr):
        for jj, scale in enumerate(scale_arr):
            prob = ss.gamma.pdf(input_data, a=alpha, scale=scale)
            llrs[ii][jj] = np.log(prob).sum()

    # Get the maximum Log Likelihood ratio (LLR)
    max_alpha_idx, max_scale_idx = np.unravel_index(llrs.argmax(), llrs.shape)

    return alpha_arr[max_alpha_idx],\
        scale_arr[max_scale_idx],\
        llrs[max_alpha_idx][max_scale_idx]


def get_best_fit_lognormal(input_data):
    """
    Find the best fit lognormal parameters that fit the data using maximum likelihood
         data fitting.
    :rtype : (1) shape parameter of best fit lognormal distribution
             (2) scale parameter of best fit lognormal distribution
             (3) Log likelihood ratio of best fit

    :param input_data: data to fit
    """
    shape_arr = np.arange(start=0.1, stop=5, step=0.1)
    scale_arr = np.arange(start=0.1, stop=10, step=0.1)

    llrs = np.ones(shape=(shape_arr.shape[0], scale_arr.shape[0])) * -10000.0

    for ii, shape in enumerate(shape_arr):
        for jj, scale in enumerate(scale_arr):
            prob = ss.lognorm.pdf(input_data, s=shape, scale=scale)
            llrs[ii][jj] = np.log(prob).sum()

    max_shape_idx, max_scale_idx = np.unravel_index(llrs.argmax(), llrs.shape)

    return shape_arr[max_shape_idx], \
        scale_arr[max_scale_idx], \
        llrs[max_shape_idx][max_scale_idx]


plt.ion()

with open("Ito95Data.pkl", 'rb') as fid:
    data = pickle.load(fid)

# *************************************************************************************************
# Population level Distribution
# *************************************************************************************************

# (1) Optimum Size Distribution
# -------------------------------------------------------------------------------------------------
optimum_size_dist = data["optSize"]
optimum_size_rf_size = data["optSizeRfSize"]

# Maximum stimulus size tested was 31x29 degrees. The distribution of optimum size (Figure 6)
# states two peaks, at 3.4 and 27 degrees. We ignore the peak at 27 degrees as it may be an
# artifact of the testing. All neurons with RFs > 27 may potentially have RF sizes greater then
# the maximum stimulus size tested and their results may have been grouped in the 27 degree bin.
#
# We assume a peak at 3.4 and a long tail ending at the RF size.

cut_off = 26  # Ignore all data >= 27 degrees
valid_idx = np.where(optimum_size_dist < cut_off)[0]  # Returns a tuple, take first element
optimum_size_dist = optimum_size_dist[valid_idx]
optimum_size_rf_size = optimum_size_rf_size[valid_idx]

f_opt_size, ax_arr = plt.subplots(1, 2)

ax_arr[0].scatter(optimum_size_rf_size, optimum_size_dist)
ax_arr[0].set_title("Preferred Size Distribution")
ax_arr[0].set_xlabel("RF Size (Square root of areal extent of Receptive Field)")
ax_arr[0].set_ylabel("Preferred Size")

ax_arr[0].set_xscale('log')
ax_arr[0].set_yscale('log')
ax_arr[0].xaxis.set_minor_formatter(FormatStrFormatter("%d"))
ax_arr[0].yaxis.set_minor_formatter(FormatStrFormatter("%d"))
ax_arr[0].set_xlim([4, 60])
ax_arr[0].set_ylim([1, 30])

x_arr = np.arange(start=4, stop=60, step=1)
ax_arr[0].plot(x_arr, x_arr, label='y=x')
ax_arr[0].legend()

# Fit the data
ax_arr[1].set_title('Histogram of Preferred Size')
ax_arr[1].set_ylabel('pdf')
ax_arr[1].set_xlabel('Stimulus Size')
ax_arr[1].hist(optimum_size_dist, bins=np.arange(0, 30, step=1), normed=True)

# Maximum Likelihood gamma fit  ------------------------------------------------
alpha_max, scale_max, llr_max = get_best_fit_gamma(optimum_size_dist)

label = 'Preferred Size Best Gamma Fit: alpha=%0.2f, scale=%0.2f, LLR=%0.2f' \
        % (alpha_max, scale_max, llr_max)

print label
x_arr = np.arange(30, step=0.01)
ax_arr[1].plot(x_arr, ss.gamma.pdf(x_arr, a=alpha_max, scale=scale_max), label=label)


# Maximum Likelihood log normal fit assume a location of 0  -------------------
shape_max, scale_max, llr_max = get_best_fit_lognormal(optimum_size_dist)

label = 'Preferred Size Best Lognormal Fit: shape=%0.2f, scale=%0.2f, LLR=%0.2f' \
        % (shape_max, scale_max, llr_max)

print label
x_arr = np.arange(30, step=0.01)
ax_arr[1].plot(x_arr, ss.lognorm.pdf(x_arr, s=shape_max, scale=scale_max), label=label)
ax_arr[1].legend(loc='best', fontsize='small')


# (2) Size Bandwidth Distribution
# ------------------------------- -----------------------------------------------------------------
size_bw_dist = data["sizeDist"]
size_bw_rf_size = data["sizeDistRfSize"]

# For bandwidths > 4 octaves the upper cutoff was not determined and it is likely that
# the results of all larger bandwidths are grouped into the 4 bw bin. We ignore the data
# in the > 4 bandwidth bin and assume it follows the same trend as in the lower bandwidths
# continues.
cutoff = 4
valid_idx = np.where(size_bw_dist < 4)[0]  # Returns a tuple, take first element
size_bw_dist = size_bw_dist[valid_idx]
size_bw_rf_size = size_bw_rf_size[valid_idx]

f_size_dist, ax_arr = plt.subplots(1, 2)
ax_arr[0].scatter(size_bw_rf_size, size_bw_dist)
ax_arr[0].set_xscale('log')
ax_arr[0].xaxis.set_minor_formatter(FormatStrFormatter("%d"))
ax_arr[0].set_xlim([4, 60])
ax_arr[0].set_ylim([1, 4])
ax_arr[0].set_xlabel('rf Size (Square root of areal extent of Receptive Field)')
ax_arr[0].set_ylabel('Size bandwidth (octaves)')
ax_arr[0].set_title("Size Bandwidth Distribution")

ax_arr[1].set_title('Distribution of Size bandwidth')
ax_arr[1].hist(size_bw_dist, bins=np.arange(0, 5, step=0.5), normed=True)

# Gamma Fit
alpha_max, scale_max, llr_max = get_best_fit_gamma(size_bw_dist)
label = 'Size BW Distribution Best Gamma Fit: alpha=%0.2f, scale=%0.2f, LLR=%0.2f' \
        % (alpha_max, scale_max, llr_max)

print label
x_arr = np.arange(7, step=0.01)
ax_arr[1].plot(x_arr, ss.gamma.pdf(x_arr, a=alpha_max, scale=scale_max), label=label)

shape_max, scale_max, llr_max = get_best_fit_lognormal(size_bw_dist)
label = 'Size BW Distribution Best Lognormal Fit: shape=%0.2f, scale=%0.2f, LLR=%0.2f' \
        % (shape_max, scale_max, llr_max)

print label
x_arr = np.arange(7, step=0.01)
ax_arr[1].plot(x_arr, ss.lognorm.pdf(x_arr, s=shape_max, scale=scale_max), label=label)
ax_arr[1].legend(loc='best', fontsize='small')

# *************************************************************************************************
# Modeling Individual Neurons
# *************************************************************************************************
#
# Give a neurons
#   (1) preferred size (pref_size) in degrees,
#   (2) size bandwidth (size_bw) in octaves,
#   (3) rf_size = areal extent of the RF
#
# A normal distribution in the log scale is generated with parameters:
#   mu      = np.log2(prefSize)
#   sigma   = (sizeBw/2) / np.sqrt(2*np.log(2))
#
# The range of stimuli that neuron responds to is given by 0 - max_stim_size. The max stimulus
# size the neuron responds to is given by: 2/np.pi * RfSize. Here RF size is area extent of the
# RF which we assume to be circular, and = sqrt(pi)*r. Stimulus Size is defined as the distance
# between the outer edges along the longest axis of the stimulus, If we again assume a circular
# stimulus, stimulus size = diameter = 2*r. there the max. stimulus size the neuron responds
# to is given by RfSize/sqrt(pi)*2

# The object pf this section is not to find best fits of the data but to show individual neuronal
#  tuning curves in the paper can be generated using this approach.

# Neuron 1 - prefSize = 5.2, and sizeBw = 1.35  ---------------------------------------------------
pref_size = 5.2
size_bw = 1.35
rf_size = 22.

# Get the maximum stimulus size
max_stim_size = 2 / np.sqrt(np.pi) * rf_size

n1_stim_size = data['n1Size']
n1_firing_rate = data['n1FiringRate']

fig_n1, ax_arr = plt.subplots(1, 2, sharey=True)
ax_arr[0].plot(n1_stim_size, n1_firing_rate, marker='o', markersize=10, label='Original Data')
ax_arr[0].set_title('Linear Scale')
ax_arr[1].plot(np.log2(n1_stim_size), n1_firing_rate, marker='o', markersize=10,
               label='Original Data')
ax_arr[1].set_title('Log Scale')

# Generate data from tuning profile
# Method 1
x_arr = np.arange(0, np.log2(max_stim_size), step=0.01)
norm_pdf = ss.norm.pdf(x_arr, loc=np.log2(pref_size), scale=(size_bw / 2) / np.sqrt(2 * np.log(2)))
ax_arr[1].plot(x_arr, norm_pdf / norm_pdf.max(), label='generated')
ax_arr[1].legend()
ax_arr[1].set_xlabel(r'$log_2(Stimulus\ Size)\ [Degrees]$')

ax_arr[0].plot(2. ** x_arr, norm_pdf / norm_pdf.max(), label='Generated-1')
ax_arr[0].legend()
ax_arr[0].set_xlabel('Stimulus Size [Degrees]')
fig_n1.suptitle("Neuron 1 - Figure 3")

# Method 2
x_arr_lin = np.arange(0, max_stim_size, step=0.01)
sigma = (size_bw / 2) / np.sqrt(2 * np.log(2))
log_domain_tuning_curve = np.exp(-(np.log2(x_arr_lin) - np.log2(pref_size))**2 / (2 * sigma**2))

ax_arr[0].plot(x_arr_lin, log_domain_tuning_curve, label='Generated-2')
ax_arr[0].legend()

# Neuron 2 - prefSize=26.1, sizeBw>5octaves ------------------------------------------------------
pref_size = 13.1
size_bw = 8
rf_size = 50
# These provide a better fit to the data, the original values from the paper are prefSize=26.1
# and  sizeBw> 5 octaves. The difference in response at26.1 and 13.1 is small.

# Get the maximum stimulus size
max_stim_size = 2 / np.sqrt(np.pi) * rf_size

n2_stim_size = data['n2Size']
n2_firing_rate = data['n2FiringRate']

fig_n2, ax_arr = plt.subplots(1, 2, sharey=True)
ax_arr[0].plot(n2_stim_size, n2_firing_rate, marker='o', markersize=10, label='Original Data')
ax_arr[0].set_title('Linear Scale')
ax_arr[1].plot(np.log2(n2_stim_size), n2_firing_rate, marker='o', markersize=10,
               label='Original Data')
ax_arr[1].set_title('Log Scale')

# Generate data from tuning profile
x_arr = np.arange(0, np.log2(max_stim_size), step=0.01)
norm_pdf = ss.norm.pdf(x_arr, loc=np.log2(pref_size), scale=(size_bw / 2) / np.sqrt(2 * np.log(2)))
ax_arr[1].plot(x_arr, norm_pdf / norm_pdf.max(), label='generated')
ax_arr[1].legend()
ax_arr[1].set_xlabel(r'$log_2(Stimulus\ Size)\ [Degrees]$')

ax_arr[0].plot(2. ** x_arr, norm_pdf / norm_pdf.max(), label='Generated')
ax_arr[0].legend()
ax_arr[0].set_xlabel('Stimulus Size [Degrees]')
fig_n2.suptitle("Neuron 2 - Figure 4 - Large Bandwidth")


# Neuron 3 - prefSize =  only largest stimulus size  ----------------------------------------------
pref_size = 56.0
size_bw = 3.3
rf_size = 28.
# These provide a better fit to the data, the original values from the paper are prefSize=26.1
# and  sizeBw> 5 octaves. The difference in response at26.1 and 13.1 is small.

# Get the maximum stimulus size
max_stim_size = 2 / np.sqrt(np.pi) * rf_size

n3_stim_size = data['n3Size']
n3_firing_rate = data['n3FiringRate']

fig_n3, ax_arr = plt.subplots(1, 2, sharey=True)
ax_arr[0].plot(n3_stim_size, n3_firing_rate, marker='o', markersize=10, label='Original Data')
ax_arr[0].set_title('Linear Scale')
ax_arr[1].plot(np.log2(n3_stim_size), n3_firing_rate, marker='o', markersize=10,
               label='Original Data')
ax_arr[1].set_title('Log Scale')

# Generate data from tuning profile
x_arr = np.arange(0, np.log2(max_stim_size), step=0.01)
norm_pdf = ss.norm.pdf(x_arr, loc=np.log2(pref_size), scale=(size_bw / 2) / np.sqrt(2 * np.log(2)))
ax_arr[1].plot(x_arr, norm_pdf / norm_pdf.max(), label='generated')
ax_arr[1].legend()
ax_arr[1].set_xlabel(r'$log_2(Stimulus\ Size)\ [Degrees]$')

ax_arr[0].plot(2. ** x_arr, 2. ** (norm_pdf / norm_pdf.max()) - 1, label='Generated')
ax_arr[0].legend()
ax_arr[0].set_xlabel('Stimulus Size [Degrees]')
fig_n3.suptitle("Neuron 3 - Figure 5 - Response only to largest Stimulus size")
