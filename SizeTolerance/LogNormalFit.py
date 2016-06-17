# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:27:31 2015

Model Size Tuning of IT Neurons as described in Ito et. al. -1995 -
Size and Position Invariances of Neuronal Responses in Monkey Inferotemporal Cortex

@author: s362khan
"""
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from matplotlib.ticker import FormatStrFormatter
import pickle
import scipy.stats as ss


def get_best_fit_gamma(input_data):
    """
    Returns the best fit gamma parameters that fit the data using maximum likelihood data fitting.
    Returns: (1) alpha parameter of best fit gamma distribution
             (2) scale parameters of best fit gamma distribution. (Location parameter = 0)
             (3) Log likelihood ratio of best fit
             :param input_data:
    """
    alpha_arr = np.arange(start=0.1, stop=10, step=0.1)
    scale_arr = np.arange(start=0.1, stop=10, step=0.1)

    llrs = np.ones(shape=(alpha_arr.shape[0], scale_arr.shape[0])) * -10000.0

    for count, alpha in enumerate(alpha_arr):
        for jj, scale in enumerate(scale_arr):
            prob = ss.gamma.pdf(input_data, a=alpha, scale=scale)
            llrs[count][jj] = np.log(prob).sum()

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

    for count, shape in enumerate(shape_arr):
        for jj, scale in enumerate(scale_arr):
            prob = ss.lognorm.pdf(input_data, s=shape, scale=scale, loc=0)
            llrs[count][jj] = np.log(prob).sum()

    max_shape_idx, max_scale_idx = np.unravel_index(llrs.argmax(), llrs.shape)

    return shape_arr[max_shape_idx], \
        scale_arr[max_scale_idx], \
        llrs[max_shape_idx][max_scale_idx]


plt.ion()

with open("Ito95Data.pkl", 'rb') as fid:
    data = pickle.load(fid)

# ---------------------------------------------------------------------------------------
# Parameter Distributions
# ---------------------------------------------------------------------------------------

# (A) Preferred Size Distribution  --------------------------

optimum_size_dist = data["optSize"]
optimum_size_rf_size = data["optSizeRfSize"]

# Maximum stimulus size tested was 31x29 degrees. The distribution of optimum size (Figure 6)
# shows 2 peaks: 3.4 and 27 degrees. However as Figure 7B shows many neurons had RF size > 50
# degrees. It is possible that their preferred sizes were larger than the maximum stimulus size
# tested and their results were grouped together into the same bin. We ignore the peak at 27
# degrees as it may be an abnormality due to testing constraints.

# Ignore all optimal sizes > 27 degrees as theses results are unreliable. Do not limit on RF sizes
# as some RFs > max stimulus sizes had small optimum sizes.
cut_off = 26  # Ignore all data >= 27 degrees
valid_idx = np.where(optimum_size_dist < cut_off)[0]  # Returns a tuple, take first element

optimum_size_dist = optimum_size_dist[valid_idx]
optimum_size_rf_size = optimum_size_rf_size[valid_idx]

# Plot (A) scatter plot or original data  and (B) histogram + ML Fit
font_size = 34
f_opt_size, ax_arr = plt.subplots(1, 2)
f_opt_size.suptitle("Size Tuning: Preferred Size Distribution", fontsize=font_size + 10)

# Scatter plot or original data. Similar to figure 7B but with the added constraint to ignore
# optimum size >= 27 degrees
ax_arr[0].scatter(optimum_size_rf_size, optimum_size_dist, label='Original Data')
ax_arr[0].set_title("Preferred Size Distribution", fontsize=font_size)
ax_arr[0].set_xlabel("RF Size (Square root of areal extent (Degrees))",
                     fontsize=font_size)
ax_arr[0].set_ylabel("Preferred Size", fontsize=font_size)

ax_arr[0].set_xscale('log')
ax_arr[0].set_yscale('log')
ax_arr[0].xaxis.set_minor_formatter(FormatStrFormatter("%d"))
ax_arr[0].yaxis.set_minor_formatter(FormatStrFormatter("%d"))
ax_arr[0].xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax_arr[0].yaxis.set_major_formatter(FormatStrFormatter("%d"))
ax_arr[0].grid(which='both')
ax_arr[0].tick_params(axis='x', which='both', labelsize=font_size)
ax_arr[0].tick_params(axis='y', which='both', labelsize=font_size)

ax_arr[0].set_xlim([4, 60])
ax_arr[0].set_ylim([1, 30])

# plot line where receptive field size = optimum stimulus size.
x_arr = np.arange(start=4, stop=60, step=1)
ax_arr[0].plot(x_arr, x_arr, label='y=x')
ax_arr[0].legend(loc='best', fontsize=font_size)

# B Plot histogram and fit(s), Figure 6.
ax_arr[1].set_title('Histogram of Preferred Size')
ax_arr[1].set_ylabel('pdf')
ax_arr[1].set_xlabel('Stimulus Size')
ax_arr[1].hist(optimum_size_dist, bins=np.arange(0, 30, step=1), normed=True)

# Maximum Likelihood gamma fit
alpha_max, scale_max, llr_max = get_best_fit_gamma(optimum_size_dist)

label = 'ML Gamma Fit: alpha=%0.2f, scale=%0.2f, LLR=%0.2f' \
        % (alpha_max, scale_max, llr_max)

print label
x_arr = np.arange(30, step=0.01)
ax_arr[1].plot(x_arr, ss.gamma.pdf(x_arr, a=alpha_max, scale=scale_max),
               label="\n".join(textwrap.wrap(label, 20)))

# Maximum Likelihood lognormal fit assume a location of 0
shape_max, scale_max, llr_max = get_best_fit_lognormal(optimum_size_dist)

label = 'ML Lognormal Fit: shape=%0.2f, scale=%0.2f, LLR=%0.2f' \
        % (shape_max, scale_max, llr_max)

print label
ax_arr[1].plot(x_arr, ss.lognorm.pdf(x_arr, s=shape_max, scale=scale_max),
               label="\n".join(textwrap.wrap(label, 20)))

ax_arr[1].legend(loc='best', fontsize=font_size)
ax_arr[1].tick_params(axis='x', labelsize=font_size)
ax_arr[1].tick_params(axis='y', labelsize=font_size)
ax_arr[1].set_title('Histogram of Preferred Size', fontsize=font_size)
ax_arr[1].set_ylabel('frequency', fontsize=font_size)
ax_arr[1].set_xlabel('Stimulus Size (Degrees)', fontsize=font_size)


# (2) Size Bandwidth Distribution -------------------------------------------------------
size_bw_dist = data["sizeDist"]
size_bw_rf_size = data["sizeDistRfSize"]

# In figure 2, two peaks in size bandwidths can be found: (1) between 1 and 2 octaves and
# (2) greater than 4 octaves. The figure additionally does not include bandwidths for cells that
# only responded to the largest stimulus size tested. Additionally the results of all bandwidths >
# than 4 octaves were grouped together into a single bin. We ignore the data in the > 4 bandwidth
# bin and assume it follows the same trend as in the lower bandwidths continues.
cutoff = 4
valid_idx = np.where(size_bw_dist < 4)[0]  # Returns a tuple, take first element

size_bw_dist = size_bw_dist[valid_idx]
size_bw_rf_size = size_bw_rf_size[valid_idx]

# Plot (A) scatter plot or original data  and (B) histogram + ML Fit
font_size = 34
f_size_dist, ax_arr = plt.subplots(1, 2)
f_size_dist.suptitle("Size Tuning: Size BW Distribution", fontsize=font_size + 10)

# Scatter plot or original data. Similar to figure 7A but with the added constraint to ignore
# bandwidths greater than 4 octaves.
ax_arr[0].scatter(size_bw_rf_size, size_bw_dist, label='Original Data')
ax_arr[0].set_title("Size Bandwidth Distribution", fontsize=font_size)
ax_arr[0].set_xlabel("RF Size (Square root of areal extent (Degrees))",
                     fontsize=font_size)
ax_arr[0].set_ylabel("Bandwidth (octaves)", fontsize=font_size)

ax_arr[0].set_xscale('log')
ax_arr[0].set_yscale('log')
ax_arr[0].xaxis.set_minor_formatter(FormatStrFormatter("%d"))
ax_arr[0].yaxis.set_minor_formatter(FormatStrFormatter("%d"))
ax_arr[0].xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax_arr[0].yaxis.set_major_formatter(FormatStrFormatter("%d"))
ax_arr[0].grid(which='both')
ax_arr[0].tick_params(axis='x', which='both', labelsize=font_size - 10)
ax_arr[0].tick_params(axis='y', which='both', labelsize=font_size)

ax_arr[0].set_xlim([4, 60])
ax_arr[0].set_ylim([1, 4])

# B Plot histogram and fit(s), Figure 2.
ax_arr[1].set_title('Distribution of Size bandwidth')
ax_arr[1].hist(size_bw_dist, bins=np.arange(0, 5, step=0.5), normed=True)

# Maximum Likelihood gamma fit
alpha_max, scale_max, llr_max = get_best_fit_gamma(size_bw_dist)
label = 'ML Gamma Fit: alpha=%0.2f, scale=%0.2f, LLR=%0.2f' \
        % (alpha_max, scale_max, llr_max)

print label
x_arr = np.arange(7, step=0.01)
ax_arr[1].plot(x_arr, ss.gamma.pdf(x_arr, a=alpha_max, scale=scale_max),
               label="\n".join(textwrap.wrap(label, 15)))

# Maximum Likelihood lognormal fit assume a location of 0
shape_max, scale_max, llr_max = get_best_fit_lognormal(size_bw_dist)
label = 'ML Lognormal Fit: shape=%0.2f, scale=%0.2f, LLR=%0.2f' \
        % (shape_max, scale_max, llr_max)

print label
x_arr = np.arange(7, step=0.01)
ax_arr[1].plot(x_arr, ss.lognorm.pdf(x_arr, s=shape_max, scale=scale_max),
               label="\n".join(textwrap.wrap(label, 15)))

ax_arr[1].legend(loc='best', fontsize=font_size)
ax_arr[1].tick_params(axis='x', labelsize=font_size)
ax_arr[1].tick_params(axis='y', labelsize=font_size)
ax_arr[1].set_title('Histogram of Size Bandwidth', fontsize=font_size)
ax_arr[1].set_ylabel('frequency', fontsize=font_size)
ax_arr[1].set_xlabel('Stimulus Size (Degrees)', fontsize=font_size)

# ---------------------------------------------------------------------------------------
# Modeling Individual Neurons
# ---------------------------------------------------------------------------------------
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

# Neuron 1 - prefSize = 5.2, and sizeBw = 1.35  ---------------------------------------------
pref_size_n1 = 5.2
size_bw_n1 = 1.35
rf_size_n1 = 22.

# Get the maximum stimulus size
max_stim_size_n1 = 2 / np.sqrt(np.pi) * rf_size_n1

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
x_arr_n1 = np.arange(0, np.log2(max_stim_size_n1), step=0.01)
norm_pdf_n1 = ss.norm.pdf(x_arr_n1,
                          loc=np.log2(pref_size_n1),
                          scale=(size_bw_n1 / 2) / np.sqrt(2 * np.log(2)))

ax_arr[1].plot(x_arr_n1, norm_pdf_n1 / norm_pdf_n1.max(), label='generated')
ax_arr[1].legend()
ax_arr[1].set_xlabel(r'$log_2(Stimulus\ Size)\ [Degrees]$')

ax_arr[0].plot(2. ** x_arr_n1, norm_pdf_n1 / norm_pdf_n1.max(), label='Generated-1')
ax_arr[0].legend()
ax_arr[0].set_xlabel('Stimulus Size [Degrees]')
fig_n1.suptitle("Neuron 1 - Figure 3")

# Method 2
x_arr_lin_n1 = np.arange(0, max_stim_size_n1, step=0.01)
sigma = (size_bw_n1 / 2) / np.sqrt(2 * np.log(2))
log_domain_tuning_curve = \
    np.exp(-(np.log2(x_arr_lin_n1) - np.log2(pref_size_n1))**2 / (2 * sigma**2))

ax_arr[0].plot(x_arr_lin_n1, log_domain_tuning_curve, label='Generated-2')
ax_arr[0].legend()

# Neuron 2 - prefSize=26.1, sizeBw>5octaves ---------------------------------------------
pref_size_n2 = 13.1
size_bw_n2 = 8
rf_size_n2 = 50
# These provide a better fit to the data, the original values from the paper are prefSize=26.1
# and  sizeBw> 5 octaves. The difference in response at 26.1 and 13.1 is very small.

# Get the maximum stimulus size
max_stim_size_n2 = 2 / np.sqrt(np.pi) * rf_size_n2

n2_stim_size = data['n2Size']
n2_firing_rate = data['n2FiringRate']

fig_n2, ax_arr = plt.subplots(1, 2, sharey=True)
ax_arr[0].plot(n2_stim_size, n2_firing_rate, marker='o', markersize=10, label='Original Data')
ax_arr[0].set_title('Linear Scale')
ax_arr[1].plot(np.log2(n2_stim_size), n2_firing_rate, marker='o', markersize=10,
               label='Original Data')
ax_arr[1].set_title('Log Scale')

# Generate data from tuning profile
x_arr_n2 = np.arange(0, np.log2(max_stim_size_n2), step=0.01)
norm_pdf_n2 = ss.norm.pdf(x_arr_n2,
                          loc=np.log2(pref_size_n2),
                          scale=(size_bw_n2 / 2) / np.sqrt(2 * np.log(2)))

ax_arr[1].plot(x_arr_n2, norm_pdf_n2 / norm_pdf_n2.max(), label='generated')
ax_arr[1].legend()
ax_arr[1].set_xlabel(r'$log_2(Stimulus\ Size)\ [Degrees]$')

ax_arr[0].plot(2. ** x_arr_n2, norm_pdf_n2 / norm_pdf_n2.max(), label='Generated')
ax_arr[0].legend()
ax_arr[0].set_xlabel('Stimulus Size [Degrees]')
fig_n2.suptitle("Neuron 2 - Figure 4 - Large Bandwidth")


# Neuron 3 - prefSize =  only largest stimulus size  ------------------------------------
pref_size_n3 = 56.0
size_bw_n3 = 3.3
rf_size_n3 = 28.

# Get the maximum stimulus size
max_stim_size_n3 = 2 / np.sqrt(np.pi) * rf_size_n3

n3_stim_size = data['n3Size']
n3_firing_rate = data['n3FiringRate']

fig_n3, ax_arr = plt.subplots(1, 2, sharey=True)
ax_arr[0].plot(n3_stim_size, n3_firing_rate, marker='o', markersize=10, label='Original Data')
ax_arr[0].set_title('Linear Scale')
ax_arr[1].plot(np.log2(n3_stim_size), n3_firing_rate, marker='o', markersize=10,
               label='Original Data')
ax_arr[1].set_title('Log Scale')

# Generate data from tuning profile
x_arr_n3 = np.arange(0, np.log2(max_stim_size_n3), step=0.01)
norm_pdf_n3 = ss.norm.pdf(x_arr_n3,
                          loc=np.log2(pref_size_n3),
                          scale=(size_bw_n3 / 2) / np.sqrt(2 * np.log(2)))

ax_arr[1].plot(x_arr_n3, norm_pdf_n3 / norm_pdf_n3.max(), label='generated')
ax_arr[1].legend()
ax_arr[1].set_xlabel(r'$log_2(Stimulus\ Size)\ [Degrees]$')

ax_arr[0].plot(2. ** x_arr_n3, 2. ** (norm_pdf_n3 / norm_pdf_n3.max()) - 1, label='Generated')
ax_arr[0].legend()
ax_arr[0].set_xlabel('Stimulus Size [Degrees]')
fig_n3.suptitle("Neuron 3 - Figure 5 - Response only to largest Stimulus size")


# -------------------------------------------------------------------------------------------------
# Plot all 3 Log scale Tuning curves in a single figure
f_size = 50
fig_n4, ax_arr = plt.subplots(1, 3, sharey=True)
fig_n4.subplots_adjust(wspace=0.1)


# ax_arr[0].plot(np.log2(n1_stim_size), n1_firing_rate,
#                marker='o', markersize=10, label='Original Data', linewidth=2)

ax_arr[0].vlines(
    np.log2(max_stim_size_n1),
    0,
    1,
    linewidth=3,
    linestyle='--',
    label='Max. size')

# ax_arr[1].plot(np.log2(n2_stim_size), n2_firing_rate,
#                marker='o', markersize=10, label='Original Data', linewidth=2)

ax_arr[1].vlines(
    np.log2(max_stim_size_n2),
    0,
    1,
    linewidth=3,
    linestyle='--',
    label='Max. size')

# ax_arr[2].plot(np.log2(n3_stim_size), n3_firing_rate,
#                marker='o', markersize=10, label='Original Data', linewidth=2)

ax_arr[2].vlines(
    np.log2(max_stim_size_n3),
    0,
    1,
    linewidth=3,
    linestyle='--',
    label='Max. size')


# Plot the Fitting Curves.
ax_arr[0].plot(x_arr_n1, norm_pdf_n1 / norm_pdf_n1.max(),
               label='', linewidth=2)
ax_arr[1].plot(x_arr_n2, norm_pdf_n2 / norm_pdf_n2.max(), label='', linewidth=2)
ax_arr[2].plot(x_arr_n3, norm_pdf_n3 / norm_pdf_n3.max(), label='Lognormal Fit', linewidth=2)


ax_arr[1].set_xlabel(r'$log_2(Stimulus\ Size)\ [Degrees]$', fontsize=f_size)
ax_arr[1].set_ylim([0, 1.1])
ax_arr[0].set_ylabel('Normalized Firing Rate (Spikes/s)', fontsize=f_size)

for ii in np.arange(3):
    ax_arr[ii].tick_params(axis='x', labelsize=f_size)
    ax_arr[ii].tick_params(axis='y', labelsize=f_size)
    # ax_arr[ii].grid()

# ax_arr[0].set_title("Narrow BW", fontsize=f_size)
# ax_arr[1].set_title("Wide BW", fontsize=f_size)
# ax_arr[2].set_title("Largest Size Only", fontsize=f_size)

#ax_arr[2].legend(loc='best', fontsize=f_size - 10)
ax_arr[1].set_xlim([0, 6])
ax_arr[2].set_xlim([0, 5.5])

# fig_n4.suptitle("Sample Size Tuning Curves", fontsize=f_size + 10)
