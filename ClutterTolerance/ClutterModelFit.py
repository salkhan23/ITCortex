# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:36:10 2015

Averaging Model Based on Zoccolan et al. 2005 - Multiple Response Normalization in Monkey
Inferotemporal Cortex The scatter distribution of figure 3. Scatter points do not have sufficient
resolution to locate each point individually. Current data collection is designed to quantify the
degree of spread from the averaging model - Mean can be ignored and only the standard deviation
is important.

@author: s362khan
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian_fcn(x, mean, sigma, amp):
    return amp * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


plt.ion()

# Load Data From File
with open('ZoccolanScatter.pkl', 'rb') as fid:
    data2 = pickle.load(fid)

with open('ZoccolanScatter3.pkl', 'rb') as fid:
    data3 = pickle.load(fid)

# Plot original data and prediction models --------------------------------------------------------

# 2 Objects
f, axArr = plt.subplots(2, 1, sharex=True)
f.subplots_adjust(hspace=0.1, wspace=0.05)
axArr[0].scatter(data2['SumIsolatedResp'], data2['RespPairs'], label='Orignal Data')

xLinear = np.arange(150)
axArr[0].plot(xLinear, 0.5 * xLinear, label='Average', color='r')
axArr[0].plot(xLinear, xLinear, label='Sum', color='g')
axArr[0].legend(loc='best')
axArr[0].set_ylim([0, 150])
axArr[0].set_ylabel("Response to object pairs R(A + B)")
axArr[0].set_title('2 Objects')

# 3 Objects
axArr[1].scatter(data3['SumIsolatedResp'], data3['RespTrips'],
                 label='Orignal Data')
axArr[1].plot(xLinear, 1.0 / 3 * xLinear, label='Average', color='r')
axArr[1].plot(xLinear, xLinear, label='Sum', color='g')
axArr[1].legend(loc='best')
axArr[1].set_ylim([0, 150])
axArr[1].set_xlim([0, 150])
axArr[1].set_xlabel("Sum Isolated Responses")
axArr[1].set_ylabel("Response to objects Trips R(A + B + C)")
axArr[1].set_title('3 Objects')

f.suptitle('Scatter Plots of Original Data', fontsize=16)

# Collect Deviations from Average model for all points --------------------------------------------
DevArray = np.array([])

normD = data2['RespPairs'] / data2['SumIsolatedResp']
deviation = normD - 0.5
DevArray = np.append(DevArray, deviation)

normD = data3['RespTrips'] / data3['SumIsolatedResp']
deviation = normD - 1.0 / 3
DevArray = np.append(DevArray, deviation)

plt.figure('Histogram')
plt.title('Histogram of Deviations from Average Model')
n, bins, patches = plt.hist(DevArray, bins=50)

# Fit Histogram to Gaussian, given the large number of points
pOpt, pCov = curve_fit(gaussian_fcn, bins[:-1], n)

xLinear = np.arange(-1, 1, step=0.01)
label = 'Best Fit Gaussian. mean = %0.2f, sigma=%0.2f, Amp=%0.2f' % (pOpt[0], pOpt[1], pOpt[2])
print(label)

plt.plot(xLinear, gaussian_fcn(xLinear, *pOpt), label=label, color='r', linewidth=2)
plt.legend(loc='best')
plt.xlabel('Normalized deviation from average model prediction')
plt.ylabel('Frequency')

# Generate Sample Data ----------------------------------------------------------------------------
nNeurons = 500
nObjArray = [2, 3]

f, axArr = plt.subplots(1, len(nObjArray), sharey=True)
f.subplots_adjust(hspace=0.05, wspace=0.05)
f.suptitle('Simulated Neuron Population Responses to Multiple Objects', size=40)

for idx, nObj in enumerate(nObjArray):
    # Random Firing Rate modifiers for individual objects objects
    # For now, replace we predictions of isolated responses of each object
    isolatedResp = np.random.uniform(size=(nNeurons, nObj))
    print isolatedResp.shape

    sumIsolatedResp = np.sum(isolatedResp, axis=1)
    RespJoint = 1.0 / nObj * sumIsolatedResp + \
        np.random.normal(loc=0, scale=pOpt[1], size=nNeurons)

    RespJoint[RespJoint < 0] = 0

    axArr[idx].set_title('%i Objects' % nObj, fontsize=34)
    axArr[idx].scatter(sumIsolatedResp, RespJoint, label='Generated Responses')
    axArr[idx].plot(sumIsolatedResp, 1.0 / nObj * sumIsolatedResp,
                    label='Average Model Prediction', color='r')
    axArr[idx].plot(sumIsolatedResp, 1.0 * sumIsolatedResp,
                    label='Linear sum', color='g')
    axArr[idx].legend(loc='best', fontsize=34)
    axArr[idx].set_xlim(0, 2)
    axArr[idx].set_ylim(0, 2)
    axArr[idx].tick_params(axis='x', labelsize=30)
    axArr[idx].tick_params(axis='y', labelsize=30)
    axArr[idx].set_xlabel('Normalized Sum of Isolated Responses', fontsize=34)
    

axArr[0].set_ylabel('Normalized Response to Multiple Objects', fontsize=34)
