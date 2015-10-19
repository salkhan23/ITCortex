# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:37:11 2015

Ref:
[1] Neilson, Logothesis & Rainer - 2006 - Dissociation between Local Field Potentials & spiking
activity in Macaque Inferior Temporal Cortex reveals diagnosticity based encoding of complex
objects.

[2] Kovacs, Vogels & Orban -1995 - Selectivity of Macaque Inferior Temporal Neurons for Partially
Occluded Shapes.

[3] Oreilly et. al. - 2013 - Recurrent processing during object recognition.

@author: s362khan
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D


def exponential(x, a):
    return a * np.exp(-a * x)


def sigmoid(x, a, b):
    """ http://computing.dcu.ie/~humphrys/Notes/Neural/sigmoid.html """
    return 1.0 / (1 + np.exp(a*(x-b)))

# Code start ----------------------------------------------------------------------------
plt.ion()

# 1. Fit Diagnosticity data -------------------------------------------------------------
# Diagnosticity is defined as the ratio of the trail by trail variance of diagnostic parts
# (at all stimulus sizes) and the total trail by trail variance. If a neuron responded only to the
# diagnostic part, its ratio = 1. While if an image  did not preferentially respond to the
# diagnostic parts its ratio = 0.
#
# The total firing rate of the neuron is computed as a weighted average of the diagnostic tuning
# curve and the non-diagnostic tuning curve weighted by the diagnosticity.
#
# In supplementary material diagnosticity metric was compared to the difference between the net
# firing rates to the diagnostic and non-diagnostic parts. For all cases there was a positive
# correlation. Indicating neurons with higher diagnostic variances fired for more diagnostic
# parts than to non-diagnostic parts.

with open('Neilson2006.pkl', 'rb') as fid:
    data = pickle.load(fid)

''' PDF of Diagnostic variance '''
diagnosticity = data['diagVariance']

hist, bins = np.histogram(diagnosticity, bins=np.arange(100, step=1), density=True)

# Clean up zeros
idxs = np.nonzero(hist)
hist = hist[idxs]
bins = bins[idxs]

plt.figure('Diagnostic Variance Distribution')
plt.scatter(bins, hist, label='Original Data')

# Fit the pdf
pOptExp, pCovExp = curve_fit(exponential, bins, hist)
plt.plot(bins, exponential(bins, *pOptExp), label='Exponential a*exp(-ax): a=%f' % pOptExp[0])

# # Plot +- 1 SD around fit
# sigma = np.sqrt(pCovExp[0])
# plt.plot(bins, exponential(bins, pOptExp[0] + sigma), 'k--')
# plt.plot(bins, exponential(bins, pOptExp[0] - sigma), 'k--')

# 2. Generate Diagnosticity Distribution ------------------------------------------------
# Given the diagnosticity pdf generate random variables that follow the distribution - inverse CDF
#
# Ref:
# [1] http://www.ece.virginia.edu/mv/edu/prob/stat/random-number-generation.pdf
# [2] SYSD 750 - Lecture Notes
#
# CDF = Fx(x) = y = 1-np.exp(-a*x)
# ln(y-1) = -a*x, and y in uniformly distributed over 0, 1
n = 100

y_arr = np.random.uniform(size=n)
genDiagVar = -np.log(y_arr) / pOptExp[0]

hist, bins = np.histogram(genDiagVar, bins=np.arange(100), density=True)
# Clean up zeros
idxs = np.nonzero(hist)
hist = hist[idxs]
bins = bins[idxs]

plt.figure('Diagnostic Variance Distribution')
plt.scatter(bins, hist, marker='+', color='red', label='Generated Data')
plt.title('Distribution of diagnosticity across IT')
plt.xlabel('Diagnosticity (%)')
plt.ylabel('Probability Density')
plt.legend()

# 3. Diagnostic and Non-Diagnostic Tuning Curves Fits ------------------------------------------
#
# Data Problems:
# Neilson 2006 - 3 points per object. Figure 1b & 2b.
# Kovacs 1995  - 4 points per object. Figure 8.
# Oreilly 2013 - Many points, but generated data. Also not firing rates, mean recognition
#                performance of Leabra model for trained objects, Figure 5(a). Only checking if a
#                sigmoid provides a good fit for this model as well.
#
#
# Neilson 2006:
# [1] For each occlusion level, diagnostic firing rates are always higher than
#     non-diagnostic rates.
# [2] Even though there are variations between the firing rates of individual neurons, not enough
#     statistical information is presented to model these variations. Results for only an exemplar
#     neuron and entire population of recorded neurons.
# [3] Population level diagnostic data is misleading, it is flat, but should be incrementing
#     (at a rate faster than non-diagnostic data). This is an artifact of the way they tested.
#     Each individual diagnostic part was tested to show the animal could correctly decode the
#     identity of the object from it. Also it be be a result of the small stimulus set that they
#     used only 4 natural scene stimuli which the monkey may be  storing in working memory and
#     influencing the results of the IT neuron.
#
# [5] We choose to optimize parameters of a sigmoid function to model occlusion responses.
#     At high occlusion levels, responses should be low as object identity is yet to be determined.
#     As more of the object is revealed stronger responses are anticipated and during the mid
#     level occlusion responses should rise sharply. At low occlusion rate of increase of
#     responses should plateau as the IT neuron will have some tolerance to small variations.

with open('Oreilly2013.pkl', 'rb') as fid:
    oreillyData = pickle.load(fid)

with open('Kovacs1995.pkl', 'rb') as fid:
    kovacsData = pickle.load(fid)

# Curve Fits to non-diagnostic tuning profiles ------------------------------------------
f, axArr = plt.subplots(3, 3, sharex='col', sharey='row')
f.subplots_adjust(hspace=0.05, wspace=0.05)
plt.suptitle('Sigmoid Fits to Non-diagnostic tuning curves 1/(1+exp(a*(x-b))) ', size=16)

x_arr = np.arange(start=0, stop=100, step=1)
pOptArray = np.array([])
pCovArray = np.array([])

# Fit Kovacs Data
for obj, perObjRates in enumerate(kovacsData['rates']):
    row = obj / 3
    col = obj - 3*row

    rMax = np.max(perObjRates)
    axArr[row][col].scatter(kovacsData['occlusion'], perObjRates / rMax)
    pOpt, pCov = curve_fit(sigmoid, kovacsData['occlusion'], perObjRates / rMax)
    pOptArray = np.append(pOptArray, pOpt)
    pCovArray = np.append(pCovArray, pCov)

    legLabel = 'Kovacs 1995 Obj %i: a=%0.4f, b=%0.4f' % (obj, pOpt[0], pOpt[1])
    axArr[row][col].plot(x_arr, sigmoid(x_arr, *pOpt), label=legLabel)

    axArr[row][col].legend(fontsize='small')
    axArr[row][col].set_xlim(0 - 5, 100 + 5)
    axArr[row][col].set_ylim(0 - 0.05, 1 + 0.05)

# Fit Oreilly Data
rMax = np.max(oreillyData['Rates'])
pOpt, pCov = curve_fit(sigmoid, oreillyData['Occ'],
                       oreillyData['Rates'] / rMax)

pOptArray = np.append(pOptArray, pOpt)
pCovArray = np.append(pCovArray, pCov)

axArr[1][2].scatter(oreillyData['Occ'], oreillyData['Rates'] / rMax)

legLabel = 'Oreilly 2013 a=%0.4f, b=%0.4f' % (pOpt[0], pOpt[1])
axArr[1][2].plot(x_arr, sigmoid(x_arr, *pOpt), label=legLabel)
axArr[1][2].legend(fontsize='small')
axArr[1][2].set_xlim(0 - 5, 100 + 5)
axArr[1][2].set_ylim(0 - 0.05, 1 + 0.05)

axArr[2][0].set_xlabel('Occlusion Level')
axArr[2][1].set_xlabel('Occlusion Level')
axArr[2][2].set_xlabel('Occlusion Level')
axArr[0][0].set_ylabel('Normalized Firing Rate')
axArr[1][0].set_ylabel('Normalized Firing Rate')
axArr[2][0].set_ylabel('Normalized Firing Rate')

# Fit Neilson Data - Non Diagnostic Single Figure 1b & Non-Diagnostic Population Figure 2b
rMax = np.max(data['singleNonDiagRate'])
pOpt, pCov = curve_fit(sigmoid, data['singleOcc'],
                       data['singleNonDiagRate'] / rMax,
                       p0=[0.05, 30])

axArr[2][0].scatter(data['singleOcc'], data['singleNonDiagRate'] / rMax)
axArr[2][0].set_xlim(0 - 5, 100 + 5)
axArr[2][0].set_ylim(0 - 0.05, 1 + 0.05)

legLabel = 'Neilson 2006 Single Neuron a=%0.4f, b=%0.4f' % (pOpt[0], pOpt[1])
axArr[2][0].plot(x_arr, sigmoid(x_arr, *pOpt), label=legLabel)
axArr[2][0].legend(fontsize='small')

pOptArray = np.append(pOptArray, pOpt)
pCovArray = np.append(pCovArray, pCov)

rMax = np.max(data['popNonDiagRate'])
pOpt, pCov = curve_fit(sigmoid, data['popOcc'],
                       data['popNonDiagRate'] / rMax, p0=[0.05, 30])

axArr[2][1].scatter(data['popOcc'], data['popNonDiagRate'] / rMax)
axArr[2][1].set_xlim(0 - 5, 100 + 5)
axArr[2][1].set_ylim(0 - 0.05, 1 + 0.05)

legLabel = 'Neilson 2006 Population a=%0.4f, b=%0.4f' % (pOpt[0], pOpt[1])
axArr[2][1].plot(x_arr, sigmoid(x_arr, *pOpt), label=legLabel)
axArr[2][1].legend(fontsize='small')

# Curve fits for diagnostic profiles ----------------------------------------------------
# Single Neuron Data Fit to exemplar Neron in Neilson 2006. Figure 1b.
plt.figure('Diagnostic Tuning Curves')
plt.suptitle('Sigmoid Fits to Diagnostic tuning curves 1/(1+exp(a*(x-b))) ', size=16)
rMax = np.max(data['singleDiagRate'])
pOpt, _ = curve_fit(sigmoid, data['singleOcc'],
                    data['singleDiagRate'] / rMax,
                    p0=[2, 100])

plt.scatter(data['singleOcc'], data['singleDiagRate'] / rMax)
legLabel = 'Neilson Single Neuron Diagnostic a=%0.4f, b=%0.4f' % (pOpt[0], pOpt[1])
plt.plot(x_arr, sigmoid(x_arr, *pOpt), label=legLabel)

# # PLot Single Non-Diagnostic Fit as well
# rMax = np.max(data['singleNonDiagRate'])
# pOpt, _1 = curve_fit(sigmoid, data['singleOcc'],
#                      data['singleNonDiagRate'] / rMax,
#                      p0=[0.05, 30])
#
# plt.scatter(data['singleOcc'], data['singleNonDiagRate'] / rMax)
# legLabel = 'Neilson Exemplar Neuron NonDiagnostic a=%0.4f, b=%0.4f' % (pOpt[0], pOpt[1])
# plt.plot(x_arr, sigmoid(x_arr, *pOpt), label=legLabel)

# Single Neuron Data Fit to population average in Neilson 2006. Figure 2b.
rMax = np.max(data['popDiagRate'])
pOpt, _2 = curve_fit(sigmoid, data['popOcc'],
                     data['popDiagRate'] / rMax,
                     p0=[2, 100])

plt.scatter(data['popOcc'], data['popDiagRate'] / rMax, marker='+', color='green')
legLabel = 'Neilson Pop Neuron Diagnostic a=%0.4f, b=%0.4f' % (pOpt[0], pOpt[1])
plt.plot(x_arr, sigmoid(x_arr, *pOpt), label=legLabel)

# # PLot population Non-Diagnostic Fit as well
# rMax = np.max(data['popNonDiagRate'])
# pOpt, _3 = curve_fit(sigmoid, data['popOcc'],
#                      data['popNonDiagRate'] / rMax,
#                      p0=[0.05, 30])
#
# plt.scatter(data['popOcc'], data['popNonDiagRate'] / rMax)
# legLabel = 'Neilson Pop Neuron NonDiagnostic a=%0.4f, b=%0.4f' \
#    % (pOpt[0], pOpt[1])
# plt.plot(x_arr, sigmoid(x_arr, *pOpt), label=legLabel)
plt.legend()

# 4. Weighted Average single tuning curve ------------------------------------------------

# Find average of all sigmoid parameters of non-diagnostic & diagnostic tuning curve fits
pOptArray = np.reshape(pOptArray, (7, 2))

nonDiagAvgParams = np.mean(pOptArray, axis=0)
print ("Average Parameters for all non-diagnostic tuning curves. a= %0.4f, b= %0.4f"
       % (nonDiagAvgParams[0], nonDiagAvgParams[1]))

# TODO: Salman where do these values come from?
diagAvgParams = np.array([0.32, 70])
print ("Average Parameters for all non-diagnostic tuning curves. a= %0.4f, b= %0.4f"
       % (nonDiagAvgParams[0], nonDiagAvgParams[1]))

# Choose an arbitrary Diagnosticity level
diagnosticity = 0.60

# Plot both average tuning curve
occ = np.arange(start=0, stop=100, step=1)

plt.figure('Mean Tuning Curves')
plt.plot(occ,
         sigmoid(occ, diagAvgParams[0], diagAvgParams[1]),
         label='Diagnostic: a=%0.4f, b=%0.4f' % (diagAvgParams[0], diagAvgParams[1]))
plt.plot(occ,
         sigmoid(occ, nonDiagAvgParams[0], nonDiagAvgParams[1]),
         label='Non-Diagnostic: a=%0.4f, b=%0.4f' % (nonDiagAvgParams[0], nonDiagAvgParams[1]))

plt.legend()
plt.xlabel('Occlusion')
plt.ylabel('Normalized Firing Rate')
plt.title('Mean Model Neuron Occlusion Tuning Profiles, F(s) = 1/(1 + exp(a*(x-b)))')

# 3D complete occlusion tuning profile
occ1, occ2 = np.meshgrid(occ, occ)

fig = plt.figure()
ax = fig.gca(projection='3d')

Z = np.ones(shape=(100, 100))
for ii in np.arange(100):
    for jj in np.arange(100):
        Z[ii][jj] = diagnosticity * sigmoid(ii, diagAvgParams[0], diagAvgParams[1]) + \
            (1 - diagnosticity) * sigmoid(jj, nonDiagAvgParams[0], nonDiagAvgParams[1])

surf = ax.plot_surface(occ1, occ2, Z)

font_size = 34
ax.set_xlabel('Non-Diagnostic occlusion', fontsize=font_size)
ax.set_ylabel('Diagnostic occlusion', fontsize=font_size)
ax.set_zlabel('Normalized Firing Rate', fontsize=font_size)
ax.set_title('Occlusion Tolerance', fontsize=font_size + 10)

ax.tick_params(axis='x', labelsize=font_size)
ax.tick_params(axis='y', labelsize=font_size)
ax.tick_params(axis='z', labelsize=font_size)

ax.annotate('Diagnosticity=%0.2f' % diagnosticity,
            xy=(0.80, 0.8),
            xycoords='axes fraction',
            fontsize=font_size,
            horizontalalignment='right',
            verticalalignment='top')
