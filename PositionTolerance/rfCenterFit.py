# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 18:50:17 2014

Receptive Field Centers distribution modeled from :
Op de Beeck & Vogels - 2000 - Spatial Sensitivities of Macaque Inferior Temporal Neurons - Fig 6.
Use maximum likelihood to fit the data.
@author: s362khan
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


def GenerateRfCenters(n, deg2Pixel=1, gazeCenter=np.array([0, 0])):
    ''' Generate RF centers distribution based on data from Op de Beeck & Vogels - 2000 -
        Spatial Sensitivities of Macaque Inferior Temporal Neurons - Fig 6.

        @param n            = number of RF centers to generate
        @param deg2Pixel    = conversion factor from degree to pixels
        @param gazeCenter   = Center of gaze in pixels

        @return Receptive Field Centers in pixels.

        Use a random variable with parameters extracted from the curve fitting of recorded
        data in the main function.  Even though the gamma provides a slightly better fit,
        the log likelihood probabilities are similar and the Gaussian RV uses less parameters.
    '''
    sigmaX = 2.02
    muX = 1.82

    sigmaY = 2.12
    muY = 0.61

    xGen = (gazeCenter[0] + ss.norm.rvs(size=n, loc=muX, scale=sigmaX)) * deg2Pixel
    yGen = (gazeCenter[0] + ss.norm.rvs(size=n, loc=muY, scale=sigmaY)) * deg2Pixel

    z = np.vstack((xGen, yGen))

    return(z.T)


def Main():
    # Load RF centers data
    with open('rfCenters.pkl', 'rb') as fId:
        data = pickle.load(fId)

    x = data['x']
    y = data['y']

#    mag = np.sqrt(x**2 + y**2)
#    angle = np.arctan2(y, x)

    plt.figure("RF Centers Distribution")
    plt.title('Population Receptive Field Center Distributions', fontsize=40)
    plt.scatter(x, y, label='Extracted Data', s=50)
    plt.grid()

    f, axArr = plt.subplots(2, sharex=True)
    f.subplots_adjust(hspace=0.2, wspace=0.05)
    f.suptitle('X, Y Distributions and Best Fits', size=16)

    nX, bins, patches = axArr[0].hist(x, bins=np.arange(-10, 10, step=0.5), normed=True)
    axArr[0].set_title('X coordinate')
    axArr[0].set_ylabel('Normalized Frequency')

    nY, bins, patches = axArr[1].hist(y, bins=np.arange(-10, 10, step=0.5), normed=True)
    axArr[1].set_title('Y coordinate')
    axArr[1].set_xlabel('Eccentricity (Degrees)')
    axArr[1].set_ylabel('Normalized Frequency')

    ''' x Data Fits ----------------------------------------------------------------------------'''
    sortedIdx = np.argsort(x)
    x = x[sortedIdx]

    ''' Gaussian '''
    xGaussianFitParams = ss.norm.fit(x)
    prob = ss.norm.pdf(x, loc=xGaussianFitParams[0], scale=xGaussianFitParams[1])
    LLR = np.log(prob).sum()
    print("(X) Best Gaussian Fit mu=%0.2f, sigma=%0.2f, LLR %f"
          % (xGaussianFitParams[0], xGaussianFitParams[1], LLR))

    axArr[0].plot(x, prob,
                  label=r'$Gaussian\ Fit:\ \mu=%0.2f,\ \sigma=%0.2f,\ LLR=%0.2f$'
                  % (xGaussianFitParams[0], xGaussianFitParams[1], LLR))

    ''' Gamma '''
    xGammaFitParams = ss.gamma.fit(x)
    prob = ss.gamma.pdf(x, a=xGammaFitParams[0], loc=xGammaFitParams[1], scale=xGammaFitParams[2])
    LLR = np.log(prob).sum()
    print ("(X) Best Gamma Fit shape=%0.2f, loc=%0.2f, scale=%0.2f, LLR=%f"
           % (xGammaFitParams[0], xGammaFitParams[1], xGammaFitParams[2], LLR))

    axArr[0].plot(x, prob,
                  label=r'$Gamma\ Fit:\ shape=%0.2f,\ loc=%0.2f,\ scale=%0.2f,\ LLR=%0.2f$'
                  % (xGammaFitParams[0], xGammaFitParams[1], xGammaFitParams[2], LLR))

    axArr[0].legend(loc='best')

    ''' y Data Fits ----------------------------------------------------------------------------'''
    sortedIdx = np.argsort(y)
    y = y[sortedIdx]

    ''' Gaussian '''
    yGaussianFitParams = ss.norm.fit(y)
    prob = ss.norm.pdf(y, loc=yGaussianFitParams[0], scale=yGaussianFitParams[1])
    LLR = np.log(prob).sum()
    print ("(Y) Best Gaussian Fit mu=%0.2f, sigma=%0.2f, LLR %f"
           % (yGaussianFitParams[0], yGaussianFitParams[1], LLR))

    axArr[1].plot(y, prob,
                  label=r'$Gaussian\ Fit:\ \mu=%0.2f,\ \sigma=%0.2f,\ LLR=%0.2f$'
                  % (yGaussianFitParams[0], yGaussianFitParams[1], LLR))

    ''' Gamma '''
    yGammaFitParams = ss.gamma.fit(y)
    prob = ss.gamma.pdf(y, a=yGammaFitParams[0], loc=yGammaFitParams[1], scale=yGammaFitParams[2])
    LLR = np.log(prob).sum()
    print("(Y) Best Gamma Fit shape=%0.2f, loc=%0.2f, scale=%0.2f, LLR=%0.2f"
          % (yGammaFitParams[0], yGammaFitParams[1], yGammaFitParams[2], LLR))

    axArr[1].plot(y, prob,
                  label=r'$Gamma\ Fit:\ shape=%0.2f,\ loc=%0.2f, scale=%0.2f, LLR=%0.2f$'
                  % (yGammaFitParams[0], yGammaFitParams[1], yGammaFitParams[2], LLR))

    axArr[1].legend(loc='best')

    ''' -----------------------------------------------------------------------------------'''
    ''' Generate Sample Data '''
    n = 100
    rfCenters = GenerateRfCenters(n)

    plt.figure("RF Centers Distribution")
    plt.scatter(0, 0, color='red', marker='+', linewidth=10, label='Gaze Center')
    plt.scatter(rfCenters[:, 0], rfCenters[:, 1], color='green', marker='o', s=50,
                label='Generated Data')
    plt.xlabel('Horizontal position (deg)', fontsize=30)
    plt.ylabel('Vertical position (deg)',  fontsize=30)
    plt.text(-3, -5, 'IPSI', fontsize=30)
    plt.text(6, -5, 'CONTRA', fontsize=30)
    plt.axvline(linewidth=1, color='k')
    plt.axhline(linewidth=1, color='k')

if __name__ == "__main__":
    plt.ion()
    Main()
