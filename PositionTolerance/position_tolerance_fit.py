# -*- coding: utf-8 -*-
""" ---------------------------------------------------------------------------------------------
Fit(s) for the Position Tolerance vs Selectivity data from Figure Figure 4.A of of
[Zoccolan et. al, 2007].

Two properties of position tolerance are modeled: (1) On average, position tolerance decreases as
selectivity (activity fraction) of neuron increases, (2) Position tolerance variations (spread)
decrease as selectivity/sparseness decreases.

First, a linear regression fit of the data is done to model property one.

Second, to model the variations about the mean position tolerance, given by the linear regression
fit, three different approaches are tried to model variations about the mean value. In all three
cases a gamma distribution is used. However there are variations in the parameters of the
distribution

  (1) Scale of distribution = found mean position tolerance
  (2) Mean of distribution = found mean position tolerance. Scale =  Mean / alpha
  (3) Mode of the distribution = found mean position tolerance. Scale = Mode / (alpha - 1)

alpha is found for all the options using ML estimation.


Best ML estimates were found with option two when the the fitted mean position tolerance was set
to mean of the gamma distribution.

Created on Thu Nov 13 16:53:46 2014

@author: s362khan
----------------------------------------------------------------------------------------------"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss


if __name__ == '__main__':
    plt.ion()

    # Load the data
    with open("Zoccolan2007.pkl") as handle:
        data = pickle.load(handle)

    zoc_sel_af = data["xScatterRaw"]
    zoc_pos_tol = data["yScatterRaw"]

    plt.figure("Original Data")
    plt.scatter(zoc_sel_af, zoc_pos_tol, label='Original Data',  marker='o', s=60, color='blue')

    # ---------------------------------------------------------------------------------------
    # Linear Regression fit to find the average position tolerance given selectivity
    # ---------------------------------------------------------------------------------------
    m, b = np.polyfit(zoc_sel_af, zoc_pos_tol, 1)

    print("[1] Mean position tolerance given activity fraction selectivity = %f*selectivity + %f"
          % (m, b))

    sel_arr = np.arange(1.05, step=0.05)
    plt.plot(sel_arr, m * sel_arr + b, label="Best Linear Fit", color='blue', linewidth=2)

    plt.legend()
    plt.xlabel('Selectivity')
    plt.xlim([0, 1])
    plt.ylabel('Position Tolerance')

    # ---------------------------------------------------------------------------------------
    # Distribution of actual position tolerance about the calculated mean position tolerance
    # ---------------------------------------------------------------------------------------
    print("[2] Gamma distributions about mean position tolerance")
    plt.figure('Alpha')
    plt.title('Gamma RV: ML Alpha estimates, scale defined by specified function in label')

    alphaArray = np.arange(start=0.01, stop=20, step=0.01)

    # Method 1 - Scale = calculated mean pos tolerance
    # -------------------------------------------------
    logLikelihood = []
    for alpha in alphaArray:
        prob = [ss.gamma.pdf(y, a=alpha, scale=(m * zoc_sel_af[idx] + b))
                for idx, y in enumerate(zoc_pos_tol)]

        logLikelihood.append(np.log(prob).sum())

    logLikelihood = np.array(logLikelihood)
    print ("\tMethod A [Scale = calculated mean pos tolerance]: alpha=%f, LLV=%f"
           % (alphaArray[logLikelihood.argmax()], logLikelihood.max()))

    plt.plot(alphaArray, logLikelihood, label="Scale = calculated mean pos tolerance")
    plt.plot(alphaArray[np.nanargmax(logLikelihood)], np.nanmax(logLikelihood),
             'r+', markeredgewidth=3)

    # Method 2 - Mean = calculated mean pos tolerance
    # -------------------------------------------------
    logLikelihood = []
    for alpha in alphaArray:
        prob = [ss.gamma.pdf(y, a=alpha, scale=(m * zoc_sel_af[idx] + b) / alpha)
                for idx, y in enumerate(zoc_pos_tol)]

        logLikelihood.append(np.log(prob).sum())

    logLikelihood = np.array(logLikelihood)
    print ("\tMethod B [Mean = calculated mean pos tolerance]: alpha=%f, LLV=%f"
           % (alphaArray[logLikelihood.argmax()], logLikelihood.max()))

    plt.plot(alphaArray, logLikelihood, label="Mean = calculated mean pos tolerance")
    plt.plot(alphaArray[np.nanargmax(logLikelihood)], np.nanmax(logLikelihood),
             'r+', markeredgewidth=3)

    # Method 3 - Mode = calculated mean pos tolerance
    # -------------------------------------------------
    logLikelihood = []
    alphaArray[alphaArray == 1] = 1.01  # Remove alpha = 1, avoid divide by zero

    for alpha in alphaArray:
        prob = [ss.gamma.pdf(y, a=alpha, scale=(m * zoc_sel_af[idx] + b) / (alpha - 1))
                for idx, y in enumerate(zoc_pos_tol)]

        logLikelihood.append(np.log(prob).sum())

    logLikelihood = np.array(logLikelihood)
    print ("\tMethod C [Mode = calculated mean pos tolerance]: alpha=%f, LLV=%f"
           % (alphaArray[np.nanargmax(logLikelihood)], np.nanmax(logLikelihood)))

    plt.plot(alphaArray, logLikelihood, label="Mode = calculated mean pos tolerance")
    plt.plot(alphaArray[np.nanargmax(logLikelihood)], np.nanmax(logLikelihood),
             'r+', markeredgewidth=3)
