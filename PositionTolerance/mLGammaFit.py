# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:53:46 2014

@author: s362khan
"""

import dill
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

plt.ion()

dill.load_session('positionToleranceData.pkl')


# ---------------------------------------------------------
# Code to plot Zoccolan Data onto a figure
zoc_pos_tol = yScatterRaw * np.pi / 180.0  # in Radians
zoc_sel_af = xScatterRaw

plt.scatter(zoc_sel_af, zoc_pos_tol, label='Original Data',  marker='o', s=60, color='blue')
plt.plot(xScatterRaw,
         (yLineFit[0]* np.pi / 180.0 * xScatterRaw + yLineFit[1] * np.pi / 180.0),
         color='blue',
         linewidth=2)
#----------------------------------------------------------


plt.figure('Original Data')

plt.scatter(xScatterRaw,yScatterRaw, label='Scatter Data')
plt.plot(xScatterRaw, yLineFit[0]*xScatterRaw+yLineFit[1],\
         label='Best linear fit', color='red', linewidth=2)
         
plt.legend()
plt.xlabel('Selectivity')
plt.ylabel('Position Tolerance')

''' -----------------------------------------------------------------------------------'''
plt.figure('Alpha')
plt.title('Best fit Gamma RV: Max.likiehood Alpha estimate, scale defined by specified function')

''' Gamma Fit, scale variable based on best fit line '''
alphaArray = np.arange(start=0.01, stop=20, step=0.01)
logLikelihood = []

for alpha in alphaArray:
    prob = [ss.gamma.pdf(y, a=alpha, scale = (yLineFit[0]*xScatterRaw[idx]+yLineFit[1])) \
            for idx, y in enumerate(yScatterRaw)]
    logLikelihood.append( np.log(prob).sum() )

logLikelihood = np.array(logLikelihood)
print ("Method: ML Gamma RV Fit, alpha fixed, scale = best linear fit of data")
print ("alpha %f, max Loglikelihood %f" %(alphaArray[logLikelihood.argmax()], logLikelihood.max()) )

plt.plot(alphaArray, logLikelihood, label = ('scale = best linear fit of data'))
plt.plot(alphaArray[np.nanargmax(logLikelihood)], np.nanmax(logLikelihood), 'r+', linewidth=3)
   
''' -----------------------------------------------------------------------------------'''
''' Gamma Fit, mean based on best fit line, scale = mean/alpha '''
alphaArray = np.arange(start=0.01, stop=20, step=0.01)
logLikelihood = []

for alpha in alphaArray:
    prob = [ss.gamma.pdf(y, a=alpha, scale = (yLineFit[0]*xScatterRaw[idx]+yLineFit[1])/(alpha)) \
            for idx, y in enumerate(yScatterRaw)]
    logLikelihood.append( np.log(prob).sum() )

logLikelihood = np.array(logLikelihood)
print ("Method: ML Gamma RV Fit, Gamma Fit, mean based on best fit line, scale = mean/alpha")
print ("alpha %f, max Loglikelihood %f" %(alphaArray[logLikelihood.argmax()], logLikelihood.max()) )

plt.plot(alphaArray, logLikelihood, label = 'mean based on best fit line, scale = mean/alpha')
plt.plot(alphaArray[np.nanargmax(logLikelihood)], np.nanmax(logLikelihood), 'r+', linewidth=3)

''' -----------------------------------------------------------------------------------'''
''' Gamma Fit, mode based on best fit line, scale = mean/alpha '''
alphaArray = np.arange(start=0.01, stop=20, step=0.01)
logLikelihood = []

for alpha in alphaArray:
    prob = [ss.gamma.pdf(y, a=alpha, scale = yLineFit[0]*xScatterRaw[idx]+yLineFit[1]/(alpha-1))\
            for idx, y in enumerate(yScatterRaw)]
    logLikelihood.append( np.log(prob).sum() )

logLikelihood = np.array(logLikelihood)
print ("Method: ML Gamma RV Fit, Gamma Fit, mean based on best fit line, scale = mode/alpha")
print ("alpha %f, max Loglikelihood %f" %(alphaArray[np.nanargmax(logLikelihood)], np.nanmax(logLikelihood)) )

plt.plot(alphaArray, logLikelihood, label = 'mode based on best fit line,  scale = mode/alpha')
plt.plot(alphaArray[np.nanargmax(logLikelihood)], np.nanmax(logLikelihood), 'r+', linewidth=3)
plt.legend(loc='lower right')
plt.xlabel('alpha')
plt.ylabel('Loglikelihood')

''' -----------------------------------------------------------------------------------'''
''' Raw Gamma Fit '''
[gFitAlpha, gFitLocation, gFitScale] = ss.gamma.fit(yScatterRaw)
prob = [ss.gamma.pdf(x,a=gFitAlpha,loc=gFitLocation,scale=gFitScale) for x in yScatterRaw]
ll= np.log(prob).sum()
print ('Method: Raw Gamma Fit of Data')
print ("Alpha=%f,loc=%f,scale=%f, logLikelihood of Data %f" %(gFitAlpha, gFitLocation, gFitScale, ll))


''' -----------------------------------------------------------------------------------'''
plt.figure('Gamma Distirbutions as selectivity Increase, alpha =4.04, scale=mean/alpha')
xArray = np.arange(30, step=0.5)
alpha=4.04
#[plt.plot(xArray, ss.gamma.pdf(xArray, a=alpha, scale =(yLineFit[0]*x+yLineFit[1])/(alpha))) for x in xArray]
[plt.plot(xArray, ss.gamma.pdf(xArray, a=alpha, scale=yLineFit[0]*x+yLineFit[1]/(alpha))) \
 for x in xScatterRaw]
plt.xlabel('Position Tolerance')
plt.ylabel('Probability of Occurrence')
plt.text(15, 0.6, "<--- Selectivity Increases")













#''' Gamma Fit Scale based on best line fit line '''
#alphaArray = np.arange(start=0.01, stop =5, step=0.01)
#logLikelihood = []
#for alpha in alphaArray:
#    prob = [ ss.gamma.pdf( x, a=alpha, scale=(yLineFit[0]*x + yLineFit[1]) ) \
#             for idx, x in enumerate(yScatterRaw) ]
#    logLikelihood.append(np.log(prob).sum())
#    
#logLikelihood = np.array(logLikelihood)
#print ("Method: Gamma RV fit, alpha fixed, scale = best linear fit of data")
#print ("max logLikelihood %f, alpha %f" \
#        %(logLikelihood.max(), alphaArray[logLikelihood.argmax()]) )
#
#plt.figure('Alpha')
#plt.plot(alphaArray, logLikelihood, label = 'Scale based on best fit line')
#plt.plot(alphaArray[logLikelihood.argmax()], logLikelihood.max(), 'r+', linewidth=3)
#    
#''' Gamma Fit'''
#[gFitAlpha, gFitLocation, gFitScale ] = ss.gamma.fit(yScatterRaw)
#prob = [ss.gamma.pdf(x,a=gFitAlpha,loc=gFitLocation,scale=gFitScale) for x in yScatterRaw]
#ll= np.log(prob).sum()
#print ('Method: Basic gamma Fit')
#print ("Alpha=%f,loc=%f,scale=%f, logLikelihood of Data %f" %(gFitAlpha, gFitLocation, gFitScale, ll))
#
#''' Gamma Fit mean based on best line fit line  '''
#alphaArray = np.arange(start=0.01, stop =5, step=0.01)
#logLikelihood = []
#for alpha in alphaArray:
#    prob = [ ss.gamma.pdf( x, a=alpha, scale=(yLineFit[0]*x + yLineFit[1])/alpha ) \
#             for idx, x in enumerate(xScatterRaw) ]
#    logLikelihood.append(np.log(prob).sum())
#    
#logLikelihood = np.array(logLikelihood)
#print ("Method: Gamma RV fit, alpha fixed, scale = mean/alpha, mean given by bist fit regression line")
#print ("max logLikelihood %f, alpha %f" \
#        %(logLikelihood.max(), alphaArray[logLikelihood.argmax()]) )
#
#plt.plot(alphaArray, logLikelihood, label = 'Mean based on best fit line')
#plt.plot(alphaArray[logLikelihood.argmax()], logLikelihood.max(), 'r+', linewidth=3)
#        
#''' Gamma Fit mode based on best line fit line  '''
#alphaArray = np.arange(start=0.01, stop =5, step=0.01)
#logLikelihood = []
#for alpha in alphaArray:
#    prob = [ ss.gamma.pdf( x, a=alpha, scale=(yLineFit[0]*x + yLineFit[1])/(alpha-1) ) \
#             for idx, x in enumerate(xScatterRaw) ]
#    logLikelihood.append(np.log(prob).sum())
#    
#logLikelihood = np.array(logLikelihood)
#print ("Method: Gamma RV fit, alpha fixed, scale = mode/(alpha-1), mode given by bist fit regression line")
#print ("max logLikelihood %f, alpha %f" \
#        %(np.nanmax(logLikelihood), alphaArray[np.nanargmax(logLikelihood)]) )
#
#plt.plot(alphaArray, logLikelihood, label = 'Mode based on best fit line')
#plt.plot(alphaArray[logLikelihood.argmax()], logLikelihood.max(), 'r+', linewidth=3)
#
#plt.legend()
#
#''' exponential distribution '''
#ss.exp



raw_input('Exit?')
#plt.close()

