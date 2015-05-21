# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 12:08:04 2014

Multiple Gaussian Fitting for Rotation Tuning Curves of Inferior Temporal Cortex Neurons
Using scipy.optimize.curve_fit, to find best fits for single, multiple, triple, quadruple
Gaussian functions, provided initial estimates of the Gaussian parameters are provided. 
curve fit function for Gaussian needs initial estimates to fit best fits.

@author: s362khan
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def correctedRotation(x, mu):
    """Given an input rotation angle [-180,180), return corrected angle to ensure
    such that angle lies within mu-180 and mu+180.
    """
    if (x < (mu-180)):
        x+=360
    elif (x> mu+180):
        x-=360
    return(x)
    
def singleGaussian(xArray, mu, sigma, A):
    xCorrected = np.array([correctedRotation(x,mu) for x in xArray])
    return(A*np.exp( -(xCorrected-mu)**2/(2.0*sigma**2) ))

def doubleGaussian(xArray, 
                   mu1, sigma1, A1,
                   mu2, sigma2, A2):
    S  = singleGaussian(xArray, mu1, sigma1, A1)
    S += singleGaussian(xArray, mu2, sigma2, A2)
    return(S)

def tripleGaussian(xArray, 
                   mu1, sigma1, A1,
                   mu2, sigma2, A2,
                   mu3, sigma3, A3):
    S  = singleGaussian(xArray, mu1, sigma1, A1)
    S += singleGaussian(xArray, mu2, sigma2, A2)
    S += singleGaussian(xArray, mu3, sigma3, A3)
    return(S)

def quadrupleGaussian(xArray, 
                      mu1, sigma1, A1,
                      mu2, sigma2, A2,
                      mu3, sigma3, A3,
                      mu4, sigma4, A4):
    S  = singleGaussian(xArray, mu1, sigma1, A1)
    S += singleGaussian(xArray, mu2, sigma2, A2)
    S += singleGaussian(xArray, mu3, sigma3, A3)
    S += singleGaussian(xArray, mu4, sigma4, A4)
    return(S)
    
def main(x, y, InitialEst):
    plt.figure('Rotation')
    plt.title('Rotation Tuning using LSE fitting (scipy.optimize.curve_fit)')
    plt.xlabel('Angle(Deg)')
    plt.ylabel('Normalized Firing Rate')
    plt.scatter(x, y, label = 'Original Data')
    
    angles = np.arange(-180, 180, step=1)
    
    ''' ----------------------------------------------------------------------------------
    Single Gaussian Curve Fitting
    -----------------------------------------------------------------------------------'''
    pFit, pCov = curve_fit(singleGaussian, x, y, p0 = InitialEst[0,:])
    
    plt.plot(angles, singleGaussian(angles, pFit[0], pFit[1], pFit[2]), \
       label = r'$1\ Gaussian:\ \mu_1=%0.2f,\ \sigma_1=%0.2f,\ A_1=%0.2f$' \
       %(pFit[0], pFit[1], pFit[2])) 
    
    print ("1 Gaussian Fit Variances %s" %str(np.diag(pCov)))
    
    ''' ----------------------------------------------------------------------------------
    Double Gaussian Curve Fitting
    -----------------------------------------------------------------------------------'''
    if -255 not in (InitialEst[1,:]):
        pFit2, pCov2 = curve_fit(doubleGaussian, x, y,  \
                         p0 = np.concatenate((InitialEst[0,:],InitialEst[1,:]), axis=0))
        
        plt.plot(angles, doubleGaussian(angles, 
                                        pFit2[0], pFit2[1], pFit2[2], 
                                        pFit2[3], pFit2[4], pFit2[5]), \
             label = r'$2\ Gaussian:\ $' +
             r'$\mu_1=%0.2f,\ \sigma_1=%0.2f,\ A_1=%0.2f\ $' %(pFit2[0],pFit2[1],pFit2[2]) +
             r'$\mu_2=%0.2f,\ \sigma_2=%0.2f,\ A_2=%0.2f\ $' %(pFit2[3],pFit2[4],pFit2[5]) )
        
        print ("2 Gaussian Fit Variances %s" %str(np.diag(pCov2)))  
        
    ''' ----------------------------------------------------------------------------------
    Triple Gaussian Curve Fitting
    -----------------------------------------------------------------------------------'''
    if -255 not in (InitialEst[2,:]):
        pFit3, pCov3 = curve_fit(tripleGaussian, x, y,  \
                         p0 = np.concatenate((InitialEst[0,:], 
                                              InitialEst[1,:], 
                                              InitialEst[2,:]), axis=0))
        
        plt.plot(angles, tripleGaussian(angles, 
                                        pFit3[0], pFit3[1], pFit3[2], 
                                        pFit3[3], pFit3[4], pFit3[5],
                                        pFit3[6], pFit3[7], pFit3[8]), \
             label = r'$3\ Gaussian:\ $' +
             r'$\mu_1=%0.2f,\ \sigma_1=%0.2f,\ A_1=%0.2f\ $' %(pFit3[0],pFit3[1],pFit3[2]) +
             r'$\mu_2=%0.2f,\ \sigma_2=%0.2f,\ A_2=%0.2f\ $' %(pFit3[3],pFit3[4],pFit3[5]) +
             r'$\mu_3=%0.2f,\ \sigma_3=%0.2f,\ A_3=%0.2f\ $' %(pFit3[6],pFit3[7],pFit3[8]) )
        
        print ("3 Gaussian Fit Variances %s" %str(np.diag(pCov3))) 
        
    ''' ----------------------------------------------------------------------------------
    Quadruple Gaussian Curve Fitting
    -----------------------------------------------------------------------------------'''
    if -255 not in (InitialEst[3,:]):
        pFit4, pCov4 = curve_fit(quadrupleGaussian, x, y,  \
                         p0 = np.concatenate((InitialEst[0,:], 
                                              InitialEst[1,:], 
                                              InitialEst[2,:],
                                              InitialEst[3,:] ), axis=0))
        
        plt.plot(angles, quadrupleGaussian(angles, 
                                           pFit4[0], pFit4[1], pFit4[2], 
                                           pFit4[3], pFit4[4], pFit4[5],
                                           pFit4[6], pFit4[7], pFit4[8],
                                           pFit4[9], pFit4[10], pFit4[11]), \
             label = r'$4\ Gaussian:\ $' +
             r'$\mu_1=%0.2f,\ \sigma_1=%0.2f,\ A_1=%0.2f\ $' %(pFit4[0],pFit4[1],pFit4[2]) +
             r'$\mu_2=%0.2f,\ \sigma_2=%0.2f,\ A_2=%0.2f\ $' %(pFit4[3],pFit4[4],pFit4[5]) +
             r'$\mu_3=%0.2f,\ \sigma_3=%0.2f,\ A_3=%0.2f\ $' %(pFit4[6],pFit4[7],pFit4[8]) +
             r'$\mu_4=%0.2f,\ \sigma_4=%0.2f,\ A_4=%0.2f\ $' %(pFit4[9],pFit4[10],pFit4[11]) )
        
        print ("4 Gaussian Fit Variances %s" %str(np.diag(pCov4)))
        
if __name__ == "__main__":
    # if you call this script from the command line (the shell) it will
    # run the 'main' function
    plt.ion()
    InitialEst = -255* np.ones(shape =(4,3))
    
    # Load extracted data
    with open('rotationalTolerance.pkl', 'rb') as handle:
        data = pickle.load(handle)
      
    # Fig 5a, logothesis, Pauls & poggio -1995 ------------------------------------------
    x = data['fig5ax']
    y = data['fig5ay']
    y = y/max(y)
            
    InitialEst[0,:] = [     100,     20,    1.00]
    InitialEst[1,:] = [     -30,     10,    0.30]
    InitialEst[2,:] = [     -90,     20,    0.25]
      
#    # Fig 5b, logothesis, Pauls & poggio -1995 ------------------------------------------
#    x = data['fig5bx']
#    y = data['fig5by']
#    y = y/max(y)
#    
#    InitialEst[0,:] = [     -10,     20,    1.00]
#    InitialEst[1,:] = [     180,     30,    0.30]
#    InitialEst[2,:] = [      90,     30,    0.20]
#    InitialEst[3,:] = [      135,    30,    0.20]
      
#    # Fig 5c, logothesis, Pauls & poggio -1995 -------------------------------------------
#    x = data['fig5cx']
#    y = data['fig5cy']
#    y = y/max(y)
#    
#    InitialEst[0,:] = [     120,     30,    1.00]
#    InitialEst[1,:] = [     -60,     30,    0.40]
#    InitialEst[2,:] = [     -145,    20,    0.20]
   
    # Fig 5d, logothesis, Pauls & poggio -1995 -------------------------------------------
#    x = data['fig5dx']
#    y = data['fig5dy']
#    y = y/max(y)
#    
#    InitialEst[0,:] = [     -90,     30,    1.00]
#    InitialEst[1,:] = [     -30,     60,    0.25]

#    # Fig 5d, logothesis, Pauls & poggio -1995 -------------------------------------------
#    x = data['fig5ex']
#    y = data['fig5ey']
#    y = y/max(y)
#    
#    InitialEst[0,:] = [     -70,     100,   1.00]
#    InitialEst[1,:] = [     -10,     80,    0.90]
#    InitialEst[2,:] = [     170,     80,    0.80]

#    # Fig 8b, Hung, Carlson & Conner -2012 -------------------------------------------
#    x = data['hungFig8bx']
#    y = data['hungFig8by']
#    y = y/max(y)
#    
#    InitialEst[0,:] = [      70,     10,   1.00]
#    InitialEst[1,:] = [       0,     40,   0.70]





    
    main(x, y, InitialEst)
    plt.legend()
    
    
    
    
    
    