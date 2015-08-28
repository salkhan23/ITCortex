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


def corrected_rotation(x_arr, mu):
    """
    Given an input rotation angle [-180,180), return corrected angle to ensure
    such that angle lies within mu-180 and mu+180.
    :param x_arr:
    :param mu:

    :rtype :
    """
    if x_arr < (mu-180):
        x_arr += 360
    elif x_arr > mu+180:
        x_arr -= 360

    return x_arr


def single_gaussian(angles, mu, sigma, amp):

    x_corrected = np.array([corrected_rotation(angle, mu) for angle in angles])

    return amp * np.exp(-(x_corrected - mu)**2/(2.0 * sigma**2))


def double_gaussian(
        angles,
        mu1, sigma1, amp1,
        mu2, sigma2, amp2):

    s = single_gaussian(angles, mu1, sigma1, amp1)
    s += single_gaussian(angles, mu2, sigma2, amp2)

    return s


def triple_gaussian(
        angles,
        mu1, sigma1, amp1,
        mu2, sigma2, amp2,
        mu3, sigma3, amp3):

    s = single_gaussian(angles, mu1, sigma1, amp1)
    s += single_gaussian(angles, mu2, sigma2, amp2)
    s += single_gaussian(angles, mu3, sigma3, amp3)

    return s


def quadruple_gaussian(
        angles,
        mu1, sigma1, amp1,
        mu2, sigma2, amp2,
        mu3, sigma3, amp3,
        mu4, sigma4, amp4):

    s = single_gaussian(angles, mu1, sigma1, amp1)
    s += single_gaussian(angles, mu2, sigma2, amp2)
    s += single_gaussian(angles, mu3, sigma3, amp3)
    s += single_gaussian(angles, mu4, sigma4, amp4)

    return s


def main(angles_org, firing_rates_org, initial_est, fig_title=''):
    """
    :param angles_org:      : Measured (original data) angles.
    :param firing_rates_org : Measured (original)Firing rates for angles specified in angles_org.
    :param initial_est      : List of initial estimates of fit parameters for each component
                              Gaussian function.
    :param fig_title        : Title of figure. Default = ''.
    """
    # Plot the original data
    plt.figure()
    plt.title('Rotation Tuning ' + fig_title)
    plt.xlabel('Angle(Deg)')
    plt.ylabel('Normalized Firing Rate')
    plt.scatter(angles_org, firing_rates_org, label='Original Data')
    
    angles_arr = np.arange(-180, 180, step=1)
    
    # -------------------------------------------------------------------------------------------
    # Single Gaussian Curve Fit
    # -------------------------------------------------------------------------------------------
    params_fit, params_cov_mat = curve_fit(
        single_gaussian,
        angles_org,
        firing_rates_org,
        p0=initial_est[0, :])

    # Standard deviation of fit parameters:
    # REF: (1) http://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-
    # parameters-using-the-optimize-leastsq-method-i
    # (2) http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    params_err_std_dev = np.sqrt(np.diag(params_cov_mat))
    
    plt.plot(
        angles_arr,
        single_gaussian(angles_arr, params_fit[0], params_fit[1], params_fit[2]),
        label=r'$1\ Gaussian:\ \mu_1=%0.2f,\ \sigma_1=%0.2f,\ Amp_1=%0.2f$'
              % (params_fit[0], params_fit[1], params_fit[2]))

    print ("1 Gaussian Fit - standard deviation of errors in parameters:" +
           "\n\tmu_1=%0.4f, sigma_1=%0.4f, Amp_1=%0.4f"
           % (params_err_std_dev[0], params_err_std_dev[1], params_err_std_dev[2]))

    # ------------------------------------------------------------------------------------------
    # Double Gaussian Curve Fitting
    # ------------------------------------------------------------------------------------------
    # 255 = initial value of the initialization parameters estimates, if any parameter
    # in initial_est list = 255, skip estimation for this curve fitting function.
    if -255 not in (initial_est[1, :]):

        params_fit_g2, params_cov_mat_g2 = curve_fit(
            double_gaussian,
            angles_org,
            firing_rates_org,
            p0=np.concatenate((initial_est[0, :], initial_est[1, :]), axis=0))

        params_err_std_dev_g2 = np.sqrt(np.diag(params_cov_mat_g2))
        
        plt.plot(
            angles_arr,
            double_gaussian(angles_arr,
                            params_fit_g2[0], params_fit_g2[1], params_fit_g2[2],
                            params_fit_g2[3], params_fit_g2[4], params_fit_g2[5]),
            label=r'$2\ Gaussian:\ $' +
                  r'$\mu_1=%0.2f,\ \sigma_1=%0.2f,\ Amp_1=%0.2f\ $'
                  % (params_fit_g2[0], params_fit_g2[1], params_fit_g2[2]) +
                  r'$\mu_2=%0.2f,\ \sigma_2=%0.2f,\ Amp_2=%0.2f\ $'
                  % (params_fit_g2[3], params_fit_g2[4], params_fit_g2[5])
        )

        print ("2 Gaussian Fit - standard deviation of errors in parameters:" +
               "\n\t mu_1=%0.4f, sigma_1=%0.4f, Amp_1=%0.4f"
               % (params_err_std_dev_g2[0], params_err_std_dev_g2[1], params_err_std_dev_g2[2]) +
               "\n\t mu_2=%0.4f, sigma_2=%0.4f, Amp_2=%0.4f"
               % (params_err_std_dev_g2[3], params_err_std_dev_g2[4], params_err_std_dev_g2[5])
               )

    # -----------------------------------------------------------------------------------------
    # Triple Gaussian Curve Fitting
    # -----------------------------------------------------------------------------------------
    if -255 not in (initial_est[2, :]):

        params_fit_g3, params_cov_mat_g3 = curve_fit(
            triple_gaussian,
            angles_org,
            firing_rates_org,
            p0=np.concatenate((initial_est[0, :], initial_est[1, :], initial_est[2, :]), axis=0))

        params_err_std_dev_g3 = np.sqrt(np.diag(params_cov_mat_g3))

        plt.plot(
            angles_arr,
            triple_gaussian(angles_arr,
                            params_fit_g3[0], params_fit_g3[1], params_fit_g3[2],
                            params_fit_g3[3], params_fit_g3[4], params_fit_g3[5],
                            params_fit_g3[6], params_fit_g3[7], params_fit_g3[8]),
            label=r'$3\ Gaussian:\ $' +
                  r'$\mu_1=%0.2f,\ \sigma_1=%0.2f,\ Amp_1=%0.2f\ $'
                  % (params_fit_g3[0], params_fit_g3[1], params_fit_g3[2]) +
                  r'$\mu_2=%0.2f,\ \sigma_2=%0.2f,\ Amp_2=%0.2f\ $'
                  % (params_fit_g3[3], params_fit_g3[4], params_fit_g3[5]) +
                  r'$\mu_3=%0.2f,\ \sigma_3=%0.2f,\ Amp_3=%0.2f\ $'
                  % (params_fit_g3[6], params_fit_g3[7], params_fit_g3[8])
        )

        print ("3 Gaussian Fit - standard deviation of errors in parameters:" +
               "\n\t mu_1=%0.4f, sigma_1=%0.4f, Amp_1=%0.4f"
               % (params_err_std_dev_g3[0], params_err_std_dev_g3[1], params_err_std_dev_g3[2]) +
               "\n\t mu_2=%0.4f, sigma_2=%0.4f, Amp_2=%0.4f"
               % (params_err_std_dev_g3[3], params_err_std_dev_g3[4], params_err_std_dev_g3[5]) +
               "\n\t mu_3=%0.4f, sigma_3=%0.4f, Amp_3=%0.4f"
               % (params_err_std_dev_g3[6], params_err_std_dev_g3[7], params_err_std_dev_g3[8])
               )
        
    ''' ----------------------------------------------------------------------------------
    Quadruple Gaussian Curve Fitting
    -----------------------------------------------------------------------------------'''
    if -255 not in (initial_est[3, :]):

        params_fit_g4, params_cov_mat_g4 = curve_fit(
            quadruple_gaussian,
            angles_org,
            firing_rates_org,
            p0=np.concatenate((initial_est[0, :],
                               initial_est[1, :],
                               initial_est[2, :],
                               initial_est[3, :]), axis=0)
        )

        params_err_std_dev_g4 = np.sqrt(np.diag(params_cov_mat_g4))
        
        plt.plot(
            angles_arr,
            quadruple_gaussian(angles_arr,
                               params_fit_g4[0], params_fit_g4[1], params_fit_g4[2],
                               params_fit_g4[3], params_fit_g4[4], params_fit_g4[5],
                               params_fit_g4[6], params_fit_g4[7], params_fit_g4[8],
                               params_fit_g4[9], params_fit_g4[10], params_fit_g4[11]),
            label=r'$4\ Gaussian:\ $' +
                  r'$\mu_1=%0.2f,\ \sigma_1=%0.2f,\ Amp_1=%0.2f\ $'
                  % (params_fit_g4[0], params_fit_g4[1], params_fit_g4[2]) +
                  r'$\mu_2=%0.2f,\ \sigma_2=%0.2f,\ A_2=%0.2f\ $'
                  % (params_fit_g4[3], params_fit_g4[4], params_fit_g4[5]) +
                  r'$\mu_3=%0.2f,\ \sigma_3=%0.2f,\ A_3=%0.2f\ $'
                  % (params_fit_g4[6], params_fit_g4[7], params_fit_g4[8]) +
                  r'$\mu_4=%0.2f,\ \sigma_4=%0.2f,\ A_4=%0.2f\ $'
                  % (params_fit_g4[9], params_fit_g4[10], params_fit_g4[11])
        )

        print ("4 Gaussian Fit - standard deviation of errors in parameters:" +
               "\n\t mu_1=%0.4f, sigma_1=%0.4f, Amp_1=%0.4f"
               % (params_err_std_dev_g4[0], params_err_std_dev_g4[1], params_err_std_dev_g4[2]) +
               "\n\t mu_2=%0.4f, sigma_2=%0.4f, Amp_2=%0.4f"
               % (params_err_std_dev_g4[3], params_err_std_dev_g4[4], params_err_std_dev_g4[5]) +
               "\n\t mu_3=%0.4f, sigma_3=%0.4f, Amp_3=%0.4f"
               % (params_err_std_dev_g4[6], params_err_std_dev_g4[7], params_err_std_dev_g4[8]) +
               "\n\t mu_4=%0.4f, sigma_4=%0.4f, Amp_4=%0.4f"
               % (params_err_std_dev_g4[9], params_err_std_dev_g4[10], params_err_std_dev_g4[11])
               )

if __name__ == "__main__":
    # if you call this script from the command line (the shell) it will
    # run the 'main' function
    plt.ion()

    # Load extracted data
    with open('rotationalTolerance.pkl', 'rb') as handle:
        data = pickle.load(handle)

    # # -------------------------------------------------------------------------------------------
    title = 'Rotation Tuning - Fig 5a, Logothesis, Pauls & Poggio -1995'

    x = data['fig5ax']
    y = data['fig5ay']
    y = y/max(y)

    InitialEst = -255 * np.ones(shape=(4, 3))
    InitialEst[0, :] = [100, 20, 1.00]
    InitialEst[1, :] = [-30, 10, 0.30]
    InitialEst[2, :] = [-90, 20, 0.25]

    print title
    main(x, y, InitialEst, title)
    plt.legend()

    # -------------------------------------------------------------------------------------------
    title = 'Fig 5b, logothesis, Pauls & poggio -1995'
    x = data['fig5bx']
    y = data['fig5by']
    y = y/max(y)

    InitialEst = -255 * np.ones(shape=(4, 3))
    InitialEst[0, :] = [-10, 20, 1.00]
    InitialEst[1, :] = [180, 30, 0.30]
    InitialEst[2, :] = [90,  30, 0.20]
    InitialEst[3, :] = [135, 30, 0.20]

    print title
    main(x, y, InitialEst, title)
    plt.legend()

    # -------------------------------------------------------------------------------------------
    title = 'Fig 5c, logothesis, Pauls & poggio -1995'
    x = data['fig5cx']
    y = data['fig5cy']
    y = y/max(y)

    InitialEst = -255 * np.ones(shape=(4, 3))
    InitialEst[0, :] = [120,  30, 1.00]
    InitialEst[1, :] = [-60,  30, 0.40]
    InitialEst[2, :] = [-145, 20, 0.20]

    print title
    main(x, y, InitialEst, title)
    plt.legend()

    # -------------------------------------------------------------------------------------------
    title = 'Fig 5d, logothesis, Pauls & poggio -1995'
    x = data['fig5dx']
    y = data['fig5dy']
    y = y/max(y)

    InitialEst = -255 * np.ones(shape=(4, 3))
    InitialEst[0, :] = [-90, 30, 1.00]
    InitialEst[1, :] = [-30, 60, 0.25]

    print title
    main(x, y, InitialEst, title)
    plt.legend()

    # -------------------------------------------------------------------------------------------
    title = 'Fig 5d, logothesis, Pauls & poggio -1995'
    x = data['fig5ex']
    y = data['fig5ey']
    y = y/max(y)

    InitialEst = -255 * np.ones(shape=(4, 3))
    InitialEst[0, :] = [-70, 100, 1.00]
    InitialEst[1, :] = [-10, 80,  0.90]
    InitialEst[2, :] = [170, 80,  0.80]

    print title
    main(x, y, InitialEst, title)
    plt.legend()

    # -------------------------------------------------------------------------------------------
    title = 'Fig 8b, Hung, Carlson & Conner -2012'
    x = data['hungFig8bx']
    y = data['hungFig8by']
    y = y/max(y)

    InitialEst = -255 * np.ones(shape=(4, 3))
    InitialEst[0, :] = [70, 10, 1.00]
    InitialEst[1, :] = [0,  40, 0.70]

    print title
    main(x, y, InitialEst, title)
    plt.legend()
