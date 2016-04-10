
import numpy as np
import matplotlib.pyplot as plt
import pickle
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


def pseudo_symmetric_gaussian(
        angles,
        mu1, sigma1, amp):

    s = single_gaussian(angles, mu1, sigma1, amp)
    s += single_gaussian(angles, mu1+180, sigma1, amp)

    return s


def main(angles_org, firing_rates_org, initial_est, fig_title=''):

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

    # -------------------------------------------------------------------------------------------
    # Mirror Symmetric Fit
    # -------------------------------------------------------------------------------------------
    params_fit, params_cov_mat = curve_fit(
        pseudo_symmetric_gaussian,
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
        pseudo_symmetric_gaussian(angles_arr, params_fit[0], params_fit[1], params_fit[2]),
        label=r'$ Pseudo Sym Gaussian:\ \mu_1=%0.2f,\ \sigma_1=%0.2f,\ Amp_1=%0.2f$'
              % (params_fit[0], params_fit[1], params_fit[2]))

    print ("Pseudo Sym - standard deviation of errors in parameters:" +
           "\n\tmu_1=%0.4f, sigma_1=%0.4f, Amp_1=%0.4f"
           % (params_err_std_dev[0], params_err_std_dev[1], params_err_std_dev[2]))



if __name__ == "__main__":
    plt.ion()

    # Load Extracted data
    with open('rotationalTolerance.pkl', 'rb') as handle:
        data = pickle.load(handle)

    # # -------------------------------------------------------------------------------------------
    title = 'Rotation Tuning - Fig 5a, Logothesis, Pauls & Poggio -1995'

    x = data['fig5ax']
    y = data['fig5ay']
    y = y/max(y)

    InitialEst = -255 * np.ones(shape=(4, 3))
    InitialEst[0, :] = [100, 20, 1.00]

    print title
    main(x, y, InitialEst, title)
    plt.legend()

    # ------------------------------------------------------------------------------
    title = 'Fig 5c, logothesis, Pauls & poggio -1995'
    x = data['fig5cx']
    y = data['fig5cy']
    y = y/max(y)

    InitialEst = -255 * np.ones(shape=(4, 3))
    InitialEst[0, :] = [120,  30, 1.00]

    print title
    main(x, y, InitialEst, title)
    plt.legend()

    # -------------------------------------------------------------------------------------------
    title = 'Fig 5e, logothesis, Pauls & poggio -1995'
    x = data['fig5ex']
    y = data['fig5ey']
    y = y/max(y)

    InitialEst = -255 * np.ones(shape=(4, 3))
    InitialEst[0, :] = [-70, 100, 1.00]

    print title
    main(x, y, InitialEst, title)
    plt.legend()





