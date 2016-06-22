# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 18:50:17 2014

@author: s362khan
"""
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt


class GaussianPositionProfile:

    def __init__(self, selectivity):
        """
        1. Models neurons position tolerance receptive field using a 2D Gaussian function.
        Ref. [Zoccolan et. al, 2007].

        2. Models the Receptive Field Center.
        Ref. Op de Beeck & Vogels - 2000 - Spatial Sensitivities of Macaque Inferior Temporal
        Neurons - Fig 6.

        :param selectivity: Activity fraction.
            Number of objects neuron responds to divided by total number of objects.

        :rtype : Object gaussian position profile class
        """
        self.type = '2d_gaussian'

        self.rf_center = self.__get_receptive_field_center()

        self.position_tolerance = self.__get_position_tolerance(selectivity)

    @staticmethod
    def __get_receptive_field_center():
        """ Generate RF centers based on data from Op de Beeck & Vogels - 2000 -
        Spatial Sensitivities of Macaque Inferior Temporal Neurons - Fig 6.

        In file rfCenterFit.py, data was fit using maximum likelihood fitting to Gaussian and
        Gamma Distributions. A Gaussian distribution with the parameters below was selected.
        Even though the gamma provides a slightly better fit, log likelihood ratios are similar
        and the Gaussian RV uses less parameters.

        :rtype : 1x2 array of the RF Center (x, y) in degrees of eccentricity. (Radians)
        """
        sigma_x = 2.02
        mu_x = 1.82

        sigma_y = 2.12
        mu_y = 0.61

        x = ss.norm.rvs(size=1, loc=mu_x, scale=sigma_x) * np.pi / 180
        y = ss.norm.rvs(size=1, loc=mu_y, scale=sigma_y) * np.pi / 180

        return np.hstack((x, y))

    @staticmethod
    def __get_position_tolerance(s_idx):
        """
        Method determines the position tolerance of the Neuron. Position Tolerance is
        defined as 2*standard deviation of the Gaussian function.

        Two properties of position tolerance are modeled: (1) Position tolerance decreases
        as selectivity/spareness of neuron increases, (2) Position tolerance variations
        (spread) decrease as selectivity/sparseness decreases.

        A set of gamma random variables with constant shape (alpha) and variable spread
        (scale) that decreases with neuron selectivity are used to fit scatter points in
        Figure 4.A of of Zoccolan et. al, 2007. A best fit linear regression line to all
        scatter points is used to model decreasing mean (scale = Mean \ alpha) of the
        gamma random variables.

        Maximum likelihood fitting (alpha value that best fits the data) is used to
        determine alpha. See mLGammaFit.py for ML fitting.

        Gamma RV Mean(sparseness) = -9.820*sparseness + 13.9730
        Gamma RV Scale(sparseness) = mean(spareness) \ alpha

        :param s_idx: Activity fraction.
            Number of objects neuron responds to divided by total number of objects.

        :rtype : Position tolerance of the neuron in degree of eccentricity (Radians).
        """
        alpha = 4.04
        mean_position_tolerance = -9.820 * s_idx + 13.9730
        pos_tol = ss.gamma.rvs(a=alpha, scale=mean_position_tolerance / alpha) * np.pi / 180

        return pos_tol

    def firing_rate_modifier(self, x, y):
        """
        Given (x,y) pixel position coordinates return how much firing rate of neuron is
        impacted by distance from neuron's receptive field center. Receptive field center
        is a function of gaze center and the receptive filed center offset from the gaze
        center.

        :param x: x coordinate of object in radians of eccentricity
        :param y: y coordinate of object in radians of eccentricity

        :rtype  : Normalized firing rate (Rate modifier)
        """
        mean_rsp = np.exp(-((x - self.rf_center[0])**2 + (y - self.rf_center[1])**2) /
                          (self.position_tolerance**2))

        # TODO: Add noise Spatial Sensitivity of TE Neurons - H.OP BEECK and R Vogels(2000)
        # 1.1*log(mean response)+ 1.5

        return mean_rsp

    def print_parameters(self):
        print("Profile            = %s" % self.type)
        print("Position Tolerance = %0.4f (Radians)" % self.position_tolerance)
        print("RF Center          = [%0.4f, %0.4f] (Radians)"
              % (self.rf_center[0], self.rf_center[1]))

    def plot_position_tolerance_contours(self,
                                         x_start=-np.pi / 2,
                                         x_stop=np.pi / 2,
                                         y_start=-np.pi / 2,
                                         y_stop=np.pi / 2,
                                         axis=None,
                                         n_contours=6,
                                         font_size=34,
                                         print_parameters=True):
        """
        Contour Plots of the spatial receptive field of the neuron.

        :param font_size    : font_size of text on plot (default=34)
        :param x_start      : Default = -np.pi/2
        :param x_stop       : Default = np.pi/2
        :param y_start      : Default = -np.pi/2
        :param y_stop       : Default = np.pi/2
        :param axis         : Python axis object for where to plot. Useful for adding contour plot
                              as a subplot in an image. Default = None
        :param n_contours   : Number of contour lines to plot. Specifying a single contour line,
                              for example when plotting the receptive fields of multiple neurons,
                              does not plot add a title.

        :rtype              : None.
        """
        n_points = 180

        x = np.linspace(start=x_start, stop=x_stop, num=n_points)
        y = np.linspace(start=y_start, stop=y_stop, num=n_points)

        xx, yy = np.meshgrid(x, y)
        zz = self.firing_rate_modifier(xx, yy)

        if axis is None:
            fig, axis = plt.subplots()

        if n_contours == 1:
            # Plot a single contour for  cell, there will likely be contours for other cells on
            # the same figure
            axis.contour(xx, yy, zz, 1, colors='blue')
        else:
            axis.contour(xx, yy, zz, n_contours, linewidths=2)

        axis.set_xlim([x_start, x_stop])
        axis.set_ylim([y_start, y_stop])

        axis.scatter(
            0, 0,
            color='black',
            marker='+',
            s=150,
            label='Gaze Centre',
            linewidth=4
        )

        axis.set_ylabel('Y (Radians)', fontsize=font_size)
        axis.set_xlabel('X(Radians)', fontsize=font_size)

        axis.tick_params(axis='x', labelsize=font_size)
        axis.tick_params(axis='y', labelsize=font_size)

        # If more details are required use a higher contour value.
        if 1 != n_contours:
            axis.scatter(
                self.rf_center[0],
                self.rf_center[1],
                color='green',
                label='RF Centre',
                marker='o',
                edgecolor='green',
                s=150
            )

            # axis.set_title('Positional Tolerance = %0.2f (Rad)' % self.position_tolerance,
            #                fontsize=font_size)

            if print_parameters:
                axis.annotate(
                    r'$\sigma_{PT}=%0.2f$' % self.position_tolerance,
                    xy=(0.95, 0.2),
                    xycoords='axes fraction',
                    fontsize=font_size,
                    horizontalalignment='right',
                    verticalalignment='top')

            axis.legend(fontsize=font_size - 5, loc='best', scatterpoints=1)

if __name__ == "__main__":
    plt.ion()

    n1 = GaussianPositionProfile(selectivity=0.1)
    n1.print_parameters()
    n1.plot_position_tolerance_contours()
