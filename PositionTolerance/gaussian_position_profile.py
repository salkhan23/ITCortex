# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 18:50:17 2014

@author: s362khan
"""
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt


class GaussianPositionProfile:
    """
    Models neurons position tolerance receptive field using a Gaussian function.
    Reference [Zoccolan et. al, 2007].
    """
    def __init__(self, **kwargs):
        """
        :param kwargs: dictionary of required and option parameters for position tuning
            REQUIRED PARAMETERS:
                selectivity = Neurons selectivity Index

            OPTIONAL PARAMETERS:
                rfCenterOffset = List [x, y] in pixel coordinates of center of receptive field
                                 relative to center of gaze. Default = (0, 0)
                imageSize = Tuple (x,y) of input image dimensions.
                            Determines the default center of gaze = center of image, (x/2, y/2).
                            Default imageSize = (1382, 512). [KITTI Data Set]

        :rtype : Object gaussian position profile class
        """
        self.type = 'gaussian'

        # Check required parameters
        required_params = ['selectivity']
        for param in required_params:
            if param not in kwargs.keys():
                raise Exception("Required Parameter '%s' not provided" % param)

        self.params = kwargs

        # Check & Set optional parameters
        if 'rfCenterOffset' not in self.params.keys():
            self.params['rfCenterOffset'] = np.array([0, 0])

        if 'imageSize' not in kwargs.keys():
            self.params['imageSize'] = np.array([1382, 512])  # KITTI Data image size

        # Get position tolerance
        self.params['posTolDeg'] = self.__get_position_tolerance()

        if not isinstance(self.params['rfCenterOffset'], np.ndarray):
            self.params['rfCenterOffset'] = np.array(self.params['rfCenterOffset'])
        if not isinstance(self.params['imageSize'], np.ndarray):
            self.params['imageSize'] = np.array(self.params['imageSize'])

    def __get_position_tolerance(self):
        """
        Method determines the position tolerance of the Neuron. Position Tolerance is
        defined as 2*standard deviation of the Gaussian function

        :rtype : Position tolerance of the neuron in degree of eccentricity

        Two properties of position tolerance are modeled: (1) Position tolerance decreases
        as selectivity/spareness of neuron increases, (2) Position tolerance variations
        (spread) decrease as selectivity/sparseness decreases.

        A set of gamma random variables with constant shape (alpha) and variable spread
        (scale) that decreases with neuron selectivity are used to fit scatter points in
        Figure 4.A of of Zoccolan et. al, 2007. A best fit linear regression line to all
        scatter points is used to model decreasing mean (scale = Mean\alpha) of the
        gamma random variables.

        Maximum likelihood fitting (alpha value that best fits the data) is used to
        determine alpha. See mLGammmaFit.py for ML fitting.

        Gamma RV Mean(sparseness) = -9.820*sparseness + 13.9730
        Gamma RV Scale(sparseness) = mean(spareness)\ alpha
        """
        alpha = 4.04
        mean_position_tolerance = -9.820*self.params['selectivity'] + 13.9730

        return ss.gamma.rvs(a=alpha, scale=mean_position_tolerance/alpha)

    def firing_rate_modifier(self, x, y, deg2pixel, gaze_center=None):
        """
        Given (x,y) pixel position coordinates return how much firing rate of neuron is
        impacted by distance from neuron's receptive field center. Receptive field center
        is a function of gaze center and the receptive filed center offset from the gaze
        center.

        :rtype : normalized firing rate.

        :param x:           x pixel coordinate of position
        :param y:           y pixel coordinate of position
        :param deg2pixel:   Degree to pixel conversion factor
        :param gaze_center: Tuple of (x,y) coordinates of center of gaze. Default = Image Center,
                            Determined from image size specified during initialization
        """
        if gaze_center is None:
            gaze_center = self.params['imageSize']/2

        x_deg = (x-(gaze_center[0]-self.params['rfCenterOffset'][0]))/deg2pixel
        y_deg = (y-(gaze_center[1]-self.params['rfCenterOffset'][1]))/deg2pixel

        sigma = self.params['posTolDeg']/2

        mean_rsp = np.exp(-(x_deg**2 + y_deg**2) / (2 * sigma**2))

        # TODO: Add noise Spatial Sensitivity of TE Neurons - H.OP BEECK and R Vogels(2000)
        # 1.1*log(mean response)+ 1.5

        return mean_rsp

    def print_parameters(self):
        print("Profile: %s" % self.type)
        print("Image Size: %s" % self.params['imageSize'])
        print("Position Tolerance: %0.2f(degrees)" % self.params['posTolDeg'])
        print("RF Center Offset (from Gaze Center): %s(degrees)" % self.params['rfCenterOffset'])
#        keys = sorted(self.params.keys())
#        for keyword in keys:
#            print ("%s : %s" %(keyword, self.params[keyword]))

    def plot_position_tolerance(self,
                                deg2pixel,
                                x_start=0, x_stop=None, x_step=1,
                                y_start=0, y_stop=None, y_step=1,
                                gaze_center=None):

        # Necessary for 3D Plot
        from mpl_toolkits.mplot3d import Axes3D

        if gaze_center is None:
            gaze_center = self.params['imageSize']/2

        if x_stop is None:
            x_stop = self.params["imageSize"][0]
        if y_stop is None:
            y_stop = self.params["imageSize"][1]

        x = np.arange(x_start, x_stop, x_step)
        y = np.arange(y_start, y_stop, y_step)

        xx, yy = np.meshgrid(x, y)
        zz = self.firing_rate_modifier(xx, yy, deg2pixel=deg2pixel, gaze_center=gaze_center)
        f1 = plt.figure()
        ax = f1.gca(projection='3d')
        ax.set_title("Position Tolerance Profile Gaze Center=(%i,%i), RF Center Offset=(%i, %i)"
                     % (gaze_center[0], gaze_center[1],
                        self.params['rfCenterOffset'][0],
                        self.params['rfCenterOffset'][1]))
        ax.plot_surface(xx, yy, zz)
        ax.scatter(gaze_center[0], gaze_center[1], 1, color='red', marker='+', linewidth=2,
                   label='Gaze Center')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Normalized Firing Rate (spikes/s)')

    def plot_position_tolerance_contours(self,
                                         deg2pixel,
                                         x_start=0, x_stop=None, x_step=0.5,
                                         y_start=0, y_stop=None, y_step=0.5,
                                         gaze_center=None, axis=None, n_contours=6):

        if gaze_center is None:
            gaze_center = self.params['imageSize']/2

        if x_stop is None:
            x_stop = self.params['imageSize'][0]
        if y_stop is None:
            y_stop = self.params['imageSize'][1]

        x = np.arange(x_start, x_stop, x_step)
        y = np.arange(y_start, y_stop, y_step)
        xx, yy = np.meshgrid(x, y)
        zz = self.firing_rate_modifier(xx, yy, deg2pixel=deg2pixel, gaze_center=gaze_center)

        if axis is None:
            f, axis = plt.subplots()

        c_plot = axis.contour(xx, yy, zz, n_contours, colors='k')
        axis.set_xlim([x_start, x_stop])
        axis.set_ylim([y_start, y_stop])
#        plt.clabel(c_plot, inline=1)

        axis.scatter(gaze_center[0], gaze_center[1], 1, color='red',
                     marker='+', linewidth=4, label='Gaze Center (%i, %i)'
                     % (gaze_center[0], gaze_center[1]))

        rf_center_x = gaze_center[0] - self.params['rfCenterOffset'][0]
        rf_center_y = gaze_center[1] - self.params['rfCenterOffset'][1]
        axis.scatter(rf_center_x, rf_center_y, 1, color='blue',
                     marker='o', linewidth=4,
                     label='Rf Center (%i, %i)' % (rf_center_x, rf_center_y))

        axis.set_ylabel('Y')
        axis.set_xlabel('X')
        axis.set_title('Positional Tolerance(Degrees) = %0.2f'
                       % (self.params['posTolDeg']))
        axis.grid()


if __name__ == "__main__":
    plt.ion()
    x1params = {'selectivity': 0.1}

    n1 = GaussianPositionProfile(**x1params)
    n1.print_parameters()
    n1.plot_position_tolerance(deg2pixel=10)

    # Create a Neuron that processes different image sizes and RF Centers
    n2 = GaussianPositionProfile(imageSize=(800, 800),
                                 rfCenterOffset=(20, 20), **x1params)
    n2.print_parameters()
    # Plot profile at a different center of gaze
    n2.plot_position_tolerance(gaze_center=(100, 100), deg2pixel=10)
