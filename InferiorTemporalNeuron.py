# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 17:40:26 2014

@author: s362khan
"""

import numpy as np
import matplotlib.pyplot as plt


class NoProfile:
    def __init__(self):
        """
        Default Tuning Profile. Complete Tolerance, No rate modification. This is overloaded for
        each property IT neuron expects but for which the parameters are not provided.
        It makes getting the firing_rate_modifier easier instead of adding multiple conditions
        that check whether a profile exist or not before getting its firing rate modifier.

        :rtype : object
        """
        self.type = 'none'

    def firing_rate_modifier(self, *args, **kwargs):
        """Return 1 no matter what inputs are provided"""
        return 1

    def print_parameters(self):
        print("Profile: %s" % self.type)


class Neuron:
    """
    ---------------------------------------------------------------------------
    Inferior Temporal Cortex Neuron

    PARAMETERS:

    ranked_obj_list   = Ranked list of preferred Objects.
    selectivity     = Activity fraction of objects neuron responds to over total number of
                      objects. As defined in [Zoccolan et. al. 2007]
                      S = {1 - [sum(Ri/n)^2 / sum(Ri^2/n)] } / (1-1/n).
    maxFireRate     = maximum firing Rate (Spikes/second). (default = 100)

    POSITION TOLERANCE -------------------------------------------------------------------
      positionProfile = [Gaussian, None(Default)]. (string)
      positionParams  = Dictionary of all parameters of the selected profile.

      A. GAUSSIAN PARAMETERS:
      -----------------------
        Required:
          None.
        Optional Parameters:
          (1) rfCenterOffset = List [x, y] in pixel co-ordinates of center of receptive
                   field relative to center of gaze. Default = (0, 0)
          (2) imageSize = Tuple (x,y) of input image dimensions.
                   Determines the default center of gaze = center of image, (x/2, y/2).
                   Default = (1382, 512). [KITTI Data Set]
          (3) deg2Pixel = pixel span of a degree of eccentricity

    Y-ROTATIONAL TOLERANCE ---------------------------------------------------------------
      yRotationProfile = [MultiGaussianSum, Uniform(TODO), None(Default)]
      yRotationParams  = Dictionary of all parameters of the selected profile.

      MULTIGAUSSIAN SUM PARAMETERS:
      ----------------------------
        Required:
          (1) nGaussian = Number of Gaussian random variables(RVs) in tuning profile.
          (2) muArray = Array of means of all Gaussian RVs.
          (3) sigmaArray = Array of standard deviations of all Gaussian RVs.
          (4) ampArray = Normalized relative amplitudes of Gaussian peaks.
        Optional:
          None.

    SIZE TOLERANCE
    ---------------

    -----------------------------------------------------------------------------------"""
    def __init__(self,
                 ranked_obj_list,
                 selectivity,
                 max_rate=100,
                 position_profile='Default',
                 position_params={},
                 y_rotation_profile='Default',
                 y_rotation_params={},
                 size_profile='Default',
                 deg2pixel=10):

        # Get Rate modification factors for objects
        """

        :rtype : Return Single Neuron Instance
        """
        ranked_obj_list = [item.lower() for item in ranked_obj_list]

        # Get object selectivity metric for each object
        self.s = selectivity
        self.objects = self.__power_law_selectivity(ranked_obj_list)
        self.selectivityProfile = 'Power Law'

        self.maxRate = max_rate
        self.deg2pixel = deg2pixel

        # POSITION TOLERANCE
        if position_profile.lower() == 'gaussian':
            from PositionTolerance import gaussianPositionProfile as pT
            self.position = pT.GaussianPositionProfile(selectivity=self.s,
                                                       deg2pixel=self.deg2pixel,
                                                       **position_params)
        else:
            self.position = NoProfile()

        # ROTATION TOLERANCE
        if y_rotation_profile.lower() == 'multigaussiansum':
            from RotationalTolerance import multiGaussianSumProfile as rT
            self.yRotation = rT.multiGaussianSumProfile(rMax=self.maxRate, **y_rotation_params)
        else:
            self.yRotation = NoProfile()

        # SIZE TOLERANCE
        if 'lognormal' == size_profile.lower():
            # first check that gaussian position profile is defined
            if 'gaussian' == self.position.type:
                import SizeTolerance.log_normal_size_profile as st
                self.size = st.LogNormalSizeProfile(self.position.params['posTolDeg'])
            else:
                raise Exception('Lognormal Size Profile requires Gaussian Position profile!')
        else:
            self.size = NoProfile()

    def __power_law_selectivity(self, ranked_obj_list):
        """
        Object preference rate modification modeled as power law distribution.
        Rate Modifier = objectIdx^(-selectivity)

        PARAMETERS:
            ranked_obj_list = ranked list of neurons preferred objects
            selectivity = Function of fraction of objects neuron responds to divided by
                          total number of objects. As Defined in [Zoccolan et. al. 2007]
                          = {1 - [sum(Ri/n)^2 / sum(Ri^2/n)] } / (1-1/n).

        RETURN:
            Dictionary of {object: rate modification factor}

        REF: Zoccolan et.al. 2007 - Fig2
        TODO: Add rest of Power Law parameters
        """
        if not (0 < self.s <= 1):
            raise Exception("Selectivity %0.2f not within [0, 1]" % self.s)

        return({item: np.power(idx, -self.s)
               for idx, item in enumerate(ranked_obj_list, start=1)})

    def get_ranked_object_list(self):
        """ Return neurons rank list of objects and rate modification factors """
        return sorted(self.objects.items(), key=lambda item: item[1], reverse=True)

    def firing_rate(self,
                    obj_list,
                    x,
                    y,
                    y_rotation,
                    stimulus_size,
                    gaze_center=None):
        """
        Return the normalized firing rate of the neuron.

        If any input is a vector, function will assume multiple object present in the visual
        scene and with return a single multiple-object response.

        Each object in the input frame is weighted by its position modifier, and the net
        response is the weighted sum of the isolated responses. Modified from Zoccolan-2005
        Multiple object response normalization in Monkey Inferotemporal cortex. In the paper
        they discuss a single average, we extend that model by using a weighted sum, where each
        object is weighted by its position modifier.

        :rtype : Normalized firing rate of neuron

        :param obj_list:
        :param x:               pixel X coordinate
        :param y:               pixel Y coordinate
        :param y_rotation:      Rotation around the vertical axis (degrees)
        :param stimulus_size:   The distance in pixels between the outer edges along the longest
                                axis of the stimulus
        :param gaze_center:     Center of gaze (x, y)
        """
        if not isinstance(obj_list, list):
            obj_list = [obj_list]

        # Get normalized object preference for each object.
        obj_pref_list = np.array([self.objects.get(obj.lower(), 0) for obj in obj_list])

        # Separate out the position modifiers, these will be reused when calculating
        # the multi-object responses
        position_weights = self.position.firing_rate_modifier(x, y,
                                                              deg2pixel=self.deg2pixel,
                                                              gaze_center=gaze_center)

        rate = self.maxRate * obj_pref_list * position_weights *\
            self.yRotation.firing_rate_modifier(y_rotation) *\
            self.size.firing_rate_modifier(stimulus_size, deg2pixel=self.deg2pixel)

        # Get multi-object response
        # TODO: Add noise to the averaged response based on Zoccolan-2005
        rate = rate * position_weights / np.sum(position_weights, axis=0)

        return np.sum(rate, axis=0)

    def print_object_list(self):
        """ Print a ranked list of neurons object preferences """
        print("Object Preferences:")
        lst = self.get_ranked_object_list()
        for obj, rate in lst:
            print ("\t%s\t\t%0.4f" % (obj, rate))

    def print_properties(self):
        """ Print all parameters of Neuron """
        print ("*"*20 + " Neuron Properties " + "*"*20)
        print("Neuron Selectivity: %0.2f" % self.s)
        print("Selectivity Profile: %s" % self.selectivityProfile)
        print("Max Firing Rate (spikes/s): %i" % self.maxRate)
        self.print_object_list()

        print("POSITION TOLERANCE %s" % ('-'*30))
        self.position.print_parameters()

        print("ROTATIONAL TOLERANCE %s" % ('-'*28))
        self.yRotation.print_parameters()

        print("SIZE TOLERANCE: %s" % ('-'*33))
        self.size.print_parameters()

        print ('*'*60)

    def plot_object_preferences(self, axis=None):
        """ Plot Neurons Object Preferences """
        lst = self.get_ranked_object_list()
        objects, rate = zip(*lst)
        x = np.arange(len(rate))

        if axis is None:
            fig_obj_pref, axis = plt.subplots()

        axis.plot(x, rate, 'x-')

        axis.set_title("Object Preferences Selectivity Index %0.2f" % self.s)
        axis.set_xticklabels(objects, size='small')
        axis.set_xlabel('Objects')
        axis.set_ylabel('Normalized Rate (Spikes/s)')
        axis.grid()
        axis.set_ylim([0, 1])


if __name__ == "__main__":
    plt.ion()

    objList = ['car',
               'van',
               'Truck',
               'bus',
               'pedestrian',
               'cyclist',
               'tram',
               'person sitting']
    # Base ----------------------------------------------------------------------------------------
    title = "Single IT Neuron: Minimum Parameters, no sensitivity to position, rotation or size"
    print('-'*100 + '\n' + title + '\n' + '-'*100)
    n1 = Neuron(ranked_obj_list=objList,
                selectivity=0.1)

    n1.print_properties()
    groundTruthArray = [[       'car', 1382/2, 512/2,   0,  50],
                        [      'tram',      0,     0, 100,  50],
                        ['pedestrian',      0,     0, 100, 300]]

    for groundTruth in groundTruthArray:
        print("N1 firing rate to %s at (%i, %i), rotation %i and size %0.2f = %0.4f"
              % (groundTruth[0], groundTruth[1], groundTruth[2], groundTruth[3], groundTruth[4],
                 n1.firing_rate(*groundTruth)))

    # Position Tolerance Tests ------------------------------------------------------------------
    # Gaussian Profile no parameters
    title = "Single IT Neuron: Gaussian position profile only"
    print('-'*100 + '\n' + title + '\n' + '-'*100)

    positionProfile = 'Gaussian'
    n2 = Neuron(ranked_obj_list=objList,
                selectivity=0.3,
                position_profile=positionProfile)

    n2.print_properties()
    n2.position.plot_position_tolerance_contours(deg2pixel=n2.deg2pixel)

    groundTruthArray = [[       'car', 1382/2, 512/2,   0,  50],
                        [      'tram', 1382/4, 512/2, 100,  50],
                        ['pedestrian',      0,     0, 100, 300]]

    for groundTruth in groundTruthArray:
        print("N2 firing rate to %s at (%i, %i), rotation %i and size %0.2f = %0.4f"
              % (groundTruth[0], groundTruth[1], groundTruth[2], groundTruth[3], groundTruth[4],
                 n2.firing_rate(*groundTruth)))

    # Test multiple inputs -----------------------------------------------------------------------
    title = 'Multiple input tests'
    print('-'*100 + '\n' + title + '\n' + '-'*100)

    # Example 1: All inputs the same dimensions
    objArr = ['car', 'bus', 'tram']
    xArr = [250, 500, 750]
    yArr = [256, 256, 256]
    rotationArr = [0, 10, 20]
    sizeArr = [50, 20, 30]
    gazeCenter = (1382/2, 512/2)

    print('Inputs: objs=%s, x=%s, y=%s, rotation=%s size=%s\nfiring rate=%s'
          % (objArr, xArr, yArr, rotationArr, sizeArr,
             n2.firing_rate(objArr, xArr, yArr, rotationArr, sizeArr, gazeCenter)))

    # Example 2: mixed input dimensions -
    # must be either of length 1 or if larger must match all >1 length inputs
    objArr = ['car']
    xArr = [250, 500, 750]
    yArr = [256, 256, 256]
    rotationArr = [0, 10, 20]
    sizeArr = [50]
    gazeCenter = (1382/2, 512/2)

    print('Inputs: objs=%s, x=%s, y=%s, rotation=%s size=%s\nfiring rate=%s'
          % (objArr, xArr, yArr, rotationArr, sizeArr,
             n2.firing_rate(objArr, xArr, yArr, rotationArr, sizeArr, gazeCenter)))

    # Example 2B
    objArr = ['car']
    xArr = [250, 500, 750]
    yArr = [256]
    rotationArr = [0, 10, 20]
    sizeArr = [50]
    gazeCenter = (1382/2, 512/2)

    print('Inputs: objs=%s, x=%s, y=%s, rotation=%s size=%s\nfiring rate=%s'
          % (objArr, xArr, yArr, rotationArr, sizeArr,
             n2.firing_rate(objArr, xArr, yArr, rotationArr, sizeArr, gazeCenter)))

    # Check the clutter response makes sense
    # Example 3 Compare single and multi-object responses
    objArr = 'car'
    xArr = 1382/2
    yArr = 512/2
    rotationArr = 0
    sizeArr = 50
    print("Single object. Inputs: obj=%s at (%i, %i), rotation %i, size=%0.2f: %0.4s"
          % (objArr, xArr, yArr, rotationArr, sizeArr,
             n2.firing_rate(objArr, xArr, yArr, rotationArr, sizeArr)))

    objArr = ['car', 'apple']
    xArr = [1382/2, 1382/4]
    print("Multiple objects. Inputs: obj=%s at (%s, %i), rotation %i, size=%0.2f: %0.4f"
          % (objArr, xArr, yArr, rotationArr, sizeArr,
             n2.firing_rate(objArr, xArr, yArr, rotationArr, sizeArr)))

    objArr = ['car', 'apple']
    xArr = [1382/2, 0]
    yArr = [512/2, 0]
    print("Multiple objects. Inputs: obj=%s at (%s, %s), rotation %i, size=%0.2f: %0.4f"
          % (objArr, xArr, yArr, rotationArr, sizeArr,
             n2.firing_rate(objArr, xArr, yArr, rotationArr, sizeArr)))

    # Rotation Tolerance Tests -----------------------------------------------------------------
    title = 'Single IT Neuron: Multi Gaussian Sum Rotation Profile'
    print('-'*100 + '\n' + title + '\n' + '-'*100)

    rotationProfile = 'multiGaussianSum'
    rotationParams = {'nGaussian' : 2,
                      'muArray'   : [  -10,  159.00],
                      'sigmaArray': [15.73,  136.74],
                      'ampArray'  : [  0.8,    0.16] }

    n3 = Neuron(ranked_obj_list=objList,
                selectivity=0.1,
                position_profile='Gaussian',
                y_rotation_profile=rotationProfile,
                y_rotation_params=rotationParams)

    n3.print_properties()

    anglesAll = np.arange(-180, 180, step=1)
    plt.figure("Rotation Tuning Profile")
    plt.scatter(anglesAll, n3.yRotation.firing_rate_modifier(anglesAll))
    plt.title("Normalized y-Rotation Tuning")
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Normalized Firing Rate')

    # Position, Rotation & Size Tolerance Tests ------------------------------------------------
    title = 'Single IT Neuron: Low Object selectivity, Rotation & Position Tuning'
    print('-'*100 + '\n' + title + '\n' + '-'*100)

    positionProfile = 'Gaussian'
    positionParams = {'rfCenterOffset': (-15, -15)}
    rotationProfile = 'multiGaussianSum'
    rotationParams = {'nGaussian' : 2,
                      'muArray'   : [-10.00,  30.00],
                      'sigmaArray': [ 15.73,  50.74],
                      'ampArray'  : [  0.60,   0.40]}
    sizeProfile = 'lognormal'

    n4 = Neuron(ranked_obj_list=objList,
                selectivity=0.1,
                position_profile=positionProfile,
                position_params=positionParams,
                y_rotation_profile=rotationProfile,
                y_rotation_params=rotationParams,
                size_profile=sizeProfile)

    n4.print_properties()

    f, axArr = plt.subplots(2, 2)
    f.subplots_adjust(hspace=0.2, wspace=0.05)

    n4.plot_object_preferences(axArr[0][0])
    n4.position.plot_position_tolerance_contours(axis=axArr[0][1],
                                                 deg2pixel=n4.deg2pixel,
                                                 gaze_center=(800, 200))
    n4.yRotation.PlotProfile(axis=axArr[1][0])
    plt.suptitle(title, size=16)

    # Example 2
    title = 'Single IT Neuron: High Object selectivity, Rotation & Position Tuning'
    print('-'*100 + '\n' + title + '\n' + '-'*100)

    positionProfile = 'Gaussian'
    positionParams = {'rfCenterOffset': (30, 15)}
    rotationProfile = 'multiGaussianSum'
    rotationParams = {'nGaussian' : 3,
                      'muArray'   : [-100.00,  80.00, 105.00 ],
                      'sigmaArray': [  15.73,  15.74,  30.04 ],
                      'ampArray'  : [   0.45,   0.45,   0.10 ]}

    n5 = Neuron(ranked_obj_list=objList,
                selectivity=0.85,
                position_profile=positionProfile,
                position_params=positionParams,
                y_rotation_profile=rotationProfile,
                y_rotation_params=rotationParams)

    n5.print_properties()

    f, axArr = plt.subplots(2, 2)
    f.subplots_adjust(hspace=0.2, wspace=0.05)

    n5.plot_object_preferences(axArr[0][0])
    n5.position.plot_position_tolerance_contours(axis=axArr[0][1], deg2pixel=n5.deg2pixel)
    n5.yRotation.PlotProfile(axis=axArr[1][0])
    plt.suptitle(title, size=16)

    # # ----------------------------------------------------------------------------------------
    # title = 'Size Tolerance'
    # print('-'*100 + '\n' + title + '\n' + '-'*100)
    # n1 = Neuron(ranked_obj_list=objList,
    #             selectivity=0.1,
    #             positionProfile = 'Gaussian',
    #             size_profile='lognormal',
    #             size_params = {})
