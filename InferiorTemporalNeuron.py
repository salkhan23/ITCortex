# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 17:40:26 2014

@author: s362khan
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings


class NoProfile:
    """ Default Tuning Profile. Complete Tolerance, No rate modification """
    def __init__(self):
        self.type = 'None'

    def FiringRateModifier(self, *args, **kwargs):
        """Return 1 no matter what inputs are provided"""
        return 1

    def PrintParameters(self):
        print("Profile: %s" % self.type)


class Neuron:
    """
    ---------------------------------------------------------------------------
    Inferior Temporal Cortex Neuron

    PARAMETERS:

    rankedObjList   = Ranked list of preferred Objects.
    Selectivity     = Activity fraction of objects neuron responds to over total number of
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

    -----------------------------------------------------------------------------------"""
    def __init__(self,
                 rankedObjList,
                 selectivity,
                 maxRate=100,
                 positionProfile='Default',
                 positionParams={},
                 yRotationProfile='Default',
                 yRotationParams={}):

        # Get Rate modification factors for objects
        rankedObjList = [item.lower() for item in rankedObjList]
        self.objects = self._PowerLawSelectivity(rankedObjList, selectivity)
        self.s = selectivity
        self.selectivityProfile = 'Power Law'

        # Maximum firing rate
        self.maxRate = maxRate

        # POSITION TOLERANCE
        if (positionProfile.lower() == 'gaussian'):
            from PositionTolerance import gaussianPositionProfile as GPP
            self.position = GPP.GaussianPositionProfile(selectivity=self.s,
                                                        **positionParams)
        else:
            self.position = NoProfile()

        # ROTATION TOLERANCE
        if (yRotationProfile.lower() == 'multigaussiansum'):
            from RotationalTolerance import multiGaussianSumProfile as MGS
            self.yRotation = MGS.multiGaussianSumProfile(rMax=self.maxRate, **yRotationParams)
        else:
            self.yRotation = NoProfile()

    def _PowerLawSelectivity(self, rankedObjList, selectivity):
        '''
        Object preference rate modification modeled as power law distribution.
        Rate Modifier = objectIdx^(-selectivity)

        PARAMETERS:
            rankedObjList = ranked list of neurons preferred objects
            selectivity = Function of fraction of objects neuron responds to divided by
                          total number of objects. As Defined in [Zoccolan et. al. 2007]
                          = {1 - [sum(Ri/n)^2 / sum(Ri^2/n)] } / (1-1/n).

        RETURN:
            Dictionary of {object: rate modification factor}

        REF: Zoccolan et.al. 2007 - Fig2
        TODO: Add rest of Power Law parameters
        '''
        if not (0 < selectivity <= 1):
            raise Exception("Selectivity %0.2f not within [0, 1]" % selectivity)

        return({item: np.power(idx, -selectivity)
               for idx, item in enumerate(rankedObjList, start=1)})

    def GetRankedObjectLists(self):
        """ Return neurons rank list of objects and rate modification factors """
        return(sorted(self.objects.items(), key=lambda item: item[1], reverse=True))

    def firing_rate(self,
                   obj_list,
                   x,
                   y,
                   y_rotation,
                   gaze_center=None):
        """ Given pixel coordinates (x, y), gazeCenter (tuple), y rotation angles, return
            firing rate(s) of neuron. Input lengths of obj_list, x, y, y_rotation, gaze_center
            should match or be equal to one.
        """
        if not isinstance(obj_list, list):
            obj_list = [obj_list]

        obj_pref_list = np.array([self.objects.get(obj.lower(), 0) for obj in obj_list])

        position_weights = self.position.FiringRateModifier(x, y, gaze_center)

        rate = self.maxRate * obj_pref_list * position_weights *\
            self.yRotation.FiringRateModifier(y_rotation)

        # Calculate a single clutter response
        # Each object in the input frame is weighted by its position modifier, and the net
        # response is the weighted sum of the isolated responses
        #
        # Modified from Zoccolan-2005 Multiple object response normalization in Monkey
        # Inferotemporal cortex. In the paper they discuss a single average, we extend that model
        # by using a weighted sum based on the spatial receptive field of the neurons
        rate = rate * position_weights / np.sum(position_weights, axis=0)

        return np.sum(rate, axis=0)

    def PrintObjectList(self):
        """ Print a ranked list of neurons object preferences """
        print("Object Preferences:")
        Lst = self.GetRankedObjectLists()
        for obj, rate in Lst:
            print ("\t%s\t\t%0.4f" % (obj, rate))

    def PrintProperties(self):
        """ Print All Parameters of Neuron """
        print ("*"*20 + " Neuron Properties " + "*"*20)
        print("Neuron Selectivity: %0.2f" % self.s)
        print("Selectivity Profile: %s" % self.selectivityProfile)
        print("Max Firing Rate (spikes/s): %i" % self.maxRate)
        self.PrintObjectList()

        print("POSITION TOLERANCE: %s" % ('-'*30))
        self.position.PrintParameters()

        print("ROTATIONAL TOLERANCE: %s" % ('-'*30))
        self.yRotation.PrintParameters()

    def PlotObjectPreferences(self, axis=None):
        """ Plot Neurons Object Preferences """
        Lst = self.GetRankedObjectLists()
        objects, rate = zip(*Lst)
        x = np.arange(len(rate))

        if axis is None:
            f, axis = plt.subplots()

        axis.plot(x, rate, 'x-')

        axis.set_title("Object Preferences Selectivity Index %0.2f" % (self.s))
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
    title = 'Single IT Neuron: Minimum Parameters'
    print('-'*100 + '\n' + title + '\n' + '-'*100)
    n1 = Neuron(rankedObjList=objList,
                selectivity=0.1)

    n1.PrintProperties()
    grndTruth = ['car', 1382/2, 512/2, 0]
    print("Neuron Firing Rate to object %s at position(%i, %i), with rotation %i: %0.2f"
          % (grndTruth[0], grndTruth[1], grndTruth[2], grndTruth[3], n1.firing_rate(*grndTruth)))

    grndTruth = ['tram', 0, 0, 100]
    print("Neuron Firing Rate to object %s at position(%i, %i), with rotation %i: %0.2f"
          % (grndTruth[0], grndTruth[1], grndTruth[2], grndTruth[3], n1.firing_rate(*grndTruth)))

    # Position Tolerance Tests --------------------------------------------------------------------
    # Gaussian Profile no Parameters
    title = 'Single IT Neuron: Gaussian Position Profile only'
    print('-'*100 + '\n' + title + '\n' + '-'*100)

    positionProfile = 'Gaussian'
    n2 = Neuron(rankedObjList=objList,
                selectivity=0.1,
                positionProfile=positionProfile)

    n2.PrintProperties()
    n2.position.PlotPositionToleranceContours()

    grndTruth = ['car', 1382/2, 512/2, 0]
    print("Neuron Firing Rate to %s at position(%i, %i), with rotation %i: %0.2f"
          % (grndTruth[0], grndTruth[1], grndTruth[2], grndTruth[3], n2.firing_rate(*grndTruth)))

    grndTruth = ['tram', 0, 0, 135]
    print("Neuron Firing Rate to %s at position(%i, %i), with rotation %i: %0.2f"
          % (grndTruth[0], grndTruth[1], grndTruth[2], grndTruth[3], n2.firing_rate(*grndTruth)))

    #Test multiple inputs -------------------------------------------------------------------------
    title = 'Multiple input tests'
    print('-'*100 + '\n' + title + '\n' + '-'*100)

    # All inputs the same dimensions
    objects = ['car', 'bus', 'tram']
    x_arr = [250, 500, 750]
    y_arr = [256, 256, 256]
    y_rotation = [0, 10, 20]
    gaze_center = (1382/2, 512/2)

    print ('Multi Obj Firing Rates: objs:%s, x=%s, y=%s, rotation=%s, firing Rates %s'
           % (objects, x_arr, y_arr, y_rotation,
              n2.firing_rate(objects, x_arr, y_arr, y_rotation, gaze_center)))

    # mixed input dimensions - must be either of length 1 or if larger
    # must match all >1 lenth inputs
    objects = ['car']
    x_arr = [250, 500, 750]
    y_arr = [256, 256, 256]
    y_rotation = [0, 10, 20]
    gaze_center = (1382/2, 512/2)

    print ('Multi Obj Firing Rates: objs:%s, x=%s, y=%s, rotation=%s, firing Rates %s'
           % (objects, x_arr, y_arr, y_rotation,
              n2.firing_rate(objects, x_arr, y_arr, y_rotation, gaze_center)))

    objects = ['car']
    x_arr = [250, 500, 750]
    y_arr = [256]
    y_rotation = [0, 10, 20]
    gaze_center = (1382/2, 512/2)

    print ('Multi Obj Firing Rates: objs:%s, x=%s, y=%s, rotation=%s, firing Rates %s'
           % (objects, x_arr, y_arr, y_rotation,
              n2.firing_rate(objects, x_arr, y_arr, y_rotation, gaze_center)))


    #Check the clutter response makes sense
    objects = 'car'
    x_arr = 1382/2
    y_arr = 512/2
    y_rotation = 0

    print("Neuron Firing Rate to %s at position(%i, %i), with rotation %i: %0.2f"
          % (objects, x_arr, y_arr,y_rotation, n2.firing_rate(objects, x_arr, y_arr,y_rotation)))

    objects = ['car', 'apple']
    x_arr = [1382/2, 1382/2]
    y_arr = 512/2
    y_rotation = 0

    print("Neuron Firing Rate to %s at position(%s, %s), with rotation %s: %0.2f"
          % (objects, x_arr, y_arr,y_rotation, n2.firing_rate(objects, x_arr, y_arr,y_rotation)))

    objects = ['car', 'apple']
    x_arr = [1382/2, 0]
    y_arr = [512/2, 0]
    y_rotation = 0

    print("Neuron Firing Rate to %s at position(%s, %s), with rotation %s: %0.2f"
          % (objects, x_arr, y_arr,y_rotation, n2.firing_rate(objects, x_arr, y_arr,y_rotation)))

    # Rotation Tolerance Tests --------------------------------------------------------------------
    title = 'Single IT Neuron: MultiGaussian Sum Rotation Profile'
    print('-'*100 + '\n' + title + '\n' + '-'*100)

    rotationProfile = 'multiGaussianSum'
    rotationParams = {'nGaussian' : 2,
                      'muArray'   : [  -10,  159.00],
                      'sigmaArray': [15.73,  136.74],
                      'ampArray'  : [  0.8,    0.16] }

    n3 = Neuron(rankedObjList=objList,
                selectivity=0.1,
                positionProfile='Gaussian',
                yRotationProfile=rotationProfile,
                yRotationParams=rotationParams)

    n3.PrintProperties()

    anglesAll = np.arange(-180, 180, step=1)
    plt.figure("Rotation Tuning Profile")
    plt.scatter(anglesAll, n3.yRotation.FiringRateModifier(anglesAll))
    plt.title("Normalized y-Rotation Tuning")
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Normalized Firing Rate')

    # Rotation & Tolerance Tolerance Tests --------------------------------------------------------
    title = 'Single IT Neuron: Low Object selectivity, Rotation & Position Tuning'
    print('-'*100 + '\n' + title + '\n' + '-'*100)

    positionProfile = 'Gaussian'
    positionParams = {'rfCenterOffset': (-15, -15)}
    rotationProfile = 'multiGaussianSum'
    rotationParams = {'nGaussian' : 2,
                      'muArray'   : [-10.00,  30.00],
                      'sigmaArray': [ 15.73,  50.74],
                      'ampArray'  : [  0.60,   0.40]}

    n4 = Neuron(rankedObjList=objList,
                selectivity=0.1,
                positionProfile=positionProfile,
                positionParams=positionParams,
                yRotationProfile=rotationProfile,
                yRotationParams=rotationParams)

    n4.PrintProperties()

    f, axArr = plt.subplots(2, 2)
    f.subplots_adjust(hspace=0.2, wspace=0.05)

    n4.PlotObjectPreferences(axArr[0][0])
    n4.position.PlotPositionToleranceContours(axis=axArr[0][1], gazeCenter=(800, 200))
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

    n5 = Neuron(rankedObjList=objList,
                selectivity=0.85,
                positionProfile=positionProfile,
                positionParams=positionParams,
                yRotationProfile=rotationProfile,
                yRotationParams=rotationParams)

    n5.PrintProperties()

    f, axArr = plt.subplots(2, 2)
    f.subplots_adjust(hspace=0.2, wspace=0.05)

    n5.PlotObjectPreferences(axArr[0][0])
    n5.position.PlotPositionToleranceContours(axis=axArr[0][1])
    n5.yRotation.PlotProfile(axis=axArr[1][0])
    plt.suptitle(title, size=16)
