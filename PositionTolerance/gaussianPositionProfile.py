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
    Models neurons position tolerance receptive field using a gaussian function.
    Reference [Zoccolan et. al, 2007]. 

    REQUIRED PARAMETERS:        
        selectivity = Neurons selectivity Index
        
     OPTIONAL PARAMTERS:
       rfCenterOffset = List [x, y] in pixel co-ordinates of center of receptive field 
                  relative to center of gaze. Default = (0, 0)
       imageSize = Tuple (x,y) of input image dimensions.
                   Determines the default center of gaze = center of image, (x/2, y/2).
                   Default imageSize = (1382, 512). [KITTI Data Set]
       deg2Pixel = pixel span of a degree of eccentricity. TODO: why 10 ?
    """
    def __init__(self, **kwargs):    
         self.type = 'Gaussian'
         
         # Check required parameters
         requiredParams = ['selectivity']
         for param in requiredParams:
            if param not in kwargs.keys():
                raise Exception("Required Parameter '%s' not provided" %(param))

         # Check & Set optional parameters
         if 'rfCenterOffset' not in kwargs.keys():
             kwargs['rfCenterOffset'] = np.array([0, 0])
         
         if 'imageSize' not in kwargs.keys():
             kwargs['imageSize'] = np.array([1382, 512]) #Kitti Data image size
             
         if 'deg2Pixel' not in kwargs.keys():
             kwargs['deg2Pixel'] = 10
             
         # Get position tolerance
         kwargs['posTolDeg'] = self.__GetPositionTolerance(kwargs['selectivity'])
         del kwargs['selectivity']
         
         if not isinstance(kwargs['rfCenterOffset'] ,np.ndarray):
             kwargs['rfCenterOffset']= np.array(kwargs['rfCenterOffset'])
         if not isinstance(kwargs['imageSize'] ,np.ndarray):
             kwargs['imageSize']= np.array(kwargs['imageSize'])
             
         self.params = kwargs

    def __GetPositionTolerance(self, selectivity):
        """
        Method determines the position tolerance of the Neuron.Position Tolerance is 
        defined as 2*standard deviation of the gaussian function
    
        Two properties of position tolerance are modeled: (1) Position tolerance decreases 
        as selectivity/spareness of neuron increases, (2) Position tolerance variations
        (spread) decrease as selectivity/sparsness decreases.
    
        A set of gamma random variables with constant shape (alpha) and variable spread 
        (scale) that decreases with neuron selectivity are used to fit scatter points in 
        Figure 4.A of of Zoccolan et. al, 2007. A best fit linear regression line to all 
        scatter points is used to model decreaseing mean (scale = Mean\alpha) of the 
        gamma random variables.
     
        Maximum likelihood fitting (alpha value that best fits the data) is used to 
        determine alpha. See mLGammmaFit.py for ML fitting.
     
        Gamma RV Mean(spareness) = -9.820*sparsness + 13.9730
        Gamma RV Scale(sparness) = mean(spareness)\ alpha

        PARAMETERS:
            selectivity: neuron selectivity
            
        RETURN:
            Position tolerance of the neuron in degree of eccentricity
        """
        alpha = 4.04;
        meanPosTol = -9.820*selectivity + 13.9730
         
        return( ss.gamma.rvs(a=alpha, scale=meanPosTol/alpha) )

    def FiringRateModifier(self, x, y, gazeCenter = None):
        ''' 
        Given (x,y) pixel position corordinates return how much firing rate of neuron is 
        impacted by distance from neuron's receptive field center. Receptive field center
        is a function of gaze center and the receptive filed center offset from the gaze 
        center 
        
        PARAMETERS:
        x = x pixel co-ordinate of position
        y = y pixel co-ordinate of position
        gazeCenter = Tuple of (x,y) co-ordinates of center of gaze. 
                     Default = Image Center, Determined from image size specified during 
                               initialization
        '''
        if (gazeCenter is None):
            gazeCenter = self.params['imageSize']/2
            
        xDeg = (x-(gazeCenter[0]-self.params['rfCenterOffset'][0]))/self.params['deg2Pixel']
        yDeg = (y-(gazeCenter[1]-self.params['rfCenterOffset'][1]))/self.params['deg2Pixel']
        
        sigma = self.params['posTolDeg']/2
       
        #TODO: Add noise Spatial Sensitivity of TE Neurons - H.OP BEECK and R Vogels(2000)
        #1.1*log(mean response)+1.5
        meanRsp = np.exp(-(xDeg**2 + yDeg**2) / (2*(sigma)**2)) 
        rsp = meanRsp
        
        return (rsp)

    def PrintParameters(self):
        print("Profile: %s" % (self.type))
        print ("Degrees to Pixel Conversion factor: %d" % (self.params['deg2Pixel']))
        print ("Image Size: %s" % (self.params['imageSize']))
        print ("Position Tolerance: %f(degrees), %f(pixels)"
               % (self.params['posTolDeg'], self.params['posTolDeg'] * self.params['deg2Pixel']))
        print ("RF Center Offset (from Gaze Center): %s(degrees), %s(pixels)"
               % (self.params['rfCenterOffset'],
                  self.params['rfCenterOffset'] * self.params['deg2Pixel']))
#        keys = sorted(self.params.keys())
#        for keyword in keys:
#            print ("%s : %s" %(keyword, self.params[keyword]))

    def PlotPositionTolerance(self,
                              xStart=0, xStop=None, xStep=1,
                              yStart=0, yStop=None, yStep=1,
                              gazeCenter=None):

        # Necessary for 3D Plot
        from mpl_toolkits.mplot3d import Axes3D

        if (gazeCenter is None):
            gazeCenter = self.params['imageSize']/2

        if xStop is None:
            xStop = self.params['imageSize'][0]
        if yStop is None:
            yStop = self.params['imageSize'][1]
        
        x = np.arange(xStart, xStop, xStep)
        y = np.arange(yStart, yStop, yStep)
        
        X, Y = np.meshgrid(x, y)
        Z = self.FiringRateModifier(X,Y,gazeCenter=gazeCenter)
        f1 = plt.figure()
        ax = f1.gca(projection='3d')
        ax.set_title("'Position Tolerance Profile Gaze Center=(%i,%i), Rf Center Offset=(%i, %i)" 
                 %(gazeCenter[0], gazeCenter[1], 
                   self.params['rfCenterOffset'][0], 
                   self.params['rfCenterOffset'][1]) )
        ax.plot_surface(X, Y, Z)
        ax.scatter(gazeCenter[0], gazeCenter[1], 1, color='red', marker='+', linewidth=2, 
                   label='Gaze Center')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Normalized Firing Rate (spikes/s)')

    def PlotPositionToleranceContours(self,
                                      xStart=0, xStop=None, xStep=0.5,
                                      yStart=0, yStop=None, yStep=0.5,
                                      gazeCenter=None, axis=None, nContours=6):

        if (gazeCenter is None):
            gazeCenter = self.params['imageSize']/2

        if xStop is None:
            xStop = self.params['imageSize'][0]
        if yStop is None:
            yStop = self.params['imageSize'][1]

        x = np.arange(xStart, xStop, xStep)
        y = np.arange(yStart, yStop, yStep)
        X, Y = np.meshgrid(x, y)
        Z = self.FiringRateModifier(X, Y, gazeCenter=gazeCenter)

        if axis is None:
            f, axis = plt.subplots()

        cPlot = axis.contour(X, Y, Z, nContours, colors='k')
        axis.set_xlim([xStart, xStop])
        axis.set_ylim([yStart, yStop])
#        plt.clabel(cPlot, inline=1)

        axis.scatter(gazeCenter[0], gazeCenter[1], 1, color='red',
                     marker='+', linewidth=4, label='Gaze Center (%i, %i)'
                     % (gazeCenter[0], gazeCenter[1]))

        rfCenterX = gazeCenter[0] - self.params['rfCenterOffset'][0]
        rfCenterY = gazeCenter[1] - self.params['rfCenterOffset'][1]
        axis.scatter(rfCenterX, rfCenterY, 1, color='blue',
                     marker='o', linewidth=4,
                     label='Rf Center (%i, %i)' % (rfCenterX, rfCenterY))

        axis.set_ylabel('Y')
        axis.set_xlabel('X')
        axis.set_title('Positional Tolerence(Degrees) = %0.2f'
                       % (self.params['posTolDeg']))

#        textstr = 'Degree to pixel conversion factor %i' % (self.params['deg2Pixel'])
#        axis.text(0.05, 0.95, textstr, transform=axis.transAxes)
#        axis.legend(loc=3, fontsize='x-small')


if __name__ == "__main__":
    plt.ion()
    x1params = {'selectivity': 0.1,
                'deg2Pixel': 10}

    n1 = GaussianPositionProfile(**x1params)
    n1.PrintParameters()
    n1.PlotPositionTolerance()

    # Create a Neuron that processes diffrent imagesizes and rf Center
    n2 = GaussianPositionProfile(imageSize=(800, 800),
                                 rfCenterOffset=(20, 20), **x1params)
    n2.PrintParameters()
    #Plot profile at a different center of gaze
    n2.PlotPositionTolerance(gazeCenter=(100, 100))
