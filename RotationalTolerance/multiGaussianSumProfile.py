# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 09:19:15 2014

@author: s362khan
"""

import matplotlib.pyplot as plt
import numpy as np

class multiGaussianSumProfile():
    
    def __init__(self, **kwargs):
        """ Model Rotational tolerance as the sum of multiple independent Gaussian 
            functions centered at the cells preferred angle(s) with sigma defining the 
            spread of the tolerance. Additionally, Poisson variation is added the 
            to model the noise around this average rate. Multi-Gaussian sum profile 
            reference [logothetis et al, 1995]. Poisson variations additionally added.

        PARAMETERS:
            nGaussian     = Number of Gaussian random variables(RV) in tuning profile.
            muArray       = Array of means of all Gaussian RVs.
            sigmaArray    = Array of standard deviations of all Gaussian RVs.
            ampArray      = Normalized relative amplitudes of Gaussian peaks. 
            rMax          = Maximum average firing rate, determines the level of Poisson 
                            variations around the average determined by the multi-Gaussian
                            sum.
        """
        
        self.type = 'Multi-Gaussian Sum with Poisson Variance'
        requiredParams = ['nGaussian', 'muArray', 'sigmaArray', 'ampArray', 'rMax']
        
        for param in requiredParams:
            # Check if all required parameters are provided
            if param not in kwargs.keys():
                raise Exception("Required Parameter '%s' not provided" %(param))
            
            # Additional general checks
            if (param in ['muArray', 'sigmaArray', 'ampArray']):
                if not isinstance( kwargs[param], (list, np.ndarray)):
                    kwargs[param] = np.array([kwargs[param]])
                    
                if (len(kwargs[param]) != kwargs['nGaussian']):
                    raise Exception("Invalid length for %s=%d, expecting=%d!" \
                            %(param, len(kwargs[param]), kwargs['nGaussian']))

        # Adjust all means to lie within -180, 180
        kwargs['muArray'] = self.__AdjustAngles(kwargs['muArray'], 0)
       
        kwargs['rMax'] = float(kwargs['rMax'])
        self.params = kwargs

    def __AdjustAngles(self, angles, mu):
        '''Return angles such that they lie within [-180, 180) of the mean, mu '''
        
        if not isinstance(angles, (np.ndarray, list)):
            angles = np.array([angles])

        angles = [angle + 360 if angle < (mu-180) else angle for angle in angles]
        angles = [angle - 360 if angle > (mu+180) else angle for angle in angles]

        return(np.array(angles))
        
    def __GetAvgFiringRate(self, x):
        '''Return average firing rate as determined by the multi-gaussian sum '''
        s = np.zeros(shape = x.shape)
        for idx in np.arange(self.params['nGaussian']):
            x = self.__AdjustAngles(x, self.params['muArray'][idx])
            s += self.params['ampArray'][idx] * \
                np.exp( -(x - self.params['muArray'][idx])**2 / 
                         (2.0*self.params['sigmaArray'][idx]**2) )

        return(s)
        
    def firing_rate_modifier(self, x):
        '''Return average firing rate as determined by the multi-gaussian sum and
            poisson variation around the average.
        '''
        if not isinstance(x, np.ndarray):
            x = np.array([x])
           
        return(np.random.poisson(self.__GetAvgFiringRate(x)*self.params['rMax']) / 
               self.params['rMax'])

    def print_parameters(self):
        print("Profile: %s" %self.type)
        keys = sorted(self.params.keys())
        for keyword in keys:
            print ("%s : %s" %(keyword, self.params[keyword]))

    def PlotProfile(self, angles=np.arange(-180, 180, step = 1), axis = None):
        if axis is None:
            f, axis = plt.subplots()

        txtStr = ''
        for idx in np.arange(self.params['nGaussian']):
            txtStr += r'$\ \mu%i=%0.2f\ \sigma%i=%0.2f\ Amp%i=%0.2f,$' \
            % (idx, self.params['muArray'][idx], 
               idx, self.params['sigmaArray'][idx],
               idx, self.params['ampArray'][idx])
        axis.plot(angles, self.firing_rate_modifier(angles), label=txtStr)

        axis.set_title('Rotational Tolerance Profile')
        axis.set_xlabel('angle')
        axis.set_ylabel('Normalized Spike Rate (Spikes/s)')
        axis.grid()
        axis.set_ylim([0, 1])
        axis.set_xlim([-180,180])
        axis.legend(loc=1, fontsize='small')


def main():
    import lse_gaussian_mixture_fit as gF
    import pickle
    
    # Load Extracted data
    with open('rotationalTolerance.pkl', 'rb') as handle:
        data = pickle.load(handle)
    
    # Fig 5a, Logothesis, Pauls & Poggio -1995 -------------------------------------------
    angles = data['fig5ax']
    original = data['fig5ay']/max(data['fig5ay'])
    
    x1Params = { 'nGaussian' : 1, 
                 'muArray'   : 103, 
                 'sigmaArray': 26.55, 
                 'ampArray'  : 0.89, 
                 'rMax': 16 }
    
    x2Params = { 'nGaussian' : 2, 
                 'muArray'   : [103,  -66], 
                 'sigmaArray': [26.55, 44], 
                 'ampArray'  : [0.89,  0.2], 
                 'rMax': 16 }
                 
    x3Params = { 'nGaussian' : 3, 
                 'muArray'   : [103,  -38,    -91], 
                 'sigmaArray': [26.55, 9.52,   20],
                 'ampArray'  : [0.89,  0.34,  0.2],
                 'rMax': 16 } 

#    # Fig 5b, Logothesis, Pauls & Poggio - 1995 ------------------------------------------
#    angles = data['fig5bx']
#    original = data['fig5by']/max(data['fig5by'])
#    
#    x1Params = { 'nGaussian' : 1, 
#                 'muArray'   : -11, 
#                 'sigmaArray': 20, 
#                 'ampArray'  : 0.82, 
#                 'rMax': 16 }
#
#    x2Params = { 'nGaussian' : 2, 
#                 'muArray'   : [-10,  159], 
#                 'sigmaArray': [15.73, 136.74], 
#                 'ampArray'  : [0.8,  0.16], 
#                 'rMax': 16 }
#                 
#    x3Params = { 'nGaussian' : 3, 
#                 'muArray'   : [-11,-157,84], 
#                 'sigmaArray': [18.71, 42.77, 28.63],
#                 'ampArray'  : [0.84,  0.21,  0.19],
#                 'rMax': 16 } 
    
#    # Fig 5c, Logothesis, Pauls & Poggio - 1995 ------------------------------------------
#    angles = data['fig5cx']
#    original = data['fig5cy']/max(data['fig5cy'])
#    
#    x1Params = { 'nGaussian' : 1, 
#                 'muArray'   : 118, 
#                 'sigmaArray': 13.36, 
#                 'ampArray'  : 1, 
#                 'rMax': 36 }
#
#    x2Params = { 'nGaussian' : 2, 
#                 'muArray'   : [118,  -67], 
#                 'sigmaArray': [13.36, 11.85], 
#                 'ampArray'  : [1,  0.58], 
#                 'rMax': 36 }
#                 
#    x3Params = { 'nGaussian' : 3, 
#                 'muArray'   : [118,  -67,    -152], 
#                 'sigmaArray': [13.36, 11.85,   8.67],
#                 'ampArray'  : [1,  0.58,  0.2],
#                 'rMax': 36 } 

#    # Fig 5c, Logothesis, Pauls & Poggio -1995 ------------------------------------------
#    angles = data['fig5cx']
#    original = data['fig5cy']/max(data['fig5cy'])
#    
#    x1Params = { 'nGaussian' : 1, 
#                 'muArray'   : 118, 
#                 'sigmaArray': 13.36, 
#                 'ampArray'  : 1, 
#                 'rMax': 36 }
#
#    x2Params = { 'nGaussian' : 2, 
#                 'muArray'   : [118,  -67], 
#                 'sigmaArray': [13.36, 11.85], 
#                 'ampArray'  : [1,  0.58], 
#                 'rMax': 36 }
#                 
#    x3Params = { 'nGaussian' : 3, 
#                 'muArray'   : [118,  -67,    -152], 
#                 'sigmaArray': [13.36, 11.85,   8.67],
#                 'ampArray'  : [1,  0.58,  0.2],
#                 'rMax': 36 }

#    # Fig 5c, Logothesis, Pauls & Poggio -1995 ------------------------------------------
#    angles = data['fig5dx']
#    original = data['fig5dy']/max(data['fig5dy'])
#    
#    x1Params = { 'nGaussian' : 1, 
#                 'muArray'   : -95, 
#                 'sigmaArray': 23.16, 
#                 'ampArray'  : 0.92, 
#                 'rMax': 52 }
#
#    x2Params = { 'nGaussian' : 2, 
#                 'muArray'   : [-94, -29], 
#                 'sigmaArray': [14.78, 87.84], 
#                 'ampArray'  : [0.82,  0.27], 
#                 'rMax': 52 }

    #-------------------------------------------------------------------------------------
    try:
        neurons = []
        neurons.append(multiGaussianSumProfile(**x1Params))
        neurons.append(multiGaussianSumProfile(**x2Params))
        neurons.append(multiGaussianSumProfile(**x3Params))

    except NameError:
        pass
        
    print ("Number of defined neurons %i" %len(neurons))
    anglesAll = np.arange(-180, 180, step =1)
    nSubplots = len(neurons)
    
    plt.figure("Rotational Tolerance")
    for idx, neuron in enumerate(neurons):
        plt.subplot(nSubplots, 1, idx+1)
        plt.scatter(angles, original, label = 'Original Data', color ='red')
        plt.scatter(angles, neuron.firing_rate_modifier(angles),
                    label = r'$Simulated\ %i\ Gaussian\ Data\ $' %(idx+1) )
       
        #Plot Multi-Gaussian sum fit
        if (idx == 0):
            plt.plot(anglesAll, gF.single_gaussian(anglesAll,
                     x1Params['muArray'], x1Params['sigmaArray'], x1Params['ampArray']),
                     label = 'Multi-Gaussian Fit') 
        elif (idx == 1):
            plt.plot(anglesAll, gF.double_gaussian(anglesAll,
                     x2Params['muArray'][0], x2Params['sigmaArray'][0], x2Params['ampArray'][0], 
                     x2Params['muArray'][1], x2Params['sigmaArray'][1], x2Params['ampArray'][1]),
                     label = 'Multi-Gaussian Fit')
        elif (idx == 2):
            plt.plot(anglesAll, gF.triple_gaussian(anglesAll,
                     x3Params['muArray'][0], x3Params['sigmaArray'][0], x3Params['ampArray'][0], 
                     x3Params['muArray'][1], x3Params['sigmaArray'][1], x3Params['ampArray'][1],
                     x3Params['muArray'][2], x3Params['sigmaArray'][2], x3Params['ampArray'][2]),
                     label = 'Multi-Gaussian Fit') 
        
        plt.xlabel('Angle (Deg)')
        plt.ylabel('Normalized Firing Rate(spikes\s)')
        plt.legend(loc=2, fontsize='small')

if __name__ == "__main__": 
    plt.ion()
    main()