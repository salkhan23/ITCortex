# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 16:06:16 2014

Simulates a population of IT Neurons

@author: s362khan
"""
import numpy as np
import InferiorTemporalNeuron as IT
import matplotlib.pyplot as plt
import random

from ObjectSelectivity import selectivityFit as SF
from PositionTolerance import rfCenterFit as RFCenter


def PlotPopulationRf(itPopulation, axis=None, gazeCenter=np.array([0, 0]), nContours=1):
    ''' Plot Receptive Fields of entire Population '''
    if axis is None:
        f, axis = plt.subplots()

    [n.position.PlotPositionToleranceContours(gazeCenter=gazeCenter,
                                              nContours=nContours, axis=axis)
     for n in itPopulation]

    axis.set_title('Population Receptive Field Sizes N=%i, gazeCenter=(%i,%i)' 
                   % (len(itPopulation), gazeCenter[0], gazeCenter[1]))


def PlotPopulationObjPreferences(itPopulation, axis=None):
    ''' PLot Selectivity Profiles of entire Population '''
    if axis is None:
        f, axis = plt.subplots()

    for neuron in itPopulation:
        Lst = neuron.GetRankedObjectLists()
        objects, rate = zip(*Lst)
        x = np.arange(len(rate))
        axis.plot(x, rate)

    axis.set_ylabel('Normalized Firing Rate')
    axis.set_xlabel('Ranked Object Preferences')
    axis.set_title('Population Object Preferences')


def Main():
    populationSize = 100

    objList = ['car',
               'van',
               'Truck',
               'bus',
               'pedestrian',
               'cyclist',
               'tram',
               'person sitting']

    #----------------------------------------------------------------------------------------------
    # Population Distributions
    # ---------------------------------------------------------------------------------------------
    # Selectivity
    selectivityDist = SF.GenerateSelectivityDistribution(populationSize)
    title = 'Population Selectivity Distribution'
    plt.figure(title)
    plt.title(title)
    plt.hist(selectivityDist, bins=np.linspace(start=0, stop=1, num=10))
    plt.xlabel('Selectivity')
    plt.ylabel('Frequency')

    # Position Population Data
    imageSize = (1382, 512)
    deg2Pixel = 10
    rfCenters = RFCenter.GenerateRfCenters(n=populationSize, deg2Pixel=deg2Pixel)
    
    # Rotation Population data - distribution of Parameters 
    # TODO: Currently not enough information 

    #----------------------------------------------------------------------------------------------
    # Generate Population
    # ---------------------------------------------------------------------------------------------
    population = np.array([])
    gazeCenter = np.array([1382/2, 512/2])

    for idx, s in enumerate(selectivityDist):
        # Randomize Objects List
        random.shuffle(objList)
        positionProfile = 'Gaussian'

        # Position Distribution - RF size determined from selectivity
        positionParams = {'rfCenterOffset': rfCenters[idx],
                          'imageSize': imageSize,
                          'deg2Pixel': deg2Pixel}

        population = np.append(population,
                               IT.Neuron(rankedObjList=objList,
                                         selectivity=s,
                                         positionProfile=positionProfile,
                                         positionParams=positionParams))

    #----------------------------------------------------------------------------------------------
    # Population Plots and Prints
    # ---------------------------------------------------------------------------------------------
    # Sample Neuron Properties
#    n = 0
#    print ("Properties of neuron %i" % n)
#    population[n].PrintProperties()

    # PLot Object Selectivities of population
    PlotPopulationObjPreferences(population)

    # Plot Spatial Receptive Fields of Population
    PlotPopulationRf(population, gazeCenter=gazeCenter, nContours=1)
    
    

if __name__ == "__main__":
    plt.ion()
    Main()
