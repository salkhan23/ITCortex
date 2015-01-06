# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 16:06:16 2014

     Simulates a population of IT Neurons

@author: s362khan
"""
import numpy as np
import dill
import InferiorTemporalNeuron as IT
import matplotlib.pyplot as plt

def PlotPositionTolerance(itPopulation):
    plt.figure()
    plt.title('Position Tolerance of IT population')
    posTol = [neuron.position.posTolDeg for neuron in itPopulation]
    plt.scatter(selectivity, posTol)
    plt.xlabel('Selectivity')
    plt.ylabel('Position Tolerenace Degrees (2*SD of a gaussian RV)')
    
def PlotObjectPreferences(itPopulation):
    plt.figure()
    plt.title("Ranked Object Preferences")
    plt.xlabel('Object Preference')
    plt.ylabel('Normalized Firing Rate')
    for neuron in itPopulation:
        Lst = neuron.GetRankedObjectLists()
        objects, rate = zip(*Lst) 
        x = np.arange(len(rate))
        plt.plot(x, rate)
    
if __name__ == "__main__":
    plt.ion()

    dill.load_session('positionToleranceData.pkl')
    popSize = len(xScatterRaw)

    # Object Preference lists
    fixedObjList = ['car', 
                    'van', 
                    'Truck', 
                    'bus', 
                    'pedestrian', 
                    'cyclist', 
                    'tram', 
                    'person sitting']

    # Selectivity
    selectivity = xScatterRaw

    # Object List
    objList = fixedObjList

    #IT Population
    itPopulation=[IT.Neuron(objList, s) for s in selectivity]
    
    PlotPositionTolerance(itPopulation)
    
    PlotObjectPreferences(itPopulation)





    
    
    









#fixedS = np.arange(1, step=1.0/popSize)
#uniformS = np.random.uniform(size=popSize)
#
#objList = fixedObjList
#selectivity = uniformS
#
#itPopulation=[IT.ITNeuron(objList, s) for s in selectivity]
#
#
##PLot Position Tolerance of IT population
#plt.figure('Position Tolerence of IT population')
#pT= [n.posTolDeg for n in itPopulation]
#plt.scatter(selectivity, pT, color='blue')
#plt.xlabel('Selectivity')
#plt.ylabel('Position Tolerenace Degrees (2*SD of a gaussian RV)')




