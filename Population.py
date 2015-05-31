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
import os

from ObjectSelectivity import selectivityFit as SF
from PositionTolerance import rfCenterFit as RFCenter


def PlotPopulationRf(itPopulation, axis=None, gazeCenter=np.array([0, 0]), nContours=1):
    ''' Plot Receptive Fields of entire Population '''
    if axis is None:
        f, axis = plt.subplots()

    [n.position.PlotPositionToleranceContours(gazeCenter=gazeCenter,
                                              nContours=nContours, axis=axis)
     for n in itPopulation]

    axis.set_title('Population Receptive Field Sizes N=%i, gazeCenter=(%i, %i)'
                   % (len(itPopulation), gazeCenter[0], gazeCenter[1]))


def PlotPopulationObjPreferences(itPopulation, axis=None):
    ''' Plot Selectivity Profiles of entire Population '''
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


def get_ground_truth_from_file(input_frame):
    """
    Return ground truth for each frame.
    :param input_frame: Complete or relative path to file from working directory that list
                        all objects and their attributes.
    :return: lists of objects, x-coordinates, y_coordinates and y_rotation in each file
    """
    objs = []
    x_array = []
    y_array = []
    y_rotation_array = []
    with open(input_frame, 'rb') as fid:
        for line in fid:
            temp = line.split(',')
            objs.append(temp[0])
            x_array.append(float(temp[1]))
            y_array.append(float(temp[2]))
            y_rotation_array.append(float(temp[3]))

    return objs, x_array, y_array, y_rotation_array


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
#     title = 'Population Selectivity Distribution'
#     plt.figure(title)
#     plt.title(title)
#     plt.hist(selectivityDist, bins=np.linspace(start=0, stop=1, num=10))
#     plt.xlabel('Selectivity')
#     plt.ylabel('Frequency')

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

    return (population)

if __name__ == "__main__":
    plt.ion()
    population = Main()

    # ---------------------------------------------------------------------------------------------
    # Population Plots and Prints
    # ---------------------------------------------------------------------------------------------
#    # Sample Neuron Properties
#    n = 0
#    print ("Properties of neuron %i" % n)
#    population[n].PrintProperties()

    # PLot object selectivities of population
    PlotPopulationObjPreferences(population)

    # Plot spatial receptive fields of population
    gazeCenter = np.array([1382/2, 512/2])
    PlotPopulationRf(population, gazeCenter=gazeCenter, nContours=1)

    # ---------------------------------------------------------------------------------------------
    # Population Firing Rates
    # ---------------------------------------------------------------------------------------------

    # Single Object in a frame
    # Create a list of Ground Truth
    #                  object      x      y        yRotation
    groundTruthLst = [['bus',      0,     512/2,   0],
                      ['bus',    100,     512/2,   0],
                      ['bus',    200,     512/2,   0],
                      ['bus',    300,     512/2,   0],
                      ['bus',    400,     512/2,   0],
                      ['bus',    500,     512/2,   0],
                      ['bus',    600,     512/2,   0],
                      ['bus',    700,     512/2,   0],
                      ['bus',    800,     512/2,   0],
                      ['bus',    900,     512/2,   0],
                      ['bus',   1000,     512/2,   0],
                      ['bus',   1100,     512/2,   0],
                      ['bus',   1200,     512/2,   0],
                      ['bus',   1300,     512/2,   0]]

    rate_time_matrix = np.zeros(shape=(len(groundTruthLst), len(population)))

    for idx, groundTruth in enumerate(groundTruthLst):
        rate_time_matrix[idx, :] = [neuron.firing_rate(*groundTruth, gaze_center=(1382/2, 512/2))
                                    for neuron in population]

    # Plot firing rates of population for all Ground truth entries
    f, axArr = plt.subplots(len(groundTruthLst), sharex=True)
    f.subplots_adjust(hspace=0.0)

    for idx, rates in enumerate(rate_time_matrix):
        axArr[idx].plot(rates)
        axArr[idx].set_ylim(0, 100)
        axArr[idx].set_yticks([])
        axArr[idx].set_ylabel("%i" % groundTruthLst[idx][1])

    axArr[-1].set_yticks(np.arange(0, 101, step=20))
    axArr[-1].set_xlabel('Neuron')
    f.suptitle('Population Firing Rates to list of ground truth.' +
               'Y axis changes in x coordinate of object', fontsize=16)

    # Multiple Objects in a frame -----------------------------------------------------------------
    # Read Ground Truth for each frame from a file
    video_dir = './SampleVideoGroundTruth'

    video = [os.path.join(video_dir, frame_data) for frame_data in os.listdir(video_dir)
             if 'frame' in frame_data.lower()]

    video.sort()

    rate_time_matrix = np.zeros(shape=(len(video), len(population)))

    for idx, frame in enumerate(video):
        objects, x_arr, y_arr, y_rotation = get_ground_truth_from_file(frame)
        rate_time_matrix[idx, :] = \
            [neuron.firing_rate(objects, x_arr, y_arr, y_rotation, gaze_center=(1382/2, 512/2))
             for neuron in population]

    # Plot firing rates of population for all frames
    f, axArr = plt.subplots(len(video), sharex=True)
    f.subplots_adjust(hspace=0.0)

    for idx, rates in enumerate(rate_time_matrix):
        axArr[idx].plot(rates)
        axArr[idx].set_ylim(0, 100)
        axArr[idx].set_yticks([])
        axArr[idx].set_ylabel("F %i" % idx)

    axArr[-1].set_yticks(np.arange(0, 101, step=20))
    axArr[-1].set_xlabel('Neuron')
    f.suptitle('Population Firing Rates to Video Sequence.', fontsize=16)
