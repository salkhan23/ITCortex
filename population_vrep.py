# -*- coding: utf-8 -*-
""" --------------------------------------------------------------------------------------------
This file is the main entry point for the VREP - IT cortex model. First a connection with
VREP is established. Objects from the scene are extracted. Second a population of Interior
Temporal (IT) neurons that respond to these objects is generated. Third, once the simulation
is started, ground truth for all objects in the field of vision of the vision sensor(s) is
continuously fed into the IT population to generate firing rates.

Created on Mon Aug  3 10:15:01 2015

@author: s362khan
----------------------------------------------------------------------------------------------"""

import matplotlib.pyplot as plt
from vrep.src import vrep

def main():
    # libsimx = CDLL("vrep/src/remoteApi.so")
    pass



if __name__ == "__main__":
    plt.ion()
    population = main()