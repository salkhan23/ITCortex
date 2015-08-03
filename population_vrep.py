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
import sys
import time
import matplotlib.pyplot as plt
from vrep.src import vrep


def connect_vrep(sim_stop_time_s):
    """ Establish connection to VREP simulation
        Add the following command to a child script in the simulation: simExtRemoteApiStart(19999)
    """
    vrep.simxFinish(-1)  # Close any open connections.

    c_id = vrep.simxStart(
        '127.0.0.1',
        19999,
        True,
        True,
        sim_stop_time_s*1000,   # TODO: Does not appear to be working.
        5)                      # Data Communication Rate (ms) (packets transferred every 5ms).

    if c_id == -1:
        print ("Failed to connect to simulation!")
        sys.exit("could not connect")
    else:
        print ('Connected to remote API server')

    return c_id


def main():
    t_stop = 5  # Simulation stop time in seconds
    client_id = connect_vrep(t_stop)


    time.sleep(10)



    # Stop Simulation
    res = vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot)
    vrep.simxFinish(-1)


if __name__ == "__main__":
    plt.ion()
    population = main()
