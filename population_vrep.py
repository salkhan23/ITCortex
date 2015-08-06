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
import numpy as np
import matplotlib.pyplot as plt
from vrep.src import vrep


# VREP CONSTANTS ----------------------------------------------------------------------------------

# VISION SENSOR PARAMETERS
# FLOAT
VS_NEAR_CLIPPING_PLANE = 1000
VS_FAR_CLIPPING_PLANE = 1001
VS_PERSPECTIVE_PROJECTION_ANGLE = 1004
# INT PARAMETERS:
VS_RESOLUTION_X = 1002
VS_RESOLUTION_Y = 1003

# All OBJECT PARAMETERS:
# FLOAT
OBJ_BOUND_BOX_MIN_X = 15  # These are relative to object reference frame
OBJ_BOUND_BOX_MIN_Y = 16
OBJ_BOUND_BOX_MIN_Z = 17
OBJ_BOUND_BOX_MAX_X = 18  # These are relative to object reference frame
OBJ_BOUND_BOX_MAX_Y = 19
OBJ_BOUND_BOX_MAX_Z = 20
# -------------------------------------------------------------------------------------------------


class VrepObject:
    def __init__(self, name, handle, max_dimension):
        self.name = name
        self.handle = handle
        self.size = max_dimension


def connect_vrep(sim_stop_time_s):
    """
    Establish connection to VREP simulation
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


def get_object_dimensions(c_id, object_handle):
    """ Return x, y, z dimensions of an object """
    max_x = 0
    max_y = 0
    max_z = 0

    min_y = 0
    min_z = 0

    res, min_x = vrep.simxGetObjectFloatParameter(
        c_id,
        object_handle,
        OBJ_BOUND_BOX_MIN_X,
        vrep.simx_opmode_oneshot_wait)

    if res == vrep.simx_return_ok:
        res, max_x = vrep.simxGetObjectFloatParameter(
            c_id,
            object_handle,
            OBJ_BOUND_BOX_MAX_X,
            vrep.simx_opmode_oneshot_wait)

    if res == vrep.simx_return_ok:
        res, min_y = vrep.simxGetObjectFloatParameter(
            c_id,
            object_handle,
            OBJ_BOUND_BOX_MIN_Y,
            vrep.simx_opmode_oneshot_wait)

    if res == vrep.simx_return_ok:
        res, max_y = vrep.simxGetObjectFloatParameter(
            c_id,
            object_handle,
            OBJ_BOUND_BOX_MAX_Y,
            vrep.simx_opmode_oneshot_wait)

    if res == vrep.simx_return_ok:
        res, min_z = vrep.simxGetObjectFloatParameter(
            c_id,
            object_handle,
            OBJ_BOUND_BOX_MIN_Z,
            vrep.simx_opmode_oneshot_wait)

    if res == vrep.simx_return_ok:
        res, max_z = vrep.simxGetObjectFloatParameter(
            c_id,
            object_handle,
            OBJ_BOUND_BOX_MAX_Z,
            vrep.simx_opmode_oneshot_wait)

    if res != vrep.simx_return_ok:
        print ('Retrieving object dimensions failed with error code %d', res)

    return (max_x - min_x,
            max_y - min_y,
            max_z - min_z)


def get_scene_objects(c_id, objects):
    """ Get all objects in the VREP scene, find their names, handles and maximum real work size """

    # Ignore all object with default, it_cortex, proxy and floor in name
    objects_to_ignore = ['default', 'floor', 'it_cortex', 'proxy']

    res, handles, i_data, f_data, s_data = vrep.simxGetObjectGroupData(
        c_id,
        vrep.sim_appobj_object_type,
        0,  # Retrieves the object names in s_data
        vrep.simx_opmode_oneshot_wait)

    if res != vrep.simx_return_ok:
        print ('get_scene_objects: Remote API function call returned with error code: %i' % res)
    else:
        for count in np.arange(len(handles)):

            if not any([word in s_data[count].lower() for word in objects_to_ignore]):
                size = get_object_dimensions(c_id, handles[count])
                obj = VrepObject(s_data[count], handles[count], max(size))
                objects.append(obj)


def print_objects(objects):
    """ Print all objects stored in objects """
    print("Number of Objects in List %d" % len(objects))
    for obj in objects:
        print("\t%s: handle=%d, max_dimension=%0.1f" % (obj.name, obj.handle, obj.size))


def set_robot_velocity(c_id, target_velocity):
    """
    The model expects to find the IT Cortex Robot. Essentially this is the Pioneer 3dx Robot
    with an extra vision sensor it_cortex_vision_sensor, whose reference frame is used to
    generate the ground truth.

    TODO: See how to make this robot into a single object or use a simpler robot
    TODO: Add more checks to see if the robot has its vision senor
    """
    res, left_motor_handle = vrep.simxGetObjectHandle(
        c_id,
        "it_cortex_robot_left_motor",
        vrep.simx_opmode_oneshot_wait)

    if res == vrep.simx_return_ok:
        res, right_motor_handle = vrep.simxGetObjectHandle(
            c_id,
            "it_cortex_robot_right_motor",
            vrep.simx_opmode_oneshot_wait)

    if res != vrep.simx_return_ok:
        print ('Failed to retrieve handles to IT cortex robot motors, error code %d' % res)

    if res == vrep.simx_return_ok:
        res = vrep.simxSetJointTargetVelocity(
            c_id,
            left_motor_handle,
            target_velocity,
            vrep.simx_opmode_oneshot)

        res = vrep.simxSetJointTargetVelocity(
            c_id,
            right_motor_handle,
            target_velocity,
            vrep.simx_opmode_oneshot)

        if res != vrep.simx_return_ok and \
           res != vrep.simx_return_novalue_flag:
            print("Failed to set velocity of IT cortex robot motors, error code %d" % res)


def main():
    t_stop = 20  # Simulation stop time in seconds
    client_id = connect_vrep(t_stop)

    # Create objects list ----------------------------------------------------------------------
    objects_array = []
    get_scene_objects(client_id, objects_array)
    print_objects(objects_array)

    # Generate a Population of IT Neurons that react to the list of objects in the scene

    # Start IT Cortex Robot --------------------------------------------------------------------
    set_robot_velocity(client_id, 0.2)
    time.sleep(15)

    # Stop Simulation
    set_robot_velocity(client_id, 0)
    time.sleep(1)
    result = vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot)
    print result
    if res != vrep.simx_return_ok and \
       res != vrep.simx_return_novalue_flag:
        print("Failed to stop simulation.")
    vrep.simxFinish(-1)


if __name__ == "__main__":
    plt.ion()
    main()
