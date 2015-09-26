# -*- coding: utf-8 -*-
""" --------------------------------------------------------------------------------------------
This file is the main entry point of the VREP - IT cortex model. First a connection with
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
import warnings
import ctypes

from vrep.src import vrep

import it_neuron_vrep as it
import population_utils as utils
# Force reload (compile) IT cortex modules to pick changes not included in cached version.
reload(it)
reload(utils)


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
        self.max_dimension = max_dimension


def connect_vrep(sim_stop_time_ms, sim_dt_ms):
    """
    Establish connection to VREP simulation.

    NOTE: The port 19997 is for synchronous operation mode and is opened by default at startup.
    To configure it differently, change ports in remoteApiConnections.txt which is located in the
     external vrep application directory.
    """
    vrep.simxFinish(-1)  # Close any open connections.

    c_id = vrep.simxStart(
        '127.0.0.1',
        19997,
        True,
        True,
        sim_stop_time_ms,   # Only closes the remote connection, does not stop simulation.
        5)                      # Data Communication Rate (ms) (packets transferred every 5ms).

    if c_id == -1:
        print ("Failed to connect to simulation!")
        sys.exit("could not connect")
    else:
        print ('Connected to remote API server')

    # Synchronous operation mode
    res = vrep.simxSynchronous(c_id, True)
    if res != vrep.simx_return_ok:
        raise Exception('Failed to set synchronous operation mode for simulation! Err %s' % res)

    # Set the simulation step size
    res = vrep.simxSetFloatingParameter(
        c_id,
        vrep.sim_floatparam_simulation_time_step,
        sim_dt_ms/1000.0,
        vrep.simx_opmode_oneshot)
    if res != vrep.simx_return_ok and \
       res != vrep.simx_return_novalue_flag:
        raise Exception('Failed to set VREP simulation time step! Err %s' % res)

    # Start the simulation
    res = vrep.simxStartSimulation(c_id, vrep.simx_opmode_oneshot_wait)
    if res != vrep.simx_return_ok:
        raise Exception('Failed to start simulation! Err %s' % res)

    return c_id


def get_object_dimensions(c_id, object_handle):
    """ Return x, y, z dimensions of an object. """
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
    """
    Get all objects in the VREP scene, find their names, handles and size (length of the side
    with the largest dimension.

    Find the number of unique objects in the list.Currently this is only displayed locally and
    not passed on to the calling function.

    :param c_id: connected scene id.
    :param objects: Empty list to which found objects are appended to.
    """

    # Ignore all object with default, it_cortex, proxy and floor in name
    objects_to_ignore = ['default', 'floor', 'it_cortex', 'proxy']

    res, handles, i_data, f_data, s_data = vrep.simxGetObjectGroupData(
        c_id,
        vrep.sim_appobj_object_type,
        0,  # Retrieves the object names in s_data
        vrep.simx_opmode_oneshot_wait)

    if res != vrep.simx_return_ok:
        raise Exception('get_scene_objects: Failed to get object names. Error Code %d' % res)
    else:
        for count in np.arange(len(handles)):

            if not any([word in s_data[count].lower() for word in objects_to_ignore]):
                size = get_object_dimensions(c_id, handles[count])
                obj = VrepObject(s_data[count], handles[count], max(size))
                objects.append(obj)

    # Filter only unique (parent) objects
    unique_objects = set()
    for obj in objects:
        res,  parent_handle = vrep.simxGetObjectParent(
            c_id,
            obj.handle,
            vrep.simx_opmode_oneshot_wait)

        if res != vrep.simx_return_ok:
            raise Exception("get_scene_objects: Failed to get %s parent handle" % obj.name)
        elif parent_handle == -1:  # no parent and therefore unique
            unique_objects.add(obj)

    print ("Number of unique objects in scene %d" % len(unique_objects))
    print_objects(unique_objects)


def print_objects(objects):
    """ Print all objects stored in objects. """

    longest_name = max([len(obj.name) for obj in objects])

    for obj in objects:
        print("\t%s: handle=%d, max_dimension=%0.1f"
              % (obj.name.ljust(longest_name), obj.handle, obj.max_dimension))


def set_robot_velocity(c_id, target_velocity):
    """
    The model expects to find the IT Cortex Robot. Essentially this is the Pioneer 3dx Robot
    with an extra vision sensor it_cortex_vision_sensor, whose reference frame is used to
    generate the ground truth.

    TODO: See how to make this robot into a single object or use a simpler robot
    """
    motors_list = ["it_cortex_robot_left_motor", "it_cortex_robot_right_motor"]
    motors_handle_list = []

    for motor in motors_list:
        res, handle = vrep.simxGetObjectHandle(
            c_id,
            motor,
            vrep.simx_opmode_oneshot_wait)

        if res != vrep.simx_return_ok:
            raise Exception("Failed to get %s handle. Error Code %d" % (motor, res))
        else:
            motors_handle_list.append(handle)

    for ii, handle in enumerate(motors_handle_list):
        res = vrep.simxSetJointTargetVelocity(
            c_id,
            handle,
            target_velocity,
            vrep.simx_opmode_oneshot)

        if res != vrep.simx_return_ok and \
           res != vrep.simx_return_novalue_flag:
            raise Exception("Failed to set velocity of %s motor. Error code %d"
                            % (motors_list[ii], res))


def get_vision_sensor_parameters(c_id):
    """
    Retrieve parameters of the vision sensor.
    :param c_id         : connected scene id.

    :return: (alpha_rad, aspect_ratio, near_z, far_z).
        angle           : Perspective angle of vision sensor in radians.
        ar              : Aspect Ratio. Screen width/height = x_resolution/y_resolution.
        n_z             : Near clipping plane of vision sensor.
        f_z             : Far clipping plane of vision sensor.
        vis_sen_handle  : vrep object handle of vision sensor.
    """
    res, vis_sen_handle = vrep.simxGetObjectHandle(
        c_id,
        'it_cortex_robot_vision_sensor',
        vrep.simx_opmode_oneshot_wait)

    if res != vrep.simx_return_ok:
        raise Exception("Failed to get it_cortex_robot_vision_sensor handle. Error Code %d" % res)

    res, angle = vrep.simxGetObjectFloatParameter(
        c_id,
        vis_sen_handle,
        VS_PERSPECTIVE_PROJECTION_ANGLE,
        vrep.simx_opmode_oneshot_wait)
    if res != vrep.simx_return_ok:
        raise Exception("Failed to get VS_PERSPECTIVE_PROJECTION_ANGLE. Error code %d" % res)

    res, resolution_x = vrep.simxGetObjectIntParameter(
        c_id,
        vis_sen_handle,
        VS_RESOLUTION_X,
        vrep.simx_opmode_oneshot_wait)
    if res != vrep.simx_return_ok:
        raise Exception("Failed to get VS_RESOLUTION_X. Error code %d" % res)

    res, resolution_y = vrep.simxGetObjectIntParameter(
        c_id,
        vis_sen_handle,
        VS_RESOLUTION_Y,
        vrep.simx_opmode_oneshot_wait)
    if res != vrep.simx_return_ok:
        raise Exception("Failed to get VS_RESOLUTION_Y. Error code %d" % res)

    ar = resolution_x / resolution_y

    res, z_n = vrep.simxGetObjectFloatParameter(
        c_id,
        vis_sen_handle,
        VS_NEAR_CLIPPING_PLANE,
        vrep.simx_opmode_oneshot_wait)
    if res != vrep.simx_return_ok:
        raise Exception("Failed to get VS_NEAR_CLIPPING_PLANE. Error code %d" % res)

    res, z_f = vrep.simxGetObjectFloatParameter(
        c_id,
        vis_sen_handle,
        VS_FAR_CLIPPING_PLANE,
        vrep.simx_opmode_oneshot_wait)
    if res != vrep.simx_return_ok:
        raise Exception("Failed to get VS_FAR_CLIPPING_PLANE. Error code %d" % res)

    return angle, ar, z_n, z_f, vis_sen_handle


def get_object_position(c_id,
                        object_handle,
                        reference_frame_handle,
                        op_mode=vrep.simx_opmode_buffer):
    """
     Get position (x,y,z) coordinates of object specified by object_handle with respect to
     the reference frame of object specified by reference_frame_handle.

    :param c_id                     : connected scene id.
    :param object_handle            : vrep handle of object.
    :param reference_frame_handle   : vrep handle of objects whose reference frame to use.
    :param op_mode                  : operation mode of the command. Default =
                                      vrep.simx_opmode_buffer. To initialize this function use
                                      vrep.simx_opmode_streaming at first call.

    :rtype: (x,y,z) co-ordinates of target object.
    """
    res, position = vrep.simxGetObjectPosition(
        c_id,
        object_handle,
        reference_frame_handle,
        op_mode)

    if vrep.simx_opmode_buffer == op_mode:
        # check the result
        if res != vrep.simx_return_ok:
            warnings.warn("Failed to retrieve position of object  handle %d, with err %s"
                          % (object_handle, res))

    # elif vrep.simx_opmode_streaming == op_mode:
    #     print('Initializing position acquisition function for object handle %d' % object_handle)

    return position


def get_object_rotations(c_id,
                         object_handle,
                         reference_frame_handle,
                         op_mode=vrep.simx_opmode_buffer):
    """
    Get euler rotations (alpha, beta, gamma) of object specified by object_handle with respect to
    the reference frame of object specified by reference_frame_handle.

    :param c_id                     : connected scene id.
    :param object_handle            : vrep handle of object.
    :param reference_frame_handle   : vrep handle of objects whose reference frame to use.
     :param op_mode                  : operation mode of the command. Default =
                                      vrep.simx_opmode_buffer. To initialize this function use
                                      vrep.simx_opmode_streaming at first call.

    :rtype: (alpha,beta,gamma) co-ordinates of target object.
    """
    res, rotations = vrep.simxGetObjectOrientation(
        c_id,
        object_handle,
        reference_frame_handle,
        op_mode)

    if vrep.simx_opmode_buffer == op_mode:
        # check the result
        if res != vrep.simx_return_ok:
            warnings.warn("Failed to retrieve rotation angles for object handle %d, with err %s"
                          % (object_handle, res))

    # elif vrep.simx_opmode_streaming == op_mode:
    #     print('Initializing rotation acquisition function for object handle %d' % object_handle)

    return rotations


def initialize_vrep_streaming_operations(c_id,
                                         all_vrep_objs,
                                         vis_sensor_handle):
    """
    Initialize all periodic VREP streaming functions. That require VREP to setup streaming
    services on the server side.

    :param c_id                 : connected scene id.
    :param all_vrep_objs        : List of all objects in scene. Each objects is of type VrepObject
    :param vis_sensor_handle    : vrep handle for vision sensor.
    """
    for obj in all_vrep_objs:
        _ = get_object_position(
            c_id,
            obj.handle,
            vis_sensor_handle,
            vrep.simx_opmode_streaming)

        _ = get_object_rotations(
            c_id,
            obj.handle,
            vis_sensor_handle,
            vrep.simx_opmode_streaming)

        _ = vrep.simxReadStringStream(
            c_id,
            "occlusionData",
            vrep.simx_opmode_streaming)

        # wait some time to allow VREP to setup streaming services.
        time.sleep(0.1)


def get_object_visibility_levels(obj_handles, c_id):
    """
    Inform the vision sensor child script which  object handles to calculate occlusion levels
    for. Retrieve occlusion levels for all object handles in obj_handles list.

    :param obj_handles: List of object handles to calculate occlusion levels for.
    :param c_id: connected scene id.

    :rtype : List of visibility levels for each specified object
    """
    visibility_levels = np.zeros(shape=len(obj_handles))

    if obj_handles:

        # Inform vision_sensor child script which objects to calculate occlusion for
        obj_handles_string = vrep.simxPackInts(obj_handles)

        raw_bytes = (ctypes.c_ubyte * len(obj_handles_string)).from_buffer_copy(obj_handles_string)
        res = vrep.simxWriteStringStream(
            c_id,
            "getOcclusionForHandles",
            raw_bytes,
            vrep.simx_opmode_oneshot)

        if res != vrep.simx_return_ok:
            warnings.warn("Failed to send object handles for occlusion. Error %d" % res)

        # Read occlusion data from child script
        res, occlusion_data = vrep.simxReadStringStream(
            c_id,
            "occlusionData",
            vrep.simx_opmode_buffer)

        occlusion_data = vrep.simxUnpackFloats(occlusion_data)

        if res != vrep.simx_return_ok:
            warnings.warn("Failed to get occlusion data, Error %d" % res)
        else:

            # The occlusion data sent down is actually for the previous time step, to the child
            # script we specify which objects we are interested in and it returns values from its
            # previous request. We therefore need to match the object handle identities.

            # For each requested handle, the child script sends down the value of the requested
            # handle followed by its visibility level.
            for idx in np.arange(len(occlusion_data)/2):
                identity = np.int(occlusion_data[2*idx])
                value = occlusion_data[2*idx + 1]

                for idx2, obj_handle in enumerate(obj_handles):
                    if identity == obj_handle:
                        visibility_levels[idx2] = value

    return visibility_levels


def get_ground_truth(c_id, objects, vis_sen_handle, proj_mat, ar, projection_angle):
    """
    Given a list of vrepObjects, Determine if they lie within the projection frame of the vision
    senor and extract ground truth if they do.

    :param c_id             : connected scene id.
    :param objects          : list of VrepObject objects.
    :param vis_sen_handle   : vrep object handle of vision sensor.
    :param proj_mat         : Camera projection matrix.
    :param ar               : Aspect Ratio. Screen width/height = x_resolution/y_resolution.
    :param projection_angle : Perspective angle of vision sensor in radians.

    :return: A list of tuples for each object that lies in the vision sensor projection frame.
    Each Tuple Entry consists of
        obj_name,           : vrep name of object.
        x,                  : object vision frame x coordinate in degree of eccentricity (radians).
        y,                  : object vision frame y coordinate in degree of eccentricity (radians).
        size,               : size of object (span of objects maximum dimension) in degree of
                              eccentricity (radians).
        rot_x,              : rotations about the x-axis  in degree of eccentricity (radians).
        rot_y,              : rotations about the x-axis  in degree of eccentricity (radians).
        rot_z,              : rotations about the x-axis  in degree of eccentricity (radians).
        vis_non_diag        : Visibility percentage of total object (non-diagnostic). Range (0,1)
    """
    objects_in_frame = []
    object_handles_in_frame = []

    for vrep_obj in objects:
        # Get real world coordinates of object in vision sensor reference frame
        pos_world = get_object_position(
            c_id,
            vrep_obj.handle,
            vis_sen_handle)

        # Convert to homogeneous world coordinates
        pos_world.append(1)
        pos_world = np.array(pos_world)

        # Part 1. Project to Video Sensor Projection Plane
        camera_homogeneous = np.dot(proj_mat, pos_world)

        # Part 2. Divide by Chw (actual z coordinate) to get vision sensor homogeneous coordinates
        e = 1.0/camera_homogeneous[-1]
        p_mat2 = np.array([[e, 0, 0, 0],
                           [0, e, 0, 0],
                           [0, 0, e, 0],
                           [0, 0, 0, 1]])

        camera_cartesian = np.dot(p_mat2, camera_homogeneous)

        # Check if object lies within projection frame
        epsilon = 1*10**-3

        if ((-1 - epsilon <= camera_cartesian[0] <= 1 + epsilon) and
                (-1 - epsilon <= camera_cartesian[1] <= 1 + epsilon) and
                (-1 - epsilon <= camera_cartesian[2] <= 1 + epsilon)):
            # x,y coordinates in degrees
            # --------------------------
            # (1) Normalize calculated coordinates so they have the same scale in
            #     the x & y direction (by multiply x by the aspect_ratio). The above
            #     transformations leave x within (-1, 1), we actually what x
            #     to range between -ar, ar.
            # (2) Convert to degrees. Assume visual span (180 degrees) covers the
            #     range of x. -ar to ar. There 2*ar = np.pi
            # size in degrees
            # ---------------
            # (3) np.pi * max_dimension / (2 * aspect_ratio * distance * tan(alpha_rad/2)).
            #     See Notes.
            x = np.pi / 2 * camera_cartesian[0]
            y = np.pi / (2 * ar) * camera_cartesian[1]

            # For correct projection in the plane we actually need the euclidean distance to
            # object , not only its z coordinate.
            distance = np.sqrt(x**2 + y**2 + camera_cartesian[-1]**2)
            size = (np.pi * vrep_obj.max_dimension) / \
                   (2 * ar * distance * np.tan(projection_angle / 2))

            # Get object orientation
            rot_alpha, rot_beta, rot_gamma = get_object_rotations(
                c_id,
                vrep_obj.handle,
                vis_sen_handle)

            # Add Ground Truth
            objects_in_frame.append([
                vrep_obj.name,
                x,                  # x image coordinate in Radians
                y,                  # y coordinates in Radians
                size,               # size in Radians
                rot_alpha,          # Rotation around the x-axis in Radians
                rot_beta,           # Rotation around the y-axis in Radians
                rot_gamma           # Rotation around the z-axis in Radians.
            ])

            object_handles_in_frame.append(vrep_obj.handle)

    # After identifying all objects that lie within the field of vision of the vision sensor,
    # get occlusion levels from child script.
    vis_array = get_object_visibility_levels(object_handles_in_frame, c_id)

    for idx, entry in enumerate(objects_in_frame):
        entry.append(vis_array[idx])

    return objects_in_frame


def main():

    t_step_ms = 5       # 5ms
    t_stop_ms = 5*1000  # 5 seconds
    client_id = connect_vrep(t_stop_ms, t_step_ms)

    try:

        # SETUP VREP  ---------------------------------------------------------------------------
        # Get list of objects in scene
        print("Initializing VREP simulation...")
        objects_array = []
        get_scene_objects(client_id, objects_array)
        print ("Number of objects in scene %d" % len(objects_array))
        print_objects(objects_array)

        # Get IT Cortex Robot Vision sensor parameters
        alpha_rad, aspect_ratio, z_near, z_far, vs_handle = get_vision_sensor_parameters(client_id)

        initialize_vrep_streaming_operations(client_id, objects_array, vs_handle)

        # Construct vision sensor projection matrix
        # This Projection Matrix, scales x, y, z axis to range (-1, 1) to make it easier to detect
        # whether the object falls within the projection frame of the vision sensor.
        # Ref: http://ogldev.atspace.co.uk/www/tutorial12/tutorial12.html
        a = 1.0/(aspect_ratio*np.tan(alpha_rad/2.0))
        b = 1.0/np.tan(alpha_rad/2.0)
        c = -(z_near + z_far) / (z_near - z_far)
        d = (2*z_near*z_far) / (z_near - z_far)

        p_mat = np.array([[a, 0, 0, 0],
                          [0, b, 0, 0],
                          [0, 0, c, d],
                          [0, 0, 1, 0]])

        # Generate IT Population ----------------------------------------------------------------
        print("Initializing IT Population...")

        population_size = 100
        list_of_objects = [obj.name for obj in objects_array]
        it_cortex = []

        for _ in np.arange(population_size):
            neuron = it.Neuron(list_of_objects,
                               sim_time_step_s=t_step_ms/1000.0,
                               selectivity_profile='Kurtosis',
                               position_profile='Gaussian',
                               size_profile='Lognormal',
                               dynamic_profile='Tamura',
                               )

            it_cortex.append(neuron)

        # Get Ground Truth  ---------------------------------------------------------------------
        print("Starting Data collection...")
        set_robot_velocity(client_id, 2)

        rates_vs_time_arr = np.zeros(shape=(t_stop_ms/t_step_ms, population_size))

        t_current_ms = 0
        while t_current_ms < t_stop_ms:

            # Step the simulation
            res = vrep.simxSynchronousTrigger(client_id)
            if res != vrep.simx_return_ok:
                print ("Failed to step simulation! Err %s" % res)
                break

            # raw_input("Continue with step %d ?" % t_current_ms)

            ground_truth = get_ground_truth(
                client_id,
                objects_array,
                vs_handle,
                p_mat,
                aspect_ratio,
                alpha_rad)

            if ground_truth:
                # Print the Ground Truth
                print("Time=%dms, Number of objects %d" % (t_current_ms, len(ground_truth)))
                for entry in ground_truth:
                    print ("\t %s, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f"
                           % (entry[0].ljust(30), entry[1], entry[2], entry[3],
                              entry[4], entry[5], entry[6], entry[7]))

            # Get IT cortex firing rates
            for n_idx, neuron in enumerate(it_cortex):
                rates_vs_time_arr[t_current_ms/t_step_ms, n_idx] = \
                    neuron.firing_rate(ground_truth)

            t_current_ms += t_step_ms

        # Plot firing rates ---------------------------------------------------------------------
        population_max_fire_rate = utils.population_max_firing_rate(it_cortex)

        if np.count_nonzero(rates_vs_time_arr):
            print("Plotting Results...")
            rates_vs_time_arr = np.array(rates_vs_time_arr)

            markers = ['+', '.', '*', '^', 'o', '8', 'd', 's']

            quotient, remainder = divmod(population_size, len(markers))
            n_subplots = quotient
            if 0 != remainder:
                n_subplots += 1

            fig_rates_vs_time, ax_array = plt.subplots(n_subplots, sharex=True)
            fig_rates_vs_time.subplots_adjust(hspace=0.0)

            for neuron_idx in np.arange(population_size):
                marker_idx = neuron_idx % len(markers)
                subplot_idx = neuron_idx / len(markers)

                ax_array[subplot_idx].plot(
                    np.arange(t_stop_ms, step=t_step_ms),
                    rates_vs_time_arr[:, neuron_idx],
                    marker=markers[marker_idx],
                    label='N%i' % neuron_idx)

            for ax in ax_array:
                ax.legend(fontsize='5')
                ax.set_ylim(0, population_max_fire_rate + 1)
                ax.set_yticks([])

            ax_array[-1].set_yticks(
                np.arange(0, population_max_fire_rate + 1, step=20))
            ax_array[-1].set_xlabel("Time (ms)")
            ax_array[-1].set_ylabel("Firing Rate")
            fig_rates_vs_time.suptitle("Population Firing Rates ", fontsize=16)

    finally:
        # Stop Simulation -----------------------------------------------------------------------
        print("Stopping Simulation...")
        set_robot_velocity(client_id, 0)
        time.sleep(1)
        result = vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)
        if result != vrep.simx_return_ok:
            print("Failed to stop simulation.")
        vrep.simxFinish(client_id)

    return it_cortex, rates_vs_time_arr


if __name__ == "__main__":
    plt.ion()

    population, rates_array = main()

    # Population Plots -------------------------------------------------------------------------
    # # Plot the selectivity distribution of the population
    # utils.plot_population_selectivity_distribution(population)
    #
    # # Plot Object preferences of population
    # utils.plot_population_obj_preferences(population)
