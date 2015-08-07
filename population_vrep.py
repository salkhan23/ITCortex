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
        self.max_dimension = max_dimension


def connect_vrep(sim_stop_time_s):
    """
    Establish connection to VREP simulation.

    NOTE: Add the following command to a child script in the simulation:
        simExtRemoteApiStart(19999)
    """
    vrep.simxFinish(-1)  # Close any open connections.

    c_id = vrep.simxStart(
        '127.0.0.1',
        19999,
        True,
        True,
        sim_stop_time_s*1000,   # Only closes the remote connection, does not stop simulation.
        5)                      # Data Communication Rate (ms) (packets transferred every 5ms).

    if c_id == -1:
        print ("Failed to connect to simulation!")
        sys.exit("could not connect")
    else:
        print ('Connected to remote API server')

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


def get_object_position(c_id, object_handle, reference_frame_handle):
    """
     Get position (x,y,z) coordinates of object specified by object_handle with respect to
     the reference frame of object specified by reference_frame_handle.

    :param c_id                     : connected scene id.
    :param object_handle            : vrep handle of object.
    :param reference_frame_handle   : vrep handle of objects whose reference frame to use.

    :return: (x,y,z) co-ordinates of target object.
    """
    res, position = vrep.simxGetObjectPosition(
        c_id,
        object_handle,
        reference_frame_handle,
        vrep.simx_opmode_buffer)

    if res != vrep.simx_return_ok:
        # print('Initializing position acquisition function for object handle %d' % object_handle)
        res, position = vrep.simxGetObjectPosition(
            c_id,
            object_handle,
            reference_frame_handle,
            vrep.simx_opmode_streaming)

        time.sleep(0.1)  # wait 100ms after initializing
        res, position = vrep.simxGetObjectPosition(
            c_id,
            object_handle,
            reference_frame_handle,
            vrep.simx_opmode_buffer)

    return position


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
        size)               : size of object (span of objects maximum dimension) in degree of
                              eccentricity (radians).
    """
    objects_in_frame = []

    for vrep_obj in objects:
        # Get read world coordinates of object in vision sensor reference frame
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

        # Check if object list within projection frame
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

            objects_in_frame.append([
                vrep_obj.name,
                x,                  # x image coordinate in radians
                y,                  # y coordinates in radians
                size])              # size in radians

    return objects_in_frame


def main():

    t_stop = 20  # Simulation stop time in seconds
    t_start = time.time()
    client_id = connect_vrep(t_stop)

    try:

        # SETUP VREP  ---------------------------------------------------------------------------
        # Get list of objects in scene
        objects_array = []
        get_scene_objects(client_id, objects_array)
        print ("Number of objects in scene %d" % len(objects_array))
        print_objects(objects_array)

        # Get IT Cortex Robot Vision sensor parameters
        alpha_rad, aspect_ratio, z_near, z_far, vs_handle = get_vision_sensor_parameters(client_id)

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

        # Get Ground Truth  ---------------------------------------------------------------------

        set_robot_velocity(client_id, 0.2)

        while time.time() < (t_start + t_stop):
            ground_truth = get_ground_truth(
                client_id,
                objects_array,
                vs_handle,
                p_mat,
                aspect_ratio,
                alpha_rad)

            if ground_truth:
                print("Number of objects %d" % len(ground_truth))
                for entry in ground_truth:
                    print ("%s, %0.2f, %0.2f, %0.2f"
                           % (entry[0].ljust(30), entry[1], entry[2], entry[3]))

    finally:
        # Stop Simulation -------------------------------------------------------------_---------
        print("Stopping Simulation")
        set_robot_velocity(client_id, 0)
        time.sleep(1)
        result = vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot)
        if result != vrep.simx_return_ok and \
           result != vrep.simx_return_novalue_flag:
            print("Failed to stop simulation.")
        vrep.simxFinish(-1)


if __name__ == "__main__":
    plt.ion()
    main()
