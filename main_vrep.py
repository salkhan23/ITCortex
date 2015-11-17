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
import traceback

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
    def __init__(self, name, handle, max_dimension, parent_handle=-1):
        """
        Initialize a Vrep object type.

        :param name             : name
        :param handle           : vrep handles of object.
        :param max_dimension    : maximum length in any dimension.
        :param parent_handle    : vrep parent handle. If -1 not parent. (Default=-1)

        :rtype                  : Instance of vrep object.
        """
        self.name = name
        self.handle = handle
        self.non_diag_children = []      # Empty list to store all non-diagnostic child handles
        self.diag_children = []          # Empty list to store diagnostic children
        self.max_dimension = max_dimension
        self.parent = parent_handle

        # Store rotation periods and mirror symmetry along x,y,z defaults. These will be updated
        # when the vrep scene is analyzed.
        self.x_rot_period = 1
        self.x_rot_mirror_symmetric = False
        self.y_rot_period = 1
        self.y_rot_mirror_symmetric = False
        self.z_rot_period = 1
        self.z_rot_mirror_symmetric = False


def connect_vrep(sim_stop_time_ms, sim_dt_ms):
    """
    Establish connection to VREP simulation.

    NOTE: The port 19997 is for synchronous operation mode and is opened by default at startup.
    To configure it differently, change ports in remoteApiConnections.txt which is located in the
     external vrep application directory.

    :param  sim_dt_ms       :  simulation time step in milliseconds
    :param  sim_stop_time_ms:  simulation stop time in milliseconds
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
        sim_dt_ms / 1000.0,
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
    """ Return x, y, z dimensions of an object.
    :param  object_handle   : vrep handle of object to get dimensions for.
    :param  c_id            : c_id = id of the vrep session.
    """
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
    Create/Append a list of objects of interest (parent objects) in the VREP scene. Elements of the
    list are VrepObject class instances. For each element fill in the parameters as well. This
    includes Vrep handles for: the target object, its parent if any, its diagnostic and
    non-diagnostic children separately and the its maximum size (length of the side with the
    largest magnitude for the object or any of its children. Diagnostic children are sent to the
    Vrep scene to calculate their visibility levels separately.

    :param c_id     : Connected scene id.
    :param objects  : Empty list to which found Vrep objects (class) are appended to.
    """

    print("Analyzing VREP Scene...")
    # Ignore all object with default, it_cortex, proxy and floor in name
    objects_to_ignore = ['default', 'floor', 'it_cortex', 'proxy']

    res, handles, i_data, f_data, s_data = vrep.simxGetObjectGroupData(
        c_id,
        vrep.sim_appobj_object_type,
        0,  # Retrieves the object names in s_data
        vrep.simx_opmode_oneshot_wait)

    if res != vrep.simx_return_ok:
        raise Exception('get_scene_objects: Failed to get object names. Error Code %d' % res)

    # # Print all objects and their handles
    # print("All objects in VREP scene:")
    # longest_name = max([len(name) for name in s_data])
    #
    # for count in np.arange(len(handles)):
    #     print("Obj: %s, handle: %d" % (s_data[count].ljust(longest_name), handles[count]))

    # Build the list of all vrep parent objects and fill in parameters
    children = []  # list of non-diagnostic child handles.

    for count in np.arange(len(handles)):

        if not any([word in s_data[count].lower() for word in objects_to_ignore]):

            res, parent_handle = vrep.simxGetObjectParent(
                c_id,
                handles[count],
                vrep.simx_opmode_oneshot_wait)

            if res != vrep.simx_return_ok:
                raise Exception("get_scene_objects: "
                                "Failed to get %s parent handle. Error %d" % (s_data[count], res))

            if -1 == parent_handle:
                size = get_object_dimensions(c_id, handles[count])
                obj = VrepObject(s_data[count], handles[count], max(size), parent_handle)
                objects.append(obj)
            else:
                # child (part) of a parent object
                children.append((handles[count], s_data[count], parent_handle))

    # Add handles of all children to all their parent.
    for c_handle, name, p_handle in children:

        top_parent_handle = p_handle
        while top_parent_handle != -1:

            p_handle = top_parent_handle
            res, top_parent_handle = vrep.simxGetObjectParent(
                c_id,
                p_handle,
                vrep.simx_opmode_oneshot_wait)

            if res != vrep.simx_return_ok:
                raise Exception("get_scene_objects: Failed to retrieve parent "
                                "of child %s. Err %d" % (c_handle, res))

        for obj in objects:
            if obj.handle == p_handle:

                max_size = max(get_object_dimensions(c_id, c_handle))
                if max_size > obj.max_dimension:
                    obj.max_dimension = max_size

                if 'diagnostic' in name.lower():
                    obj.diag_children.append(c_handle)
                else:
                    obj.non_diag_children.append(c_handle)


def print_objects(objects):
    """ Print all objects stored in objects.

    :param objects: list of VrepObjects.
    """
    longest_name = max([len(obj.name) for obj in objects])

    for obj in objects:
        print("\t%s: handle=%s, size=%0.1f, non-diag children=%s, diag children=%s"
              % (obj.name.ljust(longest_name),
                 obj.handle,
                 obj.max_dimension,
                 ",".join(str(x) for x in obj.non_diag_children),
                 ",".join(str(x) for x in obj.diag_children)))


def set_robot_velocity(c_id, target_velocity):
    """
    The model expects to find the IT Cortex Robot. Essentially this is the Pioneer 3dx Robot
    with an extra vision sensor it_cortex_vision_sensor, whose reference frame is used to
    generate the ground truth.

    :param  c_id            :  vrep session ID
    :param  target_velocity :  target velocity to mode the it cortex robot with.
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

        _ = vrep.simxWriteStringStream(
            c_id,
            "getOcclusionForHandles",
            (ctypes.c_ubyte * len('-1')).from_buffer_copy('-1'),
            vrep.simx_opmode_oneshot)

        _ = vrep.simxReadStringStream(
            c_id,
            "rotationData",
            vrep.simx_opmode_streaming)

        # wait some time to allow VREP to setup streaming services.
        time.sleep(0.1)


def get_object_visibility_levels(objects_list, c_id):
    """
    Inform the vision sensor child script which  object handles to calculate occlusion levels
    for. Retrieve occlusion levels for all object handles in obj_handles list.

    :param objects_list: List of Vrep objects to calculate occlusion levels for.
    :param c_id: connected scene id.

    :rtype : List of (non-diagnostic, diagnostic) visibility levels for each specified object
    """
    visibility_levels = np.zeros(shape=(len(objects_list), 2))
    # For diagnostic visibility, -1 = no data as no parts are labeled diagnostic
    visibility_levels[:, 1] = -1

    if objects_list:

        # Inform vision_sensor child script which objects to calculate occlusion for
        # - send parent handles & their children.
        handles_to_send = []

        for obj in objects_list:

            handles_to_send.append(obj.handle)
            handles_to_send.extend(obj.non_diag_children)
            handles_to_send.extend(obj.diag_children)
            handles_to_send.append(-1)  # separator

            # Also sent diag children as a separate object of interest
            if obj.diag_children:
                handles_to_send.extend(obj.diag_children)
                handles_to_send.append(-1)  # separator

        # print ("Sending:", handles_to_send)

        obj_handles_string = vrep.simxPackInts(handles_to_send)
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

        if res != vrep.simx_return_ok:
            warnings.warn("Failed to get occlusion data, Error %d" % res)
        else:

            occlusion_data = vrep.simxUnpackFloats(occlusion_data)
            if not occlusion_data:
                raw_input("Empty occlusion data")

            # print("Received:", occlusion_data)

            # The occlusion data sent down is actually for the previous time step, to the child
            # script we specify which objects we are interested in and it returns values from its
            # previous request. We therefore need to match the object handle identities.

            # For each requested handle, the child script sends down the value of the requested
            # handle followed by its visibility level and finally the number of pixels that are
            # visible.

            # Objects of interest sent down from VRep can either be a parent object or diagnostic
            #  part of an object.
            retrieved_data = np.reshape(occlusion_data, (len(occlusion_data) / 3, 3))

            for data_idx in np.arange(retrieved_data.shape[0]):
                identity = np.int(retrieved_data[data_idx, 0])

                for obj_list_idx, obj in enumerate(objects_list):

                    # If parent object, store combined visibility as nondiagnostic visibility.
                    if identity == obj.handle:
                        # Only adjust the visibility level if it hasn't been updated
                        if not visibility_levels[obj_list_idx][0]:
                            visibility_levels[obj_list_idx][0] = retrieved_data[data_idx, 1]
                        break

                    # if diagnostic object part, add diagnostic visibility. But also adjust
                    # nondiagnostic visibility. The returned nondiagnostic visibility is
                    # total visibility, we here convert to visibility of nondiagnostic parts only.
                    elif identity in obj.diag_children:

                        # Find the parent data index:
                        for count in np.arange(retrieved_data.shape[0]):
                            if obj.handle == np.int(retrieved_data[count, 0]):
                                parent_data_idx = count

                        # find the total and visible pixel counts for parent and diagnostic
                        parent_visible_pixels = retrieved_data[parent_data_idx, 2]
                        parent_total_pixels = parent_visible_pixels / \
                            retrieved_data[parent_data_idx, 1]  # visibility

                        diagnostic_visible_pixels = retrieved_data[data_idx , 2]
                        diagnostic_total_pixels = diagnostic_visible_pixels / \
                            retrieved_data[data_idx, 1]  # visibility

                        nondiagnostic_visible_pixels = parent_visible_pixels - \
                            diagnostic_visible_pixels

                        nondiagnostic_total_pixels = parent_total_pixels - \
                            diagnostic_total_pixels

                        # Store the correct visibilities
                        visibility_levels[obj_list_idx][0] = nondiagnostic_visible_pixels / \
                            nondiagnostic_total_pixels

                        visibility_levels[obj_list_idx][1] = retrieved_data[data_idx, 1]

    return visibility_levels


def set_object_handles_for_rotation_symmetries(objects_list, c_id):
    """

    :param objects_list: List of Vrep objects to get rotation symmetries for
    :param c_id: connected scene id.

    :return:
    """
    if objects_list:

        # Send parent handles to it cortex robot child script that retrieves rotation symmetries
        handles_to_send = []

        for obj in objects_list:
            handles_to_send.append(obj.handle)
        # print ("Sending:", handles_to_send)

        obj_handles_string = vrep.simxPackInts(handles_to_send)
        raw_bytes = (ctypes.c_ubyte * len(obj_handles_string)).from_buffer_copy(obj_handles_string)

        res = vrep.simxSetStringSignal(
            c_id,
            "getRotationSymmetryForHandles",
            raw_bytes,
            vrep.simx_opmode_oneshot)

        if res != vrep.simx_return_ok and \
           res != vrep.simx_return_novalue_flag:
            warnings.warn("Failed to send object handles for rotation symmetries. Err. %d" % res)


def get_rotation_symmetries(c_id):

    res, rotation_data = vrep.simxReadStringStream(
        c_id,
        "rotationData",
        vrep.simx_opmode_buffer)

    if vrep.simx_return_ok != res and \
       vrep.simx_return_novalue_flag != res:
            warnings.warn("Failed to get rotation symmetries, Error %d" % res)
    else:
        rotation_data = vrep.simxUnpackInts(rotation_data)

        if not rotation_data:
            raw_input("Empty rotation data data")
        else:
            print("Received:", rotation_data)
            print("Length Received %d" % len(rotation_data))

        # retrieved_data = np.reshape(occlusion_data, (len(occlusion_data) / 3, 3))


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
        obj_name,               : vrep name of object.
        x,                      : object vision frame x coordinate in degree of eccentricity
                                  (radians).
        y,                      : object vision frame y coordinate in degree of eccentricity
                                  (radians).
        size,                   : size of object (span of objects maximum dimension) in degree of
                                  eccentricity (radians).
        rot_x,                  : rotations about the x-axis  in degree of eccentricity (radians).
        rot_x_period            : rotations symmetry period around x-axis
        rot_x_mirror_symmetric  : whether the object is mirror symmetric about x-axis.
        rot_y,                  : rotations about the y-axis  in degree of eccentricity (radians).
        rot_y_period            : rotations symmetry period around y-axis
        rot_y_mirror_symmetric  : whether the object is mirror symmetric about y-axis.
        rot_z,                  : rotations about the z-axis  in degree of eccentricity (radians).
        rot_z_period            : rotations symmetry period around z-axis
        rot_z_mirror_symmetric  : whether the object is mirror symmetric about z-axis.
        vis_non_diag            : Visibility percentage of non-diagnostic parts of the object.
                                  Range (0, 1)
        vis_diag                : Visibility percentage of diagnostic parts of the object.
                                  Range (0, 1)
    """
    ground_truth_list = []
    objects_in_frame = []   # list of all Vrep Objects found in the frame

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
        e = 1.0 / camera_homogeneous[-1]
        p_mat2 = np.array([[e, 0, 0, 0],
                           [0, e, 0, 0],
                           [0, 0, e, 0],
                           [0, 0, 0, 1]])

        camera_cartesian = np.dot(p_mat2, camera_homogeneous)

        # Check if object lies within projection frame
        # The position is for the center of the object.
        epsilon = 1 * 10**-2.75

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
            ground_truth_list.append([
                vrep_obj.name,
                x,                                  # X object center image coordinate in Radians.
                y,                                  # Y object center image coordinate in Radians.
                size,                               # Size in Radians.
                rot_alpha,                          # Rotation around the x-axis in Radians.
                vrep_obj.x_rot_period,              # X rotation symmetry period.
                vrep_obj.x_rot_mirror_symmetric,    # Are x_axis rotations mirror symmetric?
                rot_beta,                           # Rotation around the y-axis in Radian.s
                vrep_obj.y_rot_period,              # Y rotation symmetry period.
                vrep_obj.y_rot_mirror_symmetric,    # Are y_axis rotations mirror symmetric?
                rot_gamma,                          # Rotation around the z-axis in Radians.
                vrep_obj.z_rot_period,              # Z rotation symmetry period.
                vrep_obj.z_rot_mirror_symmetric,    # Are z axis rotations mirror symmetric?
            ])

            objects_in_frame.append(vrep_obj)

    # After identifying all objects that lie within the field of vision of the vision sensor,
    # get occlusion levels from child script.
    vis_array = get_object_visibility_levels(objects_in_frame, c_id)

    for idx, entry in enumerate(ground_truth_list):
        entry.extend(vis_array[idx])               # Add nondiagnostic and diagnostic visibilities.

    return ground_truth_list


def main():

    t_step_ms = 5       # 5ms
    t_stop_ms = 5 * 1000  # 5 seconds
    client_id = connect_vrep(t_stop_ms, t_step_ms)

    population_size = 100
    it_cortex = []
    rates_vs_time_arr = np.zeros(shape=(t_stop_ms / t_step_ms, population_size))

    # noinspection PyBroadException
    try:

        # SETUP VREP  ---------------------------------------------------------------------------
        # Get list of objects in scene
        print("Initializing VREP simulation...")
        objects_array = []
        get_scene_objects(client_id, objects_array)

        # Pass the handles of all parent objects, to get rotation symmetries for
        set_object_handles_for_rotation_symmetries(objects_array, client_id)

        print ("%d objects of in scene." % len(objects_array))
        print_objects(objects_array)

        # Get IT Cortex Robot Vision sensor parameters
        alpha_rad, aspect_ratio, z_near, z_far, vs_handle = get_vision_sensor_parameters(client_id)

        initialize_vrep_streaming_operations(client_id, objects_array, vs_handle)

        # Construct vision sensor projection matrix
        # This Projection Matrix, scales x, y, z axis to range (-1, 1) to make it easier to detect
        # whether the object falls within the projection frame of the vision sensor.
        # Ref: http://ogldev.atspace.co.uk/www/tutorial12/tutorial12.html
        a = 1.0 / (aspect_ratio * np.tan(alpha_rad / 2.0))
        b = 1.0 / np.tan(alpha_rad / 2.0)
        c = - (z_near + z_far) / (z_near - z_far)
        d = (2 * z_near * z_far) / (z_near - z_far)

        p_mat = np.array([[a, 0, 0, 0],
                          [0, b, 0, 0],
                          [0, 0, c, d],
                          [0, 0, 1, 0]])

        # Generate IT Population ----------------------------------------------------------------
        print("Initializing IT Population...")

        # Initialize the population with parent objects only
        list_of_objects = [obj.name for obj in objects_array
                           if 'diagnostic' not in obj.name.lower()]

        for _ in np.arange(population_size):
            neuron = it.Neuron(list_of_objects,
                               sim_time_step_s=t_step_ms / 1000.0,
                               selectivity_profile='Kurtosis',
                               position_profile='Gaussian',
                               size_profile='Lognormal',
                               rotation_profile='Gaussian',
                               dynamic_profile='Tamura',
                               occlusion_profile='TwoInputSigmoid'
                               )

            it_cortex.append(neuron)

        # Get Ground Truth  ---------------------------------------------------------------------
        print("Starting Data collection...")
        set_robot_velocity(client_id, 2)

        rates_vs_time_arr = np.zeros(shape=(t_stop_ms / t_step_ms, population_size))

        t_current_ms = 0
        while t_current_ms < t_stop_ms:

            # Step the simulation
            res = vrep.simxSynchronousTrigger(client_id)
            if res != vrep.simx_return_ok:
                print ("Failed to step simulation! Err %s" % res)
                break

            # The occlusion calculate script take some time to run and send the results up to the
            # streaming buffer. Add some delay to allow the calculated occlusion data to be written
            # into the streaming buffer so it can be correctly picked up by the python side of the
            # code. Without this additional time delay, the second time step also results in an
            # empty occlusion data stream retrieved. In this case, received occlusion data is from
            # current time step - 2, instead of -1. The first empty occlusion data string is
            # because of VREPs communication mechanism for streaming data. Good system but just
            # need to be aware of 1 step delay it causes.
            time.sleep(1.25)

            if t_current_ms == 0:
                # Because object handles need to be sent to the child script and the fact that
                # the api to get custom data is not supported over python, we need to setup a
                # signal communication mechanism to get this information from the script. This
                # causes a delay of 1 time step of when we send up the object handles and when
                # data is returned.
                get_rotation_symmetries(client_id)

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
                    print ("\t %s, %0.2f, %0.2f, %0.2f, %0.2f, %d, %s, %0.2f, %d, %s, "
                           "%0.2f, %d, %s, %0.2f, %0.2f"
                           % (entry[0].ljust(30), entry[1], entry[2], entry[3],
                              entry[4], entry[5], entry[6], entry[7], entry[8],
                              entry[9], entry[10], entry[11], entry[12], entry[13],
                              entry[14]))

            # Get IT cortex firing rates
            for n_idx, neuron in enumerate(it_cortex):
                rates_vs_time_arr[t_current_ms / t_step_ms, n_idx] = \
                    neuron.firing_rate(ground_truth)

            t_current_ms += t_step_ms

        # Plot firing rates ---------------------------------------------------------------------
        population_max_fire_rate = utils.population_max_firing_rate(it_cortex)
        font_size = 34

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
                np.arange(0, population_max_fire_rate + 1, step=10))
            ax_array[-1].set_xlabel("Time (ms)", fontsize=font_size)
            ax_array[-1].set_ylabel("Firing Rate", fontsize=font_size)

            ax_array[-1].tick_params(axis='x', labelsize=font_size)
            ax_array[-1].tick_params(axis='y', labelsize=font_size - 7)

            fig_rates_vs_time.suptitle("Population (N=%d) Firing Rates " % len(it_cortex),
                                       fontsize=font_size + 10)

    except Exception:
        traceback.print_exc()

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
