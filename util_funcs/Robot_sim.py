# import struct
import time
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from util_funcs import utils
from simulation import vrep


class Robot(object):
    def __init__(self, workspace_limits,
                 num_of_obj,
                 obj_dir,
                 is_insert_task=False,
                 num_of_holes=0,
                 is_eval=False):
        self.workspace_limits = np.asarray(workspace_limits)
        self.workspace_center = np.asarray([(workspace_limits[0][1] + workspace_limits[0][0]) / 2,
                                            (workspace_limits[1][1] + workspace_limits[1][0]) / 2,
                                            0.1])
        if not is_insert_task:
            self.obj_target_home_pos = self.workspace_center
        else:

            self.obj_target_home_pos = np.asarray([self.workspace_center[0] - 0.1,
                                                   self.workspace_center[1],
                                                   0.08])
        self.obj_target_dir = os.path.join(obj_dir, 'objects')
        self.obj_files = [f for f in listdir(self.obj_target_dir) if isfile(join(self.obj_target_dir, f)) and ('.ttm' in f)]
        self.obj_files.sort()
        self.obj_target_handles = []

        if is_insert_task:
            self.hole_file_dir = os.path.join(obj_dir, 'holes')
            self.hole_files = [f for f in listdir(self.hole_file_dir) if isfile(join(self.hole_file_dir, f)) and ('.ttm' in f)]
            self.hole_files.sort()
            self.hole_home_pos = np.asarray([self.workspace_center[0] + 0.15,
                                             self.workspace_center[1],
                                             0.08])
            self.hole_handles = []
        #  These parameters are related to the vrep scenes
        self.num_of_obj = num_of_obj
        if is_insert_task:
            self.num_of_holes = num_of_holes

        self.gripper_home_pos = np.asarray([0.4, 0.0, 0.4])  # Gripper home pos
        self.gripper_home_ori = np.asarray([np.pi, 0., 0.])  # Grasp target Ori (for gripper rotation = 0)

        self.is_insert_task = is_insert_task
        self.cam_intrinsics = None
        self.UR5_target_handle = None
        self.UR5_tip_handle = None

        self.is_eval = is_eval
        self.add_obj_count = 0

        # Connect to simulator
        vrep.simxFinish(-1)  # Just in case, close all opened connections
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.stop_sim()
            self.restart_sim()

        # Setup virtual camera in simulation
        self.setup_sim_camera()
        self.open_RG2_gripper()

        self.rotate_gripper_z(0.)

    #  >>>>>  Scene Functions
    def setup_sim_camera(self, resolution_x=1024., resolution_y=1024.):
        # Get handle to camera
        perspectiveAngle = np.deg2rad(54.70)
        self.cam_intrinsics = np.asarray([[resolution_x / (2 * np.tan(perspectiveAngle / 2)), 0, resolution_x/2],
                                          [0, resolution_y / (2 * np.tan(perspectiveAngle / 2)), resolution_y/2],
                                          [0, 0, 1]])
        _, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client,
                                                      'Vision_sensor_persp_0',
                                                      vrep.simx_opmode_blocking)
        # Get camera pose and intrinsics in simulation
        _, cam_position = vrep.simxGetObjectPosition(self.sim_client,
                                                     self.cam_handle,
                                                     -1,
                                                     vrep.simx_opmode_blocking)
        _, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client,
                                                           self.cam_handle,
                                                           -1,
                                                           vrep.simx_opmode_blocking)
        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        cam_orientation = [cam_orientation[0], cam_orientation[1], cam_orientation[2]]
        cam_rotm = np.eye(4, 4)
        cam_rotm[0:3, 0:3] = utils.euler2rotm(cam_orientation)
        self.cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representating camera pose
        self.cam_depth_scale = 1

    def restart_sim(self):
        _, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,
                                                             'UR5_target',
                                                             vrep.simx_opmode_blocking)
        _, self.UR5_tip_handle = vrep.simxGetObjectHandle(self.sim_client,
                                                          'UR5_tip',
                                                          vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,
                                   self.UR5_target_handle,
                                   -1,
                                   self.gripper_home_pos,
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client,
                                      self.UR5_target_handle,
                                      -1,
                                      self.gripper_home_ori,
                                      vrep.simx_opmode_blocking)

        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(0.3)
        _, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client,
                                                          'UR5_tip',
                                                          vrep.simx_opmode_blocking)
        _, gripper_position = vrep.simxGetObjectPosition(self.sim_client,
                                                         self.RG2_tip_handle, -1,
                                                         vrep.simx_opmode_blocking)
        while gripper_position[2] > self.gripper_home_pos[2] + 0.01:  # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client,
                                                                   self.RG2_tip_handle, -1,
                                                                   vrep.simx_opmode_blocking)
        self.open_RG2_gripper()
        self.obj_target_handles = []
        if self.is_insert_task:
            self.hole_handles = []
            self.add_hole()
        self.add_objects()
        time.sleep(0.8)

    def stop_sim(self):
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(0.1)

    def check_sim(self):
        # Check if simulation is stable by checking if gripper is within workspace
        # Need to be modify, not working now
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.15 and gripper_position[0] < self.workspace_limits[0][1] + 0.15 and gripper_position[1] > self.workspace_limits[1][0] - 0.15 and gripper_position[1] < self.workspace_limits[1][1] + 0.15 and gripper_position[2] > 0. and gripper_position[2] < 0.5

        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.stop_sim()
            self.restart_sim()
        return sim_ok

    def get_camera_data(self):
        """
        Return a tuple containing (RGB_img, Depth_img)
        """
        # Get color image from simulation
        _, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client,
                                                                 self.cam_handle,
                                                                 0,
                                                                 vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float) / 255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client,
                                                                                self.cam_handle,
                                                                                vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear
        return color_img, depth_img

    #  >>>>> Task Functions  >>>>> scene basic blocks
    def add_objects(self):
        for i in range(self.num_of_obj):
            obj_range = len(self.obj_files)
            if self.is_insert_task:
                if len(self.obj_target_handles) < 1:
                    obj_idx = 4
                else:
                    obj_idx = np.random.randint(0, 4)
            else:
                obj_idx = np.random.randint(0, obj_range)
            obj_path = os.path.join(self.obj_target_dir,
                                    self.obj_files[obj_idx])
            _, obj_handle = vrep.simxLoadModel(self.sim_client,
                                               obj_path,
                                               1,
                                               vrep.simx_opmode_blocking)
            self.obj_target_handles.append(obj_handle)

            obj_ori = self.get_single_obj_orientations(obj_handle)
            if not self.is_insert_task:
                if not self.is_eval:
                    self.place_target_obj(obj_handle, self.obj_target_home_pos,
                                          obj_ori, noise=0.3)
                else:
                    self.place_target_obj(obj_handle, self.obj_target_home_pos,
                                          obj_ori, noise=0.2)
            else:
                position = self.obj_target_home_pos.copy()
                position[0] += (np.random.random_sample() - 0.5) * 0.1
                position[1] += (np.random.random_sample() - 0.5) * 0.1
                position[2] = 0.08
                self.place_target_obj(obj_handle,
                                      position,
                                      obj_ori,
                                      noise=0.1)
            self.add_obj_count += 1

    def add_hole(self, hole_id=None):
        assert self.is_insert_task, 'Not insert task, should not add hole'
        if hole_id is None:
            for hole_id in range(self.num_of_holes):
                hole_path = os.path.join(self.hole_file_dir, self.hole_files[hole_id])
                _, hole_handle = vrep.simxLoadModel(self.sim_client,
                                                    hole_path,
                                                    1,
                                                    vrep.simx_opmode_blocking)
                self.hole_handles.append(hole_handle)
                hole_ori = self.get_single_obj_orientations(hole_handle)
                # if hole_id > 0:
                #     self.hole_home_pos[1] += 0.15
                self.place_target_obj(hole_handle,
                                      self.hole_home_pos,
                                      hole_ori,
                                      noise=-0.09,
                                      is_hole=True)
                hole_ori = self.get_single_obj_orientations(hole_handle)
                hole_ori[1] += 2 * np.pi * np.random.random_sample() - np.pi
                self.set_single_obj_orientation(hole_handle, hole_ori)
        pass

    def get_single_obj_position(self, object_handle):
        _, obj_position = vrep.simxGetObjectPosition(self.sim_client,
                                                     object_handle,
                                                     -1,
                                                     vrep.simx_opmode_blocking)
        return np.asarray(obj_position)

    def get_single_obj_orientations(self, object_handle):
        _, obj_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1,
                                                           vrep.simx_opmode_blocking)
        a = []
        for i in obj_orientation:
            temp = i if i >= 0 else i + 2 * np.pi
            a.append(temp)
        return np.asarray(a)

    def set_single_obj_position(self, object_handle, goal_pos):
        _ = vrep.simxSetObjectPosition(self.sim_client,
                                       object_handle,
                                       -1,
                                       goal_pos,
                                       vrep.simx_opmode_blocking)

    def set_single_obj_orientation(self, object_handle, goal_ori):
        _ = vrep.simxSetObjectOrientation(self.sim_client,
                                          object_handle,
                                          -1,
                                          goal_ori,
                                          vrep.simx_opmode_blocking)

    def place_target_obj(self,
                         object_handle,
                         pos,
                         ori,
                         noise=-1.,
                         is_hole=False):
        if noise > 0:
            pos_noise = noise * np.random.random_sample((3,)) - noise/2
            pos_noise[2] = 0
            if not is_hole:
                ori_noise = 2 * np.pi * np.random.random_sample((3,)) - np.pi
            else:
                ori_noise = np.zeros(3)
        else:
            pos_noise, ori_noise = np.zeros(3), np.zeros(3)
        self.set_single_obj_position(object_handle, pos + pos_noise)
        self.set_single_obj_orientation(object_handle, ori + ori_noise)
        time.sleep(0.1)

    def remove_object(self,
                      object_handle,
                      obj_handle_list):
        idx = obj_handle_list.index(object_handle)
        obj_handle_list.pop(idx)
        _ = vrep.simxRemoveModel(self.sim_client,
                                 object_handle,
                                 vrep.simx_opmode_blocking)
        time.sleep(0.6)

    def check_obj_in_workspace(self, obj_handle):
        obj_pos = self.get_single_obj_position(obj_handle)
        margin = 0.05
        if self.workspace_limits[0][0] + margin < obj_pos[0] < self.workspace_limits[0][1] - margin:
            x_in = True
        else:
            x_in = False
        if self.workspace_limits[1][0] + margin < obj_pos[1] < self.workspace_limits[1][1] - margin:
            y_in = True
        else:
            y_in = False
        if self.workspace_limits[2][0] < obj_pos[2] < self.workspace_limits[2][1]:
            z_in = True
        else:
            z_in = False
        return x_in and y_in and z_in

    def remove_out_of_bound_obj(self):
        remove_list = []
        for obj_handle in self.obj_target_handles:
            obj_in_workspace = self.check_obj_in_workspace(obj_handle)
            if not obj_in_workspace:
                remove_list.append(obj_handle)
        if not remove_list:
            return False
        else:
            for obj_handle in remove_list:
                self.remove_object(obj_handle,
                                   self.obj_target_handles)
                time.sleep(0.2)
            return True

    # >>>>> Task Functions >>>>> robot basic movements
    def close_RG2_gripper(self, default_vel=-0.1, motor_force=50):
        # RG2 gripper function
        gripper_motor_velocity = default_vel
        if self.is_insert_task:
            gripper_motor_force = 200
        else:
            gripper_motor_force = motor_force
        _, gripper_handle = vrep.simxGetObjectHandle(self.sim_client,
                                                     'RG2_openCloseJoint',
                                                     vrep.simx_opmode_blocking)
        _, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client,
                                                              gripper_handle,
                                                              vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client,
                               gripper_handle,
                               gripper_motor_force,
                               vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client,
                                        gripper_handle,
                                        gripper_motor_velocity,
                                        vrep.simx_opmode_blocking)

        gripper_fully_closed = False
        close_gripper_count = 0
        while gripper_joint_position > -0.04:  # Block until gripper is fully closed
            _, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client,
                                                                      gripper_handle,
                                                                      vrep.simx_opmode_blocking)
            close_gripper_count += 1
            if new_gripper_joint_position < gripper_joint_position:
                close_gripper_count = 0
                gripper_joint_position = new_gripper_joint_position
            if close_gripper_count > 1:
                return gripper_fully_closed
        gripper_fully_closed = True
        return gripper_fully_closed

    # def close_RG2_gripper(self, default_vel=-0.1, motor_force=700):
    #     # Baxter
    #     gripper_motor_velocity = default_vel * (-0.9)
    #     gripper_motor_force = motor_force * 0.1
    #     sim_ret, gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'BaxterGripper_closeJoint',
    #                                                        vrep.simx_opmode_blocking)
    #
    #     sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, gripper_handle,
    #                                                                 vrep.simx_opmode_blocking)
    #
    #     vrep.simxSetJointForce(self.sim_client, gripper_handle,
    #                            gripper_motor_force, vrep.simx_opmode_blocking)
    #     vrep.simxSetJointTargetVelocity(self.sim_client, gripper_handle,
    #                                     gripper_motor_velocity,
    #                                     vrep.simx_opmode_blocking)
    #
    #     gripper_fully_closed = False
    #     close_gripper_count = 0
    #     while gripper_joint_position < 0.01:  # Block until gripper is fully closed
    #         time.sleep(0.1)
    #         sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client,
    #                                                                         gripper_handle,
    #                                                                         vrep.simx_opmode_blocking)
    #
    #         close_gripper_count += 1
    #         print('Gripper joint:')
    #         print(new_gripper_joint_position)
    #
    #         if new_gripper_joint_position > gripper_joint_position:
    #             close_gripper_count = 0
    #             gripper_joint_position = new_gripper_joint_position
    #         if close_gripper_count > 1:
    #             return gripper_fully_closed
    #     gripper_fully_closed = True
    #     print('Gripper joint:')
    #     print(new_gripper_joint_position)
    #     return gripper_fully_closed

    def open_RG2_gripper(self, default_vel=0.5, motor_force=100):
        # RG2 Gripper
        gripper_motor_velocity = default_vel
        gripper_motor_force = motor_force
        sim_ret, gripper_handle = vrep.simxGetObjectHandle(self.sim_client,
                                                           'RG2_openCloseJoint',
                                                           vrep.simx_opmode_blocking)

        _, _ = vrep.simxGetJointPosition(self.sim_client,
                                         gripper_handle,
                                         vrep.simx_opmode_blocking)

        vrep.simxSetJointForce(self.sim_client,
                               gripper_handle,
                               gripper_motor_force,
                               vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client,
                                        gripper_handle,
                                        gripper_motor_velocity,
                                        vrep.simx_opmode_blocking)
    # def open_RG2_gripper(self, default_vel=0.5, motor_force=100):
    #     # Baxter
    #     gripper_motor_velocity = default_vel * (-1)
    #     gripper_motor_force = motor_force
    #     sim_ret, gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'BaxterGripper_closeJoint',
    #                                                        vrep.simx_opmode_blocking)
    #     vrep.simxSetJointForce(self.sim_client, gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
    #     vrep.simxSetJointTargetVelocity(self.sim_client, gripper_handle, gripper_motor_velocity,
    #                                     vrep.simx_opmode_blocking)

    def move_linear(self, tool_position, num_steps=10):
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0],
                                     tool_position[1] - UR5_target_position[1],
                                     tool_position[2] - UR5_target_position[2]])
        num_move_steps = num_steps
        move_step = move_direction / num_move_steps

        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client,
                                       self.UR5_target_handle,
                                       -1,
                                       (UR5_target_position[0] + move_step[0],
                                        UR5_target_position[1] + move_step[1],
                                        UR5_target_position[2] + move_step[2]),
                                       vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
        time.sleep(0.05)

    def go_up(self, tool_position, up_margin=0.2):
        self.open_RG2_gripper()
        position = np.asarray(tool_position).copy()
        location_above_grasp_target = (position[0], position[1], position[2] + up_margin)
        self.move_linear(location_above_grasp_target)

    def _rotate_gripper_z(self, target_angle, num_step=None):
        _, current_angle = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                         vrep.simx_opmode_blocking)
        # if target_angle > np.pi:
        #     target_angle -= np.pi
        # if target_angle < -np.pi:
        #     target_angle += np.pi
        if target_angle > np.pi + 0.2:
            target_angle -= 2*np.pi
        if target_angle < -np.pi - 0.2:
            target_angle += 2*np.pi
        ori_direction = np.asarray([0.,
                                    0.,
                                    target_angle - current_angle[2]])
        if num_step is not None:
            num_move_step = num_step
        else:
            diff = abs(np.rad2deg(ori_direction[2]))
            num_move_step = int((diff // 5) + 1)

        ori_move_step = ori_direction / num_move_step
        for step_iter in range(num_move_step):
            vrep.simxSetObjectOrientation(self.sim_client,
                                          self.UR5_target_handle,
                                          -1,
                                          (current_angle[0] + ori_move_step[0],
                                           current_angle[1] + ori_move_step[1],
                                           current_angle[2] + ori_move_step[2]),
                                          vrep.simx_opmode_blocking)
            _, current_angle = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                             vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client,
                                      self.UR5_target_handle,
                                      -1,
                                      (current_angle[0],
                                       current_angle[1],
                                       target_angle),
                                      vrep.simx_opmode_blocking)
        time.sleep(0.1)

    def rotate_gripper_z(self, target_angle, num_step=None):
        self._rotate_gripper_z(target_angle=target_angle,
                               num_step=num_step)
        self._rotate_gripper_z(target_angle=target_angle,
                               num_step=5)

    # >>>>> Robot complex motions (motion primitives)
    def grasp(self,
              position,
              rot_angle,
              place_motion=True,
              compensate_place=None):
        workspace_limits = self.workspace_limits
        # print('Executing: grasp at (%f, %f, %f)' % (position[0],
        #                                             position[1],
        #                                             position[2]))
        tool_rotation_angle = np.deg2rad(rot_angle)
        # print('Gripper Rot Angle: ', rot_angle)

        # Avoid collision with floor
        position = np.asarray(position).copy()
        # position[2] = max(position[2] - 0.015, workspace_limits[2][0] + 0.025)
        # Move gripper to location above grasp target
        if self.is_insert_task:
            grasp_location_margin = 0.2
        else:
            grasp_location_margin = 0.2
        location_above_grasp_target = (position[0],
                                       position[1],
                                       position[2] + grasp_location_margin)

        # Ensure gripper is open
        self.open_RG2_gripper(default_vel=0.5)

        # Approach grasp target
        self.move_linear(location_above_grasp_target)
        self.rotate_gripper_z(tool_rotation_angle)
        self.move_linear(position)
        # Close gripper to grasp target
        _ = self.close_RG2_gripper(default_vel=-0.1)

        # Move gripper to location above grasp target
        self.move_linear(location_above_grasp_target)  # For robust grasp task we don't need to lift the object
        gripper_full_closed = self.close_RG2_gripper()  # Check if obj is in hand
        grasp_successful = not gripper_full_closed
        return_obj_handle = None
        simulation_fail = False
        # if grasp_successful:
        for i in range(len(self.obj_target_handles)):
            obj_pos = self.get_single_obj_position(self.obj_target_handles[i])
            if obj_pos[2] >= 0.18:
                return_obj_handle = i
        if return_obj_handle is None:
            grasp_successful = False
        if return_obj_handle and not grasp_successful:
            simulation_fail = True
        if grasp_successful and place_motion:
            if compensate_place is None:
                position[2] = position[2] - 0.005
                obj_pos, obj_ori = self.place(position=position,
                                              obj_handle=self.obj_target_handles[return_obj_handle],
                                              rot_rad=None,
                                              location_margin=grasp_location_margin)
            else:
                place_position = position.copy()
                place_position[0] += compensate_place[1]
                place_position[1] += compensate_place[2]
                if self.is_insert_task:
                    place_position[2] = 0.17 - 0.005
                else:
                    place_position[2] = place_position[2] - 0.005
                obj_pos, obj_ori = self.place(position=place_position,
                                              obj_handle=self.obj_target_handles[return_obj_handle],
                                              rot_rad=compensate_place[0],
                                              location_margin=grasp_location_margin-0.1)
            self.move_linear(location_above_grasp_target)
        elif grasp_successful:
            # ------
            self.move_linear(location_above_grasp_target)
            # ------
        return grasp_successful, return_obj_handle, simulation_fail

    def place(self,
              position,
              obj_handle=None,
              rot_rad=None,
              location_margin=0.15):
        location_margin = location_margin
        location_above_target = (position[0], position[1], position[2] + location_margin)
        place_z = max(0.02, position[2] - 0.02)
        # print('Place Height: ', place_z)
        place_pos = (position[0],
                     position[1],
                     place_z)
        self.move_linear(location_above_target)
        if rot_rad is not None:
            print('Place Gripper Rot Angle: ', np.rad2deg(rot_rad))
            self.rotate_gripper_z(rot_rad)
        self.move_linear(position)
        self.close_RG2_gripper(default_vel=-0.01, motor_force=0)
        # # self.open_RG2_gripper(default_vel=0.01, motor_force=0)
        self.move_linear(place_pos, num_steps=3)
        # >>>>>> Hack To Solve Vrep Unstable OBJ attachment
        time.sleep(0.1)
        obj_pos, obj_ori = None, None
        if obj_handle is not None:
            obj_pos = self.get_single_obj_position(obj_handle)
            obj_ori = self.get_single_obj_orientations(obj_handle)
        time.sleep(0.1)
        # >>>>>> Hack End Here
        self.open_RG2_gripper(default_vel=0.2)
        if obj_pos is not None:
            self.set_single_obj_position(obj_handle,
                                         obj_pos)
            self.set_single_obj_orientation(obj_handle,
                                            obj_ori)
        self.move_linear(location_above_target)
        # self.move_linear(self.gripper_home_pos)
        return obj_pos, obj_ori

