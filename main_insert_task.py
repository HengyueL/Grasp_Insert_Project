#!/usr/bin/env python
# This file is builds the simulation environment for the insertion subtask
import time, os
import torch
import argparse
import numpy as np
import cv2
from util_funcs.Robot_sim import Robot
from util_funcs.grasp_trainer import Trainer
from util_funcs.insert_trainer import SAC
from util_funcs.logger import Logger
from util_funcs import utils
from util_funcs import utils_gp as ugp
from scipy import ndimage
import matplotlib.pyplot as plt
from network_models.insert_model import ReplayMemory


def insert_check_terminate(gripper_position,
                           x_low, x_high,
                           y_low, y_high,
                           z_low):
    """
    Check if the simulation env should stop.

    (x_low, x_high, y_low, y_high, z_low) are threshold values for the gripper tip point.

    if gripper_position lies outside the space defined by the threshold values, the simulation shold terminate.
    """
    x_in_range = x_low < gripper_position[0] < x_high
    y_in_range = y_low < gripper_position[1] < y_high
    z_in_range = gripper_position[2] > z_low
    return not (x_in_range and y_in_range and z_in_range)


def main(args,
         logger_dir=None,
         sac_save_model_dir=None,
         load_model_dir=None,
         is_eval=False,
         is_insert_task=False,
         hard_place=False,
         sac_model_load=None):
    """
    logger_dir --- customized path to save the log files
    sac_save_model_dir --- path to save SAC model
    load_model_dir --- grasping model path
    is_eval --- True if evaluation (no SAC network updates)
    is_insert_task --- always set to True (influence Robot class behavior)
    hard_place --- True if DO NOT use SAC insertion policy (straight down insertion instead)
    sac_model_load --- path to load pretrained SAC model
    """
    if not is_insert_task:
        num_of_obj = 1
        num_of_holes = 1
    else:
        num_of_obj = 1
        num_of_holes = 1

    num_of_rotations = 16
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    random_seed = np.random.seed(args.random_seed)

    # --------------- Setup options ---------------
    heightmap_size = 500    # Workspace heightmap size
    grasp_patch_size = 160  # Img patch size that crops from the heightmap and input to the grasp net
    match_patch_size = grasp_patch_size    # Patch size to do peg-hole matching

    grasp_patch_size_out = 20   # Grasp Network Output dimension
    workspace_limits = np.asarray([[0.2, 0.7],
                                   [-0.25, 0.25],
                                   [0.0002, 0.2]])  # Vrep Workspace Range

    # Image and voxel grid parameters
    heightmap_resolution = (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_size  # in meters

    # ------ Pre-loading and logging options ------
    continue_logging = args.continue_logging if args.continue_logging else False  # If access to pretrained log files
    logging_directory = os.path.abspath('logs')
    save_visualizations = args.save_visualizations
    obj_dir = os.path.abspath(args.obj_dir)     # Path to load Vrep models
    load_model_dir = load_model_dir
    # Initialize pick-and-place system (camera and robot)
    robot = Robot(workspace_limits,
                  num_of_obj=num_of_obj,
                  obj_dir=obj_dir,
                  is_insert_task=is_insert_task,
                  num_of_holes=num_of_holes,
                  is_eval=is_eval)
    # Initialize data logger
    logger = Logger(continue_logging,
                    logging_directory,
                    logger_dir=logger_dir)
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)  # Save heightmap parameters
    # Initialize trainer (Grasp Network)
    trainer = Trainer(device=device,
                      num_of_rotation=num_of_rotations,
                      load_model_dir=load_model_dir,
                      is_eval=True)    # Eval is True as the grasp network is assumed to be pretrained and do not need updates

    # Initializing SAC insertion agent
    sac_state_num = 4 + 4
    sac_action_num = 4
    sac_memory_size = int(5e4)
    sac_batch_size = 2
    sac_critic1_loss = []
    sac_critic2_loss = []
    sac_policy_loss = []
    sac_alpha_loss = []
    reward_log = []    # 0 --- not successful insertion/ none terminal states; 1 --- successful insertion.
    sac_agent = SAC(num_inputs=sac_state_num,
                    num_actions=sac_action_num,
                    entropy_tuning=True,
                    target_update_interval=20)
    # Seperate Experience Replay Buffer
    # Intuition is that reward is sparse and reward 1 is only derived for terminal states.
    # I have hoped this could improve Replay efficiency, not sure if it will work better than standard update
    process_memory = ReplayMemory(capacity=sac_memory_size,
                                  seed=random_seed)
    terminal_memory = ReplayMemory(capacity=sac_memory_size,
                                   seed=random_seed)

    # if insertion policy is SAC and would like to load pretrained models:
    if sac_model_load is not None and not hard_place:
        actor_path = os.path.join(sac_save_model_dir,
                                  'actor_insert_%d' % sac_model_load)
        critic_path = os.path.join(sac_save_model_dir,
                                   'critic_insert_%d' % sac_model_load)
        sac_agent.load_model(actor_path,
                             critic_path)

    # These variables may not be useful anymore (initially for Python 2 multi-thread control)
    nonlocal_variables = {'grasp_position': None,
                          'grasp_success': False,
                          'rot_angle': 0.,
                          'rot_idx': 0}
    # Initialize variables for heuristic bootstrapping and exploration probability
    restart_flag = False

    # Start main training/testing loop
    while True:
        iteration_time_0 = time.time()
        grasp_obj_handle = None
        print(' --- --- --- --- --- Iteration: %d --- --- --- --- --- ' % trainer.iteration)

        # Check Simulation env because Vrep goes crazy easily
        sim_ok = robot.check_sim()
        if restart_flag:
            robot.stop_sim()
            robot.restart_sim()
            restart_flag = False
            time.sleep(1)

        # Get latest RGB-D image
        color_heightmap_1, depth_heightmap_1 = ugp.get_heightmap(robot=robot,
                                                                 heightmap_resolution=heightmap_resolution,
                                                                 workspace_limits=workspace_limits)

        # Get obj_target and hole_target position to crop img patches
        # Future Identification Method may substitute the code below:
        obj_handle = robot.obj_target_handles[0]
        obj_target_pos = robot.get_single_obj_position(obj_handle)
        hole_handle = robot.hole_handles[0]
        hole_pos = robot.get_single_obj_position(hole_handle)

        # Calculate the target object center in the workspace image
        obj_target_y = int((obj_target_pos[0] - workspace_limits[0][0]) / heightmap_resolution)
        obj_target_x = int((obj_target_pos[1] - workspace_limits[1][0]) / heightmap_resolution)
        # Crop the target object patch for grasping
        grasp_patch, grasp_color_patch, grasp_patch_row_low, grasp_patch_col_low = ugp.crop_workspace_heightmap(
                                                                                                center_x=obj_target_x,
                                                                                                center_y=obj_target_y,
                                                                                                patch_size=grasp_patch_size,
                                                                                                heightmap_size=heightmap_size,
                                                                                                depth_heightmap=depth_heightmap_1,
                                                                                                color_heightmap=color_heightmap_1)
        # Calculate the hole center for future cropping
        hole_y = int((hole_pos[0] - workspace_limits[0][0]) / heightmap_resolution)
        hole_x = int((hole_pos[1] - workspace_limits[1][0]) / heightmap_resolution)

        # -------- Calc Grasp Pose -----------
        # To ensure no loss input to the grasp network (E.g. after 45^\circ rotation)
        # Zero pad the input patch to sqrt(2)*grasp_patch_size
        pad_width = int((np.sqrt(2) - 1) * grasp_patch_size / 2) + 1
        grasp_patch = np.pad(grasp_patch,
                             ((pad_width, pad_width),
                              (pad_width, pad_width)),
                             'constant',
                             constant_values=0)
        grasp_patch_size_in = grasp_patch.shape[0]
        grasp_patch.shape = (grasp_patch_size_in,
                             grasp_patch_size_in,
                             1)
        grasp_predictions = trainer.make_predictions(grasp_patch,
                                                     output_size=grasp_patch_size_out)  # img_space_prediction (rot, robot_y, robot_x)

        # As Grasping network is pretrained, only applied Greedy policy
        grasp_explore_prob = -1
        grasp_random_rot_prob = -1
        unravel_idx = ugp.grasp_action(trainer,
                                       grasp_predictions,
                                       grasp_explore_prob,
                                       grasp_random_rot_prob)

        # ---------- Save Grasp Pose Info ---------------
        nonlocal_variables['rot_idx'] = unravel_idx[0]
        # Regulate grasp angle
        angle = unravel_idx[0] * 360.0 / num_of_rotations
        if angle > 180.:
            angle -= 360.
        nonlocal_variables['rot_angle'] = angle
        nonlocal_variables['grasp_position'] = unravel_idx[1:]  # img space pos

        # --------- Calculate the position in robot frame
        robot_space_x, robot_space_y = nonlocal_variables['grasp_position'][1], nonlocal_variables['grasp_position'][0]
        io_ratio = float(grasp_patch_size) / grasp_patch_size_out
        robot_frame_x = grasp_patch_col_low * heightmap_resolution + (robot_space_x + 0.5) * io_ratio * heightmap_resolution + workspace_limits[0][0]
        robot_frame_y = grasp_patch_row_low * heightmap_resolution + (robot_space_y + 0.5) * io_ratio * heightmap_resolution + workspace_limits[1][0]
        z_position = depth_heightmap_1[int(grasp_patch_row_low + (robot_space_y + 0.5) * io_ratio),
                                       int(grasp_patch_col_low + (robot_space_x + 0.5) * io_ratio)]
        z_position = max(z_position - 0.015, workspace_limits[2][0] + 0.02)
        # ---------- Synthesis the Grasp position
        robot_act_pos = (robot_frame_x,
                         robot_frame_y,
                         z_position)

        # ---------- Visualize executed primitive, and affordances ----------------
        if save_visualizations:
            grasp_vis = grasp_predictions
            grasp_vis = ugp.get_grasp_vis(grasp_vis,
                                          grasp_color_patch,
                                          nonlocal_variables['rot_idx'],
                                          nonlocal_variables['grasp_position'],
                                          num_rotations=num_of_rotations)
            # logger.save_visualizations(trainer.iteration, grasp_vis, 'grasp')
            cv2.imwrite('Insert_grasp_vis.png', grasp_vis)

        # ---- Execute 1st Grasp -----
        nonlocal_variables['grasp_success'], grasp_obj_handle_idx, simulation_fail = robot.grasp(robot_act_pos,
                                                                                                 nonlocal_variables['rot_angle'],
                                                                                                 place_motion=False,
                                                                                                 compensate_place=None)
        print('Grasp Successful: %r' % (nonlocal_variables['grasp_success']))
        time.sleep(0.3)
        # ---- First Grasp Done -----

        if grasp_obj_handle_idx is not None:    # If First Grasp Is successful, get the grasped object handle
            grasp_obj_handle = robot.obj_target_handles[grasp_obj_handle_idx]
        logger.save_npy(np.asarray(nonlocal_variables['grasp_success']), trainer.iteration,
                        logger.grasp_success_directory, 'first-grasp-success')

        if nonlocal_variables['grasp_success'] and grasp_obj_handle is not None:
            # Derive a img patch centered at the target object position
            _c, _d = ugp.get_heightmap(robot=robot,
                                       heightmap_resolution=heightmap_resolution,
                                       workspace_limits=workspace_limits)
            in_the_air_patch, _, _, _ = ugp.crop_workspace_heightmap(
                center_x=obj_target_x,
                center_y=obj_target_y,
                patch_size=match_patch_size,
                heightmap_size=heightmap_size,
                depth_heightmap=_d,
                color_heightmap=_c)
            # ---- Place the grasped target object back to the workspace
            # ---- With the gripper pose same to the grasping pose
            target_pos, target_ori = robot.place(robot_act_pos,
                                                 obj_handle=grasp_obj_handle)
            robot.set_single_obj_position(grasp_obj_handle,
                                          target_pos)
            robot.set_single_obj_orientation(grasp_obj_handle,
                                             target_ori)
            time.sleep(0.5)

            # After Placing, the target object pose may changed (compared to the pose before grasp)
            # However, we use a second grasp with the same gripper pose as the first grasp again
            # The displacement will be much smaller
            color_heightmap_2, depth_heightmap_2 = ugp.get_heightmap(robot,
                                                                     heightmap_resolution,
                                                                     workspace_limits)
            # >>>>>>>>>>>>> Patch Matching to align target object and hole >>>>>>>>
            # First Crop the image patch of the target object again
            post_grasp_patch, post_grasp_color_patch, grasp_patch_row_low, grasp_patch_col_low = ugp.crop_workspace_heightmap(
                center_x=obj_target_x,
                center_y=obj_target_y,
                patch_size=match_patch_size,
                heightmap_size=heightmap_size,
                depth_heightmap=depth_heightmap_2,
                color_heightmap=None)
            # Compute the image difference to get a clean target object patch
            post_grasp_obj_patch = post_grasp_patch - in_the_air_patch
            post_grasp_obj_patch = ugp.create_binary_img(post_grasp_obj_patch,
                                                         padding=None)

            # Find hole position and hole patch
            #  Only for one hole
            hole_patch, _, hole_patch_row_low, hole_patch_col_low = ugp.crop_workspace_heightmap(
                center_x=hole_x,
                center_y=hole_y,
                patch_size=grasp_patch_size,
                heightmap_size=heightmap_size,
                depth_heightmap=depth_heightmap_2,
                color_heightmap=color_heightmap_2)
            #  Crop Hole Patch
            hole_patch = ugp.derive_hole_patch(hole_patch,
                                               match_patch_size,
                                               wall_threshold=0.1,
                                               floor_threshould=0.002)

            #  ===== Now align the hole and target object by the cropped patches
            # Find target object and hole centers
            post_x, post_y, find_center_1 = ugp.find_single_blob_center(hole_patch)
            prev_x, prev_y, find_center_2 = ugp.find_single_blob_center(post_grasp_obj_patch)
            # Calculate some positions that will help later
            obj_center_x_workspace = prev_x + grasp_patch_col_low
            obj_center_y_workspace = prev_y + grasp_patch_row_low
            hole_center_x_workspace = post_x + hole_patch_col_low - match_patch_size/2
            hole_center_y_workspace = post_y + hole_patch_row_low - match_patch_size/2
            grasp_center_x = (robot_frame_x - workspace_limits[0][0]) / heightmap_resolution
            grasp_center_y = (robot_frame_y - workspace_limits[1][0]) / heightmap_resolution

            if find_center_1 and find_center_2:
                # ---- Flip images and compute the geometrical offset ---
                # ---- Post Grasp patch ===> Background/Hole ----
                # ---- Pre Grasp patch  ===> Object Target ----
                post_grasp_pos_patch = cv2.flip(hole_patch, 0)
                hole_vis = post_grasp_pos_patch.copy()
                pre_grasp_pos_patch = cv2.flip(post_grasp_obj_patch, 0)
                flipped_x, flipped_y, _ = ugp.find_single_blob_center(pre_grasp_pos_patch)
                obj_patch_width = min(flipped_x, flipped_y,
                                      match_patch_size - flipped_x,
                                      match_patch_size - flipped_y)
                pre_grasp_pos_patch = pre_grasp_pos_patch[(flipped_y-obj_patch_width):(flipped_y + obj_patch_width),
                                                          (flipped_x-obj_patch_width):(flipped_x + obj_patch_width)]
                # This following Function aligns the target object and hole (in image space)
                can_insert, insert_score, opt_row_trans, opt_col_trans, opt_rot, back_center_x, back_center_y = ugp.match_obj_hole(
                                                                                    post_grasp_pos_patch,
                                                                                    pre_grasp_pos_patch)

                if can_insert:
                    # Visualize Matching Result:
                    obj_vis = ndimage.rotate(pre_grasp_pos_patch, opt_rot, reshape=False)
                    obj_vis.dtype = np.int8
                    obj_center_x = int(pre_grasp_pos_patch.shape[1] / 2)
                    obj_center_y = int(pre_grasp_pos_patch.shape[0] / 2)
                    width = pre_grasp_pos_patch.shape[0]
                    induce_x, induce_y = int(back_center_x - obj_center_x + opt_col_trans), int(back_center_y - obj_center_y + opt_row_trans)
                    combine_img_vis = hole_vis
                    combine_img_vis.dtype = np.int8
                    combine_img_vis[induce_y:induce_y + width, induce_x:induce_x + width] -= obj_vis
                    fig_0 = plt.figure(0)
                    ax_1 = fig_0.add_subplot(1, 1, 1)
                    ax_1.imshow(combine_img_vis)
                    plt.savefig('Insert_matching_vis.png')
                    plt.close(fig_0)

                    # ---- Calculate Compensation in the workspace coordinate  (not in image space anymore)
                    compensate, compensate_rot = ugp.compensate_calc(grasp_point_x=grasp_center_x,
                                                                     grasp_point_y=grasp_center_y,
                                                                     obj_center_x=obj_center_x_workspace,
                                                                     obj_center_y=obj_center_y_workspace,
                                                                     hole_center_x=hole_center_x_workspace,
                                                                     hole_center_y=hole_center_y_workspace,
                                                                     opt_row_trans=opt_row_trans,
                                                                     opt_col_trans=opt_col_trans,
                                                                     opt_rot=opt_rot,
                                                                     grasp_gripper_angle=nonlocal_variables['rot_angle'],
                                                                     heightmap_resolution=heightmap_resolution)

                    print('Grasp Rotation Angle: ', nonlocal_variables['rot_angle'],
                          'Compensate Rotation Angle: ', np.rad2deg(compensate[0]))
                    # >>>> Execute the second grasp
                    second_grasp_success, _, simulation_fail = robot.grasp(robot_act_pos,
                                                                           nonlocal_variables['rot_angle'],
                                                                           place_motion=False)

                    if second_grasp_success:
                        if hard_place:  # Staight down insertion
                            # Move the gripper above the hole with the calculated compensations
                            home_position = np.asarray(robot_act_pos)
                            home_position[0] += compensate[1]
                            home_position[1] += compensate[2]
                            print('Gripper Rotation Angle: ', np.rad2deg(compensate[0]))
                            z_position = depth_heightmap_2[hole_patch_row_low:hole_patch_row_low + match_patch_size,
                                                           hole_patch_col_low:hole_patch_col_low + match_patch_size]
                            home_position[2] += np.amax(z_position)
                            robot.rotate_gripper_z(target_angle=compensate[0])
                            loc_above_home = home_position.copy()
                            loc_above_home[2] += 0.1
                            robot.move_linear(loc_above_home)
                            robot.move_linear(home_position)
                            gripper_pos = robot.get_single_obj_position(robot.UR5_tip_handle)
                            # Execute Starght Down Insertion
                            place_pos, place_ori = robot.place(gripper_pos,
                                                               obj_handle=grasp_obj_handle,
                                                               location_margin=0.)
                            time.sleep(0.8)
                            # Check if the object is inserted successfully into the hole
                            pos_z_2 = robot.get_single_obj_position(grasp_obj_handle)[2]
                            if 0.07 < pos_z_2 < 0.14:
                                print('Place Successful')
                                reward_log.append(1)
                            else:
                                print('Place Failed')
                                reward_log.append(0)
                        else:
                            # ==== SAC insertion Routine =====
                            sac_action_count, max_trial = 0, 10
                            updates = 0
                            updates_per_step = 2
                            # Move the gripper above the hole with the calculated compensations
                            home_position = np.asarray(robot_act_pos)
                            home_position[0] += compensate[1]
                            home_position[1] += compensate[2]
                            margin = 0.0
                            z_position = depth_heightmap_2[hole_patch_row_low:hole_patch_row_low + match_patch_size,
                                                           hole_patch_col_low:hole_patch_col_low + match_patch_size]
                            home_position[2] += np.amax(z_position) + margin
                            loc_above_home = home_position.copy()
                            loc_above_home[2] += 0.1
                            robot.rotate_gripper_z(target_angle=compensate[0])
                            robot.move_linear(loc_above_home)
                            robot.move_linear(home_position)
                            home_orientation = np.rad2deg(robot.get_single_obj_orientations(robot.UR5_target_handle))

                            #  ====> SAC manipulation boundaries
                            control_scaling = 1e-3  # unit in mm, deg control movement
                            compensate_rot = np.deg2rad(compensate_rot)
                            x_uncertainty = np.cos(compensate_rot)
                            y_uncertainty = np.sin(compensate_rot)
                            x_amount, y_amount = 5e-3 * abs(x_uncertainty) + 1e-3, 5e-3 * abs(y_uncertainty) + 1e-3
                            x_bound_low, x_bound_high = home_position[0] - x_amount, home_position[0] + x_amount
                            y_bound_low, y_bound_high = home_position[1] - y_amount, home_position[1] + y_amount
                            z_bound_low = home_position[2] - 0.01  # Gripper termination height
                            episode_reward = 0
                            grasp_bias_x = grasp_center_x - obj_center_x_workspace
                            grasp_bias_y = grasp_center_y - obj_center_y_workspace
                            print('grasp_bias_x & y:', grasp_bias_x, grasp_bias_y)

                            # ====> Initialize state
                            # ====> SAC state [x_trans, y_trans, z_trans, rot_angle, grasp_bias_x, grasp_bias_y, x_uncertainty, y_uncertainty]
                            gripper_pos = robot.get_single_obj_position(robot.UR5_tip_handle)
                            gripper_ori = np.rad2deg(robot.get_single_obj_orientations(robot.UR5_tip_handle))
                            ori_state = gripper_ori[2] - home_orientation[2]
                            if ori_state < -180:
                                ori_state += 360.
                            elif ori_state > 180.:
                                ori_state -= 360.
                            sac_state = np.asarray([1e3 * (gripper_pos[0] - home_position[0]),
                                                    1e3 * (gripper_pos[1] - home_position[1]),
                                                    1e3 * (home_position[2] - gripper_pos[2]),
                                                    1 * ori_state,
                                                    grasp_bias_x,
                                                    grasp_bias_y,
                                                    x_uncertainty,  # state conditioned on grasping pose
                                                    y_uncertainty])  # state conditioned on grasping pose

                            is_done = False
                            # Reduce the gripper force for more compliant insert dynamics
                            robot.close_RG2_gripper(default_vel=-0.05,
                                                    motor_force=50)
                            while not is_done:
                                sac_action_count += 1
                                print('Grasp iteration: %d, Sac action: %d' % (trainer.iteration, sac_action_count))
                                sac_action = sac_agent.select_action(sac_state,
                                                                     evaluate=is_eval)

                                if len(process_memory) > 4 * sac_batch_size and len(terminal_memory) > 4 * sac_batch_size and not is_eval:
                                    print(" >>>> Gradient Descent Thred >>>>> ")
                                    for i in range(updates_per_step):
                                        critic_1_loss, critic_2_loss, policy_loss, alpha_loss, _ = sac_agent.update_parameters(
                                            terminal_memory,
                                            sac_batch_size,
                                            updates)
                                        critic_1_loss, critic_2_loss, policy_loss, alpha_loss, _ = sac_agent.update_parameters(
                                            process_memory,
                                            sac_batch_size,
                                            updates)
                                        updates += 1
                                        sac_critic1_loss.append(critic_1_loss)
                                        sac_critic2_loss.append(critic_2_loss)
                                        sac_policy_loss.append(policy_loss)
                                        sac_alpha_loss.append(alpha_loss)
                                # Apply action to the robot (with max control step thresholding)
                                if sac_action[2] < 0:
                                    # gripper no going up
                                    sac_action[2] = 0
                                if sac_action[3] > 3:
                                    sac_action[3] = 3
                                elif sac_action[3] < -3:
                                    sac_action[3] = -3
                                target_move_position = [gripper_pos[0] + (sac_action[0] * control_scaling),
                                                        gripper_pos[1] + (sac_action[1] * control_scaling),
                                                        gripper_pos[2] - (sac_action[2] * control_scaling)]
                                target_rotation_angle = np.deg2rad(gripper_ori[2] + 1 * sac_action[3])
                                # Move according to the control action
                                robot.move_linear(target_move_position)
                                robot.rotate_gripper_z(target_rotation_angle)
                                time.sleep(0.1)
                                # === Get the next state
                                gripper_pos = robot.get_single_obj_position(robot.UR5_tip_handle)
                                gripper_ori = np.rad2deg(robot.get_single_obj_orientations(robot.UR5_tip_handle))
                                ori_state = gripper_ori[2] - home_orientation[2]
                                if ori_state < -180:
                                    ori_state += 360.
                                elif ori_state > 180.:
                                    ori_state -= 360.
                                next_state = np.asarray([1e3 * (gripper_pos[0] - home_position[0]),
                                                        1e3 * (gripper_pos[1] - home_position[1]),
                                                        1e3 * (home_position[2] - gripper_pos[2]),
                                                        1 * ori_state,
                                                        grasp_bias_x,
                                                        grasp_bias_y,
                                                        x_uncertainty,  # state conditioned on grasping pose
                                                        y_uncertainty])
                                # Check terminal state
                                is_terminate = insert_check_terminate(gripper_position=gripper_pos,
                                                                      x_low=x_bound_low, x_high=x_bound_high,
                                                                      y_low=y_bound_low, y_high=y_bound_high,
                                                                      z_low=z_bound_low)
                                object_position = robot.get_single_obj_position(grasp_obj_handle)
                                obj_out_of_range = insert_check_terminate(gripper_position=object_position,
                                                                          x_low=x_bound_low-0.02,
                                                                          x_high=x_bound_high+0.02,
                                                                          y_low=y_bound_low-0.02,
                                                                          y_high=y_bound_high+0.02,
                                                                          z_low=z_bound_low - 0.03
                                                                          )
                                is_done = is_terminate or (sac_action_count > max_trial) or obj_out_of_range
                                print('Is done?: ', is_done)
                                if is_done:
                                    print('Obj out of Range: ', obj_out_of_range)
                                    print('Gripper out of range: ', is_terminate)
                                    mask = 0.
                                    place_pos, place_ori = robot.place(gripper_pos,
                                                                       obj_handle=grasp_obj_handle,
                                                                       location_margin=0.)
                                    time.sleep(0.8)
                                    pos_z_2 = robot.get_single_obj_position(grasp_obj_handle)[2]
                                    if 0.07 < pos_z_2 < 0.14:
                                        reward = 1.
                                    else:
                                        reward = 0.
                                    # Append transition into the replay buffer
                                    terminal_memory.push(sac_state,
                                                         sac_action,
                                                         reward,
                                                         next_state,
                                                         mask)
                                else:
                                    mask = 1.
                                    reward = 0.
                                    process_memory.push(sac_state,
                                                        sac_action,
                                                        reward,
                                                        next_state,
                                                        mask)
                                print('Reward: ', reward)
                                print('Process Memory Size >>> ', len(process_memory))
                                print('Terminal Memory Size >>> ', len(terminal_memory))
                                sac_state = next_state
                            reward_log.append(reward)
                trainer.iteration += 1
            else:
                pass
        # Check the target object inside the workspace
        robot.remove_out_of_bound_obj()
        restart_flag = True
        if sac_model_load is not None:
            logger.save_npy(np.asarray(reward_log), sac_model_load,
                            logger.transitions_directory, 'sac-reward')
        else:
            logger.save_npy(np.asarray(reward_log), 2000,
                            logger.transitions_directory, 'sac-reward')
        if not is_eval and trainer.iteration % 10 == 0:
            #  Visualization of SAC training
            fig_1 = plt.figure()
            ax_1 = fig_1.add_subplot(5, 1, 1)
            ax_2 = fig_1.add_subplot(5, 1, 2)
            ax_3 = fig_1.add_subplot(5, 1, 3)
            ax_4 = fig_1.add_subplot(5, 1, 4)
            ax_5 = fig_1.add_subplot(5, 1, 5)
            ax_1.plot(sac_critic1_loss)
            ax_2.plot(sac_critic2_loss)
            ax_3.plot(sac_policy_loss)
            ax_3.title.set_text('Policy Loss')
            ax_4.plot(sac_alpha_loss)
            ax_5.plot(reward_log)
            plt.savefig('training-loss.png')
            plt.close(fig_1)
        if trainer.iteration % 25 == 0 and not is_eval:
            sac_agent.save_model('insert',
                                 trainer.iteration,
                                 sac_save_model_dir)

        if not is_eval:
            if trainer.iteration > 1000:
                break
        else:
            if trainer.iteration > 50:
                break


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=13,
                        help='random seed for simulation and neural net initialization')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,
                        help='continue logging from previous session?')
    parser.add_argument('--obj_dir', dest='obj_dir', action='store',
                        default='object_models')
    parser.add_argument('--load_model_dir', dest='load_model_dir', action='store',
                        default='logs/Training_data_1/transitions/models')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True,
                        help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()

    # ========= The following is an example of using straight_down insertion policy (hard_place = True)
    logger_dir = 'hard_insert_test'
    sac_save_model_dir = os.path.join(logger_dir, 'sac_model')
    main(args,
         logger_dir=logger_dir,
         load_model_dir='logs/grasp_training/transitions/DQN_models',
         sac_save_model_dir=sac_save_model_dir,
         is_eval=True,
         is_insert_task=True,
         hard_place=True)

    # ========= Below is an example of using SAC insertion policy
    # logger_dir = 'sac_insert_1'
    # load_idx = None
    # sac_save_model_dir = os.path.join(logger_dir, 'sac_model')
    # main(args,
    #      logger_dir=logger_dir,
    #      load_model_dir='logs/grasp_training/transitions/DQN_models',
    #      sac_save_model_dir=sac_save_model_dir,
    #      is_eval=False,
    #      is_insert_task=True,
    #      hard_place=False,
    #      sac_model_load=load_idx)



