#!/usr/bin/env python
# This is the simulation experiment to train a minimal displacement grasping policy
import time, torch
import os
import argparse
import numpy as np
import cv2
from util_funcs.Robot_sim import Robot
from util_funcs.grasp_trainer import Trainer
from util_funcs.logger import Logger
from util_funcs import utils
from scipy import ndimage
import matplotlib.pyplot as plt
from util_funcs import utils_gp as ugp


def main(args,
         logger_dir=None,
         load_model_dir=None,
         is_eval=False,
         is_insert_task=False):
    if not is_insert_task:
        num_of_obj = 1
    else:
        num_of_obj = 4
    num_of_rotations = 8
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if is_insert_task:
        num_of_holes = 2
    else:
        num_of_holes = 0

    empty_restart_count = 0
    restart_count = 0
    grasp_obj_count = 0
    place_motion = not is_eval  # if not testing, the robot will place the object back to the workspace
                                # with the grasp pose in order to derive the displacement label

    # --------------- Setup options ---------------
    heightmap_size = 500    # Workspace heightmap size
    grasp_patch_size = 160  # Img patch size that crops from the heightmap and input to the grasp net

    grasp_patch_size_out = 20    # Grasp Network Output dimension
    workspace_limits = np.asarray([[0.2, 0.7],
                                   [-0.25, 0.25],
                                   [0.0002, 0.12]]) # Vrep Workspace Range

    # Image and voxel grid parameters
    heightmap_resolution = (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_size  # in cm
    # Random Seed
    random_seed = np.random.seed(args.random_seed)

    # ------ Pre-loading and logging options ------
    continue_logging = args.continue_logging if args.continue_logging else False  # Continue logging iter
    logging_directory = os.path.abspath('logs')
    save_visualizations = args.save_visualizations  # Save visualizations of FCN predictions?
    obj_dir = os.path.abspath(args.obj_dir) # Path to load Vrep models
    load_model_dir = load_model_dir

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(workspace_limits,
                  num_of_obj=num_of_obj,
                  obj_dir=obj_dir,
                  is_insert_task=is_insert_task,
                  num_of_holes=num_of_holes,
                  is_eval=is_eval)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory,
                    logger_dir=logger_dir)
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)  # Save heightmap parameters

    # Initialize trainer
    trainer = Trainer(device=device,
                      num_of_rotation=num_of_rotations,
                      load_model_dir=load_model_dir,
                      is_eval=is_eval)
    # Some exploration parameters
    # Can be changed with any thing you want basically
    if not is_eval:
        if continue_logging:
            explore_prob = 0.3
        else:
            explore_prob = 0.99  # if testing: else 0.0
    else:
        explore_prob = 0.

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'grasp_position': None,
                          'grasp_success': False,
                          'rot_angle': 0.,
                          'rot_idx': 0}
    # Initialize variables for heuristic bootstrapping and exploration probability
    exit_called = False
    restart_flag = False
    action_count = 0

    # Start main training/testing loop
    while True:
        iteration_time_0 = time.time()
        print(' --- --- --- --- --- Iteration: %d --- --- --- --- --- ' % trainer.iteration)
        # Check Simulation env because Vrep goes crazy easily
        sim_ok = robot.check_sim()
        if restart_flag:
            robot.stop_sim()
            robot.restart_sim()
            restart_flag = False
            time.sleep(1)
            print('Check restart statistics: ')
            print(empty_restart_count, restart_count)
        # Get latest RGB-D image
        color_heightmap, valid_depth_heightmap = ugp.get_heightmap(robot=robot,
                                                                   heightmap_resolution=heightmap_resolution,
                                                                   workspace_limits=workspace_limits)
        # Hack into Object Position if Grasp_training
        if not is_insert_task:
            obj_target_handle = robot.obj_target_handles[0]
            obj_target_pos = robot.get_single_obj_position(obj_target_handle)
            obj_target_ori = robot.get_single_obj_orientations(obj_target_handle)
        else:
            # Here may use mask r-cnn + siamese to generate obj center pos
            pass

        # Get obj_target position to crop img patches (from Vrep Ground Truth)
        sample_center_y = int((obj_target_pos[0] - workspace_limits[0][0]) / heightmap_resolution)
        sample_center_x = int((obj_target_pos[1] - workspace_limits[1][0]) / heightmap_resolution)
        grasp_patch, grasp_color_patch, grasp_patch_row_low, grasp_patch_col_low = ugp.crop_workspace_heightmap(center_x=sample_center_x,
                                                                                                                center_y=sample_center_y,
                                                                                                                patch_size=grasp_patch_size,
                                                                                                                heightmap_size=heightmap_size,
                                                                                                                depth_heightmap=valid_depth_heightmap,
                                                                                                                color_heightmap=color_heightmap)
        #  Derive Grasp Mask Here
        grasp_mask = ugp.derive_depth_mask(grasp_patch,
                                           grasp_patch_size_out,
                                           low_thres=0.002,
                                           high_thres=10.0)

        if not is_eval:
            # save data
            logger.save_npy(grasp_mask,
                            trainer.iteration,
                            logger.mask_dir,
                            'grasp-mask')
            # ---------------------------------
            logger.save_npy(grasp_patch,
                            trainer.iteration,
                            logger.input_patch_dir,
                            'grasp-patch-input')
            logger.save_npy(valid_depth_heightmap,
                            trainer.iteration,
                            logger.depth_heightmaps_directory,
                            'workspace')

        # To ensure no loss input to the grasp network (E.g. after 45^\circ rotation)
        # Zero pad the input patch to sqrt(2)*grasp_patch_size
        pad_width = int((np.sqrt(2) - 1) * grasp_patch_size / 2) + 1
        grasp_patch = np.pad(grasp_patch,
                             ((pad_width, pad_width),
                              (pad_width, pad_width)),
                             'constant',
                             constant_values=0)
        padded_input_size = grasp_patch.shape[0]
        grasp_patch.shape = (padded_input_size,
                             padded_input_size,
                             1)

        if 'prev_grasp_patch' in locals() and not is_eval and not is_insert_task:
            # ------------- Stochastic Gradient Descent Thread ------------------
            # Compute training labels
            grasp_score = trainer.get_grasp_label_value(prev_ori_displacement,
                                                        prev_pos_displacement,
                                                        prev_grasp_success)
            print('Previous Grasp Score (Label): %f' % grasp_score)

            # On-policy Backprop with the newest sample
            training_loss = trainer.backprop_mask(img_input=prev_grasp_patch,
                                                  action_position=prev_grasp_position,
                                                  label_value=grasp_score,
                                                  prev_mask=prev_grasp_mask,
                                                  output_size=grasp_patch_size_out,
                                                  rot_idx=prev_grasp_rot_idx)
            print('Last Step Grasp Training Loss: %f' % training_loss)
            if trainer.iteration % 10 == 0:
                trainer.grasp_loss_log.append(training_loss)
            if not is_eval:
                logger.write_to_npy('grasp_training-loss',
                                    np.asarray(trainer.grasp_loss_log))

            if prev_grasp_rot_idx < (num_of_rotations / 2):
                another_idx = prev_grasp_rot_idx + num_of_rotations / 2
            else:
                another_idx = prev_grasp_rot_idx - num_of_rotations / 2
            training_loss = trainer.backprop_mask(img_input=prev_grasp_patch,
                                                  action_position=prev_grasp_position,
                                                  label_value=grasp_score,
                                                  prev_mask=prev_grasp_mask,
                                                  output_size=grasp_patch_size_out,
                                                  rot_idx=another_idx)
            # ---------- Experience Replay -----------
            sample_ind = np.argwhere(np.asarray(trainer.grasp_successful_log)[0:trainer.iteration] > 0)
            print('Check Successful Experiences: ', len(sample_ind))
            if len(trainer.grasp_successful_log) > 1:
                sample_ind = np.argwhere(np.asarray(trainer.grasp_successful_log)[0:trainer.iteration] > 0.5)
                sample_ind_neg = np.argwhere(np.asarray(trainer.grasp_successful_log)[0:trainer.iteration] < 0.5)

                sample_size = sample_ind.size
                if sample_size > 64:
                    replay_num = 8
                else:
                    replay_num = 1

                if sample_ind.size > 0 and sample_ind_neg.size > 0:
                    for i in range(replay_num):
                        print(" --->>> Experience Replay: ")
                        sample_ind_length = sample_ind.shape[0]
                        sample_iteration = sample_ind[np.random.randint(0, sample_ind_length)][0]
                        print('Success Experience replay: iteration %d' % sample_iteration)
                        ugp.experience_replay(sample_iteration,
                                              logger,
                                              trainer,
                                              output_size=grasp_patch_size_out)

                        # ---- Negative Replay
                        sample_ind_length_neg = sample_ind_neg.shape[0]
                        sample_iteration_neg = sample_ind_neg[np.random.randint(0, sample_ind_length_neg)][0]
                        print('Failed Experience replay: iteration %d' % sample_iteration_neg)
                        ugp.experience_replay(sample_iteration_neg,
                                              logger,
                                              trainer,
                                              output_size=grasp_patch_size_out)
            # save Model
            if trainer.iteration % 200 == 2 and not is_eval:
                logger.save_model(trainer.iteration,
                                  trainer.grasp_model,
                                  'grasp_model')
                trainer.grasp_model.to(device=device,
                                       dtype=torch.float)

        # -------- Enough Training, Let's Act ! ----------
        grasp_predictions = trainer.make_predictions(grasp_patch,
                                                     output_size=grasp_patch_size_out)  # img_space_prediction (rot, robot_y, robot_x)
        predicted_value = np.amax(grasp_predictions)
        print('Best grasp (least displacement) scores: %f' % predicted_value)

        # This is the e-greedy action policy
        if is_eval:
            explore_prob = -1
            random_rot_prob = -1
        else:
            explore_prob = max(explore_prob * 0.99, 0.05)
            random_rot_prob = 0.05
        unravel_idx = ugp.grasp_action(trainer,
                                       grasp_predictions,
                                       explore_prob,
                                       random_rot_prob)
        if not is_eval:
            logger.save_npy(np.asarray(unravel_idx), trainer.iteration,
                            logger.action_position_directory, 'grasp-position')

        # ---------- Save Grasp Pose Info ---------------
        nonlocal_variables['rot_idx'] = unravel_idx[0]
        nonlocal_variables['rot_angle'] = unravel_idx[0] * 360.0 / num_of_rotations
        nonlocal_variables['grasp_position'] = unravel_idx[1:]  # img space pos

        # --------- calculate the position in robot frame
        robot_space_x, robot_space_y = nonlocal_variables['grasp_position'][1], nonlocal_variables['grasp_position'][0]

        io_ratio = float(grasp_patch_size) / grasp_patch_size_out
        robot_frame_x = grasp_patch_col_low * heightmap_resolution + (robot_space_x + 0.5) * io_ratio * heightmap_resolution + workspace_limits[0][0]
        robot_frame_y = grasp_patch_row_low * heightmap_resolution + (robot_space_y + 0.5) * io_ratio * heightmap_resolution + workspace_limits[1][0]

        # Workspace Height Map Graspng Point
        # Used to Crop post grasp patch
        grasp_map_y = int((robot_frame_x - workspace_limits[0][0]) / heightmap_resolution)
        grasp_map_x = int((robot_frame_y - workspace_limits[1][0]) / heightmap_resolution)
        # -------- 2D version z calculation ----------------------------
        z_position = 1 * valid_depth_heightmap[int(grasp_patch_row_low + (robot_space_y + 0.5) * io_ratio),
                                               int(grasp_patch_col_low + (robot_space_x + 0.5) * io_ratio)]
        # ---------------------------------------------------------------
        #
        x_bias = 0
        y_bias = 0
        robot_act_pos = (robot_frame_x + x_bias,
                         robot_frame_y + y_bias,
                         z_position)

        # ---------- Visualize executed primitive, and affordances ----------------
        if save_visualizations:
            # >>>> Check Prediciton Visualization
            grasp_vis = grasp_predictions
            grasp_vis = ugp.get_grasp_vis(grasp_vis,
                                          grasp_color_patch,
                                          nonlocal_variables['rot_idx'],
                                          nonlocal_variables['grasp_position'],
                                          num_rotations=num_of_rotations)
            logger.save_visualizations(trainer.iteration, grasp_vis, 'grasp')
            cv2.imwrite('grasp_vis.png', grasp_vis)

            if trainer.iteration % 10 < 1:
                fig_1 = plt.figure()
                ax_1 = fig_1.add_subplot(1, 1, 1)
                ax_1.plot(trainer.grasp_loss_log)
                plt.savefig('training-loss.png')
                plt.close(fig_1)
        # ---- Execute Grasp
        nonlocal_variables['grasp_success'], grasp_obj_handle_idx, simulation_fail = robot.grasp(robot_act_pos,
                                                                                                 nonlocal_variables['rot_angle'],
                                                                                                 place_motion=place_motion)
        print('Grasp Successful: %r' % (nonlocal_variables['grasp_success']))
        if not is_eval and not is_insert_task:
            logger.save_npy(np.asarray(nonlocal_variables['grasp_success']), trainer.iteration,
                            logger.grasp_success_directory, 'grasp-success')
        else:
            pass

        if not is_insert_task:
            if nonlocal_variables['grasp_success']:
                # Derive Displacement Info and save the data
                trainer.grasp_successful_log.append(1)
                grasp_obj_count += 1
                action_count = 0

                post_pos = robot.get_single_obj_position(robot.obj_target_handles[grasp_obj_handle_idx])
                post_ori = robot.get_single_obj_orientations(robot.obj_target_handles[grasp_obj_handle_idx])
                pos_displacement = (obj_target_pos - post_pos) * 1e3  # in mm scale
                ori_displacement = np.rad2deg(utils.cal_relative_theta(obj_target_ori, post_ori))  # in degree scale
                if np.isnan(ori_displacement):
                    ori_displacement = 0.
            else:
                # Save Unsuccessful Data
                trainer.grasp_successful_log.append(0)
                action_count += 1
                # Arbitrarily Set the displacement to be very large value
                pos_displacement = [100., 100., 100.]
                ori_displacement = [180.]
            np.asarray(pos_displacement)
            np.asarray(ori_displacement)
            if not is_eval:
                logger.save_npy(np.asarray(pos_displacement), trainer.iteration,
                                logger.pos_displacement_directory, 'pos-displacement')
                logger.save_npy(np.asarray(ori_displacement), trainer.iteration,
                                logger.ori_displacement_directory, 'ori-displacement')
            print('Post Grasp Displacement: ', 'Pos: ', pos_displacement)
            print('Post Grasp Displacement: ', 'Ori: ', ori_displacement)
            time.sleep(0.5)
            empty_threshold = 90
            _, post_depth_map = ugp.get_heightmap(robot,
                                                  heightmap_resolution,
                                                  workspace_limits)
            pix_count = np.zeros(post_depth_map.shape)
            pix_count[post_depth_map > 0.002] = 1
            occupied_pix = np.sum(pix_count)
            print('Occupied Pixel Count: %d' % occupied_pix)
            if occupied_pix < empty_threshold or len(robot.obj_target_handles) < 1:
                empty_restart_count += 1
                action_count = 0
                grasp_obj_count = 0
                restart_count += 1
                restart_flag = True

            prev_grasp_patch = grasp_patch.copy()
            prev_grasp_success = nonlocal_variables['grasp_success']
            prev_grasp_rot_idx = nonlocal_variables['rot_idx']
            prev_grasp_position = nonlocal_variables['grasp_position']
            prev_pos_displacement = pos_displacement
            prev_ori_displacement = ori_displacement
            prev_grasp_mask = grasp_mask.copy()

        else:
            # Insertion thread has been writen in main_insert_task.py
            # This branch is not active anymore
            pass

        # Save information for next training step
        # ----- Handle Simulation Restart Criterion -----
        if action_count > 5 or simulation_fail or nonlocal_variables['grasp_success'] is True:
            print("About to restart. Check Simulation Fail: ", simulation_fail)
            restart_flag = True
            action_count = 0
            grasp_obj_count = 0
            restart_count += 1

        trainer.iteration += 1
        iteration_time_1 = time.time()

        if not is_eval and trainer.iteration >= 5000:
            robot.stop_sim()
            exit_called = True

        if is_eval and trainer.iteration >= 100:
            robot.stop_sim()
            exit_called = True
            task_complete_res = np.asarray([empty_restart_count,
                                            restart_count])
            logger.save_npy(task_complete_res,
                            iteration=250,
                            directory=logger.transitions_directory,
                            name='task-complete-info')

        if exit_called:
            break


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=13131,
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
    logger_dir = 'train'
    is_testing = False
    load_model_dir = None  # Path to load some pretrained models
    main(args,
         logger_dir=logger_dir,
         load_model_dir=load_model_dir,
         is_eval=is_testing,
         is_insert_task=False)

