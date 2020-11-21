import time
import datetime
import os
import numpy as np
import cv2
import torch


class Logger():
    def __init__(self,
                 continue_logging,
                 logging_directory,
                 logger_dir=None):

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.continue_logging = continue_logging

        if self.continue_logging:
            self.base_directory = logging_directory
            print('Pre-loading data logging session: %s' % self.base_directory)
        elif logger_dir is None:
            self.base_directory = os.path.join(logging_directory, timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
            print('Creating data logging session: %s' % self.base_directory)
        else:
            self.base_directory = os.path.join(logging_directory, logger_dir)
            print('Creating data logging session: %s' % self.base_directory)

        self.info_directory = os.path.join(self.base_directory, 'info')
        self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color-heightmaps')
        self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth-heightmaps')
        self.visualizations_directory = os.path.join(self.base_directory, 'visualizations')
        self.transitions_directory = os.path.join(self.base_directory, 'transitions')
        self.models_directory = os.path.join(self.transitions_directory, 'DQN_models')
        # -----------------------------------------------
        self.pos_displacement_directory = os.path.join(self.transitions_directory, 'pos-displacement')
        self.ori_displacement_directory = os.path.join(self.transitions_directory, 'ori-displacement')
        self.grasp_success_directory = os.path.join(self.transitions_directory, 'grasp-successful')
        self.action_position_directory = os.path.join(self.transitions_directory, 'action-position')
        self.rotation_idx_directory = os.path.join(self.transitions_directory, 'rotation-idx')
        self.input_patch_dir = os.path.join(self.transitions_directory, 'input-patch')
        self.mask_dir = os.path.join(self.transitions_directory, 'mask')
        self.grasp_patch_dir = os.path.join(self.transitions_directory, 'grasp-patch')
        self.grasp_pose_img_dir = os.path.join(self.transitions_directory, 'grasp-pose-img')

        if not os.path.exists(self.info_directory):
            os.makedirs(self.info_directory)
        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
        if not os.path.exists(self.visualizations_directory):
            os.makedirs(self.visualizations_directory)
        if not os.path.exists(self.transitions_directory):
            os.makedirs(os.path.join(self.transitions_directory, 'data'))
        # ------------------------- additional saver ------------------------
        if not os.path.exists(self.pos_displacement_directory):
            os.makedirs(self.pos_displacement_directory)
        if not os.path.exists(self.ori_displacement_directory):
            os.makedirs(self.ori_displacement_directory)
        if not os.path.exists(self.grasp_success_directory):
            os.makedirs(self.grasp_success_directory)
        if not os.path.exists(self.action_position_directory):
            os.makedirs(self.action_position_directory)
        if not os.path.exists(self.rotation_idx_directory):
            os.makedirs(self.rotation_idx_directory)
        if not os.path.exists(self.input_patch_dir):
            os.makedirs(self.input_patch_dir)
        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)
        if not os.path.exists(self.grasp_patch_dir):
            os.makedirs(self.grasp_patch_dir)
        if not os.path.exists(self.grasp_pose_img_dir):
            os.makedirs(self.grasp_pose_img_dir)

    def save_camera_info(self, intrinsics, pose, depth_scale):
        np.savetxt(os.path.join(self.info_directory, 'camera-intrinsics.txt'), intrinsics, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-pose.txt'), pose, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-depth-scale.txt'), [depth_scale], delimiter=' ')

    def save_heightmap_info(self, boundaries, resolution):
        np.savetxt(os.path.join(self.info_directory, 'heightmap-boundaries.txt'), boundaries, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'heightmap-resolution.txt'), [resolution], delimiter=' ')

    def save_img(self, iteration, img_to_save, directory, name):
        img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(directory, '%06d.%s.png' % (iteration, name)), img_to_save)

    def write_to_log(self, log_name, log):
        np.savetxt(os.path.join(self.transitions_directory, '%s.log.txt' % log_name), log, delimiter=' ')

    def write_to_npy(self, log_name, array):
        np.save(os.path.join(self.transitions_directory, '%s.npy' % log_name), array)

    def save_npy(self, array_to_save, iteration, directory, name):
        path = os.path.join(directory, '%06d.%s.npy' % (iteration, name))
        np.save(path, array_to_save)

    def save_mask(self, mask, iteration):
        path = os.path.join(self.mask_dir, '%06d.mask.npy' % iteration)
        np.save(path, mask)

    def save_model(self, iteration, model, name):
        torch.save(model.cpu().state_dict(), os.path.join(self.models_directory, '%06d.%s.pth' % (iteration, name)))

    def save_backup_model(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.models_directory, 'snapshot-backup.%s.pth' % (name)))

    def save_visualizations(self, iteration, affordance_vis, name):
        cv2.imwrite(os.path.join(self.visualizations_directory, '%06d.%s.png' % (iteration,name)), affordance_vis)

    def save_point_cloud(self, point_cloud, iteration, mode):
        """
        Save the point cloud captured by the camera
        """
        path = os.path.join(self.point_cloud_dir, '%06d.%s.point.npy' % (iteration, mode))
        np.save(path, point_cloud)

    def make_new_recording_directory(self, iteration):
        recording_directory = os.path.join(self.recordings_directory, '%06d' % (iteration))
        if not os.path.exists(recording_directory):
            os.makedirs(recording_directory)
        return recording_directory

    def save_q_functions(self, q_value, iteration):
        """
        Save the output nparray of the predicted output
        """
        path = os.path.join(self.predicted_q_func_dir,
                            '%06d.%s.qfunc' % (iteration, 'shovel'))
        np.save(path, q_value)

    def save_transition(self, iteration, transition):
        depth_heightmap = np.round(transition.state * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.transitions_directory, 'data', '%06d.0.depth.png' % (iteration)), depth_heightmap)
        next_depth_heightmap = np.round(transition.next_state * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.transitions_directory, 'data', '%06d.1.depth.png' % (iteration)), next_depth_heightmap)
