from scipy import ndimage
import cv2, os
import numpy as np
from util_funcs import utils


def get_workspace_vis(predictions,
                      color_heightmap,
                      action_position):
    img_size = color_heightmap.shape[0]
    visual_size = predictions.shape[1]
    zoom_ratio = float(img_size) / visual_size

    prediction_vis = ndimage.zoom(predictions,
                                  zoom=[zoom_ratio, zoom_ratio],
                                  mode='nearest',
                                  prefilter=False)
    prediction_vis = np.clip(prediction_vis, 0, 2)
    prediction_vis.shape = (img_size, img_size)
    prediction_vis = cv2.applyColorMap((prediction_vis*255/2).astype(np.uint8), cv2.COLORMAP_JET)
    prediction_vis = (0.5*cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
    if action_position is not None:
        cv2.circle(prediction_vis,
                   (int(action_position[1] * zoom_ratio - zoom_ratio / 2),
                    int(action_position[0] * zoom_ratio - zoom_ratio / 2)),
                   7, (0, 255, 255), 3)
    prediction_vis = cv2.flip(prediction_vis, 0)
    return prediction_vis


def get_grasp_vis(predictions,
                  color_heightmap,
                  action_rot,
                  action_position,
                  num_rotations=16):
    """
    Get the visualization of the grasping network prediction
    """
    canvas = None
    for canvas_row in range(4):
        tmp_row_canvas = None
        for canvas_col in range(int(num_rotations / 4)):
            rotate_idx = int(canvas_row * num_rotations / 4 + canvas_col)
            prediction = predictions[rotate_idx, :, :].copy()
            if rotate_idx == action_rot:
                position = action_position
            else:
                position = None
            prediction_vis = get_workspace_vis(prediction,
                                               color_heightmap,
                                               action_position=position)
            if tmp_row_canvas is None:
                tmp_row_canvas = prediction_vis
            else:
                tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
        if canvas is None:
            canvas = tmp_row_canvas
        else:
            canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)
    return canvas


def get_heightmap(robot,
                  heightmap_resolution,
                  workspace_limits):
    """
    As the function name
    """
    color_img_set, depth_img_set = robot.get_camera_data()
    depth_img_set = depth_img_set * robot.cam_depth_scale  # Apply depth scale from calibration
    color_heightmap, depth_heightmap = utils.get_heightmap(color_img_set, depth_img_set,
                                                           robot.cam_intrinsics,
                                                           robot.cam_pose, workspace_limits,
                                                           heightmap_resolution)
    depth_heightmap[np.isnan(depth_heightmap)] = 0
    kernel = np.ones([3, 3])
    color_heightmap = cv2.dilate(color_heightmap, kernel, iterations=2)
    color_heightmap = cv2.erode(color_heightmap, kernel, iterations=2)
    valid_depth_heightmap = cv2.dilate(depth_heightmap, kernel, iterations=2)
    valid_depth_heightmap = cv2.erode(valid_depth_heightmap, kernel, iterations=2)
    return color_heightmap, valid_depth_heightmap


def derive_depth_mask(input_img,
                      output_size,
                      low_thres=0.5,
                      high_thres=10.0):
    """
    :param input_img:  (N, N)
    :param height_threshold:
    :param mask_downscale: value between (0, 1)
    :return: a depth mask -- ndarray (m, m), m = N * mask_downscale
    """
    input_size = input_img.shape[0]
    if output_size == input_size:
        mask_downscale = 1
    else:
        mask_downscale = float(output_size) / input_size
    mask_1 = np.where(input_img < high_thres, 1., 0.)
    mask_2 = np.where(input_img > low_thres, 1., 0.)
    mask = np.multiply(mask_1, mask_2)
    if mask_downscale is not None:
        mask = cv2.resize(mask, (0, 0),
                          fx=mask_downscale,
                          fy=mask_downscale,
                          interpolation=cv2.INTER_NEAREST)
    kernel = np.ones([3, 3])
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    return mask


def crop_workspace_heightmap(center_x, center_y,
                             patch_size, heightmap_size,
                             depth_heightmap, color_heightmap):
    """
    Explained as the function name, cropping a patch from the workspace
    """
    if center_x > heightmap_size - patch_size / 2:
        center_x = heightmap_size - patch_size / 2
    if center_y > heightmap_size - patch_size / 2:
        center_y = heightmap_size - patch_size / 2
    grasp_patch_row_low = int(center_x - patch_size / 2)
    grasp_patch_col_low = int(center_y - patch_size / 2)

    if grasp_patch_row_low < 0:
        grasp_patch_row_low = 0
    elif grasp_patch_row_low >= (heightmap_size - patch_size):
        grasp_patch_row_low = heightmap_size - patch_size

    if grasp_patch_col_low < 0:
        grasp_patch_col_low = 0
    elif grasp_patch_col_low >= (heightmap_size - patch_size):
        grasp_patch_col_low = heightmap_size - patch_size
    depth_patch, color_patch = None, None
    if depth_heightmap is not None:
        depth_patch = depth_heightmap[grasp_patch_row_low: (grasp_patch_row_low + patch_size),
                                      grasp_patch_col_low: (grasp_patch_col_low + patch_size)]
    if color_heightmap is not None:
        color_patch = color_heightmap[grasp_patch_row_low: (grasp_patch_row_low + patch_size),
                                      grasp_patch_col_low: (grasp_patch_col_low + patch_size),
                                      :]
    return depth_patch, color_patch, grasp_patch_row_low, grasp_patch_col_low


def experience_replay(sample_iteration,
                      logger,
                      trainer,
                      output_size):
    """
    Experience Replay Training for grasping network
    """
    num_of_rotations = trainer.num_of_rotation
    sample_depth_patch = np.load(os.path.join(logger.input_patch_dir,
                                              '%06d.grasp-patch-input.npy' % sample_iteration))
    patch_size = sample_depth_patch.shape[0]
    pad_width = int((np.sqrt(2) - 1) * patch_size / 2) + 1
    sample_depth_patch = np.pad(sample_depth_patch,
                                ((pad_width, pad_width),
                                 (pad_width, pad_width)),
                                'constant',
                                constant_values=0)
    input_size = sample_depth_patch.shape[0]
    sample_depth_patch.shape = (input_size, input_size, 1)
    sample_action_position = np.load(os.path.join(logger.action_position_directory,
                                                  '%06d.grasp-position.npy' % sample_iteration))
    sample_grasp_success = np.load(os.path.join(logger.grasp_success_directory,
                                                '%06d.grasp-success.npy' % sample_iteration))
    sample_pos_disp = np.load(os.path.join(logger.pos_displacement_directory,
                                           '%06d.pos-displacement.npy' % sample_iteration))
    sample_ori_disp = np.load(os.path.join(logger.ori_displacement_directory,
                                           '%06d.ori-displacement.npy' % sample_iteration))
    sample_mask = np.load(os.path.join(logger.mask_dir,
                                       '%06d.grasp-mask.npy' % sample_iteration))
    sample_grasp_score = trainer.get_grasp_label_value(sample_ori_disp,
                                                       sample_pos_disp,
                                                       sample_grasp_success)
    loss_value = trainer.backprop_mask(img_input=sample_depth_patch,
                                       action_position=sample_action_position[1:],
                                       label_value=sample_grasp_score,
                                       prev_mask=sample_mask,
                                       output_size=output_size,
                                       rot_idx=sample_action_position[0])
    print('Experience Replay Loss: %f' % loss_value)
    if sample_action_position[0] < (num_of_rotations / 2):
        another_idx = sample_action_position[0] + num_of_rotations / 2
    else:
        another_idx = sample_action_position[0] - num_of_rotations / 2
    loss_value = trainer.backprop_mask(img_input=sample_depth_patch,
                                       action_position=sample_action_position[1:],
                                       label_value=sample_grasp_score,
                                       prev_mask=sample_mask,
                                       output_size=output_size,
                                       rot_idx=another_idx)
    print('Experience Replay Loss: ', loss_value)


def grasp_action(trainer,
                 grasp_predictions,
                 explore_prob,
                 random_rot_prob=0.1):
    # E_greedy policy for grasp selection
    num_of_rotations = trainer.num_of_rotation
    out_patch_size = grasp_predictions.shape[1]
    if np.random.random_sample() < explore_prob:
        print(" >>>> Exploration ")
        explore_rot_idx = np.random.randint(0, num_of_rotations)
        rand_0 = np.random.randint(-2, 3)
        rand_1 = np.random.randint(-2, 3)
        explore_idx = np.unravel_index(np.argmax(grasp_predictions[explore_rot_idx, :, :]),
                                       grasp_predictions[explore_rot_idx, :, :].shape)
        if explore_idx[0] + rand_0 < 0 or explore_idx[0] + rand_0 > (out_patch_size - 1):
            explore_0 = explore_idx[0]
        else:
            explore_0 = explore_idx[0] + rand_0
        if explore_idx[1] + rand_1 < 0 or explore_idx[1] + rand_1 > (out_patch_size - 1):
            explore_1 = explore_idx[1]
        else:
            explore_1 = explore_idx[1] + rand_1
        unravel_idx = (explore_rot_idx,
                       explore_0,
                       explore_1)
    elif np.random.random_sample() < random_rot_prob:
        print(" >>>>> Explore Rotation ")
        explore_rot_idx = np.random.randint(0, num_of_rotations)
        explore_idx = np.unravel_index(np.argmax(grasp_predictions[explore_rot_idx, :, :]),
                                       grasp_predictions[explore_rot_idx, :, :].shape)
        unravel_idx = (explore_rot_idx,
                       explore_idx[0],
                       explore_idx[1])
    else:
        print(" >>>>> Greedy Policy  ")
        unravel_idx = np.unravel_index(np.argmax(grasp_predictions),
                                       grasp_predictions.shape)  # Action pos in img space
    return unravel_idx


def create_binary_img(depth_patch,
                      padding=None):
    binary_height_map = np.zeros_like(depth_patch,
                                      dtype=np.uint8)
    binary_height_map[depth_patch > 0.008] = 1
    if len(depth_patch.shape) > 2:
        x = depth_patch.shape[0]
        binary_height_map.shape = (x, x)
    if padding is not None:
        binary_height_map = np.pad(binary_height_map,
                                   ((padding, padding),
                                    (padding, padding)),
                                   'constant',
                                   constant_values=0)
    return binary_height_map


def derive_hole_patch(target_patch, match_patch_size,
                      wall_threshold=0.12, floor_threshould=0.002,
                      padding=True):
    output_size = target_patch.shape[0]
    hole_mask = derive_depth_mask(target_patch,
                                  output_size=output_size,
                                  low_thres=floor_threshould + 0.02,
                                  high_thres=wall_threshold)
    if padding is True:
        hole_patch = create_binary_img(hole_mask,
                                       padding=int(match_patch_size / 2))
    else:
        hole_patch = create_binary_img(hole_mask,
                                       padding=None)
    return hole_patch


def find_single_blob_center(input_img):
    """
    :param input_img: nparray with shape (n, n), dtype = np.uint8
    :return: blob center coodinate (x, y)
    """
    find_center = False
    center_x, center_y = 0, 0
    # input_img.dtype = np.uint8
    M = cv2.moments(input_img)
    if M["m00"] != 0:
        find_center = True
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    # input_img.dtype = np.int
    return center_x, center_y, find_center


def compute_img_diff(obj_img,
                     back_img,
                     induce_x,
                     induce_y,
                     trans_row,
                     trans_col):
    width = obj_img.shape[0]
    combine_img = back_img.copy()
    induce_y = int(induce_y + trans_row)
    induce_x = int(induce_x + trans_col)
    combine_img[induce_y:induce_y + width, induce_x:induce_x + width] -= obj_img
    neg_count = len(np.argwhere(combine_img < 0))
    return neg_count


def get_opt_rotate(obj_img, back_img,
                   back_center_x, back_center_y,
                   obj_center_x, obj_center_y,
                   prev_rot_angle=0.,
                   is_erosion=False):
    """
    Binary Search for the optimal rotate angles

    input_img: nparray with shape (n, n), dtype = np.int8
    obj_img must be well aligned at the center

    Output --- a new rotate angle on the original obj image
    """
    width = obj_img.shape[0]
    rot_img = ndimage.rotate(obj_img, prev_rot_angle, reshape=False)
    induce_x, induce_y = int(back_center_x - obj_center_x), int(back_center_y - obj_center_y)
    combine_img = back_img.copy()
    combine_img[induce_y:induce_y + width, induce_x:induce_x + width] -= rot_img
    neg_count = len(np.argwhere(combine_img < 0))
    if is_erosion:
        angle_amount = 4.
    else:
        angle_amount = 16.
    # check combine_img.dtype; rot_img.dtype; back_img
    curr_angle = prev_rot_angle
    while angle_amount > 0.5:
        angle_amount /= 2.

        rotate_1 = ndimage.rotate(obj_img, curr_angle + angle_amount, reshape=False)
        combine_img = back_img.copy()
        combine_img[induce_y:induce_y+width, induce_x:induce_x+width] -= rotate_1
        neg_count_1 = len(np.argwhere(combine_img < 0))

        rotate_2 = ndimage.rotate(obj_img, curr_angle - angle_amount, reshape=False)
        combine_img = back_img.copy()
        combine_img[induce_y:induce_y + width, induce_x:induce_x + width] -= rotate_2
        neg_count_2 = len(np.argwhere(combine_img < 0))

        if neg_count_1 < neg_count_2:
            if neg_count_1 < neg_count:
                neg_count = neg_count_1
                curr_angle = curr_angle + angle_amount
        else:
            if neg_count_2 < neg_count:
                neg_count = neg_count_2
                curr_angle = curr_angle - angle_amount
        # print(curr_angle)
        # print(neg_count, neg_count_1, neg_count_2)
    # print('Negative Pix Count Rotation: %d.' % neg_count)
    # print('Optimal Rotation: ', curr_angle)
    return curr_angle, neg_count


def get_opt_translate(obj_img,
                      back_img,
                      back_center_x,
                      back_center_y,
                      obj_center_x,
                      obj_center_y,
                      prev_row_trans=0,
                      prev_col_trans=0,
                      is_erosion=False):
    """
    Binary search the optimal translation of the two image patches
    """
    width = obj_img.shape[0]
    obj_center_x = int(obj_center_x)
    obj_center_y = int(obj_center_y)
    curr_row_trans, curr_col_trans = prev_row_trans, prev_col_trans
    induce_x = int(back_center_x - obj_center_x + curr_col_trans)
    induce_y = int(back_center_y - obj_center_y + curr_row_trans)
    combine_img = back_img.copy()
    combine_img[induce_y:induce_y + width, induce_x:induce_x + width] -= obj_img
    neg_count = len(np.argwhere(combine_img < 0))
    if is_erosion:
        trans_amount = 4
    else:
        trans_amount = 8
    while trans_amount > 1:
        trans_amount = trans_amount / 2
        neg_count_1 = compute_img_diff(obj_img,
                                       back_img,
                                       induce_x,
                                       induce_y,
                                       trans_row=trans_amount,
                                       trans_col=0)
        neg_count_2 = compute_img_diff(obj_img,
                                       back_img,
                                       induce_x,
                                       induce_y,
                                       trans_row=(-trans_amount),
                                       trans_col=0)
        if neg_count_1 < neg_count_2:
            if neg_count_1 < neg_count:
                neg_count = neg_count_1
                curr_row_trans += trans_amount
        else:
            if neg_count_2 < neg_count:
                neg_count = neg_count_2
                curr_row_trans -= trans_amount

    induce_y = back_center_y - obj_center_y + curr_row_trans
    if is_erosion:
        trans_amount = 4
    else:
        trans_amount = 16
    while trans_amount > 1:
        trans_amount = trans_amount / 2
        neg_count_1 = compute_img_diff(obj_img,
                                       back_img,
                                       induce_x,
                                       induce_y,
                                       trans_row=0,
                                       trans_col=trans_amount)
        neg_count_2 = compute_img_diff(obj_img,
                                       back_img,
                                       induce_x,
                                       induce_y,
                                       trans_row=0,
                                       trans_col=(-trans_amount))
        if neg_count_1 < neg_count_2:
            if neg_count_1 < neg_count:
                neg_count = neg_count_1
                curr_col_trans += trans_amount
        else:
            if neg_count_2 < neg_count:
                neg_count = neg_count_2
                curr_col_trans -= trans_amount
    # print('Negative Pix Count Translation: %d.' % neg_count)
    # print(curr_row_trans, curr_col_trans)
    return curr_row_trans, curr_col_trans, neg_count


def match_obj_hole(post_grasp_pos_patch,
                   pre_grasp_pos_patch,
                   post_x=None, post_y=None):
    """
    This function matches two patches (binary images), with 1 on the object pixel; 0 for background

    pre_grasp_pos_patch --- manipulatable pose patches (target object patch)
    post_grasp_pos_patch --- background fixed pose patch (Hole patch)
    """
    if post_x is None:
        post_x, post_y, _ = find_single_blob_center(post_grasp_pos_patch)
    obj_center_x = int(pre_grasp_pos_patch.shape[1] / 2)
    obj_center_y = int(pre_grasp_pos_patch.shape[0] / 2)

    pre_grasp_pos_patch.dtype = np.int8
    post_grasp_pos_patch.dtype = np.int8
    opt_rot, opt_row_trans, opt_col_trans = 0., 0, 0

    old_neg_count = 2048
    neg_count = old_neg_count - 2

    is_erosion = False
    erosion_count = 0
    obj_patch = pre_grasp_pos_patch.copy()

    # Okay so the matching is actually iterative binary search
    while old_neg_count > neg_count:
        # print(old_neg_count, neg_count)

        if neg_count < 1:
            # print('ERODE')
            is_erosion = True
            erosion_count += 1
            kernel = np.ones([3, 3])
            post_grasp_pos_patch.dtype = np.uint8
            post_grasp_pos_patch = cv2.erode(post_grasp_pos_patch,
                                             kernel,
                                             iterations=1)
            post_grasp_pos_patch.dtype = np.int8
            neg_count = 2048

        old_neg_count = neg_count
        row_trans, col_trans, neg_count_1 = get_opt_translate(obj_img=obj_patch,
                                                              back_img=post_grasp_pos_patch,
                                                              back_center_x=post_x,
                                                              back_center_y=post_y,
                                                              obj_center_x=obj_center_x,
                                                              obj_center_y=obj_center_y,
                                                              prev_row_trans=opt_row_trans,
                                                              prev_col_trans=opt_col_trans,
                                                              is_erosion=is_erosion)
        if neg_count_1 < old_neg_count:
            opt_row_trans = row_trans
            opt_col_trans = col_trans
        rot_res, neg_count_2 = get_opt_rotate(obj_img=pre_grasp_pos_patch,
                                              back_img=post_grasp_pos_patch,
                                              back_center_x=post_x + opt_col_trans,
                                              back_center_y=post_y + opt_row_trans,
                                              obj_center_x=obj_center_x,
                                              obj_center_y=obj_center_y,
                                              prev_rot_angle=opt_rot,
                                              is_erosion=is_erosion)
        if neg_count_2 < neg_count_1:
            opt_rot = rot_res
        neg_count = min(neg_count_1, neg_count_2)
        obj_patch = ndimage.rotate(pre_grasp_pos_patch, opt_rot, reshape=False)
    return is_erosion, erosion_count, int(opt_row_trans), int(opt_col_trans), opt_rot, post_x, post_y


def compensate_calc(grasp_point_x, grasp_point_y,
                    obj_center_x, obj_center_y,
                    hole_center_x, hole_center_y,
                    opt_row_trans, opt_col_trans, opt_rot,
                    grasp_gripper_angle,
                    heightmap_resolution):
    """
    This function computes the obj-hole compensation in order to align them.

    grasp_point_x, grasp_point_y: gripper grasping point (in robot frame)
    obj_center_x, obj_center_y: target object center point (in robot frame)
    hole_center_x, hole_center_y: hole center point (in robot frame)
    opt_row_trans, opt_col_trans, opt_rot: Result from matching function "match_obj_hole"
    grasp_gripper_angle: gripper grasping angle
    heightmap_resolution: heightmap_resolution
    """
    # === Grasping point at the center of the obj patch
    grasp_center_x = grasp_point_x
    grasp_center_y = grasp_point_y
    # ---- Derive background center location (in workspace) ---
    post_obj_x = hole_center_x
    post_obj_y = hole_center_y
    # ---- Align grasping center point with background center point
    patch_align_x = (post_obj_x - grasp_center_x) * heightmap_resolution
    patch_align_y = (post_obj_y - grasp_center_y) * heightmap_resolution

    # === Obj center location on the obj patch
    obj_x = obj_center_x
    obj_y = obj_center_y
    # === calculate grasp point and obj center bias and angle
    if obj_x == grasp_center_x:
        theta_0 = np.pi / 2
        r_0 = abs(obj_y - grasp_center_y)
    else:
        theta_0 = np.arctan2(obj_y - grasp_center_y,
                             obj_x - grasp_center_x)
        r_0 = np.sqrt((obj_y - grasp_center_y) ** 2 + (obj_x - grasp_center_x) ** 2)
    # The total angle bias for the obj patch
    compensate_theta = theta_0 + np.deg2rad(opt_rot)
    # Now compute the compensate transilation due to the alignment
    # of grasp center and hole center (we want to align the obj center and hole center)
    grasp_bias_x = r_0 * np.cos(compensate_theta)
    grasp_bias_y = r_0 * np.sin(compensate_theta)
    # The total compensate translation:
    compensate_x = opt_col_trans - grasp_bias_x
    compensate_y = - opt_row_trans - grasp_bias_y
    # Add up all compensations (Note Y direction need to flip due to img space (x, y))
    place_pos_x = compensate_x * heightmap_resolution + patch_align_x
    place_pos_y = compensate_y * heightmap_resolution + patch_align_y
    # Gripper roation angle (z axis is flipped from the workspace z axis)
    place_angle_rad = np.deg2rad(grasp_gripper_angle - opt_rot)
    compensate = np.asarray([place_angle_rad,
                             place_pos_x,
                             place_pos_y])
    print('X compensate', grasp_bias_x, opt_col_trans, compensate_x)
    print('Y compensate', grasp_bias_y, opt_row_trans,  compensate_y)
    return compensate, opt_rot
