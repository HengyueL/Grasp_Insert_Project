# coding=utf-8
import os
import numpy as np
import torch
from network_models.grasp_model import RotModelRes
# import torch.nn.functional as F
# from sac_model import QNetwork, hard_update, GaussianPolicy, soft_update
# from torch.optim import Adam


class Trainer(object):
    def __init__(self,
                 device,
                 num_of_rotation,
                 load_model_dir=None,
                 is_eval=False):

        self.num_of_rotation = num_of_rotation  # 0 ----- 180
        self.iteration = 0
        self.device = device

        # Define the models
        self.grasp_model = RotModelRes(in_channel=1).to(device=self.device,
                                                        dtype=torch.float)
        if load_model_dir is not None:
            pre_trained_model = os.path.join(load_model_dir,
                                             'pre_trained.pth')
            if os.path.exists(pre_trained_model):
                #  Change the path here if the continue training
                self.grasp_model.load_state_dict(torch.load(pre_trained_model))
                print('Pre-trained model loaded')
                print(pre_trained_model)

        self.criterion = torch.nn.SmoothL1Loss()  # Huber loss
        self.criterion.to(device=self.device,
                          dtype=torch.float)

        # Set model to training mode
        if not is_eval:
            self.grasp_model.train()
            # Initialize optimizer
            self.grasp_optimizer = torch.optim.SGD([{'params': self.grasp_model.feature_trunk.parameters(), 'lr': 1e-5},
                                                    {'params': self.grasp_model.q_func_cnn.parameters()}
                                                    ],
                                                   lr=1e-4, momentum=0.9,
                                                   weight_decay=2e-5)
        else:
            self.grasp_model.eval()

        # Initialize lists to save execution info and RL variables
        self.grasp_loss_log = []
        self.grasp_successful_log = []

    def make_predictions(self,
                         image_patch,
                         output_size,
                         requires_grad=False):
        """
        :param image_patch: padded input (sqrt(2)) to enable rotation
        :param output_size: output that has reshaped to the action space
        :return:
        """
        model = self.grasp_model
        num_rotation = self.num_of_rotation

        q_value = []
        for i in range(num_rotation):
            # rot_idx = i
            if len(image_patch.shape) == 3:
                input_img = np.asarray([image_patch], dtype=float)
            else:
                input_img = image_patch.copy()

            # Formulate Input Tensor (Variable, Volatile = True)
            input_img = torch.from_numpy(input_img).permute(0, 3, 1, 2)
            input_img = input_img.to(device=self.device,
                                     dtype=torch.float)
            input_img.requires_grad = requires_grad

            # Feed Forward
            out = model.forward(input_img, rot_idx=i, num_rotations=num_rotation)
            out_prediction = out.data.detach().cpu().numpy()[0][0]
            pad_width = int(np.ceil((out_prediction.shape[0] - output_size)/2))
            q_value.append(out_prediction[pad_width:pad_width+output_size,
                                          pad_width:pad_width+output_size])
        return np.asarray(q_value)

    def get_grasp_label_value(self, ori_theta, pos_change, grasp_success):
        """
        :param ori_theta:  scale less than 30 deg
        :param pos_change:  scale less than 10 mm
        :param grasp_success:
        :return:
        """
        # Orientation version of labeling:
        if not grasp_success:
            return 0.
        # pos_score = np.exp(-abs(pos_change[0])) + np.exp(-abs(pos_change[1]))
        pos_score = np.exp(-0.07 * abs(np.sqrt(pos_change[0]**2 + pos_change[1]**2)))
        rot_score = np.exp(-0.04 * abs(ori_theta))
        score = pos_score * rot_score * 2
        return score

    def backprop_mask(self, img_input,
                      action_position,
                      label_value,
                      prev_mask,
                      output_size,
                      rot_idx=0):
        optimizer = self.grasp_optimizer
        optimizer.zero_grad()
        model = self.grasp_model
        num_rotation = self.num_of_rotation

        img_input = np.asarray([img_input], dtype=float)
        img_input = torch.from_numpy(img_input).permute(0, 3, 1, 2).to(device=self.device,
                                                                       dtype=torch.float)
        img_input.requires_grad = True
        prediction = model.forward(img_input,
                                   rot_idx=rot_idx,
                                   num_rotations=num_rotation)

        label_numpy = prediction.data.detach().cpu().numpy()
        # ----- Tuning the zero lable weights ----
        weight = 1e-1
        label_weights = weight * np.ones_like(label_numpy)
        # ----------------------------------------

        pad_width = int(np.ceil((label_numpy.shape[2] - output_size) / 2))
        label_numpy[0, 0,
                    pad_width:pad_width+output_size,
                    pad_width:pad_width+output_size] = np.multiply(label_numpy[0, 0,
                                                        pad_width:pad_width+output_size,
                                                        pad_width:pad_width+output_size], prev_mask)
        label_numpy[0, 0,
                    action_position[0] + pad_width,
                    action_position[1] + pad_width] = label_value
        label_weights[0, 0,
                      action_position[0] + pad_width,
                      action_position[1] + pad_width] = 1.

        label = torch.from_numpy(label_numpy).to(device=self.device,
                                                 dtype=torch.float)
        label_weights = torch.from_numpy(label_weights).to(device=self.device,
                                                           dtype=torch.float)

        loss = self.criterion(prediction, label) * label_weights
        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()
        optimizer.step()
        return loss_value









