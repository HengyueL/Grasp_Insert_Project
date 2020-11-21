from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False):
    """3x3 convolution"""
    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )


def initialize_weights(initializer):
    def initialize(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            initializer(m.weight)
            if m.bias is not None:
                torch.nn.init.constant(m.bias, 0)
    return initialize


def create_linear_network(input_dim,
                          output_dim,
                          hidden_units=[],
                          output_activation=None):
    model = []
    units = input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units))
        model.append(nn.ReLU())
        units = next_units

    model.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        model.append(nn.ReLU())

    return nn.Sequential(*model).apply(
        initialize_weights(nn.init.xavier_normal))


class ResModel(nn.Module):
    def __init__(self, in_channels):
        super(ResModel, self).__init__()
        resnet = list(torchvision.models.resnet18(pretrained=True).children())[:-2]
        resnet[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.kaiming_normal(resnet[0].weight.data)
        self.resnet = nn.Sequential(*resnet)

    def forward(self, x):
        return self.resnet(x)


class RotModelRes(nn.Module):
    def __init__(self, in_channel=1):
        super(RotModelRes, self).__init__()
        print('ResNet Created')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.feature_trunk = ResModel(in_channel)
        feature_channel = 512  # Change based on the resmodel size
        self.q_func_cnn = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(feature_channel)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(feature_channel, 64, kernel_size=1, stride=1, bias=False)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

    def forward(self, input_img, rot_idx=0, num_rotations=1):
        """
        :param input_img: (3, 300, 300)
        :param rot_idx: evenly divide pi
        """
        input_img = F.interpolate(input_img,
                                  scale_factor=2,
                                  mode='nearest')
        rot_theta = np.radians(rot_idx * 360. / num_rotations)
        affine_mat_before = np.asarray([[np.cos(-rot_theta), np.sin(-rot_theta), 0],
                                        [-np.sin(-rot_theta), np.cos(-rot_theta), 0]])
        affine_mat_before.shape = (2, 3, 1)
        affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).to(device=self.device,
                                                                                    dtype=torch.float)
        flow_grid_before = F.affine_grid(affine_mat_before,
                                         input_img.size())
        rot_img = F.grid_sample(input_img,
                                flow_grid_before,
                                mode='nearest')

        interm_feature = self.feature_trunk(rot_img)

        affine_mat_after = np.asarray([[np.cos(rot_theta), np.sin(rot_theta), 0],
                                       [-np.sin(rot_theta), np.cos(rot_theta), 0]])
        affine_mat_after.shape = (2, 3, 1)
        affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).to(device=self.device,
                                                                                  dtype=torch.float)
        flow_grid_after = F.affine_grid(affine_mat_after,
                                        interm_feature.size())
        rot_feature = F.grid_sample(interm_feature,
                                    flow_grid_after,
                                    mode='nearest')
        x = self.q_func_cnn(rot_feature)
        return x


#  # ------ Test Output Dimension ----------
# input_shape = (1, 160, 160)
# a = RotModelRes().cuda()
#
# bs = 1
# input_1 = torch.rand(bs, *input_shape).cuda()
# output_feat = a(input_1, rot_idx=1).cpu()
# n_size = output_feat.data.size()
# # torch.cuda.m
# print(n_size)
# pass
