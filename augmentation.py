import math
import torch
import torch.nn as nn
from dataset import FeatureGenerator
from copy import deepcopy
from model import center_x


class AugmentBatch(nn.Module):
    def __init__(self):
        super(AugmentBatch, self).__init__()
        self.FG = FeatureGenerator()

    @torch.no_grad()
    def forward(self, x):  # x.shape = [N, num_frames, num_features]  # TODO keep making sure this actually works
        # TODO add perspective transform
        is_nan = (x == 0).to(x.dtype)
        y_correction = 1.5 / 0.9096226349071285  # TODO find new y_correction
        x[..., 1] *= y_correction

        # x = self.augment_angles(x, spec_angle_range=(0.4, 0.6), factor_range=(0.75, 1/0.75))
        # x = self.augment_lines(x, factor_range=(0.75, 1/0.75))
        x = self.squish_stretch(x, factor_range=(0.75, 1/0.75))
        x = self.rotate(x, max_angle=0.3)  # TODO try larger rotation
        x = self.point_shift(x, max_shift=0.005)
        # x = self.frame_dropout(x, dropout=0.5)  # must do this last

        x[..., 1] /= y_correction
        x = (1 - is_nan) * x
        return x

    def squish_stretch(self, x, factor_range):
        factor = factor_range[0] + \
                 torch.rand(x.shape[0], 1, 1, self.FG.num_axes, dtype=x.dtype, device=x.device) * \
                 (factor_range[1] - factor_range[0])
        x *= factor
        return x

    def rotate(self, x, max_angle):  # angle in radians
        angles = 2 * (torch.rand(x.shape[0], dtype=x.dtype, device=x.device) - 0.5) * max_angle
        rot_matrices = torch.stack([
            torch.cos(angles), -torch.sin(angles),
            torch.sin(angles), torch.cos(angles),
        ])  # shape [4, N]
        rot_matrices = rot_matrices.reshape(2, 2, -1)  # shape [2, 2, N]
        rot_matrices = torch.permute(rot_matrices, dims=(2, 1, 0))  # shape [N, 2, 2], also fixes rows vs columns
        rot_matrices = rot_matrices.unsqueeze(1).unsqueeze(1)  # shape [N, 1, 1, 2, 2]
        x[:, :, :, :2] = torch.matmul(rot_matrices, x[:, :, :, :2].unsqueeze(-1)).squeeze(-1)
        return x

    # def p_warp(x, factor_range):
    #     normal = lambda x: 1/
    #     weights = factor_range[0] + torch.rand(x.shape[:2], device=x.device) * (factor_range[1] - factor_range[0])
    #     kernel = torch.Tensor([])
    #     weights = torch.cat(
    #
    #     )

    def point_shift(self, x, max_shift):  # TODO noise proportionally to std
        x += 2 * (torch.rand(x.shape[0], 1, x.shape[2], self.FG.num_axes, dtype=x.dtype, device=x.device) - 0.5) * \
             max_shift
        return x  # TODO different noises for different body parts and axes? don't noise z-axis?

    def frame_dropout(self, x, dropout):
        x *= torch.rand(size=(x.shape[0], x.shape[1], 1, 1), device=x.device) > dropout
        return x