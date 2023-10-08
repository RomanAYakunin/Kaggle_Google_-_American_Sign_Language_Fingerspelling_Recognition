import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import FeatureGenerator


class AugmentX(nn.Module):
    def __init__(self):
        super(AugmentX, self).__init__()
        self.FG = FeatureGenerator()

    @torch.no_grad()  # TODO try inference mode approach
    def forward(self, x):  # x.shape = [N, L, P, A]  # TODO keep making sure this actually works
        # TODO add perspective transform
        x = x.to(torch.float32)
        is_nan = (x == 0).to(x.dtype)
        x -= torch.sum((1 - is_nan) * x, dim=(1, 2), keepdim=True) / \
             (torch.sum((1 - is_nan), dim=(1, 2), keepdim=True) + 1e-5)
        y_correction = 1.5 / 0.9096226349071285  # TODO find new y_correction
        x[:, :, :, 1] *= y_correction

        # x = self.augment_angles(x, spec_angle_range=(0.4, 0.6), factor_range=(0.75, 1/0.75))
        # x = self.augment_lines(x, factor_range=(0.75, 1/0.75))  # TODO try re-enabling
        x, is_nan = self.flip(x, is_nan)  # TODO re-enable
        x = self.squish_stretch(x, factor_range=(0.75, 1/0.75))
        x = self.rotate(x, max_angle=0.3)
        # x, is_nan = self.squish_stretch_time(x, is_nan, factor_range=(0.75, 1.25))
        x = self.point_shift(x, max_shift=0.005)
        # x = self.frame_dropout(x, dropout=0.5)  # must do this last

        x[:, :, :, 1] /= y_correction
        x *= (1 - is_nan)
        return x

    def augment_angles(self, x, spec_angle_range, factor_range):  # [N, num_frames, num_points, num_axes]
        for angles in self.FG.angle_levels:  # [num_angles, 3]
            angle_coords = x[:, :, angles, :2]  # [N, num_frames, num_angles, 3, 2]
            s, c, m = angle_coords[..., 0, :], angle_coords[..., 1, :], angle_coords[..., 2, :]
            # shapes: [N, num_frames, num_angles, 2]
            s_vec, m_vec = s - c, m - c
            s_vec /= (torch.linalg.vector_norm(s_vec, dim=-1, keepdim=True) + 1e-5)
            m_vec /= (torch.linalg.vector_norm(m_vec, dim=-1, keepdim=True) + 1e-5)
            angle_values = torch.arccos(torch.clip(torch.sum(s_vec * m_vec, dim=-1),
                                                   min=-1, max=1))  # [N, num_frames, num_angles]
            signs = torch.det(torch.cat([s_vec.unsqueeze(-2), m_vec.unsqueeze(-2)], dim=-2))
            signs = torch.where(signs < 0, -torch.ones_like(signs), torch.ones_like(signs))

            spec_angles = spec_angle_range[0] + \
                          torch.rand(x.shape[0], 1, angles.shape[0], device=x.device) * \
                          (spec_angle_range[1] - spec_angle_range[0])  # [N, 1, num_angles]
            ampls = factor_range[0] + \
                    torch.rand(x.shape[0], 1, angles.shape[0], device=x.device) * \
                    (factor_range[1] - factor_range[0])  # [N, 1, num_angles]
            ampls = torch.clip(ampls * spec_angles, max=1)
            spec_angles *= math.pi
            ampls *= math.pi

            new_left = ampls / spec_angles * angle_values  # TODO can use factors for slope
            new_right_slope = (math.pi - ampls) / (math.pi - spec_angles)
            new_right = new_right_slope * (angle_values - spec_angles) + ampls
            new_angle_values = torch.where(angle_values < spec_angles, new_left, new_right)
            angle_deltas = new_angle_values - angle_values
            angle_deltas *= signs  # [N, num_frames, num_angles]

            anc_matrix = self.FG.anc_matrix[:, angles[:, -1]].to_sparse()  # [num_points, num_angles]

            angle_deltas = angle_deltas.reshape(-1, angle_deltas.shape[-1]).t()  # [num_angles, -1]
            angle_deltas = torch.sparse.mm(anc_matrix, angle_deltas)
            angle_deltas = angle_deltas.t().reshape(x.shape[0], x.shape[1], -1)  # [N, num_frames, num_points]
            nonzero_idxs = torch.argwhere((angle_deltas != 0).any(dim=0).any(dim=0)).reshape(-1)
            angle_deltas = angle_deltas[:, :, nonzero_idxs]  # [N, num_frames, num_nonzero]

            c = c.transpose(-1, -2)  # [N, num_frames, 2, num_angles]
            c = c.reshape(-1, c.shape[-1]).t()  # [num_angles, -1]
            c = torch.sparse.mm(anc_matrix, c)
            c = c.t().reshape(x.shape[0], x.shape[1], 2, -1).transpose(-1, -2)  # [N, num_frames, num_points, 2]
            c = c[:, :, nonzero_idxs, :]  # [N, num_frames, num_nonzero, 2]

            rot_matrices = torch.stack([
                angle_deltas.cos(), -angle_deltas.sin(),
                angle_deltas.sin(), angle_deltas.cos(),
            ])  # [4, N, num_frames, num_nonzero]
            rot_matrices = rot_matrices.permute(1, 2, 3, 0)  # [N, num_frames, num_nonzero, 4]
            rot_matrices = rot_matrices.reshape(rot_matrices.shape[0], rot_matrices.shape[1], rot_matrices.shape[2],
                                                2, 2)  # [N, num_frames, num_nonzero, 2, 2]

            m_vec = x[:, :, nonzero_idxs, :2] - c  # [N, num_frames, num_nonzero, 2]
            rot_m_vec = torch.matmul(rot_matrices, m_vec.unsqueeze(-1)).squeeze(-1)
            x[:, :, nonzero_idxs, :2] += rot_m_vec - m_vec
        return x

    def augment_lines(self, x, factor_range):
        x = x.transpose(2, 3)  # [N, num_frames, num_axes, num_points]

        lines = self.FG.lines  # [num_lines, 2]
        line_factors = factor_range[0] + \
                       torch.rand(x.shape[0], 1, 1, lines.shape[0], device=x.device) * (
                               factor_range[1] - factor_range[0])  # [N, 1, 1, num_lines]
        line_coords = x[:, :, :, lines]  # [N, num_frames, num_axes, num_lines, 2]
        s, m = line_coords[..., 0], line_coords[..., 1]  # [N, num_frames, num_axes, num_lines]
        deltas = (m - s) * (line_factors - 1)  # [N, num_frames, num_axes, num_lines]

        anc_matrix = self.FG.anc_matrix[:, lines[:, -1]].to_sparse()
        deltas = deltas.reshape(-1, deltas.shape[-1]).t()
        deltas = torch.sparse.mm(anc_matrix, deltas).t().reshape(x.shape)
        x += deltas

        x = x.transpose(2, 3)
        return x

    def flip(self, x, is_nan):
        p = torch.randint(high=2, size=(x.shape[0],), dtype=torch.long, device=x.device)  # 0 = don't flip, 1 = flip
        x[:, :, :, 0] *= -(2 * p - 1).reshape(-1, 1, 1)  # reflecting chosen samples
        reflect_arr = self.FG.reflect_arr
        flipped_samples = torch.argwhere(p).flatten()
        x[flipped_samples] = torch.index_select(x[flipped_samples],  # permuting indexes
                                                dim=2,
                                                index=reflect_arr)
        is_nan[flipped_samples] = torch.index_select(is_nan[flipped_samples],  # permuting indexes
                                                     dim=2,
                                                     index=reflect_arr)
        return x, is_nan

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

    def squish_stretch_time(self, x, is_nan, factor_range):  # [N, L, P, A]
        x *= (1 - is_nan)
        factor = (factor_range[0] + torch.rand(1) * (factor_range[1] - factor_range[0])).item()
        def interp(x):
            x_shape = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = F.interpolate(x.transpose(1, 2), size=(round(factor * x.shape[1])), mode='linear').transpose(1, 2)
            return x.reshape(x_shape[0], -1, x_shape[2], x_shape[3])
        nan_portion = interp((x == 0).to(x.dtype))
        x = interp(x)
        nan_threshold = 1 - 1e-5
        x /= (nan_portion < nan_threshold) * (1 - nan_portion) + (nan_portion >= nan_threshold).to(torch.float32)
        x = (nan_portion < nan_threshold) * x
        is_nan = (x == 0).to(x.dtype)
        return x, is_nan

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

    # def frame_dropout(self, x, dropout):
    #     x *= torch.rand(size=(x.shape[0], x.shape[1], 1, 1), device=x.device) > dropout
    #     return x


class AugmentY(nn.Module):
    def __init__(self, dataloader):
        super(AugmentY, self).__init__()
        self.train_counts = torch.zeros(59, dtype=torch.float32)
        self.supp_counts = torch.zeros(59, dtype=torch.float32)
        for batch in (pbar := tqdm(dataloader, file=sys.stdout)):
            pbar.set_description('collecting train/supp counts')
            for _, chunk_y in batch:
                y = chunk_y.clone().detach()
                train_y = y[y[:, 0] == 60].flatten()
                train_y = train_y[train_y < 59]
                supp_y = y[y[:, 0] == 61].flatten()
                supp_y = supp_y[supp_y < 59]
                self.train_counts.put_(train_y, torch.ones(train_y.shape, dtype=torch.float32), accumulate=True)
                self.supp_counts.put_(supp_y, torch.ones(supp_y.shape, dtype=torch.float32), accumulate=True)

    @torch.no_grad()
    def forward(self, y):
        return self.noise_labels(y, p=0.3)

    def noise_labels(self, y, p):
        noise_train = torch.multinomial(self.train_counts, num_samples=y.numel(), replacement=True).reshape(y.shape)
        noise_supp = torch.multinomial(self.supp_counts, num_samples=y.numel(), replacement=True).reshape(y.shape)
        noise_arr = torch.where((y[:, 0] == 60).unsqueeze(1).repeat(1, y.shape[1]),
                                noise_train.to(y.device),
                                noise_supp.to(y.device))
        noise_arr = torch.where(y < 59, noise_arr, y)  # preserves special tokens (SOT, EOT, gislr)
        noise_y = torch.where(torch.rand(y.shape, device=y.device) < p, noise_arr, y)
        return noise_y
