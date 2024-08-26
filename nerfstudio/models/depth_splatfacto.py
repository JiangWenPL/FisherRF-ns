# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union, Iterable

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal
from typing import Dict, List, Optional, Tuple, Type, Union, Iterable


from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.model_components import losses
from nerfstudio.model_components.losses import DepthLossType, EdgeAwareTV, TVLoss, basic_depth_loss, depth_ranking_loss, depth_uncertainty_weighted_loss, pearson_correlation_depth_loss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE
import torch.nn.functional as F


from einops import repeat, reduce, rearrange
# from modified_diff_gaussian_rasterization import GaussianRasterizer as ModifiedGaussianRasterizer
# from modified_diff_gaussian_rasterization import GaussianRasterizationSettings

from modified_diff_gaussian_rasterization_depth import GaussianRasterizer as ModifiedGaussianRasterizer
from modified_diff_gaussian_rasterization_depth import GaussianRasterizationSettings

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, random_quat_tensor


@dataclass
class DepthSplatfactoModelConfig(SplatfactoModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: DepthSplatfactoModel)
    depth_loss_mult: float = 0.001
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    use_depth_smooth_loss: bool = True
    """Whether to enable depth smooth loss or not"""
    smooth_loss_lambda: float = 10
    """Regularizer for smooth loss"""
    depth_sigma: float = 0.01
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = False
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    depth_loss_type: DepthLossType = DepthLossType.PEARSON_LOSS
    """Depth loss type."""
    uncertainty_weight: float = 0.0
    """Weight of the uncertainty in the loss if uncertainty weighted loss is used."""
    lift_depths_to_3d: bool = True
    """Whether to lift the depths to 3D."""

class DepthSplatfactoModel(SplatfactoModel):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: DepthSplatfactoModelConfig
    
    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        
        self.lifted_depths = np.zeros(100)
        self.new_gauss_params = None
        
        super().__init__(*args, **kwargs)
        

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma
    
    def depth_to_point_cloud(self, depth_image, intrinsics, R, t):
        """
        Convert a depth image to a point cloud using camera intrinsics and extrinsics.

        Parameters:
        depth_image (np.array): Depth image (H x W)
        intrinsics (np.array): Camera intrinsics matrix (3 x 3)
        extrinsics (np.array): Camera extrinsics matrix (4 x 4)

        Returns:
        np.array: Point cloud (N x 3)
        """
        # convert depth image to np
        depth_image = depth_image.cpu().numpy()
        depth_image = np.squeeze(depth_image)
        height, width = depth_image.shape
        
        # resize depth image to double
        depth_image = cv2.resize(depth_image, (2 * width, 2 * height), interpolation=cv2.INTER_NEAREST)
        cx = intrinsics[0, 2] * 2
        cy = intrinsics[1, 2] * 2
        fx = intrinsics[0, 0] * 2
        fy = intrinsics[1, 1] * 2
        
        points = []
        for v in range(depth_image.shape[0]):
            for u in range(depth_image.shape[1]):
                if depth_image[v, u] == 0 or depth_image[v, u] < 0:
                    continue  # Skip no depth or negative depth
                Z = depth_image[v, u] 
                if Z == 0: continue  # Skip no depth
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                points.append([X, Y, Z])
        
        t = np.array([[t[0]], [t[1]], [t[2]]])  # translation vector
        R = R @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # Example rotation matrix

        point_cloud = np.dot(R, np.transpose(points)) + t
        point_cloud = np.transpose(point_cloud)
        point_cloud_tensor = torch.tensor(point_cloud).to(self.device).float()
        
        # only take a subset of the points (0.1 percent randomly)
        num_points = point_cloud_tensor.shape[0]
        num_points_to_take = int(0.01 * num_points)
        indices = torch.randperm(num_points)[:num_points_to_take]
        point_cloud_tensor = point_cloud_tensor[indices]
            
        # lift gaussians to 3d
        means = point_cloud_tensor
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True).float()
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)
        
        features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
        features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        
        means = means.to(self.device)
        scales = scales.to(self.device)
        quats = quats.to(self.device)
        features_dc = features_dc.to(self.device)
        features_rest = features_rest.to(self.device)
        opacities = opacities.to(self.device)
        
        # create normals.
        normals = F.one_hot(torch.argmin(scales, dim=-1), num_classes=3).float()
        rots = quat_to_rotmat(quats)
        normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)
        normals = F.normalize(normals, dim=1)
        normals = torch.nn.Parameter(normals.detach())
        
        params = {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
                "normals": normals,
            }
        
        if self.config.learn_object_mask:
            sam_mask = torch.nn.Parameter(-10 * torch.ones(num_points, 1))
            sam_mask = sam_mask.to(self.device)
            params["sam_mask"] = sam_mask
        
        self.new_gauss_params = torch.nn.ParameterDict(params)
    
    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        
        if self.config.use_depth_smooth_loss:
            self.smooth_loss = TVLoss()
        
        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])
            
    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            if (
                losses.FORCE_PSEUDODEPTH_LOSS
                and self.config.depth_loss_type not in losses.PSEUDODEPTH_COMPATIBLE_LOSSES
            ):
                raise ValueError(
                    f"Forcing pseudodepth loss, but depth loss type ({self.config.depth_loss_type}) must be one of {losses.PSEUDODEPTH_COMPATIBLE_LOSSES}"
                )
            mask = None
            if "mask" in batch:
                # batch["mask"] : [H, W, 1]
                mask = self._downscale_if_required(batch["mask"])
                mask = mask.to(self.device)
                # invert mask
                mask = ~mask
                assert mask.shape[:2] == outputs["depth"].shape[:2] == outputs["depth"].shape[:2]
            mask = None
            if self.config.depth_loss_type in (DepthLossType.SIMPLE_LOSS,):
                metrics_dict["depth_loss"] = torch.Tensor([0.0]).to(self.device)
                
                termination_depth = batch["depth_image"].to(self.device)
                metrics_dict["depth_loss"] = basic_depth_loss(
                    termination_depth, outputs["depth"], mask)
                
            elif self.config.depth_loss_type in (DepthLossType.PEARSON_LOSS,):
                metrics_dict["depth_loss"] = torch.Tensor([0.0]).to(self.device)
                termination_depth = batch["depth_image"].to(self.device)
                metrics_dict["depth_loss"] = pearson_correlation_depth_loss(
                    termination_depth, outputs["depth"], mask)
                
            elif self.config.depth_loss_type in (DepthLossType.DEPTH_UNCERTAINTY_WEIGHTED_LOSS,):
                metrics_dict["depth_loss"] = torch.Tensor([0.0]).to(self.device)
                termination_depth = batch["depth_image"].to(self.device)
                termination_uncertainty = batch["depth_uncertainty"].to(self.device)
                
                metrics_dict["depth_loss"] = depth_uncertainty_weighted_loss(
                    None, termination_depth, outputs["depth"], termination_uncertainty, None, None, uncertainty_weight=self.config.uncertainty_weight)
                
            elif self.config.depth_loss_type in (DepthLossType.SPARSENERF_RANKING,):
                metrics_dict["depth_ranking"] = depth_ranking_loss(
                    outputs["depth"], batch["depth_image"].to(self.device)
                )
            else:
                raise NotImplementedError(f"Unknown depth loss type {self.config.depth_loss_type}")
        return metrics_dict

    def lift_depths_to_3d(self, camera: Cameras, batch, force=False):
        if self.training:
            if self.config.lift_depths_to_3d:
                termination_depth = batch["depth_image"].to(self.device)
                
                idx = (camera.metadata["cam_idx"]) # type: ignore
                # check if gaussians from depth image were lifted to 3D
                if self.lifted_depths[idx] == 0 or force:
                    self.lifted_depths[idx] = 1
                    print(f"Added gaussians to 3D-GS for camera {idx}")
                    optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
                    # get copy of optimized camera to world
                    optimized_camera_to_world = optimized_camera_to_world.clone()
                    # remove grad
                    optimized_camera_to_world = optimized_camera_to_world.detach()
                    extrinsics = optimized_camera_to_world
                    
                    fx = camera.fx.cpu().numpy()[0][0]
                    fy = camera.fy.cpu().numpy()[0][0]
                    cx = camera.cx.cpu().numpy()[0][0]
                    cy = camera.cy.cpu().numpy()[0][0]
                    intrinsics = np.array([[fx, 0, cx],
                                          [0, fy, cy],
                                          [0,  0,  1]])
                    
                    extrinsics = extrinsics.cpu().numpy()
                    extrinsics = extrinsics.squeeze()
                    extrinsics = np.vstack((extrinsics, np.array([0, 0, 0, 1])))
                    
                    # invert transformation matrix
                    R = extrinsics[:3, :3]
                    t = extrinsics[:3, 3]
                    # extrinsics[:3, :3] = R.T
                    # extrinsics[:3, 3] = -R.T @ t
                    
                    R = extrinsics[:3, :3]
                    t = extrinsics[:3, 3]
                    
                    # lift depths to 3d
                    self.depth_to_point_cloud(termination_depth, intrinsics, R, t)
                else: 
                    # first time this it hit, set self.new_gauss_params None
                    self.new_gauss_params = None
   
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            assert metrics_dict is not None and ("depth_loss" in metrics_dict or "depth_ranking" in metrics_dict)
            if "depth_ranking" in metrics_dict:
                loss_dict["depth_ranking"] = (
                    self.config.depth_loss_mult
                    * np.interp(self.step, [0, 2000], [0, 0.2])
                    * metrics_dict["depth_ranking"]
                )
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
        # if self.config.depth_loss_mult >= 0.005:
        #     self.config.depth_loss_mult = max(0.005, self.config.depth_loss_mult * 0.9995)
            if self.config.use_depth_smooth_loss:
                loss_dict["depth_smooth_loss"] = self.config.smooth_loss_lambda * self.smooth_loss(outputs["depth"])
            
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        metrics, images = super().get_image_metrics_and_images(outputs, batch)
        
        return metrics, images
        
       