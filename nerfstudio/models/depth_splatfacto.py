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

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.model_components import losses
from nerfstudio.model_components.losses import DepthLossType, basic_depth_loss, depth_ranking_loss, depth_uncertainty_weighted_loss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

from einops import repeat, reduce, rearrange
# from modified_diff_gaussian_rasterization import GaussianRasterizer as ModifiedGaussianRasterizer
# from modified_diff_gaussian_rasterization import GaussianRasterizationSettings

from modified_diff_gaussian_rasterization_depth import GaussianRasterizer as ModifiedGaussianRasterizer
from modified_diff_gaussian_rasterization_depth import GaussianRasterizationSettings

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig

@dataclass
class DepthSplatfactoModelConfig(SplatfactoModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: DepthSplatfactoModel)
    depth_loss_mult: float = 0.005
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.01
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = False
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    depth_loss_type: DepthLossType = DepthLossType.SIMPLE_LOSS
    """Depth loss type."""
    uncertainty_weight: float = 0.0
    """Weight of the uncertainty in the loss if uncertainty weighted loss is used."""

class DepthSplatfactoModel(SplatfactoModel):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: DepthSplatfactoModelConfig

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma
    
    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        
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
            if self.config.depth_loss_type in (DepthLossType.SIMPLE_LOSS,):
                metrics_dict["depth_loss"] = torch.Tensor([0.0]).to(self.device)
                
                termination_depth = batch["depth_image"].to(self.device)
                metrics_dict["depth_loss"] = basic_depth_loss(
                    termination_depth, outputs["depth"])
                
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
        
       