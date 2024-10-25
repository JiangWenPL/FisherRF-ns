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
import torch.nn.functional as F
from typing_extensions import Literal


from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.model_components.losses import TVLoss, mean_angular_error
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

from einops import repeat, reduce, rearrange

from modified_diff_gaussian_rasterization_depth import GaussianRasterizer as ModifiedGaussianRasterizer
from modified_diff_gaussian_rasterization_depth import GaussianRasterizationSettings

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def equal_hist(uncern):
    H, W = uncern.shape

    # Histogram equalization for visualization
    uncern = uncern.flatten()
    median = np.median(uncern)
    bins = np.append(np.linspace(uncern.min(), median, len(uncern)), 
                            np.linspace(median, uncern.max(), len(uncern)))
    # Do histogram equalization on uncern  
    # bins = np.linspace(uncern.min(), uncern.max(), len(uncern) // 20)
    hist, bins2 = np.histogram(uncern, bins=bins)
    # Compute CDF from histogram
    cdf = np.cumsum(hist, dtype=np.float64)
    cdf = np.hstack(([0], cdf))
    cdf = cdf / cdf[-1]
    # Do equalization
    binnum = np.digitize(uncern, bins, True) - 1
    neg = np.where(binnum < 0)
    binnum[neg] = 0
    uncern_aeq = cdf[binnum] * bins[-1]

    uncern_aeq = uncern_aeq.reshape(H, W)
    uncern_aeq = (uncern_aeq - uncern_aeq.min()) / (uncern_aeq.max() - uncern_aeq.min())
    return uncern_aeq 

def pcd_to_normal(xyz: Tensor):
    hd, wd, _ = xyz.shape
    bottom_point = xyz[..., 2:hd, 1 : wd - 1, :]
    top_point = xyz[..., 0 : hd - 2, 1 : wd - 1, :]
    right_point = xyz[..., 1 : hd - 1, 2:wd, :]
    left_point = xyz[..., 1 : hd - 1, 0 : wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(
        xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant"
    ).permute(1, 2, 0)
    return xyz_normal


def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5) -> Tensor:
    """Generates camera pixel coordinates [W,H]

    Returns:
        stacked coords [H*W,2] where [:,0] corresponds to W and [:,1] corresponds to H
    """

    # img size is (w,h)
    image_coords = torch.meshgrid(
        torch.arange(img_size[0]),
        torch.arange(img_size[1]),
        indexing="xy",  # W = u by H = v
    )
    image_coords = (
        torch.stack(image_coords, dim=-1) + pixel_offset
    )  # stored as (x, y) coordinates
    image_coords = image_coords.view(-1, 2)
    image_coords = image_coords.float()
    
    return image_coords
    


def get_means3d_backproj(
    depths: torch.Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    c2w: torch.Tensor,
    device: torch.device,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List]:
    """Backprojection using camera intrinsics and extrinsics

    image_coords -> (x,y,depth) -> (X, Y, depth)

    Returns:
        Tuple of (means: Tensor, image_coords: Tensor)
    """

    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
        c2w = c2w.float()
    if c2w.device != device:
        c2w = c2w.to(device)

    image_coords = get_camera_coords(img_size)
    image_coords = image_coords.to(device)  # note image_coords is (H,W)

    # TODO: account for skew / radial distortion
    means3d = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
    means3d[:, 2] = depths[:, 0]  # z

    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        means3d = means3d[mask]
        image_coords = image_coords[mask]

    if c2w is None:
        c2w = torch.eye((means3d.shape[0], 4, 4), device=device)

    # to world coords
    means3d = means3d @ torch.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
    return means3d, image_coords


def normal_from_depth_image(
    depths: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_size: tuple,
    c2w: torch.Tensor,
    device: torch.device,
    smooth: bool = False,
):
    """estimate normals from depth map"""
    if smooth:
        if torch.count_nonzero(depths) > 0:
            print("Input depth map contains 0 elements, skipping smoothing filter")
        else:
            kernel_size = (9, 9)
            depths = torch.from_numpy(
                cv2.GaussianBlur(depths.cpu().numpy(), kernel_size, 0)
            ).to(device)
    means3d, _ = get_means3d_backproj(depths, fx, fy, int(cx), int(cy), img_size, c2w, device)
    means3d = means3d.view(img_size[1], img_size[0], 3)
    normals = pcd_to_normal(means3d)
    return normals

to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)

@dataclass
class SplatfactoModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: SplatfactoModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = True
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    render_uncertainty: bool = False
    """whether or not to render uncertainty during GS training. NOTE: This will slow down training significantly."""
    depth_uncertainty_weight: float = 1.0
    """weight of depth uncertainty with the Hessian"""
    rgb_uncertainty_weight: float = 0.0
    """weight of rgb uncertainty with the Hessian"""
    max_gauss_ratio: float = 2.5
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""
    learn_object_mask: bool = True
    """If True, compute the Gaussians associated with an object based on the object mask"""
    predict_normals: bool = True
    """If True, predict normals for each gaussian"""
    normal_lambda: float = 0.2
    """Regularizer for normal loss"""


class SplatfactoModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SplatfactoModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        self.reduction_factor = 1

        super().__init__(*args, **kwargs)

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)
        self.normal_smooth_loss = TVLoss()

        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))
        
        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        
        # create normals.
        normals = F.one_hot(torch.argmin(scales, dim=-1), num_classes=3).float()
        rots = quat_to_rotmat(quats)
        normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)
        normals = F.normalize(normals, dim=1)
        normals = torch.nn.Parameter(normals.detach())
        
        param_dict = {
            "means": means,
            "scales": scales,
            "quats": quats,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacities": opacities,
            "normals": normals,
        }
        if self.config.learn_object_mask:
            # each gaussian has a single value for mask
            sam_mask = torch.nn.Parameter(-10 * torch.ones(num_points, 1))
            param_dict["sam_mask"] = sam_mask
            
        self.gauss_params = torch.nn.ParameterDict(param_dict)
        
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

    @property
    def normals(self):
        return self.gauss_params["normals"]
    

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]
    
    @property
    def sam_mask(self):
        return self.gauss_params["sam_mask"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            param_list = ["means", "scales", "quats", "features_dc", "features_rest", "opacities", "normals"]
            if self.config.learn_object_mask:
                param_list.append("sam_mask")
                
            for p in param_list:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param
        
    def dup_in_optim_add_gaussians(self, optimizer, num_gaussians, new_params):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        
        avg_shape = param_state["exp_avg"].shape
        sq_shape = param_state["exp_avg_sq"].shape
        
        # to add is based on the second and on dimensions of the existing exp_avg and exp_avg_sq
        if len(avg_shape) == 2:
            avg_add = torch.zeros((num_gaussians, avg_shape[1]), device=param.device)
        else:
            avg_add = torch.zeros((num_gaussians, avg_shape[1], avg_shape[2]), device=param.device)
        
        if "exp_avg" in param_state:
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    avg_add,
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    avg_add,
                ],
                dim=0,
            )
            
        
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n, add_gaussians=False):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            if group != "normals":
                if add_gaussians:
                    self.dup_in_optim_add_gaussians(optimizers.optimizers[group], dup_mask.shape[0], param)
                else:
                    self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, optimizers: Optimizers, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            assert self.xys.grad is not None
            grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )
            
        if self.new_gauss_params is not None:
            for name, param in self.gauss_params.items():
                self.gauss_params[name] = torch.nn.Parameter(
                    torch.cat([param.detach(), self.new_gauss_params[name]], dim=0) # type: ignore
                )
            idcs = torch.ones(self.new_gauss_params["means"].shape[0], dtype=torch.bool).to(self.device) # type: ignore
            
            self.dup_in_all_optim(optimizers, idcs, self.config.n_split_samples, add_gaussians=True)

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval > self.num_train_data + self.config.refine_every
            )
            if do_densification and self.xys_grad_norm is not None:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        if self.config.learn_object_mask:
            # step 3.5 sample new sam masks
            new_sam_masks = self.sam_mask[split_mask].repeat(samps, 1) # type: ignore
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        if self.config.learn_object_mask:
            out["sam_mask"] = new_sam_masks
            
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
                args=[training_callback_attributes.optimizers]
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        param_list = ["means", "scales", "quats", "features_dc", "features_rest", "opacities", "normals"]
        if self.config.learn_object_mask:
            param_list.append("sam_mask")
        return {
            name: [self.gauss_params[name]]
            for name in param_list
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        return image

    def get_background(self):
        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)
        return background

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        
        assert camera.shape[0] == 1, "Only one camera at a time"
        
        if self.training:
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else :
            optimized_camera_to_world = camera.camera_to_worlds

        background = self.get_background()
        
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}
        else:
            crop_ids = None
            
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = optimized_camera_to_world[0, :3, :3]  # 3 x 3 # type: ignore
        T = optimized_camera_to_world[0, :3, 3:4]  # 3 x 1 # type: ignore
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            if self.config.learn_object_mask:
                sam_masks_crop = self.sam_mask[crop_ids] # type: ignore
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            if self.config.learn_object_mask:
                sam_masks_crop = self.sam_mask
            
        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
        )  # type: ignore

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            rgb = background.repeat(H, W, 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)

            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - optimized_camera_to_world.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        sam_mask = None
        
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
            if self.config.learn_object_mask:
                sam_mask = sam_masks_crop * comp[:, None] # type: ignore
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
            if self.config.learn_object_mask:
                sam_mask = sam_masks_crop # type: ignore
                
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)
        
        # remove last dim of sam_mask
        if self.config.learn_object_mask:
            sam_mask = sam_mask.squeeze(-1) # type: ignore
            
        rgb, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )  # type: ignore
        alpha = alpha[..., None]
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        depth_im = None
        # perform depth rendering by 
        # use radii no grad
        # radii = self.radii.detach().clone().requires_grad_(False)
        # conics_depth = conics.detach().clone().requires_grad_(False)
        if self.config.output_depth_during_training or not self.training:
            depth_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths[:, None].repeat(1, 3),
                opacities,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())

        if self.config.learn_object_mask:
            mask = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                sam_mask[:, None].repeat(1, 3), # type: ignore
                opacities,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )
            mask = mask[..., 0:1]
            # take one channel for mask
        rgb_weight = self.config.rgb_uncertainty_weight
        depth_weight = self.config.depth_uncertainty_weight
        
        # normal rendering
        normals_im = torch.full(rgb.shape, 0.0)
        if self.config.predict_normals:
            quats_crop = quats_crop / quats_crop.norm(dim=-1, keepdim=True)
            normals = F.one_hot(
                torch.argmin(scales_crop, dim=-1), num_classes=3
            ).float()
            rots = quat_to_rotmat(quats_crop)
            normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)
            normals = F.normalize(normals, dim=1)
            viewdirs = (
                -means_crop.detach() + optimized_camera_to_world.detach()[..., :3, 3]
            )
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            dots = (normals * viewdirs).sum(-1)
            negative_dot_indices = dots < 0
            normals[negative_dot_indices] = -normals[negative_dot_indices]
            # update parameter group normals
            self.gauss_params["normals"] = normals
            # convert normals from world space to camera space
            normals = normals @ optimized_camera_to_world.squeeze(0)[:3, :3]
            
            xys = self.xys[0, ...].detach()
            
            normals_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                normals,
                opacities,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )
            # convert normals from [-1,1] to [0,1]
            normals_im = normals_im / normals_im.norm(dim=-1, keepdim=True)
            normals_im = (normals_im + 1) / 2
            
        if self.config.render_uncertainty:
            # render uncertainty as in FisherRF (ECCV 2024, Jiang et al.)
            uncertainties = self.render_uncertainty_rgb_depth([camera], [camera], rgb_weight=rgb_weight, depth_weight=depth_weight)
            uncertainty = uncertainties[0].unsqueeze(2)
            
        if self.config.render_uncertainty:
            return {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "background": background, "uncertainty": uncertainty, "mask": mask} # type: ignore
        else:
            return {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "background": background, "mask": mask, "normal": normals_im}  # type: ignore
    
    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        if len(outputs) == 0:
            return {}
        H_per_gaussian = torch.zeros(self.opacities.shape[0], device=self.opacities.device, dtype=self.opacities.dtype)
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]
        depth_out = outputs["depth"].detach()
        
        # SAM2 Mask section
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            # use the SAM2 mask to compute Gaussians associated with the mask

            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            
            # conver to binary mask or 1 or 0
            mask = (mask > 0.5).float()
            
            # compute binary cross entropy loss
            mask_loss = F.binary_cross_entropy_with_logits(outputs["mask"], mask)
        else:
            mask_loss = torch.tensor(0.0).to(self.device)  
        
        if "normals_image" in batch:
            batch["normals_image"] = batch["normals_image"] / 255
            batch["normals_image"] = (batch["normals_image"] * 2) - 1
            batch["normals_image"] = self.get_gt_img(batch["normals_image"])
            gt_normal = batch["normals_image"]
            
        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)
            
        main_loss = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss + (0.1 * mask_loss)
        
        # get normal loss
        normal_loss = 0
        if "normals_image" in batch and self.config.predict_normals:
            pred_normal = outputs["normal"]
            
            gt_normal = normal_from_depth_image(
                    depths=batch["depth_image"].cuda(),
                    fx=self.camera.fx.item(), # type: ignore
                    fy=self.camera.fy.item(), # type: ignore
                    cx=self.camera.cx.item(), # type: ignore
                    cy=self.camera.cy.item(), # type: ignore
                    img_size=(self.camera.width.item(), self.camera.height.item()), # type: ignore
                    c2w=torch.eye(4, dtype=torch.float, device=depth_out.device),
                    device=depth_out.device,
                    smooth=False,
                )
            gt_normal = gt_normal @ torch.diag(
                torch.tensor(
                    [1, -1, -1], device=depth_out.device, dtype=depth_out.dtype
                )
            )
            gt_normal = (1 + gt_normal) / 2
            
            if gt_normal is not None:
                # normal map loss
                normal_loss += torch.abs(gt_normal - pred_normal).mean()
                normal_loss += mean_angular_error(
                    pred=(pred_normal.permute(2, 0, 1) - 1) / 2,
                    gt=(gt_normal.permute(2, 0, 1) - 1) / 2,
                ).mean()
                
                normal_loss += self.normal_smooth_loss(pred_normal)
                
        loss_dict = {
            "main_loss": main_loss,
            "scale_reg": scale_reg,
            "normal_loss": self.config.normal_lambda * normal_loss,
        }
        
        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
        
        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore


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
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            predicted_rgb = outputs["rgb"]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict


    @torch.no_grad()
    def prepare_rasterizer(self, camera: Cameras) -> Tuple[ModifiedGaussianRasterizer, List[torch.Tensor]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {} #type: ignore
        assert camera.shape[0] == 1, "Only one camera at a time"
        
        if self.training:
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else :
            optimized_camera_to_world = camera.camera_to_worlds
        

        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = optimized_camera_to_world[0, :3, :3]  # 3 x 3
        T = optimized_camera_to_world[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            
            # sam mask
            if self.config.learn_object_mask:
                sam_masks_crop = self.sam_mask

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - optimized_camera_to_world.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        opacities = torch.sigmoid(opacities_crop)

        fovx = 2 * torch.atan(camera.width / (2 * camera.fx))
        fovy = 2 * torch.atan(camera.height / (2 * camera.fy))
        tanfovx = math.tan(fovx * 0.5)
        tanfovy = math.tan(fovy * 0.5)
        bg_color = torch.tensor([0., 0., 0.], dtype=torch.float32, device="cuda")
        scaling_modifier = 1.
        projmat = getProjectionMatrix(0.01, 100., fovx, fovy).cuda()

        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewmat.t(),
            projmatrix=viewmat.t() @ projmat.t(),
            sh_degree=n,
            campos=viewmat.inverse()[:3, 3],
            prefiltered=False,
            debug=False
        )
        rasterizer = ModifiedGaussianRasterizer(raster_settings=raster_settings)

        # Create temporary varaibles to avoid side effects of the backward engine
        # this also addresses the issues of normalization for quaterions
        means3D = means_crop.clone().requires_grad_(True)
        shs = colors_crop.clone().requires_grad_(True)
        opacities = opacities.clone().requires_grad_(True)
        scales = torch.exp(scales_crop.clone()).requires_grad_(True)
        rotations = quats_crop / quats_crop.norm(dim=-1, keepdim=True)
        rotations.requires_grad_(True)
        sam_masks = sam_masks_crop.clone().requires_grad_(True) if self.config.learn_object_mask else None

        params = [means3D, shs, opacities, scales, rotations, sam_masks]

        return rasterizer, params


    @torch.no_grad()
    def compute_diag_H_rgb_depth(self, camera: Cameras, compute_rgb_H=False, is_touch=False):
        """
        Compute diagonal hessian, on rgb or depth.

        return dict:
        rgb: rendering from 3D-GS renderer (H, W, C)
        depth: depth map from 3D-GS renderer (H, W)
        H: list of diag hessian on gaussians in the order of:
            means3D, shs, opacities, scales, rotations
        """
        rasterizer, params = self.prepare_rasterizer(camera)
        means3D, shs, opacities, scales, rotations, sam_masks = params

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        with torch.enable_grad():
            rendered_image, rendered_depth, radii = rasterizer(
                means3D=means3D,
                means2D=screenspace_points,
                shs=shs,
                colors_precomp=None,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None)
            if compute_rgb_H:
                rendered_image.backward(gradient=torch.ones_like(rendered_image))
            else:
                rendered_depth.backward(gradient=torch.ones_like(rendered_depth))
        # don't get grad of sam_masks, but use it to filter out
        cur_H = [p.grad.detach().clone() for p in params[:-1]] #type: ignore
        if is_touch:
            # only take grad of SAM2 masked object.
            sam_masks_prob = torch.sigmoid(sam_masks)
            sam_masks_prob = torch.where(sam_masks_prob > 0.5, torch.ones_like(sam_masks_prob), torch.zeros_like(sam_masks_prob))
            mask = sam_masks_prob.squeeze(-1) == 0
            
            for i in range(len(cur_H)):
                # set the grad to 0 if not part of the object
                cur_H[i][mask] = 0.
        
        for p in params:
            p.grad = None
            
        rgb = rearrange(rendered_image, "c h w -> h w c")
        
        return {"rgb": rgb, "H": cur_H, "depth": rendered_depth}  # type: ignore


    @torch.no_grad()
    def compute_EIG(self, train_cameras: Iterable[Cameras], candidate_cameras: List[Cameras]) -> torch.Tensor:
        """
        compute EIG for single camera
        """
        EIG = torch.zeros(len(candidate_cameras), device=candidate_cameras[0].device)

        H_train = None

        for train_cam in train_cameras:
            H_info = self.compute_diag_H_rgb_depth(train_cam)
            cur_H_train = torch.cat([p.reshape(-1) for p in H_info["H"]])
            H_train = H_train + cur_H_train if H_train is not None else cur_H_train

        REG_LAMBDA = 1e-6 # NOTE: might need to adjust this regularizer
        H_inv = torch.reciprocal(H_train + REG_LAMBDA)  #type: ignore
        for i, candidate_cam in enumerate(candidate_cameras):
            H_info = self.compute_diag_H_rgb_depth(candidate_cam)
            H_candidate = torch.cat([p.reshape(-1) for p in H_info["H"]])
            EIG[i] = torch.sum(H_inv * H_candidate) 

        return EIG
    
    
    @torch.no_grad()
    def render_uncertainty_rgb_depth(self, train_cameras: Iterable[Cameras], test_cameras: Iterable[Cameras], rgb_weight=1.0, depth_weight=1.0):
        H_per_gaussian = torch.zeros(self.opacities.shape[0], device=self.opacities.device, dtype=self.opacities.dtype)
        
        # go through provided training cameras
        for train_cam in train_cameras:
            # get rgb uncertainty
            H_info_rgb = self.compute_diag_H_rgb_depth(train_cam, compute_rgb_H=True)
            H_info_rgb['H'] = [p * rgb_weight for p in H_info_rgb['H']]
            H_per_gaussian += sum([reduce(p, "n ... -> n", "sum") for p in H_info_rgb['H']])
        
            # get depth uncertainty
            H_info_depth = self.compute_diag_H_rgb_depth(train_cam, compute_rgb_H=False)
            H_info_depth['H'] = [p * depth_weight for p in H_info_depth['H']]
            H_per_gaussian += sum([reduce(p, "n ... -> n", "sum") for p in H_info_depth['H']])
        
        hessian_color = repeat(H_per_gaussian.detach(), "n -> n c", c=3)
        uncern_maps = []
        for test_cam in test_cameras:
            rasterizer, params = self.prepare_rasterizer(test_cam)
            means3D, shs, opacities, scales, rotations, sam_masks = params

            # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
            screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass

            # cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params]) # type: ignore
            pts3d_homo = to_homo(means3D)
            pts3d_cam = pts3d_homo @ rasterizer.raster_settings.viewmatrix
            gaussian_depths = pts3d_cam[:, 2, None]

            cur_hessian_color = hessian_color * gaussian_depths.clamp(min=0)
            rendered_image, rendered_depth, radii = rasterizer(
                means3D=means3D,
                means2D=screenspace_points,
                shs=None,
                colors_precomp=cur_hessian_color,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None)
            
            # denormalize by rendered_depth
            rendered_image[0] = rendered_image[0]
            
            # plt.imshow(rendered_image[0].cpu().numpy())
            # plt.show()
            
            # plot rendered image
            # based on position of camera and bounding box on uncertainty, compute a crop of the image
            
            uncern_maps.append(rendered_image[0])

            # log_pix_uncern = torch.log(rendered_image[0]) # C-dim is the same            
            # import matplotlib.pyplot as plt
            # import seaborn as sns
            # sns.heatmap(log_pix_uncern.cpu(), square=True)
            # plt.savefig("/mnt/kostas_home/wen/tmp/ns-debug/log-uncern.png")
            # log_pix_uncern = torch.where(rendered_image > 0, torch.log(rendered_image), 0.)
            # breakpoint()

        return uncern_maps 