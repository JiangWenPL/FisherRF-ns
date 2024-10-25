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
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import json
import cv2

import random
import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch
import torch.distributed as dist
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.ros_utils import convertPose2Numpy

from nerfstudio.cameras.cameras import Cameras

from einops import repeat, reduce, rearrange

from tqdm import tqdm
import copy

# ROS related imports
import rospy
from std_msgs.msg import String
from gaussian_splatting.srv import NBVPoses, NBVPosesRequest, NBVPosesResponse, NBVResultRequest, NBVResultResponse, NBVResult
from geometry_msgs.msg import PoseStamped, Pose, Transform, TransformStamped
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest

import numpy as np


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


class Pipeline(nn.Module):
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization. Each model
    class should be 1:1 with a pipeline that can act as a standardized interface and hide
    differences in how each model takes in and outputs data.

    This class's function is to hide the data manager and model classes from the trainer,
    worrying about:
    1) Fetching data with the data manager
    2) Feeding the model the data and fetching the loss
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes, which each may have different forward() methods
    and so on.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'train': loads train/eval datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    datamanager: DataManager
    _model: Model
    world_size: int

    @property
    def model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._model)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: Optional[bool] = None):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}

        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.model.load_state_dict(model_state, strict=False)
            else:
                raise

        super().load_state_dict(pipeline_state, strict=False)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.world_size > 1:
            assert self.datamanager.eval_sampler is not None
            self.datamanager.eval_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @abstractmethod
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """

    @abstractmethod
    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.
        """

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """

    @abstractmethod
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """


@dataclass
class VanillaPipelineConfig(InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VanillaPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=DataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""


class VanillaPipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        
        # ROS specific code
        rospy.init_node('nerf_pipeline', anonymous=True)
        rospy.loginfo("Starting the pipeline node!")
        self.touch_phase = False
        
        # import pdb; pdb.set_trace()

        # get random or fisher param
        self.view_selection_method = rospy.get_param('view_selection_method', 'fisher')

        # get views to add 
        self.views_to_add = int(rospy.get_param('views_to_add', 10)) # type: ignore
        
        self.touches_to_add = int(rospy.get_param('touches_to_add', 5)) # type: ignore
 
        self.added_views_so_far = 0
        self.added_touches_so_far = 0
        
        self.new_view_ready = False
        # create service to receive 
        self.continue_training_srv = rospy.Service('continue_training', Trigger, self.continue_training)
        
        # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device
    
    def continue_training(self, req: TriggerRequest) -> TriggerResponse:
        rospy.loginfo(f"Received request to continue training: {req}")
        self.new_view_ready = True
        
        # add new view to the training set
        res = TriggerResponse()
        res.success = True
        return res
    
    
    def fisher_single_view(self, training_views: List[int], candidate_views: List[np.ndarray],
                           rgb_weight=1.0, depth_weight=1.0, camera_info=None) -> Tuple[int, List[float]]:
        # construct initial Hessian matrix from the current training data
        H_train = None
        sample_cam: Cameras = None # type: ignore
        scale_factor = self.datamanager.train_dataparser_outputs.dataparser_scale  # type: ignore
        
        for view in training_views:
            # get full camera from view idx
            cam, batch = self.datamanager.get_cam_data_from_idx(view)
            if sample_cam is None:
                sample_cam = cam
                
            cur_H: torch.tensor = self.compute_hessian(cam, rgb_weight, depth_weight) # type: ignore
            
            if H_train is None:
                H_train = cur_H
            else:
                H_train += cur_H
                
        H_train = H_train.to(self.device) # type: ignore
        reg_lambda = 1e-6
        I_train = torch.reciprocal(H_train + reg_lambda)
        
        acq_scores = torch.zeros(len(candidate_views)) # type: ignore
        rospy.loginfo("Hessian computed! Computing NBV...")
        
        # go through each candidate camera and calculate the acq score
        for idx, view_pos in enumerate(tqdm(candidate_views, desc="Computing acq scores")):
            # copy sample_cam and update the pose
            cam = copy.deepcopy(sample_cam)
            if camera_info is not None:
                cam.fx[0][0] = camera_info['fx']
                cam.fy[0][0] = camera_info['fy']
                cam.cx[0][0] = camera_info['cx']
                cam.cy[0][0] = camera_info['cy']
                cam.width[0][0] = camera_info['w']
                cam.height[0][0] = camera_info['h']
                
                self.dt_cam = cam
                
            cam.metadata = None
            
            view_pos = torch.from_numpy(view_pos.astype(np.float32)).to(self.device)
            view_pos[:3, 3] *= scale_factor
            cam.camera_to_worlds = view_pos[:3, :4].unsqueeze(0)
            
            # if camera_info is not None, it is a touch view
            is_touch = camera_info is not None
            
            cur_H: torch.tensor = self.compute_hessian(cam, rgb_weight, depth_weight, is_touch=is_touch) # type: ignore
            I_acq = cur_H
            acq_scores[idx] = torch.sum(I_acq * I_train).item()

        # arbitrary scaling to have the scales be readable
        if camera_info is not None:
            acq_scores /= 100000
        else:
            acq_scores /= 1000000000000
        print(f"acq_scores: {acq_scores.tolist()}")
        
        # take max of acq score and return the view idx
        _, indices = torch.sort(acq_scores, descending=True)
        top_idx = indices[0]
        
        selected_idx = int(top_idx)
        print('Selected views:', selected_idx)
        return selected_idx, acq_scores.tolist()
        
    
    def view_selection(self, training_views: List[int], candidate_views: List[np.ndarray], option='random',
                       rgb_weight=1.0, depth_weight=1.0,
                       camera_info=None) -> Tuple[int, List[float]]: # type: ignore
        """
        Args:
            training_views (List[int]): List of training views by index.
            candidate_views (List[np.ndarray]): Candidate views to select from -- a list of poses.
            option (str, optional): The method of view selection. Either random or fisher. Defaults to 'random'.
            rgb_weight (float, optional): RGB weight for Fisherrf. Defaults to 1.0.
            depth_weight (float, optional): Depth weight for Fisherrf. Defaults to 1.0.
            is_touch (bool, optional): whether the view is a touch view. Defaults to False.

        Returns:
            Tuple[int, List[float]]: _description_
        """
        if option == 'random':
            # construct scores based on number of candidate views
            scores = [random.random() for _ in range(len(candidate_views))]
            # get argmax of scores
            max_idx = scores.index(max(scores))
            return max_idx, scores
        
        elif option == "fisher":
            # use fisher information to select the next view
            selected_view, acq_scores = self.fisher_single_view(training_views, candidate_views, rgb_weight, depth_weight,
                                                                camera_info=camera_info) # type: ignore
            return selected_view, acq_scores # type: ignore 

        elif option == "fisher-multi-view":
            # batch fisher information to select the next views
            scores = [random.random() for _ in range(len(candidate_views))]
            # get argmax of scores
            max_idx = scores.index(max(scores))
            return max_idx, scores
        else:
            # otherwise, select the first view in list. Not recommended.
            max_idx = 0
            scores = [100.0 - i for i in range(len(candidate_views))]
            return max_idx, scores
        
    def call_get_nbv_poses(self) -> List[np.ndarray]:
        rospy.wait_for_service('get_poses')
        rospy.loginfo("Calling service to get NBV poses")
        poses = []
        
        # make service call to get the list of avail poses
        try:
            get_nbv_poses = rospy.ServiceProxy('get_poses', NBVPoses)
            req = NBVPosesRequest()
            response: NBVPosesResponse = get_nbv_poses(req)
            if response.success:
                rospy.loginfo(f"Response Message: {response.message}")
                poses: List[PoseStamped] = response.poses
                ns_poses: List[np.ndarray] = [convertPose2Numpy(pose) for pose in poses] # type: ignore
                # transform to play nice with nerfstudio
                final_poses = []
                for pose in ns_poses:
                    new_pose = pose
                    new_pose[0:3, 1:3] *= -1
                    final_poses.append(new_pose)
            else: 
                rospy.loginfo("Failed to get poses. Adding no views")
            
        except rospy.ROSException as e:
            rospy.loginfo(f"Service call failed: {e}")
            
        return final_poses
    
    def send_nbv_scores(self, scores) -> bool:
        rospy.wait_for_service('receive_nbv_scores')
        poses = []
        
        # make service call to get the list of avail poses
        try:
            receive_nbv_scores_service = rospy.ServiceProxy('receive_nbv_scores', NBVResult)
            req = NBVResultRequest()
            req.scores = list(scores)
            response: NBVPosesResponse = receive_nbv_scores_service(req)
            
            if response.success:
                rospy.loginfo(f"Response Message: {response.message}")
                return True
            else: 
                rospy.loginfo("Failed send NBV scores. Adding no views")
            
        except rospy.ROSException as e:
            rospy.loginfo(f"Service call failed: {e}")
        return False
    
    def run_nbv(self, rgb_weight=1.0, depth_weight=1.0, is_touch=False, touch_poses=None,
                camera_info=None):
        if not is_touch:
            avail_views = self.call_get_nbv_poses()
        else:
            avail_views = touch_poses
        rospy.loginfo("Selecting new view for training")
        current_views_idxs = self.datamanager.get_current_views()
        
        next_view, acq_scores = self.view_selection(current_views_idxs, avail_views, option=self.view_selection_method, # type: ignore
                                                rgb_weight=rgb_weight, depth_weight=depth_weight, camera_info=camera_info) # type: ignore
        
        return next_view, acq_scores


    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        
        ray_bundle, batch = self.datamanager.next_train(step)
        
        self.model.lift_depths_to_3d(ray_bundle, batch) # type: ignore
        self.model.camera = ray_bundle # type: ignore
        if not self.touch_phase:
            model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        else: 
            # touch phase
            metrics_dict = self.model.get_metrics_dict({}, batch)
            loss_dict = self.model.get_loss_dict({}, batch, metrics_dict)
            model_outputs = {}
            model_outputs['depth'] = metrics_dict['depth']
        if step > 10000 and step % 100 == 99:
            if False:
            # if self.added_views_so_far < self.views_to_add:
                # add views to the training set if we can
                next_view, acq_scores = self.run_nbv(rgb_weight=0.0, depth_weight=1.0, is_touch=False)
                
                # send acquired scores in ROS
                success = self.send_nbv_scores(acq_scores)
                
                rate = rospy.Rate(1)  # 1 Hz    
                while not rospy.is_shutdown() and not self.new_view_ready:
                    rospy.loginfo("Waiting for Robot Node to Be Done...")
                    rate.sleep()
                rospy.loginfo("GS taking new view!")
                if success:
                    self.datamanager.add_new_view(next_view) # type: ignore
                    self.model.camera_optimizer.add_camera() # type: ignore
                    print("Added new view succesfully.")
                    # new view is not ready now
                    self.new_view_ready = False
                    self.added_views_so_far += 1

            else:
                if self.added_touches_so_far < self.touches_to_add:
                    # add new touch!! 
                    self.touch_phase = True
                    rospy.loginfo("Adding new touch view!")
                    
                    # for now, get from the list of touches
                    touch_data_dir = "/home/user/NextBestSense/data/ridge_touch/touch"
                    
                    with open(f'{touch_data_dir}/transforms.json', "r") as f:
                        data = json.load(f)
                        
                    import pdb; pdb.set_trace()
                        
                    frames = data['frames']
                    
                    poses = []
                    diff = 0.25  # 25 cm away from the touch in the negative z direction
                    orig_touch_poses = []
                    for frame in frames:
                        pose = frame['transformation']
                        pose_np = np.array(pose).reshape(4, 4)
                        # add poses along the z in the negative direction
                        z = pose_np[0:3, 2]
                        
                        orig_pose = pose_np.copy()
                        orig_pose[0:3, 1:3] *= -1
                        orig_touch_poses.append(orig_pose)
                        
                        # update the position in pose_np
                        pose_np[0:3, 3] -= (diff * z)
                        pose_np[0:3, 1:3] *= -1
                        poses.append(pose_np)
                        
                    camera_info  = {
                        'fx': 200,
                        'fy': 200,
                        'cx': 320,
                        'cy': 320,
                    }
                    camera_info['h'] = data['h']
                    camera_info['w'] = data['w']
                        
                    # get next best touch
                    next_touch, acq_scores = self.run_nbv(rgb_weight=0.0, depth_weight=1.0, is_touch=True, 
                                                        touch_poses=poses, camera_info=camera_info)
                    
                    # get depth of the touch
                    touch_file_path = frames[next_touch]['file_path']
                    touch_pose = poses[next_touch]
                    touch_file_path = touch_file_path.split('/')[-1]
                    touch_file_path = touch_file_path.split('.')[0]
                    touch_file_path = f"{touch_data_dir}/{touch_file_path}_zmap.png"
                    
                    # read in depth
                    depth = cv2.imread(touch_file_path, cv2.IMREAD_UNCHANGED) / 1000.0
                    depth = depth / 1000.0
                    
                    # set low values to zero
                    depth[depth < 0.005] = 0.0
                    
                    # we have computed the best touch, now add it
                    # real surface is higher than current surface: add the gaussians. no need to squash
                    # start from the touch pose and slowly move the camera to the actual touch pose
                    is_real_surface_lower = True
                    
                    # if trajectory is stored, break up trajectory into discrete sections and prune Gaussians
                    if is_real_surface_lower:
                        import pdb; pdb.set_trace()
                        ros_touch_pose = poses[next_touch]
                        ros_touch_pose[0:3, 1:3] *= -1
                        z = ros_touch_pose[0:3, 2]
                        top_position = ros_touch_pose[0:3, 3]
                        bottom_position = top_position + (0.25 * z)
                        
                        # prune the Gaussians in the cylinder from n meters out to the touch
                        self.model.prune_gaussians(top_position, bottom_position, 0.03) # type: ignore
                        
                    sample_cam, _ = self.datamanager.get_cam_data_from_idx(0) # type: ignore
                    # copy the camera
                    dt_cam = copy.deepcopy(sample_cam)
                    dt_cam.metadata = None 
                    dt_cam.cx[0][0] = camera_info['cx']
                    dt_cam.cy[0][0] = camera_info['cy']
                    dt_cam.fx[0][0] = camera_info['fx']
                    dt_cam.fy[0][0] = camera_info['fy']
                    dt_cam.width = 640
                    dt_cam.height = 640
                    
                    touch_pose_torch = torch.from_numpy(touch_pose.astype(np.float32)).to(self.device)
                    dt_cam.camera_to_worlds = touch_pose_torch[:3, :4].unsqueeze(0)
                    depth = torch.tensor(depth).to(self.device)
                    self.model.add_touch_cam(dt_cam, depth) # type: ignore
                    
                    # now add the Gaussians for touch! These Gaussians are less likely to be culled as they are reasonably close to the surface
                    # self.model.add_touch_gaussians(touch_pose, camera_info, depth) # type: ignore
                    self.added_touches_so_far += 1
                    
                    # add to touch dataset with dt cam and provided depth
                    # self.model.add_touch_cam(self.dt_cam, depth) # type: ignore
                    
                    
                    # add a new touch camera to GS
                    
                    # update all depth maps with the touch depth
                    
        return model_outputs, loss_dict, metrics_dict
    
    
    def compute_hessian(self, ray_bundle, rgb_weight, depth_weight, is_touch=False):
        if hasattr(self.model, 'compute_diag_H_rgb_depth') and callable(getattr(self.model, 'compute_diag_H_rgb_depth')):
            # compute the Hessian
            H_info_rgb = self.model.compute_diag_H_rgb_depth(ray_bundle, compute_rgb_H=True, is_touch=is_touch) # type: ignore
            H_info_rgb['H'] = [p * rgb_weight for p in H_info_rgb['H']]
            H_per_gaussian = sum([reduce(p, "n ... -> n", "sum") for p in H_info_rgb['H']])
            
            H_info_depth = self.model.compute_diag_H_rgb_depth(ray_bundle, compute_rgb_H=False, is_touch=is_touch) # type: ignore
            H_info_depth['H'] = [p * depth_weight for p in H_info_depth['H']]
            H_per_gaussian += sum([reduce(p, "n ... -> n", "sum") for p in H_info_depth['H']])
            return H_per_gaussian   

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
            # disable=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    raise NotImplementedError("Saving images is not implemented yet")

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
