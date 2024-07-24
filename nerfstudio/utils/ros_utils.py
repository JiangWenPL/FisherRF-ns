"""
ROS utils for converting between ROS and numpy
"""


from typing import Union

from geometry_msgs.msg import PoseStamped, Pose, Transform, TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as sciR


def convertPose2Numpy(pose:Union[PoseStamped, Pose]) -> np.ndarray:
        if isinstance(pose, PoseStamped):
            pose = pose.pose

        c2w = np.eye(4)
        quat = sciR.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        c2w[:3, :3] = quat.as_matrix()

        # position
        c2w[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

        return c2w