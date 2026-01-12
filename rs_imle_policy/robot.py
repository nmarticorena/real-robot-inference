"""
Pending: Add backends for other robots
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray
from enum import Enum

from frankx import (
    Robot,
    Waypoint,
    WaypointMotion,
    JointMotion,
    Affine,
    Gripper,
)

from rs_imle_policy.utils import transforms


class GripperState(Enum):
    OPEN = 0
    CLOSED = 1


class FrankxRobot:
    def __init__(self, ip: str = "172.16.0.2", dynamic_rel: float = 1):
        self.gripper = Gripper(fci_ip=ip, speed=0.1)
        self.gripper.open(True)
        self.gripper_state = GripperState.OPEN
        self.robot = Robot(ip, dynamic_rel=dynamic_rel, repeat_on_error=True)
        self.move_async = None
        self.motion = None

        self.X_BE = self.robot.get_pose()
        self.pos = self.X_BE[:3, 3]
        self.rot = self.X_BE[:3, :3]

    def get_gripper_width(self) -> float:
        return self.gripper.width()

    def initialize_cartesian_impedance(self):
        impedance = [400.0, 400.0, 400.0, 40.0, 40.0, 40.0]

        self.robot.set_cartesian_impedance(impedance)
        self.robot.accel_rel = 0.1
        self.robot.jerk_rel = 0.1

    def move_to_start(self, home_config: Optional[NDArray]):
        if home_config is not None:
            self.robot.move(JointMotion(goal=home_config))
        self.close_gripper()

    def close_gripper(self):
        if self.gripper_state != GripperState.CLOSED:
            self.gripper.close(blocking=False)
            self.gripper_state = GripperState.CLOSED

    def open_gripper(self):
        if self.gripper_state != GripperState.OPEN:
            self.gripper.open(blocking=False)
            self.gripper_state = GripperState.OPEN

    def stop_motion(self, release: bool = True):
        if self.motion is not None:
            self.motion.finish()
        if release:
            self.open_gripper()
        return

    def get_state(self) -> NDArray:
        """
        Get current robot state: [pos, rot_6d, gripper_width]
        Returns:
            NDArray: shape (10,), [x, y, z, r1, r2, r3, r4, r5, r6, gripper_width]
        """
        if self.motion is None:
            raise RuntimeError("Motion not initialized")
        state = self.motion.get_robot_state()

        X_BE = np.array(state.O_T_EE).reshape(4, 4, order="F")

        rot = transforms.matrix_to_rotation_6d(X_BE[:3, :3])
        pos = X_BE[:3, 3]
        width = self.get_gripper_width()

        # Store values for further use
        self.X_BE = X_BE
        self.pos = pos
        self.rot = X_BE[:3, :3]
        # rr.log(
        #     "/state/O_T_ee",
        #     rr.Transform3D(
        #         translation=X_BE[:3, 3], mat3x3=X_BE[:3, :3], axis_length=0.1
        #     ),
        # )
        return np.concatenate([pos, rot, [width]])

    def set_next_waypoints(self, translations, orientations, relative: bool = False):
        """
        Set the next waypoints for the robot to follow
        Args:
            translations (NDArray): shape (N, 3)
            orientations (NDArray): shape (N, 4) in quaternion format [w, x, y, z]
            relative (bool): whether the waypoints are relative to the current pose
        """
        waypoints = [
            Waypoint(Affine(trans[0], trans[1], trans[2], q[0], q[1], q[2], q[3]))
            for trans, q in zip(translations, orientations)
        ]
        self.motion.set_next_waypoints(waypoints)

    def init_waypoint_motion(self):
        """
        Initialize the waypoint motion
        """
        current_pose = self.robot.get_pose()
        self.motion = WaypointMotion(
            [Waypoint(goal=current_pose)],
            return_when_finished=False,
        )
        self.move_async = self.robot.move_async(self.motion)
