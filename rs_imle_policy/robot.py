"""Robot control module for Franka Panda robot.

This module provides a high-level interface for controlling the Franka Panda
robot using the frankx library. It handles motion control, gripper operations,
and state monitoring.
"""

from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from frankx import Affine, Gripper, JointMotion, Robot, Waypoint, WaypointMotion

from rs_imle_policy.utils import transforms

# Constants
DEFAULT_ROBOT_IP = "172.16.0.2"
DEFAULT_GRIPPER_SPEED = 0.1
DEFAULT_DYNAMIC_REL = 1.0
CARTESIAN_IMPEDANCE = [400.0, 400.0, 400.0, 40.0, 40.0, 40.0]
ACCEL_REL = 0.1
JERK_REL = 0.1


class GripperState(Enum):
    """Enumeration of gripper states."""
    
    OPEN = 0
    CLOSED = 1


class FrankxRobot:
    """High-level interface for Franka Panda robot control.
    
    This class provides methods for robot motion control, gripper operations,
    and state monitoring using the frankx library.
    
    Attributes:
        gripper: Gripper controller instance
        robot: Robot controller instance
        gripper_state: Current state of the gripper
        move_async: Async motion handle
        motion: Current motion instance
        X_BE: End-effector pose transformation matrix
        pos: Current end-effector position
        rot: Current end-effector rotation matrix
    """
    
    def __init__(
        self,
        ip: str = DEFAULT_ROBOT_IP,
        dynamic_rel: float = DEFAULT_DYNAMIC_REL,
    ):
        """Initialize the robot controller.
        
        Args:
            ip: Robot IP address
            dynamic_rel: Dynamic scaling factor for robot motion
        """
        self.gripper = Gripper(fci_ip=ip, speed=DEFAULT_GRIPPER_SPEED)
        self.gripper.open(True)
        self.gripper_state = GripperState.OPEN
        self.robot = Robot(ip, dynamic_rel=dynamic_rel, repeat_on_error=True)
        self.move_async = None
        self.motion = None

        self.X_BE = self.robot.get_pose()
        self.pos = self.X_BE[:3, 3]
        self.rot = self.X_BE[:3, :3]

    def get_gripper_width(self) -> float:
        """Get current gripper width.
        
        Returns:
            Current gripper width in meters
        """
        return self.gripper.width()

    def initialize_cartesian_impedance(self):
        """Initialize Cartesian impedance control parameters."""
        self.robot.set_cartesian_impedance(CARTESIAN_IMPEDANCE)
        self.robot.accel_rel = ACCEL_REL
        self.robot.jerk_rel = JERK_REL

    def move_to_start(self, home_config: Optional[NDArray]):
        """Move robot to starting configuration.
        
        Args:
            home_config: Joint configuration for home position, or None to skip
        """
        if home_config is not None:
            self.robot.move(JointMotion(goal=home_config))
        self.close_gripper()

    def close_gripper(self):
        """Close the gripper if not already closed."""
        if self.gripper_state != GripperState.CLOSED:
            self.gripper.close(blocking=False)
            self.gripper_state = GripperState.CLOSED

    def open_gripper(self):
        """Open the gripper if not already open."""
        if self.gripper_state != GripperState.OPEN:
            self.gripper.open(blocking=False)
            self.gripper_state = GripperState.OPEN

    def stop_motion(self, release: bool = True):
        """Stop current robot motion.
        
        Args:
            release: If True, open the gripper after stopping
        """
        if self.motion is not None:
            self.motion.finish()
        if release:
            self.open_gripper()

    def get_state(self) -> NDArray:
        """Get current robot state.
        
        Returns state vector containing end-effector position, rotation (6D),
        and gripper width.
        
        Returns:
            NDArray: shape (10,), [x, y, z, r1, r2, r3, r4, r5, r6, gripper_width]
            
        Raises:
            RuntimeError: If motion has not been initialized
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

        return np.concatenate([pos, rot, [width]])

    def set_next_waypoints(
        self, translations, orientations, relative: bool = False
    ):
        """Set the next waypoints for the robot to follow.
        
        Args:
            translations: Array of shape (N, 3) containing target positions
            orientations: Array of shape (N, 4) in quaternion format [w, x, y, z]
            relative: If True, waypoints are relative to current pose (not yet implemented)
        """
        waypoints = [
            Waypoint(Affine(trans[0], trans[1], trans[2], q[0], q[1], q[2], q[3]))
            for trans, q in zip(translations, orientations)
        ]
        self.motion.set_next_waypoints(waypoints)

    def init_waypoint_motion(self):
        """Initialize waypoint motion controller.
        
        Creates a waypoint motion starting from the current robot pose and
        starts asynchronous motion execution.
        """
        current_pose = self.robot.get_pose()
        self.motion = WaypointMotion(
            [Waypoint(goal=current_pose)],
            return_when_finished=False,
        )
        self.move_async = self.robot.move_async(self.motion)
