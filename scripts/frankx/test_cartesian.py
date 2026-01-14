from frankx import Affine, WaypointMotion, Waypoint, Robot, JointMotion
import numpy as np
import os

breakpoint()

robot = Robot(
    os.environ["PANDA_IP"], repeat_on_error=True, user="franka", password="franka123"
)
# robot.recover_from_errors()
robot.set_dynamic_rel(0.05)
robot.move(JointMotion(np.deg2rad([-90, 0, 0, -90, 0, 90, 45])))


front = Waypoint(Affine(0.1, 0.0, 0.0), Waypoint.Relative)
up = Waypoint(Affine(0.0, 0.0, 0.1), Waypoint.Relative)
back = Waypoint(Affine(-0.1, 0.0, 0.0), Waypoint.Relative)
down = Waypoint(Affine(0.0, 0.0, -0.1), Waypoint.Relative)

# I should be in the same original position

motion = WaypointMotion([front, up, back, down])
robot.move(motion)

print("Final configuration (radians):", robot.get_joint_positions())
print("Final config (degrees):", np.rad2deg(robot.get_joint_positions()))

robot.disconnect()
