import roboticstoolbox as rtb
import swift
import spatialgeometry as sg
import spatialmath as sm
import numpy as np


class RobotViz:
    def __init__(self):
        self.env = swift.Swift()
        self.env.launch()
        self.robot = rtb.models.Panda()
        self.gello = rtb.models.Panda()
        self.env.add(self.robot, robot_alpha=0.5)
        self.env.add(self.gello, robot_alpha=0.5)
        self.object_pose = sg.Axes(0.1, pose=sm.SE3(1, 1, 1))
        self.policy_pose = sg.Axes(0.3, pose=sm.SE3(1, 1, 1))
        # self.policy_pose = sg.Arrow(0.3,0.005, pose = sm.SE3(1,1,1), color = np.random.rand(3))
        self.ee_pose = sg.Axes(0.1, pose=sm.SE3(1, 1, 1))
        self.orientation_frame = sg.Axes(0.3, pose=sm.SE3(1, 1, 1))
        self.cup_handle = sg.Axes(0.1, pose=sm.SE3(1, 1, 1))

        self.env.add(self.object_pose)
        self.env.add(self.ee_pose)
        self.env.add(self.policy_pose)
        self.env.add(self.orientation_frame)
        self.env.add(self.cup_handle)

        self.robot.grippers[0].q = [0.03, 0.03]

        # tranform to finray tcp
        X_FE = np.array(
            [
                [0.70710678, 0.70710678, 0.0, 0.0],
                [-0.70710678, 0.70710678, 0, 0],
                [0.0, 0.0, 1.0, 0.2],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.X_FE = sm.SE3(X_FE, check=False).norm()
        # import pdb; pdb.set_trace()

    def step(self, q, gello_q=None):
        self.robot.q = q
        if gello_q is not None:
            self.gello.q = gello_q
        self.env.step()
