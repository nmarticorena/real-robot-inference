import roboticstoolbox as rtb
import swift
import spatialgeometry as sg
import spatialmath as sm
import numpy as np


class RobotViz:
    def __init__(self, action_horizon = 1, teleop = True):
        self.env = swift.Swift()
        self.env.launch()
        self.robot = rtb.models.Panda()
        self.env.add(self.robot, robot_alpha=1.0)
        if teleop:
            self.gello = rtb.models.Panda()
            self.env.add(self.gello, robot_alpha=0.5)
        

        self.ee_pose = sg.Axes(0.1, pose=sm.SE3(1, 1, 1))
        self.env.add(self.ee_pose)

        self.action_viz = []
        for _ in range(action_horizon):
            ap = sg.Axes(0.05, pose=sm.SE3(1, 1, 1))
            self.action_viz.append(ap)
            self.env.add(ap)

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
    def update_actions(self, action_poses):
        for ap, pose in zip(self.action_viz, action_poses):
            ap.T = pose

    def step(self, q, gello_q=None):
        # self.robot.q = q
        if gello_q is not None:
            self.gello.q = gello_q
        self.env.step()
