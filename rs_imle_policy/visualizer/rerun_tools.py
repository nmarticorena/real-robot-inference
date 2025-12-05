from numpy.typing import NDArray
from typing import Optional
import rerun as rr
import spatialgeometry as sg
import trimesh
import roboticstoolbox as rtb
import numpy as np


class ReRunRobot:
    def __init__(self, robot: rtb.Robot, leader=False):
        self.robot = robot
        self.robot_name = robot.name
        self.leader_name = robot.name + "_leader"
        self.leader = leader
        self.load_meshes()

    def load_meshes(self):
        transforms = self.robot.fkine_all(self.robot.qr)
        for ix, link in enumerate(self.robot.links):
            R = transforms[ix + 1].R
            O_T_ix = transforms[ix + 1]
            if len(link.geometry) == 0:
                continue
            geom = link.geometry[0]
            if isinstance(geom, sg.Mesh):
                mesh_path = geom.filename
                mesh = trimesh.load_mesh(mesh_path, process=False)
                vertex_colour = mesh.visual.to_color().vertex_colors
                rr.log(
                    self.robot_name + "/" + link.name,
                    rr.Transform3D(translation=O_T_ix.t, mat3x3=R),
                )
                rr.log(
                    self.robot_name + "/" + link.name,
                    rr.Mesh3D(
                        vertex_positions=mesh.vertices,
                        triangle_indices=mesh.faces,
                        vertex_normals=mesh.vertex_normals,
                        vertex_colors=vertex_colour,
                    ),
                )
                if self.leader:
                    rr.log(
                        self.leader_name + "/" + link.name,
                        rr.Transform3D(translation=O_T_ix.t, mat3x3=R),
                    )
                    vertex_colour_leader = vertex_colour.copy()
                    vertex_colour_leader = vertex_colour_leader.reshape(-1, 4)
                    vertex_colour_leader[:, -1] = int(0.5 * 255)

                    rr.log(
                        self.leader_name + "/" + link.name,
                        rr.Mesh3D(
                            vertex_positions=mesh.vertices,
                            triangle_indices=mesh.faces,
                            vertex_normals=mesh.vertex_normals,
                            vertex_colors=vertex_colour_leader,
                        ),
                    )

    def step_robot(self, q, leader_q: Optional[NDArray] = None):
        transforms = self.robot.fkine_all(q)
        rr.log(self.robot_name + "/q", rr.Scalars(q))
        for ix, link in enumerate(self.robot.links):
            O_T_ix: NDArray[np.float64] = transforms[ix + 1].A
            rr.log(
                self.robot_name + "/" + link.name,
                rr.Transform3D(translation=O_T_ix[:3, 3], mat3x3=O_T_ix[:3, :3]),
            )
        if leader_q is not None:
            rr.log(self.leader_name + "/q", rr.Scalars(q))
            transforms = self.robot.fkine_all(leader_q)
            for ix, link in enumerate(self.robot.links):
                O_T_ix: NDArray[np.float64] = transforms[ix + 1].A
                rr.log(
                    self.leader_name + "/" + link.name,
                    rr.Transform3D(translation=O_T_ix[:3, 3], mat3x3=O_T_ix[:3, :3]),
                )


if __name__ == "__main__":
    robot = rtb.models.Panda()
    rr.init("urdf_test", recording_id="testing", spawn=True)
    rr.log(f"/{robot.name}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rerun_robot = ReRunRobot(robot)
    rerun_robot.load_meshes()
    for _ in range(20):
        rerun_robot.step_robot(rerun_robot.robot.random_q())
