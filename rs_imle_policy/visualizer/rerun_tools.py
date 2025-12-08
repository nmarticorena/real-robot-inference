from numpy.typing import NDArray
import time
import av
from typing import Optional
import rerun as rr
import spatialgeometry as sg
import trimesh
import roboticstoolbox as rtb
import numpy as np

from rs_imle_policy.configs.train_config import VisionConfig


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
                # rr.log(
                #     self.robot_name + "/" + link.name,
                #     rr.Transform3D(translation=O_T_ix.t, mat3x3=R),
                #     static=True,
                # )
                rr.log(
                    self.robot_name + "/" + link.name,
                    rr.Mesh3D(
                        vertex_positions=mesh.vertices,
                        triangle_indices=mesh.faces,
                        vertex_normals=mesh.vertex_normals,
                        vertex_colors=vertex_colour,
                    ),
                    static=True,
                )
                if self.leader:
                    # rr.log(
                    #     self.leader_name + "/" + link.name,
                    #     rr.Transform3D(translation=O_T_ix.t, mat3x3=R),
                    #     static=True,
                    # )
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
                        static=True,
                    )
 
    def initialize_video_stream(self, cams: VisionConfig):
        container = av.open("/dev/null", "w", format="hevc")
        self.streams = {
            cam_name: container.add_stream("libx265", rate = 1) # This needs to be the same obs
            for cam_name in cams.cameras
        }
        for name, stream in self.streams.items():
            assert isinstance(stream, av.video.stream.VideoStream)
            stream.width = cams.cameras_params[0].resolution[0]
            stream.height = cams.cameras_params[0].resolution[1]
            stream.max_b_frames = 0  # according to rerun docs
            fake_frame = np.zeros(
                (cams.cameras_params[0].resolution[1], cams.cameras_params[0].resolution[0], 3),
                dtype=np.uint8,
            )
            av_frame = av.VideoFrame.from_ndarray(fake_frame, format="rgb24")
            packet = stream.encode(av_frame)
            for packet in packet:
                rr.set_time("time", duration=float(pkt.pts * pkt.time_base))
                rr.log(
                    "video_stream/" + name,
                    rr.VideoStream(codec=rr.VideoCodec.H265, sample=bytes(packet)),
                )

    def log_frame(self, frame: NDArray, cam_name: str):
        """
        Log a frame of the camera feeds
        frame [w, h , channels]
        """
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        packet = self.streams[cam_name].encode(av_frame)
        for pkt in packet:
            rr.set_time("time", duration=float(pkt.pts * pkt.time_base))
            rr.log(
                "/video_stream/" + cam_name,
                rr.VideoStream(codec = rr.VideoCodec.H265,sample=bytes(pkt)),
            )

    def log_frame_encoded(self, frame: NDArray, cam_name:str):
        """
        Alternative approach until the av is working
        """
        rr.log(
            "/video_stream/" + cam_name,
            rr.Image(frame).compress()
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
