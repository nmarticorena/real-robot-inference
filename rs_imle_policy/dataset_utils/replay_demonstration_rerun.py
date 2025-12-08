import time
import roboticstoolbox as rtb
from rs_imle_policy.dataset import PolicyDataset, unnormalize_data
from rs_imle_policy.configs.train_config import LoaderConfig, VisionConfig
from rs_imle_policy.visualizer import rerun_tools

import rerun as rr
import tyro

args = tyro.cli(LoaderConfig)

vision_config = tyro.extras.from_yaml(
    VisionConfig, open(args.path / "vision_config.yml")
)

dataset = PolicyDataset(
    args.path,
    vision_config=vision_config,
    visualize=True,
)
rlds = dataset.rlds

rr.init("replay_demonstration_rerun", recording_id="test", spawn=True)
gui = rerun_tools.ReRunRobot(rtb.models.Panda(), leader= True)
gui.load_meshes()


for episode in rlds:
    ep_data = rlds[episode]
    ep_data_video = dataset.cached_dataset[str(episode)]

    for idx in range(len(ep_data["robot_pos"])):
        if idx % 2 != 0:
            continue

        gello_q = unnormalize_data(ep_data["gello_q"][idx], dataset.stats["gello_q"])
        robot_q = unnormalize_data(ep_data["robot_q"][idx], dataset.stats["robot_q"])

        progress = unnormalize_data(ep_data["progress"][idx], dataset.stats["progress"])

        gui.step_robot(robot_q, gello_q)
        for camera in vision_config.cameras:
            rr.log(camera, rr.Image(ep_data_video[camera][idx]))

        rr.log("action/progress", rr.Scalars(progress))

        time.sleep(0.1)  # 10 hz is the frecuency we loged the data
