import os
import spatialmath as sm
import tqdm
import pims
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import importlib.util
import torch
import torch.nn.functional as F

from matplotlib.widgets import Slider


def get_config(config_file):
    config_file += ".py"
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one directory up
    config_file = os.path.join(script_dir, "configs", config_file)
    # config_file = os.path.join(script_dir, config_file)
    module_name = os.path.basename(config_file).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def adjust_hsv_ranges(image_path):
    # Load the image and convert to HSV
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Initial HSV ranges
    hsv_ranges = {
        "H_low": 0,
        "H_high": 255,
        "S_low": 130,
        "S_high": 216,
        "V_low": 150,
        "V_high": 230,
    }

    # Create figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.4)
    mask = cv2.inRange(
        hsv_img,
        (hsv_ranges["H_low"], hsv_ranges["S_low"], hsv_ranges["V_low"]),
        (hsv_ranges["H_high"], hsv_ranges["S_high"], hsv_ranges["V_high"]),
    )
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    masked_img = ax.imshow(result)

    # Define slider axes
    axcolor = "lightgoldenrodyellow"
    ax_h_low = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor=axcolor)
    ax_h_high = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_s_low = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_s_high = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_v_low = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_v_high = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    # Create sliders
    s_h_low = Slider(ax_h_low, "H Low", 0, 255, valinit=hsv_ranges["H_low"], valstep=1)
    s_h_high = Slider(
        ax_h_high, "H High", 0, 255, valinit=hsv_ranges["H_high"], valstep=1
    )
    s_s_low = Slider(ax_s_low, "S Low", 0, 255, valinit=hsv_ranges["S_low"], valstep=1)
    s_s_high = Slider(
        ax_s_high, "S High", 0, 255, valinit=hsv_ranges["S_high"], valstep=1
    )
    s_v_low = Slider(ax_v_low, "V Low", 0, 255, valinit=hsv_ranges["V_low"], valstep=1)
    s_v_high = Slider(
        ax_v_high, "V High", 0, 255, valinit=hsv_ranges["V_high"], valstep=1
    )

    # Update function for sliders
    def update(val):
        # Get current slider values
        h_low = s_h_low.val
        h_high = s_h_high.val
        s_low = s_s_low.val
        s_high = s_s_high.val
        v_low = s_v_low.val
        v_high = s_v_high.val

        # Create new mask and update the displayed image
        lower_bound = np.array([h_low, s_low, v_low])
        upper_bound = np.array([h_high, s_high, v_high])
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        masked_img.set_data(result)
        fig.canvas.draw_idle()

    # Connect update function to sliders
    s_h_low.on_changed(update)
    s_h_high.on_changed(update)
    s_s_low.on_changed(update)
    s_s_high.on_changed(update)
    s_v_low.on_changed(update)
    s_v_high.on_changed(update)

    # Show the plot
    plt.show()

    # Print final HSV values after closing the plot
    print(
        f"Final HSV Ranges: H:({s_h_low.val}, {s_h_high.val}), S:({s_s_low.val}, {s_s_high.val}), V:({s_v_low.val}, {s_v_high.val})"
    )


def cache_pims_video(dataset_path):
    # Open an HDF5 file in write mode
    with h5py.File("t_block_1.h5", "w") as h5f:
        episodes = sorted(
            os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x)
        )
        for episode in tqdm.tqdm(episodes):
            video_wrist = os.path.join(
                dataset_path, "episodes", episode, "video", "0.mp4"
            )
            video_side = os.path.join(
                dataset_path, "episodes", episode, "video", "2.mp4"
            )

            # Use PIMS to read the video files
            pims_video_wrist = pims.PyAVReaderIndexed(video_wrist)
            pims_video_side = pims.PyAVReaderIndexed(video_side)

            # Convert frames to NumPy arrays and resize
            wrist_frames = np.array(
                [cv2.resize(np.array(frame), (320, 240)) for frame in pims_video_wrist]
            )
            side_frames = np.array(
                [cv2.resize(np.array(frame), (320, 240)) for frame in pims_video_side]
            )

            # Create a group for each episode in HDF5
            grp = h5f.create_group(f"{episode}")

            # Chunking size is critical: Use (chunk_frames, height, width, channels)
            # Assuming typical access of 16 frames at a time.
            chunk_size = (16, 240, 320, 3)

            # Store wrist and side video frames in separate datasets with chunking
            grp.create_dataset(
                "wrist", data=wrist_frames, compression="gzip", chunks=chunk_size
            )
            grp.create_dataset(
                "side", data=side_frames, compression="gzip", chunks=chunk_size
            )

    print("Video data saved to video_data.h5")


def extract_robot_pos_orien(poses):
    xyz = []
    oriens = []
    for pose in poses:
        pose = np.array(pose)
        xyz.append(pose[:3, 3])
        rot = pose[:3, :3]
        oriens.append(matrix_to_rotation_6d(rot))
    return xyz, oriens


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)

    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if isinstance(d6, np.ndarray):
        d6 = torch.tensor(d6)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)


def rotation_6d_to_quat(rot: torch.Tensor) -> torch.Tensor:
    mat = rotation_6d_to_matrix(rot)
    return matrix_to_quaternion(mat)


def pos_rot_to_se3(pos: torch.Tensor, rot: torch.Tensor) -> sm.SE3:
    """Convert position and 6D rotation to SE3 object.

    Args:
        pos: Position tensor of shape (..., 3).
        rot: 6D rotation tensor of shape (..., 6).

    Returns:
        SE3 object representing the pose.
    """
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(pos)
    if not isinstance(rot, torch.Tensor):
        rot = torch.tensor(rot)

    if len(pos.shape) == 1:
        pose = np.eye(4)
    else:
        pose = np.repeat((np.eye(4)[None, :, :]), pos.shape[0], axis=0)

    rot_m = rotation_6d_to_matrix(rot).cpu().numpy()
    t = pos.cpu().numpy()

    pose[..., :3, :3] = rot_m
    pose[..., :3, 3] = t

    if len(pos.shape) > 1:
        se3 = [sm.SE3(p) for p in pose]
    else:
        se3 = sm.SE3(pose)

    return se3


def get_delta() -> sm.SE3:
    import panda_py
    import frankx
    import numpy as np
    from roboticstoolbox.models import Panda

    panda = Panda()

    q = panda.random_q()
    T_0p = sm.SE3(panda_py.fk(q), check=False)
    T_0f = sm.SE3(np.array(frankx.Kinematics.forward(q)).reshape(4, 4, order="F"))

    T_pf = T_0p.inv() * T_0f
    return T_pf


if __name__ == "__main__":
    # cache_pims_video('data/t_block_1')

    # Example usage
    image_path = "./evaluation/target_mask.png"
    adjust_hsv_ranges(image_path)
