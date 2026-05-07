"""Viser visualization for InterHand retargeted datasets.

Visualizes:
- Human hand joints from HDF5 key `world_hand_joints`.
- Robot hand from URDF under $DEX_URDF_PATH, using per-demo `robot_type` and `mano_side`.

The robot and human are shown in the *same* world coordinate frame:
- Human joints are rendered in `world_hand_joints` coordinates.
- Robot base pose is taken from `Ts_world_root` (4x4 world-from-robot-root).
- Robot joint configuration is taken from `robot_joint_pose` (normalized 0..1) and
  denormalized with [`calibration_net.utils.AngleNormalizer`](calibration_net/utils.py:300).

Example
-------
python \
  visualization/visualize_inter.py \
  --dataset-path visualization/InterHand_retargeted_30fps_signed_aug_shape.hdf5 \
  --port 8091
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import tyro
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from calibration_net.utils import AngleNormalizer


_DEBUG = os.environ.get("VISER_INTERHAND_DEBUG", "") not in ("", "0", "false", "False")


def _log(*args: object) -> None:
    if _DEBUG:
        print("[visualize_inter]", *args, flush=True)


DEX_URDF_PATH = os.environ.get("DEX_URDF_PATH", "")


MANO_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _decode_scalar(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.ndarray):
        # Common HDF5 patterns:
        # - object scalar: array( b'omni', dtype=object )
        # - length-1: array([b'right'], dtype=object)
        if x.shape == ():
            return _decode_scalar(x[()])
        if x.size == 1:
            return _decode_scalar(x.reshape(-1)[0])
    return str(x)


def _make_bone_segments(joints: np.ndarray) -> np.ndarray:
    """(21,3) -> (E,2,3) for viser line segments."""
    segs = []
    for a, b in MANO_CONNECTIONS:
        segs.append(np.stack([joints[a], joints[b]], axis=0))
    return np.stack(segs, axis=0)


def _load_urdf(robot_type: str, side: str) -> yourdfpy.URDF:
    if not DEX_URDF_PATH:
        raise EnvironmentError(
            "DEX_URDF_PATH is not set. Please export DEX_URDF_PATH to point to the dex-urdf repository."  # noqa: E501
        )

    robot_urdf_path = (
        Path(DEX_URDF_PATH)
        / "robots"
        / "hands"
        / f"{robot_type}_hand"
        / f"{robot_type}_hand_{side}.urdf"
    )

    def filename_handler(fname: str) -> str:
        base_path = robot_urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)

    return yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)


def _extract_robot_cfg(
    grp: h5py.Group,
    frame_idx: int,
    normalizer: Optional[AngleNormalizer],
) -> Optional[np.ndarray]:
    """Return robot joint angles for a single frame, or None if missing."""
    if "denormed_robot_joint_pose" in grp:
        return np.asarray(grp["denormed_robot_joint_pose"][frame_idx], dtype=np.float32)

    # The retarget pipeline writes normalized `robot_joint_pose`.
    if "robot_joint_pose" in grp:
        if normalizer is None:
            raise RuntimeError(
                "robot_joint_pose is present but no normalizer was provided; cannot denormalize."  # noqa: E501
            )
        q01 = np.asarray(grp["robot_joint_pose"][frame_idx], dtype=np.float32)
        return np.asarray(normalizer.denormalize(q01), dtype=np.float32)

    # Fallbacks to be tolerant to other datasets.
    if "qpos" in grp:
        return np.asarray(grp["qpos"][frame_idx], dtype=np.float32)

    return None


@dataclass
class Args:
    dataset_path: str
    port: int = 8091

    demo_id: Optional[str] = None

    # Playback
    fps: float = 30.0

    # Rendering
    show_grid: bool = True
    show_axes: bool = False
    point_size: float = 0.004
    line_width: float = 2.0


def main(args: Args) -> None:
    f = h5py.File(args.dataset_path, "r")

    demo_keys = list(f.keys())
    try:
        demo_keys = sorted(demo_keys, key=lambda k: int(k))
    except Exception:
        demo_keys = sorted(demo_keys)

    if len(demo_keys) == 0:
        raise ValueError(f"No demos found in {args.dataset_path}")

    if args.demo_id is not None:
        if args.demo_id not in f:
            raise KeyError(
                f"demo_id '{args.demo_id}' not found. Available: {demo_keys[:10]}..."
            )
        demo_keys = [args.demo_id]

    server = viser.ViserServer(port=args.port)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Reasonable default view.
        client.camera.position = (0.25, 0.0, 0.25)
        client.camera.look_at = (0.0, 0.0, 0.0)

    if args.show_grid:
        server.scene.add_grid("/grid", 1.0, 1.0)

    # UI
    demo_slider = server.gui.add_slider(
        "demo",
        min=0,
        max=max(0, len(demo_keys) - 1),
        step=1,
        initial_value=0,
        disabled=(len(demo_keys) == 1),
    )

    playing = server.gui.add_checkbox("playing", True)
    frame_slider = server.gui.add_slider("frame", min=0, max=1, step=1, initial_value=0)

    # Cache current demo (to keep slider interaction snappy).
    cache: dict[str, Any] = {
        "demo_idx": None,
        "grp": None,
        "T": None,
        "robot_type": None,
        "mano_side": None,
        "normalizer": None,
        "urdf_vis": None,
        "base_frame": None,
    }

    def load_demo(demo_idx: int) -> None:
        demo_key = demo_keys[demo_idx]
        grp = f[demo_key]

        required = ["world_hand_joints"]
        missing = [k for k in required if k not in grp]
        if missing:
            raise KeyError(f"Demo '{demo_key}' missing required keys: {missing}")

        world_hand_joints = grp["world_hand_joints"]
        if world_hand_joints.ndim != 3 or world_hand_joints.shape[1:] != (21, 3):
            raise ValueError(
                f"Expected world_hand_joints shape (T,21,3), got {world_hand_joints.shape}"
            )

        T = int(world_hand_joints.shape[0])
        frame_slider.max = max(0, T - 1)
        frame_slider.value = 0

        robot_type = _decode_scalar(np.asarray(grp.get("robot_type", "")))
        if not robot_type:
            robot_type = "inspire"

        mano_side = _decode_scalar(np.asarray(grp.get("mano_side", np.array(["right"], dtype=object))))
        mano_side = mano_side.lower().strip()
        if mano_side not in ("left", "right"):
            mano_side = "right"

        # Load URDF + normalizer.
        urdf = _load_urdf(robot_type, mano_side)
        urdf_path = (
            Path(DEX_URDF_PATH)
            / "robots"
            / "hands"
            / f"{robot_type}_hand"
            / f"{robot_type}_hand_{mano_side}.urdf"
        )
        normalizer = AngleNormalizer(urdf_path)

        # (Re-)create robot nodes.
        base_frame = server.scene.add_frame(
            "/robot/base",
            show_axes=args.show_axes,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot/base")

        cache.update(
            {
                "demo_idx": demo_idx,
                "grp": grp,
                "T": T,
                "robot_type": robot_type,
                "mano_side": mano_side,
                "normalizer": normalizer,
                "urdf_vis": urdf_vis,
                "base_frame": base_frame,
            }
        )

        server.gui.add_markdown(
            "".join([
                f"**demo**: `{demo_key}`  \n"
                f"**frames**: `{T}`  \n"
                f"**mano_side**: `{mano_side}`  \n"
                f"**robot_type**: `{robot_type}`"
            ]),
        )

    def render(demo_idx: int, frame_idx: int) -> None:
        if cache["demo_idx"] != demo_idx:
            load_demo(demo_idx)

        grp: h5py.Group = cache["grp"]
        base_frame = cache["base_frame"]
        urdf_vis: ViserUrdf = cache["urdf_vis"]
        normalizer: AngleNormalizer = cache["normalizer"]

        # Human joints.
        joints = np.asarray(grp["world_hand_joints"][frame_idx], dtype=np.float32)
        joints = joints - joints[0] + base_frame.position  # root-relative -> world

        # Robot base pose.
        if "Ts_world_root" in grp:
            T_world_root = np.asarray(grp["Ts_world_root"][frame_idx], dtype=np.float32)
            if T_world_root.shape != (4, 4):
                raise ValueError(f"Ts_world_root expected (4,4), got {T_world_root.shape}")
            pos = T_world_root[:3, 3]
            wxyz = R.from_matrix(T_world_root[:3, :3]).as_quat(scalar_first=True)
        else:
            # Fallback: identity base.
            pos = np.zeros(3, dtype=np.float32)
            wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Robot joints.
        robot_cfg = _extract_robot_cfg(grp, frame_idx, normalizer)

        with server.atomic():
            # Human
            server.scene.add_point_cloud(
                name="/human/joints",
                points=joints,
                colors=np.array([1.0, 0.0, 0.0]),
                point_size=float(args.point_size),
            )
            server.scene.add_line_segments(
                name="/human/bones",
                points=_make_bone_segments(joints),
                colors=np.array([1.0, 0.0, 0.0]),
                line_width=float(args.line_width),
            )

            # Robot
            base_frame.position = pos
            base_frame.wxyz = wxyz
            if robot_cfg is not None:
                urdf_vis.update_cfg(np.asarray(robot_cfg, dtype=np.float32))

    @demo_slider.on_update
    def _(_evt) -> None:
        # Programmatic writes (client_id=None) can happen during `load_demo()` and
        # during autoplay; ignore those to avoid recursive callback loops.
        if getattr(_evt, "client_id", None) is None:
            _log("demo_slider programmatic update ignored", "demo=", demo_slider.value)
            return
        _log("demo_slider user update", "demo=", demo_slider.value)
        render(int(demo_slider.value), int(frame_slider.value))

    @frame_slider.on_update
    def _(_evt) -> None:
        if getattr(_evt, "client_id", None) is None:
            _log("frame_slider programmatic update ignored", "frame=", frame_slider.value)
            return
        _log("frame_slider user update", "frame=", frame_slider.value)
        render(int(demo_slider.value), int(frame_slider.value))

    # Initial draw.
    render(int(demo_slider.value), int(frame_slider.value))

    server.gui.add_markdown(
        "help",
        """
Use **demo** and **frame** sliders, or enable **playing**.

Legend:
- **red**: human `world_hand_joints`
- URDF: robot hand from `$DEX_URDF_PATH` (pose from `Ts_world_root` + joints from `robot_joint_pose`)
""".strip(),
    )

    dt = 1.0 / max(1e-6, float(args.fps))
    while True:
        if playing.value:
            frame_slider.value = (int(frame_slider.value) + 1) % int(frame_slider.max + 1)
            render(int(demo_slider.value), int(frame_slider.value))
        time.sleep(dt)


if __name__ == "__main__":
    main(tyro.cli(Args))
