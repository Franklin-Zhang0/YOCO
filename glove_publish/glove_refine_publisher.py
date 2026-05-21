#!/usr/bin/env python3
"""CyberGlove keypoint capture, four-pose calibration, and refinement.

This module keeps the runtime pieces together:

* connect to OpenCyberGlove;
* convert glove angles to 21 3D keypoints using the hand FK model;
* collect four interactive calibration snapshots while showing reference
  skeletons in Viser;
* pass the calibration pairs to ``RefineModelInferencer``;
* refine live glove keypoints with the loaded refine model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import time
from typing import Optional

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/calibration_net_matplotlib")

from calibration_net.policy.refine_inferencer import RefineModelInferencer
from calibration_net.utils import (
    mano_recon,
    transform_joint3d as joint_3d_world_to_local,
)


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


# Captured from data/avp_calib_gt_poses/20260521_232739 on 2026-05-21.
PRESET_CALIBRATION_KEYPOINTS = {
    "pinch": np.array(
        [
            [0.181695864, 0.183354139, 0.864475071],
            [0.160942152, 0.173284814, 0.881526589],
            [0.123618007, 0.143140957, 0.904101074],
            [0.096644193, 0.128157109, 0.898272514],
            [0.076207235, 0.119316891, 0.880364776],
            [0.091054872, 0.182852104, 0.891104400],
            [0.056098081, 0.158630610, 0.883230031],
            [0.061438359, 0.135222629, 0.879648685],
            [0.073241875, 0.117374621, 0.877713978],
            [0.092275746, 0.185288519, 0.868916035],
            [0.044847302, 0.178319380, 0.862562835],
            [0.032303862, 0.152543783, 0.861076653],
            [0.025597701, 0.131274298, 0.860345721],
            [0.093948722, 0.182282820, 0.849314094],
            [0.052940983, 0.182595089, 0.844167113],
            [0.034086354, 0.163746014, 0.841792107],
            [0.020354236, 0.147083595, 0.840033889],
            [0.104224741, 0.179237083, 0.831498981],
            [0.072588265, 0.181850269, 0.820845246],
            [0.056114145, 0.173986167, 0.815463841],
            [0.039322332, 0.165122926, 0.809994340],
        ],
        dtype=np.float32,
    ),
    "fully_open": np.array(
        [
            [0.177247092, 0.184021324, 0.883406997],
            [0.156592324, 0.174016014, 0.900383174],
            [0.119640611, 0.172941238, 0.938204288],
            [0.093258373, 0.175467402, 0.954924703],
            [0.072785690, 0.176745474, 0.974883437],
            [0.086403936, 0.182150826, 0.908225536],
            [0.044998415, 0.194125801, 0.911784530],
            [0.020851677, 0.194968298, 0.913708329],
            [-0.000666103, 0.195157811, 0.915409088],
            [0.088171765, 0.184541374, 0.886877179],
            [0.041490082, 0.196643725, 0.882074535],
            [0.013026956, 0.196438491, 0.878785729],
            [-0.009183953, 0.196710438, 0.876019180],
            [0.089858025, 0.183521405, 0.868301988],
            [0.051079057, 0.195554391, 0.860686958],
            [0.024810592, 0.196156651, 0.855916679],
            [0.003489700, 0.196178526, 0.852062404],
            [0.099885240, 0.181983262, 0.851412117],
            [0.071354493, 0.191930383, 0.837072730],
            [0.054332934, 0.193610653, 0.828790307],
            [0.036578305, 0.194427446, 0.820211411],
        ],
        dtype=np.float32,
    ),
    "fist": np.array(
        [
            [0.178439468, 0.205908567, 0.859727502],
            [0.158183903, 0.196097761, 0.876377761],
            [0.120095275, 0.162103847, 0.883850276],
            [0.095265701, 0.149813578, 0.871112645],
            [0.079232410, 0.147822738, 0.848507047],
            [0.090422928, 0.208052486, 0.885532260],
            [0.074049808, 0.170796126, 0.876567364],
            [0.097189412, 0.167715982, 0.875922680],
            [0.107819118, 0.185018510, 0.880098045],
            [0.091188811, 0.209414214, 0.862360656],
            [0.081495129, 0.163958818, 0.859342217],
            [0.109049596, 0.165070668, 0.860312939],
            [0.114114307, 0.185745627, 0.862486720],
            [0.093970813, 0.204496056, 0.843188286],
            [0.094082221, 0.164541662, 0.848741531],
            [0.119357273, 0.164415911, 0.850903511],
            [0.120823860, 0.184993118, 0.847822547],
            [0.103616022, 0.199947238, 0.827607930],
            [0.106226109, 0.167977646, 0.834172726],
            [0.123621061, 0.166680127, 0.838690102],
            [0.126791984, 0.185382575, 0.836004615],
        ],
        dtype=np.float32,
    ),
    "little_pinch": np.array(
        [
            [0.173600957, 0.194350928, 0.873397231],
            [0.153327376, 0.184533328, 0.890064716],
            [0.115056440, 0.150060222, 0.895715117],
            [0.092417553, 0.138034195, 0.879047394],
            [0.076832525, 0.133070141, 0.856545746],
            [0.085166752, 0.193369582, 0.899925888],
            [0.048365004, 0.174229115, 0.892536759],
            [0.047633450, 0.151012748, 0.888747036],
            [0.057226747, 0.132590219, 0.886938453],
            [0.086243317, 0.197667077, 0.877742112],
            [0.044401493, 0.177832395, 0.870571136],
            [0.048593413, 0.150418997, 0.869943440],
            [0.063322857, 0.134813860, 0.871817231],
            [0.088632837, 0.193498701, 0.858917594],
            [0.053162944, 0.174392119, 0.856094837],
            [0.058424342, 0.149243698, 0.856882095],
            [0.071990348, 0.133302942, 0.858112037],
            [0.098291941, 0.188933030, 0.842800498],
            [0.070600860, 0.171611458, 0.844764233],
            [0.068431213, 0.153774738, 0.847725570],
            [0.075386763, 0.136143893, 0.850977302],
        ],
        dtype=np.float32,
    ),
}


@dataclass
class GloveKeypointSample:
    """One synchronized glove sample before and after refinement."""

    raw_keypoints: np.ndarray
    refined_keypoints: np.ndarray
    angles: np.ndarray
    tensile_data: np.ndarray
    timestamp: int


class GloveForwardKinematics:
    """Convert OpenCyberGlove model angles to 21 3D keypoints."""

    def __init__(self, hand_model_path: str | Path) -> None:
        from open_cyber_glove.utils import DEFAULT_GT_ORDER, forward_kinematics

        self.default_gt_order = DEFAULT_GT_ORDER
        self.forward_kinematics = forward_kinematics
        with Path(hand_model_path).expanduser().open("rb") as f:
            self.hand_model = pickle.load(f)

    def angles_to_joints(self, angles: np.ndarray, side: str) -> np.ndarray:
        angle_dict: dict[str, dict[str, dict[str, float]]] = {}
        for i, angle_name in enumerate(self.default_gt_order):
            finger, joint, dof = angle_name.split("_")
            angle_dict.setdefault(finger, {}).setdefault(joint, {})[dof] = float(
                angles[i]
            )

        joints, _ = self.forward_kinematics(
            self.hand_model[side],
            angle_dict,
            side,
            np,
        )
        return (np.asarray(joints, dtype=np.float32) / 1000.0).astype(np.float32)

    def zeros(self) -> np.ndarray:
        return np.zeros(len(self.default_gt_order), dtype=np.float32)

    def vector_from_named_values(self, values: dict[str, float]) -> np.ndarray:
        angles = self.zeros()
        for i, name in enumerate(self.default_gt_order):
            angles[i] = float(values.get(name, 0.0))
        return angles


def _to_local_keypoints(joints: np.ndarray) -> np.ndarray:
    local = joint_3d_world_to_local(np.asarray(joints, dtype=np.float32))
    return np.asarray(local, dtype=np.float32)


class ViserCalibrationView:
    """Small Viser scene for calibration target/current skeletons."""

    def __init__(self, port: int) -> None:
        import viser

        self.server = viser.ViserServer(port=port)
        self.server.scene.add_grid("/grid", width=0.4, height=0.4)
        self.info = self.server.gui.add_markdown("calibration", "")
        self._pose_handles = []

    @staticmethod
    def _segments(joints: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.stack([joints[a], joints[b]], axis=0) for a, b in MANO_CONNECTIONS],
            axis=0,
        )

    def update(
        self,
        pose_name: str,
        target: np.ndarray,
        current: Optional[np.ndarray] = None,
    ) -> None:
        self.clear_pose()
        self.info.content = (
            f"### Calibration pose: `{pose_name}`\n"
            "Match the green skeleton, then press Enter in the terminal."
        )
        self._draw("/target", target, np.array([0.1, 0.8, 0.2]))
        if current is not None:
            self._draw("/current", current, np.array([0.1, 0.25, 1.0]))

    def clear_pose(self) -> None:
        while self._pose_handles:
            handle = self._pose_handles.pop()
            handle.remove()

    def _draw(self, prefix: str, joints: np.ndarray, color: np.ndarray) -> None:
        joints = _to_local_keypoints(joints)
        self._pose_handles.append(
            self.server.scene.add_point_cloud(
                f"{prefix}/joints",
                points=joints,
                colors=color,
                point_size=0.004,
            )
        )
        self._pose_handles.append(
            self.server.scene.add_line_segments(
                f"{prefix}/bones",
                points=self._segments(joints),
                colors=color,
                line_width=2.0,
            )
        )


class GloveRefinePublisher:
    """Connect a CyberGlove, calibrate four poses, and refine keypoints."""

    CALIBRATION_POSE_ORDER = ("pinch", "fully_open", "fist", "little_pinch")

    def __init__(
        self,
        *,
        side: str,
        glove_port: str | None = None,
        hand_model_path: str | Path,
        glove_model_path: str | Path | None = None,
        refine_model_dir: str | Path,
        device: str | None = None,
        viser_port: int | None = 8097,
        fixed_noise: bool = True,
        dummy: bool = False,
        dummy_noise_std: float = 0.003,
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        if not dummy and glove_port is None:
            raise ValueError("glove_port is required unless dummy=True.")
        if not dummy and glove_model_path is None:
            raise ValueError("glove_model_path is required unless dummy=True.")

        self.side = side
        self.glove_port = glove_port
        self.hand_model_path = Path(hand_model_path).expanduser()
        self.glove_model_path = (
            Path(glove_model_path).expanduser()
            if glove_model_path is not None
            else None
        )
        self.refine_model_dir = Path(refine_model_dir).expanduser()
        self.dummy = dummy
        self.dummy_noise_std = float(dummy_noise_std)
        self._dummy_phase = 0
        self._dummy_pose_name: str | None = None
        self.fk = GloveForwardKinematics(self.hand_model_path)
        self.refiner = RefineModelInferencer(
            model_dir=str(self.refine_model_dir),
            side=side,
            device=device,
            paired_poses=None,
            fixed_noise=fixed_noise,
        )
        self.sdk = None
        self._connected = False
        self.view = ViserCalibrationView(viser_port) if viser_port is not None else None
        self.calibration_pairs: dict[str, np.ndarray] | None = None

    def connect(self, calibrate_glove: bool = True) -> None:
        """Connect the glove and optionally run OpenCyberGlove baseline calibration."""
        if self.dummy:
            print("Dummy glove mode enabled; skipping OpenCyberGlove connection.")
            self._connected = True
            return

        from open_cyber_glove.sdk import OpenCyberGlove

        kwargs = {
            f"{self.side}_port": self.glove_port,
            "model_path": str(self.glove_model_path),
        }
        self.sdk = OpenCyberGlove(**kwargs)
        self.sdk.start()
        self._connected = True
        if calibrate_glove:
            self.sdk.calibrate()

    def close(self) -> None:
        if self.sdk is not None:
            self.sdk.stop()
            self.sdk = None
        self._connected = False

    def get_glove_keypoints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Return raw FK keypoints, model angles, tensile data, and device timestamp."""
        if not self._connected:
            raise RuntimeError("Call connect() before reading the glove.")
        if self.dummy:
            return self._get_dummy_keypoints()
        if self.sdk is None:
            raise RuntimeError("OpenCyberGlove SDK is not initialized.")

        glove = getattr(self.sdk, f"{self.side}_glove")
        raw = self.sdk.get_data(hand_type=self.side)
        angles = np.asarray(
            glove.inference(raw, method="model", model=self.sdk.model),
            dtype=np.float32,
        )
        keypoints = self.fk.angles_to_joints(angles, self.side)
        tensile = np.asarray(raw.tensile_data, dtype=np.float32)
        return keypoints, angles, tensile, int(raw.timestamp)

    def refine_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Run the refine model and return 21 3D keypoints."""
        pred = np.asarray(self.refiner.inference(keypoints, batched=False))
        if pred.shape == (21, 3):
            return pred.astype(np.float32)
        if pred.size == 63:
            return pred.reshape(21, 3).astype(np.float32)
        if pred.shape == (45,):
            return self._mano_theta_to_keypoints(pred)
        raise ValueError(f"Unsupported refine output shape: {pred.shape}")

    def get_sample(self) -> GloveKeypointSample:
        raw_keypoints, angles, tensile, timestamp = self.get_glove_keypoints()
        refined = self.refine_keypoints(raw_keypoints)
        return GloveKeypointSample(
            raw_keypoints=raw_keypoints,
            refined_keypoints=refined,
            angles=angles,
            tensile_data=tensile,
            timestamp=timestamp,
        )

    def calibrate_refine_condition(self) -> dict[str, np.ndarray]:
        """Collect four snapshots and update the refine model's paired condition."""
        if not self._connected:
            raise RuntimeError("Call connect() before calibration.")

        biased = []
        target = []
        for pose_name in self.CALIBRATION_POSE_ORDER:
            target_joints = self.reference_keypoints(pose_name)
            if self.view is not None:
                self.view.update(pose_name, target_joints)

            self._dummy_pose_name = pose_name
            input(f"[{pose_name}] Match the shown skeleton, then press Enter...")
            keypoints, _, _, _ = self.get_glove_keypoints()
            if self.view is not None:
                self.view.update(pose_name, target_joints, current=keypoints)
                time.sleep(0.2)

            biased.append(keypoints)
            target.append(target_joints)

        self.calibration_pairs = {
            "world_hand_joints_biased": np.stack(biased).astype(np.float32),
            "world_hand_joints": np.stack(target).astype(np.float32),
        }
        self.refiner.update_condition(self.calibration_pairs)
        self._dummy_pose_name = None
        return self.calibration_pairs

    def reference_keypoints(self, pose_name: str) -> np.ndarray:
        if pose_name not in PRESET_CALIBRATION_KEYPOINTS:
            raise KeyError(f"Unknown calibration pose: {pose_name}")
        keypoints = PRESET_CALIBRATION_KEYPOINTS[pose_name].copy()
        if self.side == "left":
            keypoints[:, 0] *= -1.0
        return keypoints

    def _mano_theta_to_keypoints(self, theta: np.ndarray) -> np.ndarray:
        import torch

        theta_t = torch.as_tensor(
            theta[None],
            dtype=torch.float32,
            device=self.refiner.device,
        )
        _, joints = mano_recon(
            theta_t,
            side=self.side,
            apply_left_shapedirs_fix=True,
        )
        return joints[0].detach().cpu().numpy().astype(np.float32)

    def _get_dummy_keypoints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        pose_name = self._dummy_pose_name
        if pose_name is None:
            pose_name = self.CALIBRATION_POSE_ORDER[
                self._dummy_phase % len(self.CALIBRATION_POSE_ORDER)
            ]
            self._dummy_phase += 1

        keypoints = self.reference_keypoints(pose_name)
        rng = np.random.default_rng(seed=10_000 + self._dummy_phase)
        if self.dummy_noise_std > 0:
            keypoints = keypoints + rng.normal(
                loc=0.0,
                scale=self.dummy_noise_std,
                size=keypoints.shape,
            ).astype(np.float32)

        angles = self.fk.zeros()
        tensile = np.zeros(19, dtype=np.float32)
        timestamp = time.monotonic_ns()
        return keypoints, angles.astype(np.float32), tensile, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect CyberGlove, run four-pose calibration, and preview "
            "refinement."
        )
    )
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--glove-port")
    parser.add_argument("--hand-model-path", required=True)
    parser.add_argument("--glove-model-path")
    parser.add_argument("--refine-model-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--viser-port", type=int, default=8097)
    parser.add_argument("--no-glove-calibration", action="store_true")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use synthetic glove poses without opening the serial glove.",
    )
    parser.add_argument(
        "--dummy-noise-std",
        type=float,
        default=0.003,
        help="Stddev of synthetic keypoint noise in meters for dummy mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    publisher = GloveRefinePublisher(
        side=args.side,
        glove_port=args.glove_port,
        hand_model_path=args.hand_model_path,
        glove_model_path=args.glove_model_path,
        refine_model_dir=args.refine_model_dir,
        device=args.device,
        viser_port=args.viser_port,
        dummy=args.dummy,
        dummy_noise_std=args.dummy_noise_std,
    )
    try:
        publisher.connect(calibrate_glove=not args.no_glove_calibration)
        publisher.calibrate_refine_condition()
        sample = publisher.get_sample()
        print("Raw keypoints:", sample.raw_keypoints.shape)
        print("Refined keypoints:", sample.refined_keypoints.shape)
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
