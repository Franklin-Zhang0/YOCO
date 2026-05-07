import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_bias_generator_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "calibration_net" / "policy" / "bias_generator.py"

    package_backup = sys.modules.get("calibration_net")
    utils_backup = sys.modules.get("calibration_net.utils")

    fake_package = types.ModuleType("calibration_net")
    fake_package.__path__ = [str(repo_root / "calibration_net")]

    fake_utils = types.ModuleType("calibration_net.utils")

    def fake_mano_recon(
        mano_theta_pred,
        mano_orient=None,
        transl=None,
        side=None,
        betas=None,
    ):
        batch = mano_theta_pred.shape[0]
        joints = torch.zeros((batch, 21, 3), dtype=mano_theta_pred.dtype, device=mano_theta_pred.device)
        return None, joints

    fake_utils.mano_recon = fake_mano_recon
    sys.modules["calibration_net"] = fake_package
    sys.modules["calibration_net.utils"] = fake_utils

    try:
        spec = importlib.util.spec_from_file_location(
            "bias_generator_under_test",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("bias_generator_under_test", None)
        if package_backup is not None:
            sys.modules["calibration_net"] = package_backup
        else:
            sys.modules.pop("calibration_net", None)
        if utils_backup is not None:
            sys.modules["calibration_net.utils"] = utils_backup
        else:
            sys.modules.pop("calibration_net.utils", None)


@pytest.fixture(scope="module")
def bias_module():
    return _load_bias_generator_module()


def _joint_slice(finger_idx: int, joint_idx: int) -> slice:
    start = (finger_idx * 3 + joint_idx) * 3
    return slice(start, start + 3)


def test_zero_config_is_identity(bias_module):
    cfg = bias_module.BiasConfig(
        per_finger_offset_prob=0.0,
        per_joint_residual_prob=0.0,
        global_scale_prob=0.0,
        per_finger_scale_prob=0.0,
        per_joint_scale_prob=0.0,
        finger_correlation_prob=0.0,
        joint_correlation_prob=0.0,
        seed=123,
    )
    generator = bias_module.BiasGenerator(cfg)
    pose = np.random.RandomState(0).randn(4, 45).astype(np.float32)

    biased = generator.apply_bias_to_hand_pose(pose, demo_id=7)

    np.testing.assert_allclose(biased, pose)


def test_bias_sampling_is_deterministic_per_demo(bias_module):
    cfg = bias_module.BiasConfig(
        per_finger_offset_prob=1.0,
        per_finger_offset_std=0.08,
        per_joint_residual_prob=1.0,
        per_joint_residual_std=0.03,
        global_scale_prob=1.0,
        global_scale_std=0.05,
        per_finger_scale_prob=1.0,
        per_finger_scale_std=0.04,
        per_joint_scale_prob=1.0,
        per_joint_scale_std=0.02,
        finger_correlation_prob=1.0,
        finger_correlation_std=0.2,
        joint_correlation_prob=1.0,
        joint_correlation_std=0.2,
        seed=11,
    )
    generator = bias_module.BiasGenerator(cfg)

    params_a = generator.get_bias_params(3)
    generator.clear_cache()
    params_b = generator.get_bias_params(3)
    params_c = generator.get_bias_params(4)

    for key in params_a:
        np.testing.assert_allclose(params_a[key], params_b[key])
    assert any(not np.allclose(params_a[key], params_c[key]) for key in params_a)


def test_per_joint_scaling_respects_mask_and_composes_scales(bias_module):
    cfg = bias_module.BiasConfig(
        per_finger_offset_prob=0.0,
        per_joint_residual_prob=0.0,
        global_scale_prob=1.0,
        global_scale_std=0.12,
        per_finger_scale_prob=1.0,
        per_finger_scale_std=0.10,
        per_joint_scale_prob=1.0,
        per_joint_scale_std=0.15,
        finger_correlation_prob=0.0,
        joint_correlation_prob=0.0,
        biased_joint_indices=[4],
        seed=5,
    )
    generator = bias_module.BiasGenerator(cfg)
    pose = np.zeros((45,), dtype=np.float32)
    pose[_joint_slice(1, 1)] = np.array([0.5, -1.0, 0.25], dtype=np.float32)
    pose[_joint_slice(1, 2)] = np.array([0.2, 0.4, -0.3], dtype=np.float32)

    params = generator.get_bias_params(8)
    biased = generator.apply_bias_to_hand_pose(pose, demo_id=8).reshape(15, 3)
    expected = pose.reshape(15, 3).copy()
    expected[4] *= params["joint_scale"][4, 0]

    np.testing.assert_allclose(biased, expected)
    assert not np.isclose(params["joint_scale"][4, 0], 1.0)
    assert np.isclose(params["joint_scale"][5, 0], 1.0)


def test_adjacent_finger_crosstalk_stays_local(bias_module):
    cfg = bias_module.BiasConfig(
        per_finger_offset_prob=0.0,
        per_joint_residual_prob=0.0,
        global_scale_prob=0.0,
        per_finger_scale_prob=0.0,
        per_joint_scale_prob=0.0,
        finger_correlation_prob=1.0,
        finger_correlation_std=0.25,
        joint_correlation_prob=0.0,
        seed=19,
    )
    generator = bias_module.BiasGenerator(cfg)
    pose = np.zeros((45,), dtype=np.float32)
    pose[_joint_slice(1, 1)] = np.array([1.0, -0.5, 0.25], dtype=np.float32)

    biased = generator.apply_bias_to_hand_pose(pose, demo_id=2)

    assert np.linalg.norm(biased[_joint_slice(0, 1)]) > 0.0
    assert np.linalg.norm(biased[_joint_slice(2, 1)]) > 0.0
    np.testing.assert_allclose(biased[_joint_slice(3, 1)], 0.0)
    np.testing.assert_allclose(biased[_joint_slice(4, 1)], 0.0)
    np.testing.assert_allclose(biased[_joint_slice(0, 0)], 0.0)
    np.testing.assert_allclose(biased[_joint_slice(0, 2)], 0.0)


def test_proximal_to_distal_coupling_has_no_backward_leak(bias_module):
    cfg = bias_module.BiasConfig(
        per_finger_offset_prob=0.0,
        per_joint_residual_prob=0.0,
        global_scale_prob=0.0,
        per_finger_scale_prob=0.0,
        per_joint_scale_prob=0.0,
        finger_correlation_prob=0.0,
        joint_correlation_prob=1.0,
        joint_correlation_std=0.3,
        seed=23,
    )
    generator = bias_module.BiasGenerator(cfg)

    proximal_pose = np.zeros((45,), dtype=np.float32)
    proximal_pose[_joint_slice(2, 0)] = np.array([0.6, 0.2, -0.4], dtype=np.float32)
    proximal_biased = generator.apply_bias_to_hand_pose(proximal_pose, demo_id=4)
    assert np.linalg.norm(proximal_biased[_joint_slice(2, 1)]) > 0.0
    assert np.linalg.norm(proximal_biased[_joint_slice(2, 2)]) > 0.0
    np.testing.assert_allclose(proximal_biased[_joint_slice(1, 1)], 0.0)

    distal_pose = np.zeros((45,), dtype=np.float32)
    distal_pose[_joint_slice(2, 2)] = np.array([0.4, -0.3, 0.1], dtype=np.float32)
    distal_biased = generator.apply_bias_to_hand_pose(distal_pose, demo_id=4)
    np.testing.assert_allclose(distal_biased[_joint_slice(2, 0)], 0.0)
    np.testing.assert_allclose(distal_biased[_joint_slice(2, 1)], 0.0)


def test_zero_pose_stays_zero_with_multiplicative_terms_only(bias_module):
    cfg = bias_module.BiasConfig(
        per_finger_offset_prob=0.0,
        per_joint_residual_prob=0.0,
        global_scale_prob=1.0,
        global_scale_std=0.2,
        per_finger_scale_prob=1.0,
        per_finger_scale_std=0.15,
        per_joint_scale_prob=1.0,
        per_joint_scale_std=0.15,
        finger_correlation_prob=1.0,
        finger_correlation_std=0.3,
        joint_correlation_prob=1.0,
        joint_correlation_std=0.3,
        seed=31,
    )
    generator = bias_module.BiasGenerator(cfg)

    biased = generator.apply_bias_to_hand_pose(np.zeros((3, 45), dtype=np.float32), demo_id=9)

    np.testing.assert_allclose(biased, 0.0)


def test_clamp_delta_norm_limits_joint_delta(bias_module):
    cfg = bias_module.BiasConfig(
        per_finger_offset_prob=0.0,
        per_joint_residual_prob=1.0,
        per_joint_residual_std=1.0,
        global_scale_prob=0.0,
        per_finger_scale_prob=0.0,
        per_joint_scale_prob=0.0,
        finger_correlation_prob=0.0,
        joint_correlation_prob=0.0,
        clamp_delta_norm=0.05,
        seed=37,
    )
    generator = bias_module.BiasGenerator(cfg)

    biased = generator.apply_bias_to_hand_pose(np.zeros((45,), dtype=np.float32), demo_id=1)
    norms = np.linalg.norm(biased.reshape(15, 3), axis=-1)

    assert np.all(norms <= 0.05001)
