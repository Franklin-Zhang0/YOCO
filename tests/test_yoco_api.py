"""Focused public-API tests without MANO or large checkpoint dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import torch

from calibration_net.yoco import YOCO, _inference_precision, _strict_checkpoint_state


def fake_inferencer() -> SimpleNamespace:
    model = mock.Mock()
    inferencer = SimpleNamespace(
        model=model,
        cfg={"dataset": {"num_hypernet_pairs": 4}},
        device=torch.device("cpu"),
        last_hyper_input=None,
    )

    def condition(pairs):
        inferencer.last_hyper_input = torch.ones(1, 8)

    def predict(frames, *, batched):
        return (
            np.arange(len(frames) * 45, dtype=np.float32)
            .reshape(len(frames), 45)
            .squeeze()
        )

    inferencer.update_condition = mock.Mock(side_effect=condition)
    inferencer.inference = mock.Mock(side_effect=predict)
    return inferencer


class YOCOAPITest(unittest.TestCase):
    def setUp(self):
        self.inferencer = fake_inferencer()
        self.model = YOCO(self.inferencer, side="right")
        self.pairs = np.zeros((4, 21, 3), dtype=np.float32)

    def test_calibration_requires_exact_pair_count(self):
        for count in (0, 1, 3, 5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.model.calibrate(np.zeros((count, 21, 3)), self.pairs)
        self.inferencer.update_condition.assert_not_called()

    def test_conditioning_unmerges_and_uses_eval_first(self):
        events = mock.Mock()
        events.attach_mock(self.inferencer.model.hyperlora.unmerge_weight, "unmerge")
        events.attach_mock(self.inferencer.model.eval, "eval")
        events.attach_mock(self.inferencer.update_condition, "condition")
        self.assertIs(self.model.calibrate(self.pairs, self.pairs), self.model)
        self.model.calibrate(self.pairs + 1, self.pairs)
        self.assertEqual(
            [call[0] for call in events.mock_calls],
            ["unmerge", "eval", "condition", "unmerge", "eval", "condition"],
        )

    def test_predict_requires_calibration(self):
        with self.assertRaisesRegex(RuntimeError, "calibrate"):
            self.model.predict(np.zeros((21, 3)))

    def test_single_and_batch_output_shapes(self):
        self.model.calibrate(self.pairs, self.pairs)
        with mock.patch.object(
            self.model,
            "_reconstruct_joints",
            side_effect=lambda theta, betas: np.zeros((len(theta), 21, 3), np.float32),
        ):
            single = self.model.predict(np.zeros((21, 3)))
            one = self.model.predict(np.zeros((1, 21, 3)))
            batch = self.model.predict(np.zeros((5, 21, 3)), batch_size=2)
        self.assertEqual(single.theta.shape, (45,))
        self.assertEqual(single.joints.shape, (21, 3))
        self.assertEqual(one.theta.shape, (1, 45))
        self.assertEqual(one.joints.shape, (1, 21, 3))
        self.assertEqual(batch.theta.shape, (5, 45))
        self.assertEqual(batch.joints.shape, (5, 21, 3))

    def test_frame_specific_betas_follow_prediction_chunks(self):
        self.model.calibrate(self.pairs, self.pairs)
        betas = np.arange(50, dtype=np.float32).reshape(5, 10)
        with mock.patch.object(
            self.model,
            "_reconstruct_joints",
            side_effect=lambda theta, betas: np.zeros((len(theta), 21, 3), np.float32),
        ) as reconstruct:
            self.model.predict(np.zeros((5, 21, 3)), betas=betas, batch_size=2)
        for call, expected in zip(
            reconstruct.call_args_list, (betas[:2], betas[2:4], betas[4:])
        ):
            np.testing.assert_array_equal(call.args[1], expected)

    def test_rejects_nonfinite_and_incompatible_shapes(self):
        bad_pairs = self.pairs.copy()
        bad_pairs[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.model.calibrate(bad_pairs, self.pairs)
        self.model.calibrate(self.pairs, self.pairs)
        for frames in (
            np.zeros((20, 3)),
            np.zeros((0, 21, 3)),
            np.full((21, 3), np.inf),
        ):
            with self.subTest(shape=frames.shape), self.assertRaises(ValueError):
                self.model.predict(frames)
        with self.assertRaisesRegex(ValueError, "betas"):
            self.model.predict(np.zeros((21, 3)), betas=np.zeros(9))
        with self.assertRaisesRegex(ValueError, "batch_size"):
            self.model.predict(np.zeros((21, 3)), batch_size=0)

    def test_unavailable_cuda_does_not_fall_back(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(ValueError, "CUDA"),
        ):
            YOCO.from_pretrained("missing", canonicalizer_dir="missing", device="cuda")

    def test_mano_path_is_required(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "MANO_PATH"),
        ):
            YOCO.from_pretrained("missing", canonicalizer_dir="missing")

    def test_mano_path_must_be_absolute(self):
        with (
            mock.patch.dict(os.environ, {"MANO_PATH": "~/models"}),
            self.assertRaisesRegex(ValueError, "absolute"),
        ):
            YOCO.from_pretrained("missing", canonicalizer_dir="missing")

    def test_checkpoint_state_load_is_strict(self):
        model = mock.Mock()
        state = {"weight": torch.ones(1)}
        with mock.patch("torch.load", return_value=state) as load:
            _strict_checkpoint_state(model, Path("checkpoint"))
        model.load_state_dict.assert_called_once_with(state, strict=True)
        self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_legacy_checkpoint_still_rejects_missing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.save(
                {"weight": torch.ones(1, 1)},
                path / "model.pth",
                _use_new_zipfile_serialization=False,
            )
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                _strict_checkpoint_state(torch.nn.Linear(1, 1), path)


class YOCOExampleInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/run_yoco.py"
        spec = importlib.util.spec_from_file_location("run_yoco_example", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_hdf5_demo_uses_valid_prefix_pairs_and_variant_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.hdf5"
            raw = np.arange(2 * 220 * 21 * 3, dtype=np.float32).reshape(2, 220, 21, 3)
            target = raw[0] + 2
            with h5py.File(path, "w") as data:
                group = data.create_group("1")
                group["mano_side"] = np.bytes_("right")
                group["world_hand_joints_biased"] = raw
                group["world_hand_joints"] = target
                group["betas"] = np.zeros(10, np.float32)
                bad = np.zeros(220, bool)
                bad[:20] = True
                group["bad_frame"] = bad
            args = argparse.Namespace(
                input=None,
                data_path=path,
                group="1",
                side="right",
                sample_idx=1,
                calibration_window=200,
                frames=32,
                seed=0,
            )
            inputs = self.module.load_inputs(args, 4)
            self.assertTrue(np.all(inputs["calib_idx"] >= 20))
            self.assertTrue(np.all(inputs["calib_idx"] < 200))
            np.testing.assert_array_equal(
                inputs["raw_pairs"], raw[1, inputs["calib_idx"]]
            )
            np.testing.assert_array_equal(
                inputs["target_pairs"], target[inputs["calib_idx"]]
            )
            self.assertEqual(inputs["raw_frames"].shape, (20, 21, 3))

    def test_npz_missing_array_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npz"
            np.savez(path, raw_frames=np.zeros((21, 3)))
            with self.assertRaisesRegex(ValueError, "raw_pairs"):
                self.module.load_inputs(argparse.Namespace(input=path), 4)


class YOCOCUDAPrecisionTest(unittest.TestCase):
    @staticmethod
    def snapshot():
        return {
            "precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }

    def setUp(self):
        self.original = self.snapshot()

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.original["matmul_tf32"]
        torch.set_float32_matmul_precision(self.original["precision"])
        torch.backends.cudnn.allow_tf32 = self.original["cudnn_tf32"]
        torch.backends.cudnn.enabled = self.original["cudnn_enabled"]
        torch.backends.cudnn.benchmark = self.original["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = self.original["cudnn_deterministic"]

    def configure_caller(self, precision="medium", cudnn_tf32=True):
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        return self.snapshot()

    def assert_full_precision(self, caller):
        active = self.snapshot()
        self.assertEqual(active["precision"], "highest")
        self.assertFalse(active["matmul_tf32"])
        self.assertFalse(active["cudnn_tf32"])
        for key in (
            "cudnn_enabled",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
        ):
            self.assertEqual(active[key], caller[key])
        self.assertFalse(torch.is_autocast_enabled("cuda"))

    def test_cuda_context_preserves_high_medium_and_untouched_flags(self):
        for precision in ("high", "medium"):
            for cudnn_tf32 in (True, False):
                with self.subTest(precision=precision, cudnn_tf32=cudnn_tf32):
                    caller = self.configure_caller(precision, cudnn_tf32)
                    with _inference_precision("cuda"):
                        self.assert_full_precision(caller)
                    self.assertEqual(self.snapshot(), caller)

    def test_nested_cuda_context_is_reentrant_and_restores_outer_precision(self):
        caller = self.configure_caller()
        with _inference_precision("cuda"):
            with _inference_precision("cuda"):
                self.assert_full_precision(caller)
            self.assert_full_precision(caller)
        self.assertEqual(self.snapshot(), caller)

    def test_cpu_context_does_not_modify_precision(self):
        caller = self.configure_caller()
        with _inference_precision("cpu"):
            self.assertEqual(self.snapshot(), caller)
        self.assertEqual(self.snapshot(), caller)

    def test_cuda_autocast_is_disabled_and_restored_without_gpu_operations(self):
        caller = self.configure_caller()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            torch.autocast("cuda"),
        ):
            self.assertTrue(torch.is_autocast_enabled("cuda"))
            with _inference_precision("cuda"):
                self.assert_full_precision(caller)
            self.assertTrue(torch.is_autocast_enabled("cuda"))
        self.assertEqual(self.snapshot(), caller)

    def test_public_calibrate_predict_run_in_full_precision_and_restore(self):
        caller = self.configure_caller()
        inferencer = fake_inferencer()
        inferencer.device = torch.device("cuda")
        model = YOCO(inferencer, side="right")

        def condition(pairs):
            self.assert_full_precision(caller)
            inferencer.last_hyper_input = torch.ones(1, 8)

        def inference(frames, *, batched):
            self.assert_full_precision(caller)
            return np.zeros((len(frames), 45), np.float32)

        def reconstruct(theta, betas):
            self.assert_full_precision(caller)
            return np.zeros((len(theta), 21, 3), np.float32)

        inferencer.update_condition.side_effect = condition
        inferencer.inference.side_effect = inference
        model.calibrate(np.zeros((4, 21, 3)), np.zeros((4, 21, 3)))
        self.assertEqual(self.snapshot(), caller)
        with mock.patch.object(model, "_reconstruct_joints", side_effect=reconstruct):
            result = model.predict(np.zeros((3, 21, 3)), batch_size=2)
        self.assertEqual(result.theta.shape, (3, 45))
        self.assertEqual(self.snapshot(), caller)

    def test_calibration_exception_restores_caller_precision(self):
        caller = self.configure_caller()
        inferencer = fake_inferencer()
        inferencer.device = torch.device("cuda")
        model = YOCO(inferencer, side="right")

        def fail(pairs):
            self.assert_full_precision(caller)
            raise RuntimeError("conditioning failed")

        inferencer.update_condition.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "conditioning failed"):
            model.calibrate(np.zeros((4, 21, 3)), np.zeros((4, 21, 3)))
        self.assertEqual(self.snapshot(), caller)
        self.assertFalse(model._calibrated)

    def test_prediction_exception_restores_caller_precision(self):
        caller = self.configure_caller("high")
        inferencer = fake_inferencer()
        inferencer.device = torch.device("cuda")
        model = YOCO(inferencer, side="right")
        model.calibrate(np.zeros((4, 21, 3)), np.zeros((4, 21, 3)))

        def fail(frames, *, batched):
            self.assert_full_precision(caller)
            raise RuntimeError("inference failed")

        inferencer.inference.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            model.predict(np.zeros((3, 21, 3)))
        self.assertEqual(self.snapshot(), caller)


if __name__ == "__main__":
    unittest.main()
