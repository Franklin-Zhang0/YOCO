import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn

from scripts import (
    export_hypernet_finetune_predictions as hypernet,
    run_finetune_baselines as baseline,
)


class SavedPredictionMetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.gt = rng.normal(size=(204, 21, 3)).astype(np.float32) * 0.05
        self.target = np.asarray(baseline.transform_joint3d(self.gt), dtype=np.float32)
        self.saved = self.target + np.asarray([0.002, 0.0, 0.0], dtype=np.float32)
        self.mask = np.zeros((204, 21), dtype=bool)
        self.mask[200, 4] = True
        self.cfg = {"dataset": {"preprocessing_x": [], "preprocessing_y": []}}

    def run_session(
        self,
        *,
        save=True,
        reconstruction=True,
        target_source="dataset",
        dataset_gt=True,
    ):
        with patch.object(
            sys,
            "argv",
            [
                "baseline.py",
                "--model-dir",
                "unused",
                "--data-path",
                "unused",
                "--output-dir",
                "unused",
                "--method",
                "ridge",
                "--calibration-num",
                "200",
                "--target-source",
                target_source,
            ],
        ):
            args = baseline.parse_args()
        args.save_predictions = save
        args.no_reconstruction = not reconstruction
        inputs_seen = []

        def predict(model, inputs):
            inputs_seen.append(len(inputs))
            return torch.full(
                (len(inputs), 45), float(len(inputs)), device=inputs.device
            )

        with (
            h5py.File(
                "saved-pose-test.h5", "w", driver="core", backing_store=False
            ) as data,
            ExitStack() as stack,
        ):
            group = data.create_group("1")
            group.create_dataset("world_hand_joints_biased", data=self.gt)
            group.create_dataset(
                "mano_angles", data=np.zeros((204, 45), dtype=np.float32)
            )
            group.create_dataset("mano_side", data=np.bytes_("right"))
            group.create_dataset("betas", data=np.zeros(10, dtype=np.float32))
            group.create_dataset("gt_joint_mask", data=self.mask)
            if dataset_gt:
                group.create_dataset("world_hand_joints", data=self.gt)
            stack.enter_context(
                patch.object(
                    baseline, "load_model", return_value=(nn.Identity(), {}, self.cfg)
                )
            )
            stack.enter_context(
                patch.object(
                    baseline, "build_preprocessing", return_value=lambda value: value
                )
            )
            stack.enter_context(
                patch.object(
                    baseline, "configure_baseline_model", return_value=nn.Identity()
                )
            )
            adapt = stack.enter_context(
                patch.object(baseline, "adapt_model", return_value=([0.5, 0.25], 0.1))
            )
            stack.enter_context(patch.object(baseline, "predict", side_effect=predict))
            old_metrics = {
                "mpjpe_mm": 100.0,
                "pa_mpjpe_mm": 100.0,
                "tip_mpjpe_mm": 100.0,
                "hdf5_true_mpjpe": 0.1,
                "hdf5_true_mpjpe_mm": 100.0,
                "mse": 1.0,
                "num_metric_joints": 84,
                "num_masked_bad_joints": 0,
            }
            stack.enter_context(
                patch.object(
                    baseline, "reconstruction_metrics", return_value=old_metrics
                )
            )
            recon = stack.enter_context(
                patch.object(
                    baseline, "reconstruct_prediction_joints", return_value=self.saved
                )
            )
            persisted = stack.enter_context(
                patch.object(baseline, "save_session_predictions")
            )
            metric = stack.enter_context(
                patch.object(
                    baseline,
                    "compute_joint_metrics",
                    wraps=baseline.compute_joint_metrics,
                )
            )
            row = baseline.run_session(
                group,
                "1/sample_0",
                args,
                self.cfg,
                {},
                torch.device("cpu"),
                num_calib_actions=4,
                sample_idx=0,
                output_dir=None,
            )
            return row, inputs_seen, adapt, recon, persisted, metric

    def test_receipt_uses_persisted_predictions_not_heldout_forward(self):
        row, seen, adapt, recon, persisted, metric = self.run_session()
        self.assertEqual(seen, [4, 204])
        adapt.assert_called_once()
        self.assertEqual(adapt.call_args.kwargs["steps"], 200)
        self.assertEqual(adapt.call_args.kwargs["lr"], 1e-4)
        recon.assert_called_once()
        persisted.assert_called_once()
        metric.assert_called_once()
        saved = persisted.call_args.kwargs
        np.testing.assert_array_equal(saved["pred_joints"], self.saved)
        np.testing.assert_array_equal(
            saved["pred_theta"], np.full((204, 45), 204.0, dtype=np.float32)
        )
        np.testing.assert_array_equal(saved["eval_idx"], np.arange(200, 204))
        np.testing.assert_array_equal(
            metric.call_args.args[0], saved["pred_joints"][saved["eval_idx"]]
        )
        expected = baseline.compute_joint_metrics(
            self.saved[200:], self.target[200:], ~self.mask[200:]
        )
        for key, value in expected.items():
            self.assertEqual(row[key], value)
        self.assertEqual(row["num_metric_joints"], 83)
        self.assertEqual(row["num_masked_bad_joints"], 1)
        self.assertEqual(row["hdf5_true_mpjpe"], row["mpjpe"])
        self.assertEqual(row["hdf5_true_mpjpe_mm"], row["mpjpe_mm"])
        self.assertEqual(row["paper_pose_metric_source"], "saved_predictions")
        self.assertEqual(row["auxiliary_metric_source"], "heldout_prediction")
        self.assertEqual(row["mse"], 1.0)
        self.assertEqual(row["pred_beta_source"], "group")

    def test_metrics_only_workflow_keeps_existing_behavior(self):
        row, seen, _, recon, persisted, metric = self.run_session(save=False)
        self.assertEqual(seen, [4])
        recon.assert_not_called()
        persisted.assert_not_called()
        metric.assert_not_called()
        self.assertEqual(row["mpjpe_mm"], 100.0)
        self.assertNotIn("paper_pose_metric_source", row)

    def test_no_reconstruction_does_not_enable_pose_metrics(self):
        row, seen, _, _, persisted, metric = self.run_session(reconstruction=False)
        self.assertEqual(seen, [4, 204])
        persisted.assert_called_once()
        metric.assert_not_called()
        self.assertNotIn("mpjpe_mm", row)
        self.assertNotIn("paper_pose_metric_source", row)

    def test_missing_dataset_ground_truth_does_not_change_metric_source(self):
        row, _, _, _, persisted, metric = self.run_session(dataset_gt=False)
        persisted.assert_called_once()
        metric.assert_not_called()
        self.assertEqual(row["mpjpe_mm"], 100.0)
        self.assertNotIn("paper_pose_metric_source", row)

    def test_clean_model_targets_are_not_replaced_by_dataset_ground_truth(self):
        row, seen, _, _, persisted, metric = self.run_session(
            target_source="clean_model"
        )
        self.assertEqual(seen, [204, 4, 204])
        persisted.assert_called_once()
        metric.assert_not_called()
        self.assertEqual(row["mpjpe_mm"], 100.0)
        self.assertNotIn("paper_pose_metric_source", row)

    def test_shared_helper_preserves_yoco_float32_metric_semantics(self):
        self.assertIs(hypernet.compute_joint_metrics, baseline.compute_joint_metrics)
        pred, gt, valid = self.saved[200:], self.target[200:], ~self.mask[200:]
        aligned = hypernet.batch_similarity_align_masked(pred, gt, valid)
        expected_pa = (
            float(np.mean(np.linalg.norm(aligned - gt, axis=-1)[valid])) * 1000.0
        )
        state = torch.random.get_rng_state().clone()
        metrics = baseline.compute_joint_metrics(pred, gt, valid)
        self.assertEqual(metrics["pa_mpjpe_mm"], expected_pa)
        torch.testing.assert_close(torch.random.get_rng_state(), state)


if __name__ == "__main__":
    unittest.main()
