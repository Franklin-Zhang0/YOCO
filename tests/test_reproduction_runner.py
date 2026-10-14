from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import reproduce_paper as runner


def flag(command, name):
    return command[command.index(name) + 1]


class ReproductionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.args = runner.parser().parse_args(
            [
                "--asset-root",
                str(self.root / "assets"),
                "--output-dir",
                str(self.root / "results"),
            ]
        )
        self.manifest = {
            "status": "published",
            "repositories": {
                "models": {
                    "repo_id": "test/yoco",
                    "repo_type": "model",
                    "revision": "a" * 40,
                },
            },
            "files": [
                {
                    "repository": "models",
                    "path": "yoco/model.pth",
                    "local_path": "models/yoco/model.pth",
                    "bytes": 5,
                    "sha256": hashlib.sha256(b"model").hexdigest(),
                }
            ],
        }

    def test_release_plan_has_all_fixed_seeds_and_automatic_verification(self):
        steps = runner.build_plan(self.args, ["yoco"])
        self.assertEqual(len(steps), 8)
        exports = [step for step in steps if step.verification_path is None]
        self.assertEqual(
            [flag(step.command, "--seed") for step in exports], ["0", "1", "2"] * 2
        )
        for step in exports:
            self.assertEqual(step.command[0], sys.executable)
            self.assertEqual(
                flag(step.command, "--seed"), flag(step.command, "--inference-seed")
            )
            self.assertEqual(flag(step.command, "--num-calib-actions"), "4")
            self.assertEqual(flag(step.command, "--calibration-num"), "200")
            self.assertEqual(flag(step.command, "--sample-idx"), "-1")
            self.assertEqual(flag(step.command, "--side-filter"), "right")
            self.assertEqual(flag(step.command, "--pred-beta-source"), "group")
            self.assertEqual(
                flag(step.command, "--input-key"), "world_hand_joints_biased"
            )
            self.assertEqual(
                flag(step.command, "--paired-target-key"), "world_hand_joints"
            )
            self.assertIn("--legacy-conditioning", step.command)
            self.assertNotIn("--max-sessions", step.command)
        verifications = [step for step in steps if step.verification_path is not None]
        self.assertEqual([step.setting for step in verifications], ["interhand", "avp"])
        self.assertTrue(
            all(step.reference_profile == "release" for step in verifications)
        )

    def test_eval_mode_selects_eval_reference_without_legacy_flag(self):
        self.args.conditioning_mode = "eval"
        for step in runner.build_plan(self.args, ["yoco"]):
            if step.verification_path:
                self.assertEqual(flag(step.command, "--conditioning-mode"), "eval")
            else:
                self.assertNotIn("--legacy-conditioning", step.command)

    def test_baseline_plan_preserves_legacy_cohort_and_hyperparameters(self):
        self.args.table = "2"
        steps = runner.build_plan(self.args, list(runner.METHODS))
        legacy = str(self.args.output_dir / "assets/avp_glove_table2_legacy.hdf5")
        self.assertEqual(
            Path(steps[0].command[1]).name, "prepare_table2_legacy_data.py"
        )
        self.assertEqual(flag(steps[0].command, "--output"), legacy)
        for step in steps[1:]:
            method = (
                step.method
                if step.verification_path
                else (
                    flag(step.command, "--method")
                    if "--method" in step.command
                    else flag(step.command, "--variants")
                )
            )
            if method in ("lora", "best_world_hand_joints_biased"):
                self.assertEqual(flag(step.command, "--data-path"), legacy)
            else:
                self.assertEqual(
                    flag(step.command, "--data-path"),
                    str(self.args.asset_root / runner.AVP),
                )
            if step.verification_path:
                self.assertEqual(
                    step.reference_profile,
                    "release" if method == "hypernet" else "paper",
                )
            elif method == "ridge":
                self.assertEqual(flag(step.command, "--ridge-alpha"), "1e-3")
            elif method in ("lora", "last_layer", "full_ft"):
                for name, expected in (
                    ("--steps", "200"),
                    ("--lr", "1e-4"),
                    ("--weight-decay", "0.0"),
                    ("--lora-rank", "8"),
                    ("--lora-alpha", "8"),
                    ("--lora-dropout", "0.1"),
                    ("--lora-gate-init", "0.0"),
                ):
                    self.assertEqual(flag(step.command, name), expected)
                self.assertIn("--lora-use-gates", step.command)

    def test_legacy_inputs_are_required_only_for_legacy_baselines(self):
        self.assertNotIn(
            runner.AVP_TRAIN_NEW, runner.required_assets("2", ["yoco", "ridge"])
        )
        self.assertIn(
            runner.AVP_TRAIN_NEW, runner.required_assets("2", ["no_calibration"])
        )
        self.assertNotIn(runner.AVP_TRAIN_NEW, runner.required_assets("1", ["lora"]))

    def test_manifest_rejects_mutable_revision_incomplete_and_duplicate_metadata(self):
        invalid = []
        mutable = copy.deepcopy(self.manifest)
        mutable["repositories"]["models"]["revision"] = "main"
        invalid.append(mutable)
        incomplete = copy.deepcopy(self.manifest)
        del incomplete["files"][0]["sha256"]
        invalid.append(incomplete)
        duplicate = copy.deepcopy(self.manifest)
        duplicate["files"].append(duplicate["files"][0])
        invalid.append(duplicate)
        traversal = copy.deepcopy(self.manifest)
        traversal["files"][0]["local_path"] = "../outside"
        invalid.append(traversal)
        path = self.root / "manifest.json"
        for value in invalid:
            with self.subTest(value=value):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    runner.release_manifest(path)

    def test_asset_hash_validation_preserves_mismatched_files(self):
        target = self.args.asset_root / "models/yoco/model.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"model")
        verified = runner.verify_assets(
            self.args.asset_root, ["models/yoco/model.pth"], self.manifest
        )
        self.assertEqual(verified[0]["sha256"], self.manifest["files"][0]["sha256"])
        target.write_bytes(b"other")
        with self.assertRaises(ValueError):
            runner.verify_assets(
                self.args.asset_root, ["models/yoco/model.pth"], self.manifest
            )
        self.assertEqual(target.read_bytes(), b"other")

    def test_asset_symlink_escape_is_rejected(self):
        target = self.args.asset_root / "models/yoco/model.pth"
        target.parent.mkdir(parents=True)
        outside = self.root / "outside"
        outside.write_bytes(b"model")
        target.symlink_to(outside)
        with self.assertRaises(ValueError):
            runner.verify_assets(
                self.args.asset_root, ["models/yoco/model.pth"], self.manifest
            )

    def test_dry_run_does_not_launch_or_create_outputs(self):
        with (
            patch.object(
                sys,
                "argv",
                [
                    "reproduce_paper.py",
                    "--output-dir",
                    str(self.args.output_dir),
                    "--dry-run",
                ],
            ),
            patch.object(runner.subprocess, "run") as execute,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            runner.main()
        execute.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_existing_output_is_never_reused_even_if_empty(self):
        self.args.output_dir.mkdir()
        with (
            patch.object(
                sys,
                "argv",
                ["reproduce_paper.py", "--output-dir", str(self.args.output_dir)],
            ),
            patch.object(runner.subprocess, "run") as execute,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as result,
        ):
            runner.main()
        self.assertEqual(result.exception.code, 2)
        execute.assert_not_called()
        self.assertEqual(list(self.args.output_dir.iterdir()), [])

    def verification(self, step):
        return {
            "accuracy_passed": True,
            "setting": step.setting,
            "method": step.method,
            "reference_profile": step.reference_profile,
            "conditioning_mode": "legacy",
            "issues": [],
            "per_seed": [
                {"seed": seed, "num_units": 3, "num_eval_frames": 4128}
                for seed in (0, 1, 2)
            ],
            "assets": [
                {"path": f"asset-{index}", "sha256": "a" * 64, "bytes": 1}
                for index in range(7)
            ],
            "mean": {key: 1.0 for key in runner.METRICS},
        }

    def test_verification_rejects_missing_seed_and_false_success(self):
        self.args.table = "2"
        step = runner.build_plan(self.args, ["yoco"])[-1]
        path = Path(step.verification_path)
        path.parent.mkdir(parents=True)
        report = self.verification(step)
        path.write_text(json.dumps(report))
        runner.verify_receipt(step, "legacy")
        report["per_seed"].pop()
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            runner.verify_receipt(step, "legacy")
        report = self.verification(step)
        report["accuracy_passed"] = False
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            runner.verify_receipt(step, "legacy")

    def test_mocked_execution_requires_each_verifier_receipt(self):
        self.args.table = "2"
        step = runner.build_plan(self.args, ["yoco"])[-1]
        mano = self.root / "mano"
        mano.mkdir()
        for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
            (mano / name).write_bytes(b"fixture")

        def execute(command, **kwargs):
            self.assertTrue(kwargs["check"])
            if Path(command[1]).name == "verify_paper_reproduction.py":
                path = Path(step.verification_path)
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(self.verification(step)))

        with (
            patch.object(
                sys,
                "argv",
                [
                    "reproduce_paper.py",
                    "--table",
                    "2",
                    "--output-dir",
                    str(self.args.output_dir),
                ],
            ),
            patch.object(runner, "verify_assets", return_value=[]),
            patch.dict("os.environ", {"MANO_PATH": str(mano)}),
            patch.object(runner.subprocess, "run", side_effect=execute) as calls,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            runner.main()
        self.assertEqual(calls.call_count, 4)
        report = json.loads(
            (self.args.output_dir / "reproduction_run.json").read_text()
        )
        self.assertTrue(report["accuracy_passed"])
        self.assertEqual(report["completed_steps"], 4)
        self.assertEqual(len(report["verifications"]), 1)

    def test_method_selection_rejects_duplicates_or_combined_all(self):
        with self.assertRaises(ValueError):
            runner.selected_methods(["yoco", "yoco"])
        with self.assertRaises(ValueError):
            runner.selected_methods(["all", "ridge"])
        self.assertEqual(runner.selected_methods(["all"]), list(runner.METHODS))

    def test_failed_command_keeps_an_unsuccessful_receipt(self):
        mano = self.root / "mano"
        mano.mkdir()
        for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
            (mano / name).write_bytes(b"fixture")
        with (
            patch.object(
                sys,
                "argv",
                [
                    "reproduce_paper.py",
                    "--table",
                    "2",
                    "--output-dir",
                    str(self.args.output_dir),
                ],
            ),
            patch.object(runner, "verify_assets", return_value=[]),
            patch.dict("os.environ", {"MANO_PATH": str(mano)}),
            patch.object(
                runner.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "failed-evaluation"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            runner.main()
        report = json.loads(
            (self.args.output_dir / "reproduction_run.json").read_text()
        )
        self.assertFalse(report["accuracy_passed"])
        self.assertEqual(report["completed_steps"], 0)
        self.assertEqual(report["verifications"], [])
        self.assertIn("CalledProcessError", report["error"])


if __name__ == "__main__":
    unittest.main()
