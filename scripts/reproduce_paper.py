"""Run and verify YOCO's complete, fixed three-seed release experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproduction/huggingface_release.json"
INTERHAND = (
    "datasets/interhand/"
    "InterHand_retargeted_30fps_biased_super_mano_fk20cm_bad_joint_marked.hdf5"
)
AVP = "datasets/avp/aligned_avp_glove_val.hdf5"
AVP_TRAIN_NEW = "datasets/avp/aligned_avp_glove_train_new.hdf5"
METHODS = {
    "yoco": "hypernet",
    "ridge": "ridge",
    "last_layer": "last_layer",
    "full_ft": "full_ft",
    "lora": "lora",
    "no_calibration": "best_world_hand_joints_biased",
}
METRICS = ("mpjpe_mm", "pa_mpjpe_mm", "tip_mpjpe_mm", "aucj", "pinch_distance_l2_mm")
SEEDS = (0, 1, 2)


@dataclass
class Step:
    name: str
    command: list[str]
    verification_path: str | None = None
    setting: str | None = None
    method: str | None = None
    reference_profile: str | None = None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--table", choices=("1", "2", "all"), default="all")
    result.add_argument("--asset-root", type=Path, default=ROOT / "assets")
    result.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "eval/results/reproduction",
        help="Fresh directory; existing directories are never reused or overwritten.",
    )
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument(
        "--methods",
        nargs="+",
        choices=(*METHODS, "all"),
        default=["yoco"],
        help="YOCO by default. 'all' also runs the five original paper baselines.",
    )
    result.add_argument(
        "--conditioning-mode",
        choices=("legacy", "eval"),
        default="legacy",
        help="Select the matching seeded release reference; applies only to YOCO.",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command plan only; does not inspect local assets or evaluate.",
    )
    return result


def selected_methods(names: list[str]) -> list[str]:
    if "all" in names:
        if len(names) != 1:
            raise ValueError("Use --methods all by itself.")
        return list(METHODS)
    if len(names) != len(set(names)):
        raise ValueError("A method may only be selected once.")
    return names


def settings(table: str) -> list[str]:
    return (
        ["interhand", "avp"]
        if table == "all"
        else [{"1": "interhand", "2": "avp"}[table]]
    )


def required_assets(table: str, methods: list[str]) -> list[str]:
    files = [
        f"models/frozen_estimator/{name}"
        for name in ("model.pth", "config.yaml", "norm_stats.pkl")
    ]
    if "yoco" in methods:
        files.extend(
            f"models/yoco/{name}"
            for name in ("model.pth", "config.yaml", "norm_stats.pkl")
        )
    selected = settings(table)
    if "interhand" in selected:
        files.append(INTERHAND)
    if "avp" in selected:
        files.append(AVP)
        if {"lora", "no_calibration"} & set(methods):
            files.append(AVP_TRAIN_NEW)
    return files


def release_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "published":
        raise ValueError("The asset manifest is not a published release.")
    repositories = manifest.get("repositories", {})
    entries = manifest.get("files", [])
    if not repositories or not entries:
        raise ValueError("The asset manifest is incomplete.")
    for repository in repositories.values():
        if (
            not re.fullmatch(r"[0-9a-f]{40}", repository.get("revision", ""))
            or not repository.get("repo_id")
            or repository.get("repo_type") not in ("model", "dataset")
        ):
            raise ValueError("Every Hub repository must have a pinned commit and type.")
    local_paths = set()
    for entry in entries:
        local = Path(entry.get("local_path", ""))
        remote = Path(entry.get("path", ""))
        if (
            not local.parts
            or local.is_absolute()
            or ".." in local.parts
            or not remote.parts
            or remote.is_absolute()
            or ".." in remote.parts
            or entry.get("repository") not in repositories
            or not isinstance(entry.get("bytes"), int)
            or entry["bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))
            or str(local) in local_paths
        ):
            raise ValueError("The asset manifest contains invalid or duplicate files.")
        local_paths.add(str(local))
    return manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(asset_root: Path, names: list[str], manifest: dict) -> list[dict]:
    indexed = {entry["local_path"]: entry for entry in manifest["files"]}
    verified = []
    root = asset_root.resolve()
    for name in names:
        entry = indexed.get(name)
        if entry is None:
            raise ValueError(
                f"Required asset is absent from the release manifest: {name}"
            )
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Asset resolves outside the asset directory: {name}")
        if not path.is_file():
            raise ValueError(
                f"Missing asset: {path}; run scripts/download_reproduction_assets.py"
            )
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Release asset size or SHA256 differs: {path}")
        verified.append(
            {"path": str(path), "bytes": entry["bytes"], "sha256": entry["sha256"]}
        )
    return verified


def command(script: str, *arguments: object) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / script),
        *(str(value) for value in arguments),
    ]


def build_plan(args: argparse.Namespace, methods: list[str]) -> list[Step]:
    asset_root = args.asset_root.resolve()
    output = args.output_dir.resolve()
    base = asset_root / "models/frozen_estimator"
    yoco = asset_root / "models/yoco"
    legacy = output / "assets/avp_glove_table2_legacy.hdf5"
    steps = []
    if "avp" in settings(args.table) and {"lora", "no_calibration"} & set(methods):
        steps.append(
            Step(
                "Prepare legacy Table 2 cohort",
                command(
                    "prepare_table2_legacy_data.py",
                    "--val-data",
                    asset_root / AVP,
                    "--train-new-data",
                    asset_root / AVP_TRAIN_NEW,
                    "--output",
                    legacy,
                ),
            )
        )
    for setting in settings(args.table):
        data = asset_root / (INTERHAND if setting == "interhand" else AVP)
        for seed in SEEDS:
            for name in methods:
                method_data = (
                    legacy
                    if setting == "avp" and name in ("lora", "no_calibration")
                    else data
                )
                common = [
                    "--data-path",
                    method_data,
                    "--output-dir",
                    output / setting / f"seed_{seed}",
                    "--num-calib-actions",
                    4,
                    "--calibration-num",
                    200,
                    "--seed",
                    seed,
                    "--sample-idx",
                    -1,
                    "--side-filter",
                    "right",
                    "--pred-beta-source",
                    "group",
                    "--device",
                    args.device,
                ]
                if name == "yoco":
                    arguments = [
                        "--model-dir",
                        yoco,
                        "--canonicalizer-extractor-path",
                        base,
                        "--variants",
                        "hypernet",
                        "--inference-seed",
                        seed,
                        "--calib-selection",
                        "random",
                        "--input-key",
                        "world_hand_joints_biased",
                        "--true-key",
                        "world_hand_joints_true",
                        "--paired-target-key",
                        "world_hand_joints",
                        "--metric-gt-key",
                        "world_hand_joints",
                        *common,
                    ]
                    if args.conditioning_mode == "legacy":
                        arguments.append("--legacy-conditioning")
                    invocation = command(
                        "export_hypernet_finetune_predictions.py", *arguments
                    )
                elif name == "no_calibration":
                    invocation = command(
                        "export_best_model_predictions.py",
                        "--model-dir",
                        base,
                        "--variants",
                        METHODS[name],
                        "--gt-key",
                        "world_hand_joints",
                        *common,
                    )
                else:
                    arguments = [
                        "--model-dir",
                        base,
                        "--method",
                        name,
                        "--true-joints-key",
                        "world_hand_joints",
                        *common,
                    ]
                    if name == "ridge":
                        arguments.extend(["--ridge-alpha", "1e-3"])
                    else:
                        arguments.extend(
                            [
                                "--steps",
                                200,
                                "--lr",
                                "1e-4",
                                "--weight-decay",
                                "0.0",
                                "--lora-rank",
                                8,
                                "--lora-alpha",
                                8,
                                "--lora-dropout",
                                "0.1",
                                "--lora-use-gates",
                                "--lora-gate-init",
                                "0.0",
                            ]
                        )
                    invocation = command("run_finetune_baselines.py", *arguments)
                steps.append(Step(f"{setting}: {name}, seed {seed}", invocation))
        for name in methods:
            method_data = (
                legacy
                if setting == "avp" and name in ("lora", "no_calibration")
                else data
            )
            method = METHODS[name]
            profile = "release" if name == "yoco" else "paper"
            invocation = command(
                "verify_paper_reproduction.py",
                "--output-root",
                output / setting,
                "--setting",
                setting,
                "--method",
                method,
                "--data-path",
                method_data,
                "--model-dir",
                yoco if name == "yoco" else base,
                "--base-model-dir",
                base,
                "--reference-profile",
                profile,
                "--conditioning-mode",
                args.conditioning_mode,
            )
            filename = (
                "verification.json" if name == "yoco" else f"verification-{method}.json"
            )
            steps.append(
                Step(
                    f"Verify {setting}: {name}",
                    invocation,
                    str(output / setting / filename),
                    setting,
                    method,
                    profile,
                )
            )
    return steps


def verify_receipt(step: Step, conditioning_mode: str) -> dict:
    report = json.loads(Path(step.verification_path).read_text())
    expected_units = (
        20
        if step.setting == "interhand"
        else (4 if step.method in ("lora", "best_world_hand_joints_biased") else 3)
    )
    expected_frames = (
        216340
        if step.setting == "interhand"
        else (5315 if expected_units == 4 else 4128)
    )
    if (
        report.get("accuracy_passed") is not True
        or report.get("setting") != step.setting
        or report.get("method") != step.method
        or report.get("reference_profile") != step.reference_profile
        or report.get("issues") != []
        or (
            step.reference_profile == "release"
            and report.get("conditioning_mode") != conditioning_mode
        )
    ):
        raise ValueError(
            f"Verification did not certify the selected experiment: {step.verification_path}"
        )
    seeds = report.get("per_seed", [])
    if [row.get("seed") for row in seeds] != list(SEEDS) or any(
        row.get("num_units") != expected_units
        or row.get("num_eval_frames") != expected_frames
        for row in seeds
    ):
        raise ValueError(
            f"Verification has incomplete seed/unit metadata: {step.verification_path}"
        )
    if len(report.get("assets", [])) != 7 or any(
        not entry.get("path")
        or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))
        or not isinstance(entry.get("bytes"), int)
        or entry["bytes"] <= 0
        for entry in report["assets"]
    ):
        raise ValueError(
            f"Verification has incomplete asset metadata: {step.verification_path}"
        )
    for key in METRICS:
        value = report.get("mean", {}).get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(
                f"Verification has incomplete metrics: {step.verification_path}"
            )
    return report


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    try:
        methods = selected_methods(args.methods)
        manifest = release_manifest()
        steps = build_plan(args, methods)
        if args.dry_run:
            print("Dry run only: assets and results have not been evaluated.")
            for step in steps:
                print(f"\n{step.name}\n{shlex.join(step.command)}")
            return
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError(
                "Use a fresh --output-dir; existing outputs are never reused."
            )
        assets = verify_assets(
            args.asset_root, required_assets(args.table, methods), manifest
        )
        mano = Path(os.environ.get("MANO_PATH", ""))
        if not os.environ.get("MANO_PATH") or any(
            not (mano / name).is_file() for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl")
        ):
            raise ValueError(
                "Set MANO_PATH to a licensed directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl."
            )
        if args.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ValueError("--device cuda requires an available CUDA device.")
        elif {"lora", "full_ft", "last_layer"} & set(methods):
            print(
                "Note: paper fine-tuning references were validated on an H100; CPU dropout RNG can differ.",
                file=sys.stderr,
            )
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError) as error:
        cli.error(str(error))
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("MKL_NUM_THREADS", "8")
    receipt = {
        "accuracy_passed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "table": args.table,
        "methods": methods,
        "device": args.device,
        "conditioning_mode": args.conditioning_mode,
        "seeds": list(SEEDS),
        "scope": "Release YOCO accuracy and selected original paper baselines; not the full original Table 2 YOCO row or timing.",
        "asset_manifest_sha256": sha256(MANIFEST),
        "assets": assets,
        "plan": [asdict(step) for step in steps],
        "completed_steps": 0,
        "verifications": [],
    }
    destination = args.output_dir / "reproduction_run.json"

    def save() -> None:
        destination.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        for index, step in enumerate(steps):
            print(
                f"\n[{index + 1}/{len(steps)}] {step.name}\n{shlex.join(step.command)}",
                flush=True,
            )
            subprocess.run(step.command, cwd=ROOT, env=environment, check=True)
            if step.verification_path is not None:
                report = verify_receipt(step, args.conditioning_mode)
                receipt["verifications"].append(
                    {
                        "path": step.verification_path,
                        "sha256": sha256(Path(step.verification_path)),
                        "setting": step.setting,
                        "method": step.method,
                        "reference_profile": step.reference_profile,
                        "mean": report["mean"],
                    }
                )
            receipt["completed_steps"] = index + 1
            save()
    except (Exception, KeyboardInterrupt) as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    receipt["accuracy_passed"] = True
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"\nAll selected accuracy checks passed. Receipt: {destination}")


if __name__ == "__main__":
    main()
