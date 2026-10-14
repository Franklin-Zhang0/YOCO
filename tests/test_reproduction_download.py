from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_reproduction_assets import download_assets


class ReleaseDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.payload = self.root / "cache.bin"
        self.payload.write_bytes(b"verified-checkpoint")
        self.manifest = {
            "status": "published",
            "repositories": {
                "models": {
                    "repo_id": "test/yoco",
                    "repo_type": "model",
                    "revision": "a" * 40,
                }
            },
            "files": [
                {
                    "repository": "models",
                    "path": "yoco/model.pth",
                    "local_path": "models/yoco/model.pth",
                    "bytes": self.payload.stat().st_size,
                    "sha256": hashlib.sha256(self.payload.read_bytes()).hexdigest(),
                }
            ],
        }

    def test_pinned_download_and_existing_payload(self):
        with patch(
            "scripts.download_reproduction_assets.hf_hub_download",
            return_value=str(self.payload),
        ) as fetch:
            paths = download_assets(self.manifest, self.root / "assets")
            self.assertEqual(paths[0].read_bytes(), self.payload.read_bytes())
            self.assertEqual(fetch.call_args.kwargs["revision"], "a" * 40)
            self.assertEqual(
                fetch.call_args.kwargs["endpoint"], "https://huggingface.co"
            )
            download_assets(self.manifest, self.root / "assets")
            self.assertEqual(fetch.call_count, 1)

    def test_rejects_mismatched_cache(self):
        self.manifest["files"][0]["sha256"] = "0" * 64
        with (
            patch(
                "scripts.download_reproduction_assets.hf_hub_download",
                return_value=str(self.payload),
            ),
            self.assertRaises(ValueError),
        ):
            download_assets(self.manifest, self.root / "assets")

    def test_does_not_overwrite_existing_invalid_file(self):
        target = self.root / "assets/models/yoco/model.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"user-owned-file")
        with (
            patch("scripts.download_reproduction_assets.hf_hub_download") as fetch,
            self.assertRaises(ValueError),
        ):
            download_assets(self.manifest, self.root / "assets")
        self.assertEqual(target.read_bytes(), b"user-owned-file")
        fetch.assert_not_called()

    def test_rejects_mutable_revision(self):
        self.manifest["repositories"]["models"]["revision"] = "main"
        with self.assertRaises(ValueError):
            download_assets(self.manifest, self.root / "assets")

    def test_rejects_path_traversal(self):
        self.manifest["files"][0]["local_path"] = "../outside.bin"
        with self.assertRaises(ValueError):
            download_assets(self.manifest, self.root / "assets")

    def test_optional_historical_assets(self):
        self.manifest["files"][0]["optional"] = True
        self.assertEqual(download_assets(self.manifest, self.root / "assets"), [])
        with patch(
            "scripts.download_reproduction_assets.hf_hub_download",
            return_value=str(self.payload),
        ):
            self.assertEqual(
                len(
                    download_assets(
                        self.manifest,
                        self.root / "assets",
                        include_historical=True,
                    )
                ),
                1,
            )

    def test_rejects_unpublished_release(self):
        self.manifest["status"] = "pending_publication"
        with self.assertRaises(ValueError):
            download_assets(self.manifest, self.root / "assets")

    def test_private_release_requires_authentication(self):
        self.manifest["private"] = True
        with (
            patch("scripts.download_reproduction_assets.get_token", return_value=None),
            self.assertRaisesRegex(ValueError, "Private assets require"),
        ):
            download_assets(self.manifest, self.root / "assets")

    def test_private_token_is_passed_to_hub(self):
        self.manifest["private"] = True
        with (
            patch(
                "scripts.download_reproduction_assets.get_token",
                return_value="test-token",
            ),
            patch(
                "scripts.download_reproduction_assets.hf_hub_download",
                return_value=str(self.payload),
            ) as fetch,
        ):
            download_assets(self.manifest, self.root / "assets")
            self.assertEqual(fetch.call_args.kwargs["token"], "test-token")

    def test_models_only_skips_dataset_repository(self):
        self.manifest["files"].append(
            dict(
                self.manifest["files"][0],
                repository="datasets",
                path="avp/validation.hdf5",
                local_path="datasets/avp/validation.hdf5",
            )
        )
        with patch(
            "scripts.download_reproduction_assets.hf_hub_download",
            return_value=str(self.payload),
        ) as fetch:
            paths = download_assets(
                self.manifest, self.root / "assets", models_only=True
            )
            self.assertEqual(len(paths), 1)
            self.assertEqual(fetch.call_count, 1)
            self.assertFalse((self.root / "assets/datasets").exists())


if __name__ == "__main__":
    unittest.main()
