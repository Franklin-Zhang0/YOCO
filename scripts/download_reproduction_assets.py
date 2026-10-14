"""Download immutable YOCO release assets and verify their payload hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import get_token, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_payload(path: Path, item: dict) -> None:
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise ValueError(f"Payload size or SHA256 differs: {path}")


def download_assets(
    manifest: dict,
    output_dir: Path,
    *,
    models_only: bool = False,
) -> list[Path]:
    if manifest.get("status") != "published":
        raise ValueError("The Hugging Face release has not been published yet.")
    token = get_token()
    if manifest.get("private") and token is None:
        raise ValueError("Private assets require HF_TOKEN or Hugging Face login().")
    root = output_dir.resolve()
    downloaded = []
    for item in manifest["files"]:
        if models_only and item["repository"] != "models":
            continue
        repository = manifest["repositories"][item["repository"]]
        revision = repository["revision"]
        if (
            not revision
            or len(revision) != 40
            or any(char not in "0123456789abcdef" for char in revision)
        ):
            raise ValueError("Asset revisions must be immutable Hub commit hashes.")
        relative = Path(item["local_path"])
        target = (root / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root):
            raise ValueError(f"Asset path escapes the output directory: {relative}")
        if target.exists():
            check_payload(target, item)
            print(f"Verified existing: {relative}")
        else:
            cache = hf_hub_download(
                repo_id=repository["repo_id"],
                repo_type=repository["repo_type"],
                revision=revision,
                filename=item["path"],
                cache_dir=root / ".cache/huggingface",
                token=token,
                endpoint="https://huggingface.co",
            )
            check_payload(Path(cache), item)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                shutil.copyfile(cache, temporary)
                check_payload(temporary, item)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Downloaded and verified: {relative}")
        downloaded.append(target)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "reproduction/huggingface_release.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Download model files without the prepared evaluation datasets.",
    )
    args = parser.parse_args()
    try:
        paths = download_assets(
            json.loads(args.manifest.read_text()),
            args.output_dir,
            models_only=args.models_only,
        )
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        parser.exit(
            1,
            f"Hub request failed (HTTP {status}); check authentication and access "
            "to the selected release repositories.\n",
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Asset download failed: {error}\n")
    print(f"Verified {len(paths)} release files in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
