import os
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/Franklin-Zhang0/YOCO"
WEBSITE_URL = "https://franklin-zhang0.github.io/YOCO/"
TRAINING_SCRIPTS = ("train_grpo.sh", "train_hyperlora_calib.sh")
CHECKPOINT = (
    "expdata/test_combined_cnn_mlp_leaky_global_mixer/"
    "test_combined_cnn_mlp_leaky_global_mixer_20260111_134713/best"
)


class WebsiteLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class RepositoryPathTests(unittest.TestCase):
    def website_links(self):
        parser = WebsiteLinks()
        parser.feed((ROOT / "docs/index.html").read_text())
        return parser.elements

    def test_clone_instructions_use_yoco(self):
        for name in ("README.md", "INSTALL.md"):
            with self.subTest(file=name):
                text = (ROOT / name).read_text()
                self.assertIn(
                    f"git clone --branch release/yoco-website {REPOSITORY_URL}.git",
                    text,
                )
                self.assertIn("\ncd YOCO\n", text)
                self.assertNotIn("/Calibration_net.git", text)
        self.assertTrue((ROOT / "calibration_net").is_dir())
        self.assertIn(WEBSITE_URL, (ROOT / "README.md").read_text())

    def test_website_metadata_and_code_link(self):
        elements = self.website_links()
        metadata = {
            attrs.get("property"): attrs.get("content")
            for tag, attrs in elements
            if tag == "meta"
        }
        self.assertEqual(metadata["og:url"], WEBSITE_URL)
        self.assertEqual(
            metadata["og:image"], WEBSITE_URL + "assets/images/overview.png"
        )
        self.assertIn(("link", {"rel": "canonical", "href": WEBSITE_URL}), elements)
        self.assertTrue(
            any(
                attrs.get("href") == REPOSITORY_URL + "/tree/release/yoco-website"
                for _, attrs in elements
            )
        )

    def test_local_assets_resolve_under_yoco_prefix(self):
        checked = 0
        for _, attrs in self.website_links():
            for key in ("href", "src", "poster"):
                value = attrs.get(key, "")
                if not value or value.startswith("#") or urlsplit(value).scheme:
                    continue
                resolved = urlsplit(urljoin(WEBSITE_URL, value))
                self.assertTrue(resolved.path.startswith("/YOCO/"), value)
                relative = unquote(resolved.path.removeprefix("/YOCO/"))
                self.assertTrue((ROOT / "docs" / relative).is_file(), value)
                checked += 1
        self.assertGreater(checked, 10)

    def test_sitemap_and_robots_use_yoco(self):
        sitemap = ET.parse(ROOT / "docs/sitemap.xml")
        locations = [
            node.text
            for node in sitemap.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
        ]
        self.assertEqual(locations, [WEBSITE_URL])
        self.assertIn(
            f"Sitemap: {WEBSITE_URL}sitemap.xml",
            (ROOT / "docs/robots.txt").read_text().splitlines(),
        )
        self.assertIn(WEBSITE_URL, (ROOT / "docs/README.md").read_text())

    def training_arguments(self, name, load_path=None):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "YOCO checkout"
            scripts = checkout / "scripts"
            scripts.mkdir(parents=True)
            script = scripts / name
            shutil.copyfile(ROOT / "scripts" / name, script)
            environment = dict(os.environ, MANO_PATH=str(checkout))
            environment.pop("LOAD_PATH", None)
            if load_path is not None:
                environment["LOAD_PATH"] = load_path
            # Capture the invocation without starting Python or GPU training.
            wrapper = 'python() { printf "%s\\n" "$@"; }\nexport -f python\nbash "$1"\n'
            result = subprocess.run(
                ["bash", "-c", wrapper, "bash", str(script)],
                env=environment,
                cwd=checkout,
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            )
            return checkout, result.stdout.splitlines()

    def test_training_paths_follow_renamed_checkout(self):
        for name in TRAINING_SCRIPTS:
            with self.subTest(script=name):
                checkout, arguments = self.training_arguments(name)
                self.assertIn(f"trainer.load_path={checkout / CHECKPOINT}", arguments)
                if name == "train_grpo.sh":
                    self.assertIn(f"config={checkout / 'config/grpo.yaml'}", arguments)

    def test_training_checkpoint_override_preserves_spaces(self):
        path = "/external/checkpoint with spaces/best"
        for name in TRAINING_SCRIPTS:
            with self.subTest(script=name):
                _, arguments = self.training_arguments(name, load_path=path)
                self.assertIn(f"trainer.load_path={path}", arguments)


if __name__ == "__main__":
    unittest.main()
