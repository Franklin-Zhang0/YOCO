import ast
import json
import os
import re
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
        self.subtitle = []
        self.subtitle_depth = 0
        self.subtitle_breaks = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.elements.append((tag, attrs))
        if self.subtitle_depth:
            if tag not in {"br", "img", "input", "hr"}:
                self.subtitle_depth += 1
            if tag == "br":
                self.subtitle_breaks += 1
        elif tag == "span" and "title-subtitle" in attrs.get("class", "").split():
            self.subtitle_depth = 1

    def handle_endtag(self, tag):
        if self.subtitle_depth and tag not in {"br", "img", "input", "hr"}:
            self.subtitle_depth -= 1

    def handle_data(self, data):
        if self.subtitle_depth:
            self.subtitle.append(data)


class AuthorLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.depth = 0
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if not self.depth:
            if tag == "div" and "authors" in attrs.get("class", "").split():
                self.depth = 1
            return
        if tag not in {"br", "img", "input", "hr"}:
            self.depth += 1
        if tag == "a":
            self.current = (attrs, [])

    def handle_endtag(self, tag):
        if not self.depth:
            return
        if tag == "a" and self.current is not None:
            attrs, text = self.current
            self.links.append(("".join(text).strip(), attrs))
            self.current = None
        if tag not in {"br", "img", "input", "hr"}:
            self.depth -= 1

    def handle_data(self, data):
        if self.current is not None:
            self.current[1].append(data)


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

    def test_setup_guides_use_locked_uv_without_manual_install_commands(self):
        for name in ("README.md", "INSTALL.md"):
            with self.subTest(file=name):
                text = (ROOT / name).read_text()
                self.assertIn("uv sync --frozen --extra cpu", text)
                self.assertIn("--extra cu124", text)
                self.assertNotRegex(
                    text,
                    r"\b(?:conda\s+(?:create|activate|install)|pip[3]?\s+install)\b",
                )
        for name in ("README.md", "INSTALL.md", "reproduction/README.md"):
            for block in re.findall(
                r"```bash\n(.*?)```", (ROOT / name).read_text(), re.DOTALL
            ):
                for line in block.splitlines():
                    if re.search(r"\buv\s+(?:run|sync)\b", line):
                        with self.subTest(file=name, command=line):
                            self.assertIn("--frozen", line)
                            self.assertRegex(line, r"--extra (?:cpu|cu124)\b")

    def test_readme_exposes_public_api_and_inference_entrypoint(self):
        text = (ROOT / "README.md").read_text()
        python_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        self.assertTrue(python_blocks)
        parsed = [ast.parse(block) for block in python_blocks]
        self.assertTrue(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "calibration_net.yoco"
                and any(alias.name == "YOCO" for alias in node.names)
                for module in parsed
                for node in ast.walk(module)
            )
        )
        attributes = {
            node.func.attr
            for module in parsed
            for node in ast.walk(module)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue({"from_pretrained", "calibrate", "predict"} <= attributes)
        self.assertIn('"assets/models/yoco"', text)
        self.assertIn('canonicalizer_dir="assets/models/frozen_estimator"', text)
        self.assertIn("python scripts/run_yoco.py --input observations.npz", text)
        self.assertIn("[the inference guide](docs/inference.md)", text)
        for name in (
            "calibration_net/yoco.py",
            "scripts/run_yoco.py",
            "docs/inference.md",
        ):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_readme_reproduction_links_scripts_without_exporter_loops(self):
        text = (ROOT / "README.md").read_text()
        section = text.split("## Paper Reproduction\n", 1)[1].split("\n## ", 1)[0]
        self.assertIn(
            "[`scripts/reproduce_paper.py`](scripts/reproduce_paper.py)", section
        )
        self.assertIn("[reproduction guide](reproduction/README.md)", section)
        self.assertNotIn("```", section)
        self.assertNotIn("export_hypernet_finetune_predictions.py", text)
        self.assertNotIn("run_finetune_baselines.py", text)
        self.assertNotRegex(text, r"(?m)^\s*for\s+(?:seed|method)\s+in\b")
        for name in ("scripts/reproduce_paper.py", "reproduction/README.md"):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_readme_retains_correct_metric_and_release_scope(self):
        text = " ".join((ROOT / "README.md").read_text().split())
        self.assertRegex(
            text,
            r"Tip-PA.*main paper is a typo.*tip_mpjpe_mm.*unaligned fingertip MPJPE",
        )
        self.assertIn("rounded MPJPE", text)
        self.assertIn("not its full original row", text)
        self.assertIn("private for now", text)

    def test_readme_local_links_resolve(self):
        text = (ROOT / "README.md").read_text()
        checked = 0
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            relative = unquote(urlsplit(target).path)
            self.assertTrue((ROOT / relative).is_file(), target)
            checked += 1
        self.assertGreaterEqual(checked, 4)

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

    def test_website_author_links(self):
        parser = AuthorLinks()
        parser.feed((ROOT / "docs/index.html").read_text())
        expected = [
            ("Yu Zhang", "https://franklin-zhang0.github.io/"),
            ("Yunqi Li", "https://annie-liyunqi.github.io"),
            ("Yushi Du", "https://yushi-du.github.io"),
            ("Yi Ma", "https://www.cs.hku.hk/index.php/people/academic-staff/mayi"),
            ("Yanchao Yang", "https://yanchaoyang.github.io"),
        ]
        self.assertEqual(
            [(name, attrs.get("href")) for name, attrs in parser.links], expected
        )
        for name, attrs in parser.links:
            with self.subTest(author=name):
                self.assertEqual(attrs.get("target"), "_blank")
                self.assertEqual(
                    set(attrs.get("rel", "").split()), {"noopener", "noreferrer"}
                )

    def test_website_subtitle_has_no_forced_line_break(self):
        parser = WebsiteLinks()
        parser.feed((ROOT / "docs/index.html").read_text())
        self.assertEqual(
            "".join(parser.subtitle),
            "Fast Mocap Calibration for Dexterous Teleoperation",
        )
        self.assertEqual(parser.subtitle_breaks, 0)

    def test_website_huggingface_link_follows_code(self):
        links = [attrs for tag, attrs in self.website_links() if tag == "a"]
        code_index = next(
            index
            for index, attrs in enumerate(links)
            if attrs.get("href") == REPOSITORY_URL + "/tree/release/yoco-website"
        )
        manifest = json.loads(
            (ROOT / "reproduction/huggingface_release.json").read_text()
        )
        link = links[code_index + 1]
        self.assertEqual(link["href"], manifest["collection"]["url"])
        self.assertEqual(link["target"], "_blank")
        self.assertEqual(set(link["rel"].split()), {"noopener", "noreferrer"})

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
