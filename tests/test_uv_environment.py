import unittest
from pathlib import Path
from urllib.parse import urlparse

import tomli

ROOT = Path(__file__).resolve().parents[1]


class UVEnvironmentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = tomli.loads((ROOT / "pyproject.toml").read_text())
        cls.lock = tomli.loads((ROOT / "uv.lock").read_text())

    def test_reference_python_and_package_namespace(self):
        self.assertEqual((ROOT / ".python-version").read_text().strip(), "3.10.12")
        self.assertEqual(self.project["project"]["name"], "calibration_net")
        self.assertEqual(self.project["project"]["requires-python"], ">=3.10,<3.11")

    def test_reference_numerical_stack_is_locked(self):
        versions = {
            package["name"]: package["version"] for package in self.lock["package"]
        }
        self.assertEqual(versions["numpy"], "1.23.5")
        self.assertEqual(versions["scipy"], "1.10.1")
        self.assertEqual(versions["h5py"], "3.16.0")
        self.assertEqual(versions["chumpy"], "0.70")
        torch_versions = {
            package["version"]
            for package in self.lock["package"]
            if package["name"] == "torch"
        }
        self.assertEqual(torch_versions, {"2.6.0+cpu", "2.6.0+cu124"})

    def test_accelerator_profiles_are_mutually_exclusive(self):
        uv = self.project["tool"]["uv"]
        self.assertIn([{"extra": "cpu"}, {"extra": "cu124"}], uv["conflicts"])
        sources = uv["sources"]["torch"]
        self.assertEqual(
            {item["extra"]: item["index"] for item in sources},
            {"cpu": "pytorch-cpu", "cu124": "pytorch-cu124"},
        )

    def test_sources_are_portable_and_mano_is_not_a_payload(self):
        allowed_hosts = {
            "pypi.org",
            "files.pythonhosted.org",
            "download.pytorch.org",
            "download-r2.pytorch.org",
            "github.com",
        }
        for package in self.lock["package"]:
            source = package["source"]
            if "editable" in source:
                self.assertEqual(source["editable"], ".")
            for key in ("registry", "git"):
                if key in source:
                    self.assertIn(urlparse(source[key]).hostname, allowed_hosts)
            for wheel in package.get("wheels", []):
                self.assertIn(urlparse(wheel["url"]).hostname, allowed_hosts)
            if "sdist" in package:
                self.assertIn(urlparse(package["sdist"]["url"]).hostname, allowed_hosts)
        source = self.project["tool"]["uv"]["sources"]["manopth"]
        self.assertEqual(source["rev"], "4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b")
        self.assertNotIn("MANO_LEFT.pkl", (ROOT / "uv.lock").read_text())
        self.assertNotIn("MANO_RIGHT.pkl", (ROOT / "uv.lock").read_text())


if __name__ == "__main__":
    unittest.main()
