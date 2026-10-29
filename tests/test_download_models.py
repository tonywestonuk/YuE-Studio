"""The model downloader's argument parsing and pinned-parent discovery (issues #8, #11)."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[1] / "tools" / "download_models.py"
SPEC = importlib.util.spec_from_file_location("download_models", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ParseRepo(unittest.TestCase):
    def test_plain_repo(self):
        self.assertEqual(MODULE.parse_repo("m-a-p/SheetSage2"), ("m-a-p/SheetSage2", None))

    def test_pinned_repo(self):
        self.assertEqual(MODULE.parse_repo("m-a-p/MERT-v2-FullSong@d8ba1c7"), ("m-a-p/MERT-v2-FullSong", "d8ba1c7"))

    def test_last_at_splits(self):
        self.assertEqual(MODULE.parse_repo("org/name@tag@rev"), ("org/name@tag", "rev"))

    def test_empty_revision_or_repo(self):
        self.assertEqual(MODULE.parse_repo("org/name@"), ("org/name@", None))
        self.assertEqual(MODULE.parse_repo("@rev"), ("@rev", None))


class PinnedParent(unittest.TestCase):
    def snapshot(self, root, config):
        snapshot = root / "snap"
        snapshot.mkdir()
        if config is not None:
            (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return snapshot

    def test_sheetsage_style_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = self.snapshot(Path(tmp), {"base_model_name_or_path": "m-a-p/MERT-v2-FullSong",
                                                 "base_model_revision": "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"})
            self.assertEqual(MODULE.pinned_parent(snapshot),
                             ("m-a-p/MERT-v2-FullSong", "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"))

    def test_unpinned_missing_or_broken_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(MODULE.pinned_parent(self.snapshot(Path(tmp), {"model_type": "yue2"})))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(MODULE.pinned_parent(self.snapshot(Path(tmp), None)))
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = self.snapshot(Path(tmp), None)
            (snapshot / "config.json").write_text("{not json", encoding="utf-8")
            self.assertIsNone(MODULE.pinned_parent(snapshot))
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = self.snapshot(Path(tmp), {"base_model_name_or_path": tmp, "base_model_revision": "abc"})
            self.assertIsNone(MODULE.pinned_parent(snapshot), "a local parent directory is not fetched")


class MainFollowsPins(unittest.TestCase):
    """main() downloads the parent at its pinned revision and grows `total` for it."""

    def test_parent_appended_once_with_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sizes = {("m-a-p/SheetSage2", None): 700, ("m-a-p/MERT-v2-FullSong", "d8ba"): 1300}
            downloaded = []

            class Info:
                def __init__(self, size):
                    self.siblings = [type("F", (), {"size": size})()]

            class Api:
                def model_info(self, repo, revision=None, files_metadata=False):
                    return Info(sizes[(repo, revision)])

            def snapshot_download(repo, revision=None, max_workers=1):
                downloaded.append((repo, revision))
                snapshot = root / repo.replace("/", "--") / (revision or "main")
                snapshot.mkdir(parents=True, exist_ok=True)
                if repo == "m-a-p/SheetSage2":
                    (snapshot / "config.json").write_text(json.dumps(
                        {"base_model_name_or_path": "m-a-p/MERT-v2-FullSong", "base_model_revision": "d8ba"}))
                return str(snapshot)

            lines = []
            with patch.object(MODULE, "HfApi", Api), patch.object(MODULE, "snapshot_download", snapshot_download), \
                 patch("builtins.print", lambda s, **_: lines.append(json.loads(s))):
                MODULE.main(["m-a-p/SheetSage2"])
            self.assertEqual(downloaded, [("m-a-p/SheetSage2", None), ("m-a-p/MERT-v2-FullSong", "d8ba")])
            self.assertEqual(lines[0]["total"], 700)
            self.assertEqual(lines[-1], {"done": True, "total": 2000})

    def test_explicit_pin_not_downloaded_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloaded = []

            class Info:
                siblings = []

            class Api:
                def model_info(self, repo, revision=None, files_metadata=False):
                    return Info()

            def snapshot_download(repo, revision=None, max_workers=1):
                downloaded.append((repo, revision))
                snapshot = root / repo.replace("/", "--") / (revision or "main")
                snapshot.mkdir(parents=True, exist_ok=True)
                if repo == "m-a-p/SheetSage2":
                    (snapshot / "config.json").write_text(json.dumps(
                        {"base_model_name_or_path": "m-a-p/MERT-v2-FullSong", "base_model_revision": "d8ba"}))
                return str(snapshot)

            with patch.object(MODULE, "HfApi", Api), patch.object(MODULE, "snapshot_download", snapshot_download), \
                 patch("builtins.print", lambda *a, **k: None):
                MODULE.main(["m-a-p/SheetSage2", "m-a-p/MERT-v2-FullSong@d8ba"])
            self.assertEqual(downloaded, [("m-a-p/SheetSage2", None), ("m-a-p/MERT-v2-FullSong", "d8ba")])


if __name__ == "__main__":
    unittest.main()
