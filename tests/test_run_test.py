from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.run_test import (
    _image_candidates,
    _image_sources,
    build_parser,
    configure_environment,
)


class ImageSourceTests(unittest.TestCase):
    def test_folder_source_includes_every_supported_image_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            nested = folder / "nested"
            nested.mkdir()
            expected = {
                folder / "first.jpg",
                folder / "second.PNG",
                nested / "third.webp",
            }
            for image_path in expected:
                image_path.touch()
            (folder / "notes.txt").touch()

            args = build_parser().parse_args(
                ["--mode", "0", "--image-dir", str(folder)]
            )

            self.assertEqual(set(_image_sources(args)), expected)
            self.assertEqual(set(_image_candidates(folder)), expected)

    def test_explicit_image_takes_precedence_over_image_directory(self) -> None:
        args = build_parser().parse_args(
            ["--mode", "0", "--image", "chosen.jpg", "--image-dir", "unused"]
        )

        self.assertEqual(_image_sources(args), [Path("chosen.jpg")])

    def test_empty_image_directory_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = build_parser().parse_args(
                ["--mode", "0", "--image-dir", temporary_directory]
            )

            with self.assertRaises(FileNotFoundError):
                _image_sources(args)

    def test_optional_models_keep_configured_defaults_without_flags(self) -> None:
        args = build_parser().parse_args(["--mode", "0"])

        with patch.dict(os.environ, {}, clear=True):
            configure_environment(args)
            self.assertNotIn("ENABLE_RISK_MODEL", os.environ)
            self.assertNotIn("ENABLE_BARCODE_MODEL", os.environ)

    def test_optional_models_can_be_explicitly_disabled(self) -> None:
        args = build_parser().parse_args(
            ["--mode", "0", "--disable-risk", "--disable-barcode"]
        )

        with patch.dict(os.environ, {}, clear=True):
            configure_environment(args)
            self.assertEqual(os.environ["ENABLE_RISK_MODEL"], "0")
            self.assertEqual(os.environ["ENABLE_BARCODE_MODEL"], "0")


if __name__ == "__main__":
    unittest.main()
