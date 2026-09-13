"""The committed figures must be one vintage, and all of them must be there.

Sixteen figures are committed and two scripts produce them: src/eda_analysis.py writes the
twelve under results/figures, scripts/figures/generate_figures.py writes the four under
docs/figures. Regenerating one set and not the other leaves the repository publishing two
vintages of the same analysis, and nothing said so. CI checked four of the sixteen for
existence and nothing about the other twelve, so a deleted figure or a half-regenerated
set both passed.

Comparing the matplotlib version recorded in each PNG catches the stale set without
failing merely because matplotlib moved: a whole-set regeneration under a new version
still agrees with itself. That distinction is the point, because the floor in
requirements.txt is `matplotlib>=3.7.0` and today it resolves to a version that renders
different pixels from the one that produced the committed files.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.check_repository import figures_were_rendered_together, renderer_of

ROOT = Path(__file__).resolve().parents[1]

#: A minimal PNG carrying a matplotlib-style Software tEXt chunk.
def png_with_version(version: str | None) -> bytes:
    import struct
    import zlib

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    parts = [b"\x89PNG\r\n\x1a\n", chunk(b"IHDR", header)]
    if version is not None:
        parts.append(chunk(b"tEXt",
                           b"Software\x00Matplotlib version" + version.encode()
                           + b", https://matplotlib.org/"))
    parts.append(chunk(b"IDAT", zlib.compress(b"\x00\x00")))
    parts.append(chunk(b"IEND", b""))
    return b"".join(parts)


class ReadingTheRecordedVersion(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, name: str, version: str | None) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png_with_version(version))
        return path

    def test_it_reads_the_version_out_of_the_png(self):
        self.assertEqual(renderer_of(self.write("a.png", "3.10.8")), "3.10.8")

    def test_a_two_part_version_is_read_whole(self):
        self.assertEqual(renderer_of(self.write("b.png", "3.11")), "3.11")

    def test_a_png_without_the_chunk_is_unknown_not_an_error(self):
        self.assertEqual(renderer_of(self.write("c.png", None)), "unknown")

    def test_one_vintage_passes(self):
        self.write("docs/x.png", "3.10.8")
        self.write("results/y.png", "3.10.8")
        consistent, grouped = figures_were_rendered_together(self.root)
        self.assertTrue(consistent)
        self.assertEqual(list(grouped), ["3.10.8"])

    def test_two_vintages_fail_and_name_both(self):
        self.write("docs/x.png", "3.11.2")
        self.write("results/y.png", "3.10.8")
        consistent, grouped = figures_were_rendered_together(self.root)
        self.assertFalse(consistent)
        self.assertEqual(sorted(grouped), ["3.10.8", "3.11.2"])
        self.assertIn("docs/x.png", grouped["3.11.2"])
        self.assertIn("results/y.png", grouped["3.10.8"])

    def test_no_figures_at_all_is_not_reported_as_a_mismatch(self):
        """An empty tree is a different failure, and the count check is what catches it."""
        consistent, grouped = figures_were_rendered_together(self.root)
        self.assertTrue(consistent)
        self.assertEqual(grouped, {})


class TheCommittedFigures(unittest.TestCase):
    """The repository as it stands, not a fixture."""

    def test_all_sixteen_share_one_vintage(self):
        consistent, grouped = figures_were_rendered_together(ROOT)
        self.assertTrue(consistent, f"more than one matplotlib produced them: {list(grouped)}")

    def test_the_expected_number_is_present_in_each_directory(self):
        for directory, expected in (("docs/figures", 4), ("results/figures", 12)):
            with self.subTest(directory=directory):
                self.assertEqual(len(list((ROOT / directory).glob("*.png"))), expected)

    def test_every_committed_figure_records_a_version(self):
        """"unknown" would mean something other than matplotlib wrote it."""
        for png in sorted(ROOT.glob("**/figures/*.png")):
            with self.subTest(figure=png.name):
                self.assertNotEqual(renderer_of(png), "unknown")
