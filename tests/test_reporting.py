import tempfile
import unittest
from pathlib import Path

from rc_photonics.reporting import (
    format_markdown_table,
    mean_confidence_interval,
    write_csv_table,
    write_svg_line_plot,
)


class ReportingTests(unittest.TestCase):
    def test_confidence_interval_contains_mean(self) -> None:
        interval = mean_confidence_interval([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(interval.mean, 2.5)
        self.assertLess(interval.lower, interval.mean)
        self.assertGreater(interval.upper, interval.mean)

    def test_markdown_table(self) -> None:
        table = format_markdown_table(("model", "nmse"), (("ESN", 0.1),))

        self.assertIn("| model | nmse |", table)
        self.assertIn("| ESN | 0.1 |", table)

    def test_csv_and_svg_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = write_csv_table(
                Path(directory) / "results.csv",
                ("x", "score"),
                ((1, 0.2), (2, 0.1)),
            )
            svg_path = write_svg_line_plot(
                Path(directory) / "results.svg",
                [1.0, 2.0],
                {"model": [0.2, 0.1]},
                title="Results",
                x_label="x",
                y_label="NMSE",
            )

            self.assertIn("x,score", csv_path.read_text(encoding="utf-8"))
            svg = svg_path.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn(">0.1</text>", svg)
            self.assertIn(">2</text>", svg)


if __name__ == "__main__":
    unittest.main()
