import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from rc_photonics.sensor_data import (
    SensorSeries,
    load_uci_air_quality,
    prepare_sensor_series,
)


class SensorDataTests(unittest.TestCase):
    def test_loader_selects_longest_contiguous_valid_run(self) -> None:
        contents = (
            "Date;Time;PT08.S1(CO);\n"
            "10/03/2004;18.00.00;1000;\n"
            "10/03/2004;19.00.00;1001;\n"
            "10/03/2004;20.00.00;-200;\n"
            "10/03/2004;21.00.00;1100;\n"
            "10/03/2004;22.00.00;1101;\n"
            "10/03/2004;23.00.00;1102;\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "AirQualityUCI.csv"
            path.write_text(contents, encoding="latin-1")

            series = load_uci_air_quality(path)

        np.testing.assert_array_equal(series.values, [1100.0, 1101.0, 1102.0])
        self.assertEqual(len(series.timestamps), 3)

    def test_preparation_uses_training_statistics_only(self) -> None:
        values = np.arange(20, dtype=np.float64)
        start = datetime(2025, 1, 1)
        series = SensorSeries(
            values=values,
            timestamps=tuple(start + timedelta(hours=index) for index in range(20)),
            name="example",
            unit="units",
        )

        prepared = prepare_sensor_series(series)

        self.assertAlmostEqual(float(np.mean(prepared.split.train)), 0.0)
        self.assertAlmostEqual(float(np.std(prepared.split.train)), 1.0)
        expected_mean = float(np.mean(values[:12]))
        self.assertEqual(prepared.training_mean, expected_mean)

    def test_sensor_series_rejects_mismatched_timestamps(self) -> None:
        with self.assertRaises(ValueError):
            SensorSeries(
                values=np.arange(3, dtype=np.float64),
                timestamps=tuple(),
                name="example",
                unit="units",
            )


if __name__ == "__main__":
    unittest.main()
