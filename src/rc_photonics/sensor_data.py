"""Loading and chronological preparation of the UCI Air Quality dataset."""

import csv
import io
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rc_photonics.datasets import ChronologicalSplit, chronological_split


UCI_AIR_QUALITY_URL = (
    "https://cdn.uci-ics-mlr-prod.aws.uci.edu/360/air%2Bquality.zip"
)
UCI_AIR_QUALITY_DOI = "10.24432/C59K5F"


@dataclass(frozen=True)
class SensorSeries:
    """A regularly sampled real sensor sequence with timestamps."""

    values: NDArray[np.float64]
    timestamps: tuple[datetime, ...]
    name: str
    unit: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("values must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must contain only finite samples")
        if len(self.timestamps) != values.size:
            raise ValueError("timestamps must contain one entry per sample")
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.unit:
            raise ValueError("unit must not be empty")
        object.__setattr__(self, "values", values.copy())


@dataclass(frozen=True)
class PreparedSensorData:
    """Chronological standardized splits using training statistics only."""

    split: ChronologicalSplit
    training_mean: float
    training_standard_deviation: float


def download_uci_air_quality(destination_directory: str | Path) -> Path:
    """Download and safely extract the official UCI CSV file."""
    destination = Path(destination_directory)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "AirQualityUCI.csv"
    if output_path.exists():
        return output_path

    with urllib.request.urlopen(UCI_AIR_QUALITY_URL, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        matching_names = [
            name
            for name in archive.namelist()
            if Path(name).name == "AirQualityUCI.csv"
        ]
        if len(matching_names) != 1:
            raise ValueError("UCI archive does not contain one AirQualityUCI.csv")
        with archive.open(matching_names[0]) as source, output_path.open(
            "wb"
        ) as target:
            shutil.copyfileobj(source, target)
    return output_path


def load_uci_air_quality(
    path: str | Path,
    *,
    column: str = "PT08.S1(CO)",
) -> SensorSeries:
    """Load the longest hourly run without UCI's ``-200`` missing marker."""
    source_path = Path(path)
    runs: list[list[tuple[datetime, float]]] = []
    current_run: list[tuple[datetime, float]] = []

    with source_path.open("r", encoding="latin-1", newline="") as source:
        reader = csv.DictReader(source, delimiter=";")
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"column {column!r} was not found")
        for row in reader:
            raw_date = (row.get("Date") or "").strip()
            raw_time = (row.get("Time") or "").strip()
            raw_value = (row.get(column) or "").strip()
            if not raw_date or not raw_time or not raw_value:
                if current_run:
                    runs.append(current_run)
                    current_run = []
                continue
            try:
                timestamp = datetime.strptime(
                    f"{raw_date} {raw_time}",
                    "%d/%m/%Y %H.%M.%S",
                )
                value = float(raw_value.replace(",", "."))
            except ValueError:
                if current_run:
                    runs.append(current_run)
                    current_run = []
                continue
            if value == -200.0 or not np.isfinite(value):
                if current_run:
                    runs.append(current_run)
                    current_run = []
                continue
            if current_run and timestamp - current_run[-1][0] != timedelta(hours=1):
                runs.append(current_run)
                current_run = []
            current_run.append((timestamp, value))
    if current_run:
        runs.append(current_run)
    if not runs:
        raise ValueError("dataset contains no valid hourly sensor samples")

    longest_run = max(runs, key=len)
    return SensorSeries(
        values=np.asarray([value for _, value in longest_run], dtype=np.float64),
        timestamps=tuple(timestamp for timestamp, _ in longest_run),
        name=column,
        unit="sensor response",
    )


def prepare_sensor_series(series: SensorSeries) -> PreparedSensorData:
    """Split chronologically and standardize without future leakage."""
    raw_split = chronological_split(series.values)
    training_mean = float(np.mean(raw_split.train))
    training_std = float(np.std(raw_split.train, ddof=0))
    if training_std <= 0.0:
        raise ValueError("sensor training split must have non-zero variance")
    return PreparedSensorData(
        split=ChronologicalSplit(
            train=(raw_split.train - training_mean) / training_std,
            validation=(raw_split.validation - training_mean) / training_std,
            test=(raw_split.test - training_mean) / training_std,
        ),
        training_mean=training_mean,
        training_standard_deviation=training_std,
    )
