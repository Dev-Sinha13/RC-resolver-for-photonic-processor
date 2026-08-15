"""Dependency-free tables, confidence intervals, CSV, and SVG figures."""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    n: int


def mean_confidence_interval(
    values: ArrayLike,
    *,
    z_score: float = 1.96,
) -> ConfidenceInterval:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(samples)):
        raise ValueError("values must contain only finite values")
    if not np.isfinite(z_score) or z_score < 0.0:
        raise ValueError("z_score must be finite and non-negative")
    mean = float(np.mean(samples))
    if samples.size == 1:
        margin = 0.0
    else:
        standard_error = float(np.std(samples, ddof=1) / np.sqrt(samples.size))
        margin = float(z_score * standard_error)
    return ConfidenceInterval(mean, mean - margin, mean + margin, samples.size)


def format_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> str:
    if not headers:
        raise ValueError("headers must not be empty")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("every row must match the header count")
    header_line = "| " + " | ".join(str(value) for value in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = [
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    ]
    return "\n".join((header_line, separator, *row_lines))


def write_csv_table(
    path: str | Path,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> Path:
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("every row must match the header count")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
    return output_path


def write_svg_line_plot(
    path: str | Path,
    x_values: ArrayLike,
    series: Mapping[str, ArrayLike],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> Path:
    """Write a compact deterministic SVG line plot without extra packages."""
    x = np.asarray(x_values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("x_values must contain at least two finite values")
    if not series:
        raise ValueError("series must not be empty")
    converted = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in series.items()
    }
    if any(
        values.shape != x.shape or not np.all(np.isfinite(values))
        for values in converted.values()
    ):
        raise ValueError("each series must be finite and match x_values")

    width, height = 800, 500
    left, right, top, bottom = 80, 30, 55, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_y = np.concatenate(tuple(converted.values()))
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    if x_min == x_max:
        raise ValueError("x_values must span a non-zero range")
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    def x_pixel(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_pixel(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" '
        f'font-family="sans-serif" font-size="20">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="black"/>',
        f'<text x="{left}" y="{top + plot_height + 20}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="12">{x_min:.3g}</text>',
        f'<text x="{left + plot_width}" y="{top + plot_height + 20}" '
        f'text-anchor="middle" font-family="sans-serif" font-size="12">'
        f'{x_max:.3g}</text>',
        f'<text x="{left - 8}" y="{top + plot_height + 4}" text-anchor="end" '
        f'font-family="sans-serif" font-size="12">{y_min:.3g}</text>',
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" '
        f'font-family="sans-serif" font-size="12">{y_max:.3g}</text>',
    ]
    for index, (name, values) in enumerate(converted.items()):
        color = colors[index % len(colors)]
        points = " ".join(
            f"{x_pixel(float(x_value)):.2f},{y_pixel(float(y_value)):.2f}"
            for x_value, y_value in zip(x, values)
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2"/>'
        )
        elements.append(
            f'<text x="{left + 10}" y="{top + 20 + index * 20}" '
            f'font-family="sans-serif" font-size="13" fill="{color}">'
            f'{escape(name)}</text>'
        )
    elements.extend(
        (
            f'<text x="{left + plot_width / 2}" y="{height - 15}" '
            f'text-anchor="middle" font-family="sans-serif">{escape(x_label)}</text>',
            f'<text x="18" y="{top + plot_height / 2}" '
            f'transform="rotate(-90 18 {top + plot_height / 2})" '
            f'text-anchor="middle" font-family="sans-serif">{escape(y_label)}</text>',
            "</svg>",
        )
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")
    return output_path
