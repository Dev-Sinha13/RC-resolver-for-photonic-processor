"""Console entry points for reservoir experiments and reporting."""

import argparse
from pathlib import Path

import numpy as np

from rc_photonics.experiments import (
    run_gaussian_baseline_experiment,
    run_impulse_baseline_experiment,
    run_missing_gap_experiment,
)
from rc_photonics.hardware import HardwareImpairments
from rc_photonics.model_evaluation import (
    ReservoirCandidate,
    default_esn_candidates,
    default_photonic_candidates,
    run_photonic_robustness_experiment,
    run_reservoir_gap_experiment,
    run_reservoir_gaussian_experiment,
    run_reservoir_impulse_experiment,
    run_reservoir_on_split,
)
from rc_photonics.photonic_delay import PhotonicDelayConfig
from rc_photonics.reporting import (
    format_markdown_table,
    write_csv_table,
    write_svg_line_plot,
)
from rc_photonics.sensor_data import (
    download_uci_air_quality,
    load_uci_air_quality,
    prepare_sensor_series,
)


def _print_model_benchmarks(
    title: str,
    candidates: tuple[ReservoirCandidate, ...],
) -> None:
    gaussian_baselines = run_gaussian_baseline_experiment()
    gap_baselines = run_missing_gap_experiment()
    impulse_baselines = run_impulse_baseline_experiment()
    gaussian_results = run_reservoir_gaussian_experiment(candidates)
    gap_results = run_reservoir_gap_experiment(candidates)
    impulse_results = run_reservoir_impulse_experiment(candidates)

    print(f"{title}: Gaussian denoising")
    print(
        format_markdown_table(
            ("noise", "identity", "moving average", "AR", title, "selected"),
            tuple(
                (
                    f"{baseline.noise_standard_deviation:.3f}",
                    f"{baseline.identity_nmse:.6f}",
                    f"{baseline.moving_average_nmse:.6f}",
                    f"{baseline.autoregressive_nmse:.6f}",
                    f"{result.nmse:.6f}",
                    result.selected_candidate,
                )
                for baseline, result in zip(gaussian_baselines, gaussian_results)
            ),
        )
    )
    print(f"\n{title}: missing intervals")
    print(
        format_markdown_table(
            ("gap", "carried", "AR", title, "selected"),
            tuple(
                (
                    baseline.gap_length,
                    f"{baseline.carried_forward_nmse:.6f}",
                    f"{baseline.autoregressive_nmse:.6f}",
                    f"{result.nmse:.6f}",
                    result.selected_candidate,
                )
                for baseline, result in zip(gap_baselines, gap_results)
            ),
        )
    )
    print(f"\n{title}: impulse denoising")
    print(
        format_markdown_table(
            ("probability", "identity", "median", "AR", title, "selected"),
            tuple(
                (
                    f"{baseline.impulse_probability:.3f}",
                    f"{baseline.identity_nmse:.6f}",
                    f"{baseline.median_nmse:.6f}",
                    f"{baseline.autoregressive_nmse:.6f}",
                    f"{result.nmse:.6f}",
                    result.selected_candidate,
                )
                for baseline, result in zip(impulse_baselines, impulse_results)
            ),
        )
    )


def esn_main() -> None:
    _print_model_benchmarks("ESN", default_esn_candidates())


def photonic_main() -> None:
    _print_model_benchmarks("Photonic", default_photonic_candidates())


def robustness_main() -> None:
    base_config = PhotonicDelayConfig(
        n_virtual_nodes=100,
        feedback_gain=0.8,
        leak_rate=0.2,
        input_scaling=1.0,
        phase_bias=np.pi / 4.0,
        seed=42,
    )
    cases = (
        ("ideal", HardwareImpairments()),
        ("noise_0.005", HardwareImpairments(internal_noise_std=0.005)),
        ("noise_0.02", HardwareImpairments(internal_noise_std=0.02)),
        ("attenuation_0.1", HardwareImpairments(feedback_attenuation=0.1)),
        ("attenuation_0.4", HardwareImpairments(feedback_attenuation=0.4)),
        ("quantization_8bit", HardwareImpairments(quantization_bits=8)),
        ("quantization_4bit", HardwareImpairments(quantization_bits=4)),
        ("drift_0.01", HardwareImpairments(drift_std=0.01)),
        ("jitter_0.01", HardwareImpairments(timing_jitter_std=0.01)),
    )
    results = run_photonic_robustness_experiment(base_config, cases)
    print(
        format_markdown_table(
            ("hardware case", "test NMSE"),
            tuple((result.label, f"{result.nmse:.6f}") for result in results),
        )
    )


def sensor_main() -> None:
    parser = argparse.ArgumentParser(description="Run the UCI sensor benchmark")
    parser.add_argument("--path", type=Path)
    parser.add_argument("--column", default="PT08.S1(CO)")
    arguments = parser.parse_args()
    path = arguments.path or download_uci_air_quality("data/raw")
    series = load_uci_air_quality(path, column=arguments.column)
    prepared = prepare_sensor_series(series)
    esn = run_reservoir_on_split(
        prepared.split,
        default_esn_candidates(),
    )
    photonic = run_reservoir_on_split(
        prepared.split,
        default_photonic_candidates(),
    )
    print(f"Dataset: UCI Air Quality {series.name} ({series.values.size} samples)")
    print(
        format_markdown_table(
            ("model", "identity NMSE", "restored NMSE", "selected"),
            (
                (
                    "ESN",
                    f"{esn.identity_nmse:.6f}",
                    f"{esn.nmse:.6f}",
                    esn.selected_candidate,
                ),
                (
                    "Photonic",
                    f"{photonic.identity_nmse:.6f}",
                    f"{photonic.nmse:.6f}",
                    photonic.selected_candidate,
                ),
            ),
        )
    )


def figures_main() -> None:
    output_directory = Path("results")
    baselines = run_gaussian_baseline_experiment()
    gap_baselines = run_missing_gap_experiment()
    impulse_baselines = run_impulse_baseline_experiment()
    esn_gaussian = run_reservoir_gaussian_experiment(default_esn_candidates())
    photonic_gaussian = run_reservoir_gaussian_experiment(
        default_photonic_candidates()
    )
    esn_gaps = run_reservoir_gap_experiment(default_esn_candidates())
    photonic_gaps = run_reservoir_gap_experiment(default_photonic_candidates())
    esn_impulse = run_reservoir_impulse_experiment(default_esn_candidates())
    photonic_impulse = run_reservoir_impulse_experiment(
        default_photonic_candidates()
    )

    noise = [result.noise_standard_deviation for result in baselines]
    gaussian_series = {
        "Identity": [result.identity_nmse for result in baselines],
        "Autoregressive": [result.autoregressive_nmse for result in baselines],
        "ESN": [result.nmse for result in esn_gaussian],
        "Photonic": [result.nmse for result in photonic_gaussian],
    }
    write_svg_line_plot(
        output_directory / "gaussian_nmse.svg",
        noise,
        gaussian_series,
        title="Gaussian denoising",
        x_label="Noise standard deviation",
        y_label="NMSE",
    )
    write_csv_table(
        output_directory / "gaussian_nmse.csv",
        ("noise", *gaussian_series.keys()),
        tuple(
            (noise[index], *(values[index] for values in gaussian_series.values()))
            for index in range(len(noise))
        ),
    )

    gaps = [result.gap_length for result in gap_baselines]
    gap_series = {
        "Carried forward": [
            result.carried_forward_nmse for result in gap_baselines
        ],
        "Autoregressive": [
            result.autoregressive_nmse for result in gap_baselines
        ],
        "ESN": [result.nmse for result in esn_gaps],
        "Photonic": [result.nmse for result in photonic_gaps],
    }
    write_svg_line_plot(
        output_directory / "gap_nmse.svg",
        gaps,
        gap_series,
        title="Missing-interval restoration",
        x_label="Gap length (samples)",
        y_label="Gap-only NMSE",
    )
    write_csv_table(
        output_directory / "gap_nmse.csv",
        ("gap", *gap_series.keys()),
        tuple(
            (gaps[index], *(values[index] for values in gap_series.values()))
            for index in range(len(gaps))
        ),
    )

    probabilities = [result.impulse_probability for result in impulse_baselines]
    impulse_series = {
        "Identity": [result.identity_nmse for result in impulse_baselines],
        "Median": [result.median_nmse for result in impulse_baselines],
        "Autoregressive": [
            result.autoregressive_nmse for result in impulse_baselines
        ],
        "ESN": [result.nmse for result in esn_impulse],
        "Photonic": [result.nmse for result in photonic_impulse],
    }
    write_svg_line_plot(
        output_directory / "impulse_nmse.svg",
        probabilities,
        impulse_series,
        title="Impulse-noise denoising",
        x_label="Impulse probability",
        y_label="NMSE",
    )
    write_csv_table(
        output_directory / "impulse_nmse.csv",
        ("probability", *impulse_series.keys()),
        tuple(
            (
                probabilities[index],
                *(values[index] for values in impulse_series.values()),
            )
            for index in range(len(probabilities))
        ),
    )
    print(f"Wrote results to {output_directory.resolve()}")
