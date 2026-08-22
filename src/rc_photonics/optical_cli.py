"""Command-line optical-link simulation, equalization, and plotting."""

import argparse
from pathlib import Path

from rc_photonics.optical_channel import OpticalLinkConfig
from rc_photonics.optical_experiment import run_optical_equalization_experiment
from rc_photonics.reporting import (
    format_markdown_table,
    write_csv_table,
    write_svg_line_plot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate and causally equalize an OOK optical fibre link"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--train-bits", type=int, default=4_000)
    parser.add_argument("--validation-bits", type=int, default=2_000)
    parser.add_argument("--test-bits", type=int, default=4_000)
    parser.add_argument("--symbol-rate-gbaud", type=float, default=10.0)
    parser.add_argument("--samples-per-symbol", type=int, default=8)
    parser.add_argument("--fibre-km", type=float, default=25.0)
    parser.add_argument("--launch-power-dbm", type=float, default=10.0)
    parser.add_argument("--snr-db", type=float, default=18.0)
    parser.add_argument("--bandwidth-ghz", type=float, default=7.5)
    parser.add_argument("--jitter-ui", type=float, default=0.02)
    parser.add_argument("--ssfm-steps", type=int, default=32)
    parser.add_argument("--ffe-taps", type=int, default=17)
    parser.add_argument(
        "--decision-delay",
        type=int,
        default=1,
        help="Causal receiver latency in symbols (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/optical"))
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    train_bits = 800 if arguments.quick else arguments.train_bits
    validation_bits = 400 if arguments.quick else arguments.validation_bits
    test_bits = 800 if arguments.quick else arguments.test_bits
    washout = 50 if arguments.quick else 200
    config = OpticalLinkConfig(
        symbol_rate_gbaud=arguments.symbol_rate_gbaud,
        samples_per_symbol=arguments.samples_per_symbol,
        fibre_length_km=arguments.fibre_km,
        launch_power_dbm=arguments.launch_power_dbm,
        transmitter_bandwidth_ghz=arguments.bandwidth_ghz,
        receiver_bandwidth_ghz=arguments.bandwidth_ghz,
        detector_snr_db=arguments.snr_db,
        timing_jitter_std_ui=arguments.jitter_ui,
        ssfm_steps=arguments.ssfm_steps,
        seed=arguments.seed,
    )
    result = run_optical_equalization_experiment(
        config,
        n_train_bits=train_bits,
        n_validation_bits=validation_bits,
        n_test_bits=test_bits,
        ffe_taps=arguments.ffe_taps,
        decision_delay_symbols=arguments.decision_delay,
        washout=washout,
        seed=arguments.seed + 1_000,
    )
    scores = (result.raw, result.feed_forward, result.esn, result.photonic)
    print(
        format_markdown_table(
            ("equalizer", "test BER", "test NMSE", "validation threshold"),
            tuple(
                (
                    score.name,
                    f"{score.bit_error_rate:.6f}",
                    f"{score.nmse:.6f}",
                    f"{score.threshold:.6f}",
                )
                for score in scores
            ),
        )
    )
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv_table(
        output / "equalizer_scores.csv",
        ("equalizer", "ber", "nmse", "threshold"),
        tuple(
            (score.name, score.bit_error_rate, score.nmse, score.threshold)
            for score in scores
        ),
    )
    displayed = min(100, result.test_bits.size)
    write_svg_line_plot(
        output / "recovered_symbols.svg",
        list(range(displayed)),
        {
            "Target bits": result.test_bits[:displayed],
            "Received": result.test_received_samples[:displayed],
            "FFE": result.test_predictions["FFE"][:displayed],
            "ESN": result.test_predictions["ESN"][:displayed],
            "Photonic": result.test_predictions["Photonic"][:displayed],
        },
        title="Causal OOK recovery",
        x_label="Held-out symbol index",
        y_label="Normalized decision value",
    )
    print(f"Wrote optical results to {output.resolve()}")


if __name__ == "__main__":
    main()
