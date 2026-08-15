"""Command-line entry point for the fixed baseline benchmark suite."""

from rc_photonics.experiments import (
    run_gaussian_baseline_experiment,
    run_missing_gap_experiment,
)


def main() -> None:
    print("Gaussian denoising (test NMSE)")
    print(
        "noise\twindow\tcurrent_alpha\tar_lags\tar_alpha\t"
        "identity\tcurrent\tmoving_avg\tautoregressive"
    )
    for result in run_gaussian_baseline_experiment():
        print(
            f"{result.noise_standard_deviation:.3f}\t"
            f"{result.selected_window_size}\t"
            f"{result.selected_current_regularization:g}\t"
            f"{result.selected_ar_lags}\t"
            f"{result.selected_ar_regularization:g}\t"
            f"{result.identity_nmse:.6f}\t"
            f"{result.current_sample_nmse:.6f}\t"
            f"{result.moving_average_nmse:.6f}\t"
            f"{result.autoregressive_nmse:.6f}"
        )

    print("\nMissing intervals (gap-only test NMSE)")
    print("gap\twindow\tar_lags\tar_alpha\tcarried\tmasked_avg\tautoregressive")
    for result in run_missing_gap_experiment():
        print(
            f"{result.gap_length}\t"
            f"{result.selected_window_size}\t"
            f"{result.selected_ar_lags}\t"
            f"{result.selected_ar_regularization:g}\t"
            f"{result.carried_forward_nmse:.6f}\t"
            f"{result.masked_moving_average_nmse:.6f}\t"
            f"{result.autoregressive_nmse:.6f}"
        )


if __name__ == "__main__":
    main()
