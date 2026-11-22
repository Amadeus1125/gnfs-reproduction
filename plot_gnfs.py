import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pandas as pd
from gnfs import confirmed_warning_mask

matplotlib.use("Agg")

from matplotlib import font_manager, pyplot as plt, ticker


DEFAULT_THRESHOLD = 0.015
GNFS_COLOR = "#2E86AB"
WARNING_COLOR = "#DC3545"
FIRST_WARNING_COLOR = "#FFA500"


def read_scores(path, threshold_override):
    frame = pd.read_csv(path, float_precision="round_trip")
    if frame.empty or not {"window_k1", "score"}.issubset(frame.columns):
        raise ValueError("The score file needs window_k1 and score columns and at least one row.")

    window_values = pd.to_numeric(frame["window_k1"], errors="raise").to_numpy()
    scores = pd.to_numeric(frame["score"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(window_values).all() or (window_values < 1).any():
        raise ValueError("Window IDs must be positive integers.")
    windows = window_values.astype(np.int64)
    if not np.array_equal(windows, window_values) or (np.diff(windows) != 1).any():
        raise ValueError("Window IDs must be consecutive integers in ascending order.")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("GNFS scores must be finite and nonnegative.")

    threshold = threshold_override
    if threshold is None and "threshold" in frame:
        thresholds = pd.to_numeric(frame["threshold"], errors="raise").to_numpy(dtype=float)
        threshold = float(thresholds[0])
        if (thresholds != threshold).any():
            raise ValueError("The plot needs a fixed warning threshold.")
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The warning threshold must be finite and positive.")
    return windows, scores, threshold


def draw_scores(windows, scores, threshold, fault_windows, legend_location):
    if len(set(fault_windows)) != len(fault_windows) or not np.isin(fault_windows, windows).all():
        raise ValueError("Fault window IDs must be distinct IDs present in the score file.")
    faults = np.array(sorted(fault_windows), dtype=int)
    confirmed = confirmed_warning_mask(scores, threshold)
    confirmed_positions = np.flatnonzero(confirmed)
    first = int(confirmed_positions[0]) if len(confirmed_positions) else None

    font = font_manager.FontProperties(family="Times New Roman")
    try:
        font_manager.findfont(font, fallback_to_default=False)
        font_name = "Times New Roman"
    except ValueError:
        font_name = "DejaVu Serif"

    with plt.rc_context({
        "font.family": font_name,
        "axes.unicode_minus": False,
        "figure.dpi": 100,
    }):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(windows, scores, color=GNFS_COLOR, linewidth=1, alpha=0.9, label="GNFS")
        ax.fill_between(windows, scores, color=GNFS_COLOR, alpha=0.15)
        ax.axhline(
            threshold, color=WARNING_COLOR, linestyle="--", linewidth=1.5,
            label=f"Warning Threshold ({threshold:.7g})",
        )

        if len(faults):
            if first is not None:
                first_window = int(windows[first])
                ax.scatter(
                    first_window, scores[first], color=FIRST_WARNING_COLOR, s=110, marker="*",
                    edgecolors="black", linewidths=1, zorder=7,
                    label=f"First Warning Window ({first_window})",
                )
                others = confirmed & (windows != first_window) & (windows < faults[-1])
                if others.any():
                    ax.scatter(
                        windows[others], scores[others], color=WARNING_COLOR, s=12,
                        zorder=5, label="Warning Window",
                    )
            else:
                ax.text(
                    0.02, 0.96, "No confirmed warning", transform=ax.transAxes, va="top",
                    fontsize=11, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
                )
            label = "Fault Locations" if len(faults) > 1 else "Fault Location"
            ax.scatter(
                faults, scores[np.searchsorted(windows, faults)], color=WARNING_COLOR, s=110,
                marker="*", edgecolors="black", linewidths=1, zorder=6,
                label=f"{label} ({', '.join(map(str, faults))})",
            )

        ax.set_xlabel("Window ID", fontsize=20)
        ax.set_ylabel("GNFS", fontsize=20)
        ax.tick_params(labelsize=17)
        ax.set_xlim(0, int(windows[-1]) + 5)
        if windows[-1] == 401:
            ax.set_xticks(np.arange(0, 407, 50))
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(8, integer=True))
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        if not len(faults):
            upper_limit = max(0.015, threshold * 1.06, float(scores.max()) * 1.07)
            ax.set_ylim(0, upper_limit)
            locator = ticker.MultipleLocator(0.005) if upper_limit <= 0.04 else ticker.MaxNLocator(5)
            ax.yaxis.set_major_locator(locator)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        elif ax.get_ylim()[1] <= 0.09:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(7))
        location = legend_location or ("upper right" if len(faults) == 1 else "upper left")
        ax.legend(loc=location, fontsize=11, framealpha=0.9)
        fig.tight_layout(pad=0.9, h_pad=1.3, w_pad=1.3)

    return fig, None if first is None else int(windows[first])


def main():
    parser = argparse.ArgumentParser(description="Plot a GNFS score curve and its warning markers.")
    parser.add_argument("scores", type=Path, help="GNFS global_scores.csv.")
    parser.add_argument("output", type=Path, help="New PNG, PDF, or SVG file.")
    parser.add_argument("--threshold", type=float, help="Override the input or default warning threshold.")
    parser.add_argument("--fault-window", type=int, nargs="+", default=[], help="Mapped fault window IDs.")
    parser.add_argument("--legend-location", choices=["upper left", "upper right", "lower left", "lower right"])
    parser.add_argument("--dpi", type=int, default=600, help="Resolution of PNG output.")
    args = parser.parse_args()

    image_format = args.output.suffix.lower().lstrip(".")
    if image_format not in {"png", "pdf", "svg"} or args.dpi <= 0:
        parser.error("Use a .png, .pdf, or .svg output path and positive DPI.")
    if args.output.exists():
        parser.error("The output file already exists.")
    windows, scores, threshold = read_scores(args.scores, args.threshold)
    fig, first_warning = draw_scores(windows, scores, threshold, args.fault_window, args.legend_location)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as temporary:
            staged = Path(temporary) / args.output.name
            with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "path"}):
                fig.savefig(staged, format=image_format, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
            staged.rename(args.output)
    finally:
        plt.close(fig)
    print(f"First warning window: {first_warning}" if first_warning is not None else "No confirmed warning.")
    print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
