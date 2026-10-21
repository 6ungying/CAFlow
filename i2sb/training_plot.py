import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np


# Training often runs on a machine without a display.
matplotlib.use("Agg")
from matplotlib import pyplot as plt


_LOSS_PATTERN = re.compile(
    r"train_it\s+(?P<step>\d+)/\d+.*?loss:(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def read_training_errors(log_path):
    """Read the iteration and loss values emitted by Runner.train."""
    steps = []
    errors = []
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = _LOSS_PATTERN.search(line)
            if match is None:
                continue
            steps.append(int(match.group("step")))
            errors.append(float(match.group("loss")))
    return steps, errors


def _moving_average(values, window):
    if len(values) < window:
        return np.array([], dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    return (cumulative[window:] - cumulative[:-window]) / window


def _recent_trend(errors):
    """Return a robust recent trend based on two adjacent loss windows."""
    count = len(errors)
    window = min(100, count // 2)
    if window < 5:
        return "collecting data", "#6b7280", None

    previous = float(np.median(errors[-2 * window:-window]))
    recent = float(np.median(errors[-window:]))
    if previous <= 0:
        return "collecting data", "#6b7280", None

    change = (recent / previous - 1.0) * 100.0
    if change <= -10.0:
        return "decreasing (converging)", "#15803d", change
    if change >= 10.0:
        return "increasing (possible divergence)", "#b91c1c", change
    return "stable / plateau", "#a16207", change


def save_training_error_plot(steps, errors, output_path, smooth_window=50):
    """Save a log-scale training-error plot with a smoothed trend line."""
    if len(steps) != len(errors):
        raise ValueError("steps and errors must have the same length")
    if smooth_window < 1:
        raise ValueError("smooth_window must be at least 1")

    steps = np.asarray(steps, dtype=int)
    errors = np.asarray(errors, dtype=float)
    valid = np.isfinite(errors) & (errors >= 0)
    steps = steps[valid]
    errors = errors[valid]
    if errors.size == 0:
        raise ValueError("no finite, non-negative errors to plot")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep very long runs responsive while preserving the overall raw-loss shape.
    stride = max(1, int(np.ceil(errors.size / 5000)))
    plot_steps = steps[::stride]
    plot_errors = errors[::stride]
    positive_errors = errors[errors > 0]
    floor = positive_errors.min() * 0.5 if positive_errors.size else np.finfo(float).tiny

    smooth_window = min(smooth_window, errors.size)
    smoothed = _moving_average(errors, smooth_window)
    smooth_steps = steps[smooth_window - 1:] if smoothed.size else np.array([])
    trend, trend_color, recent_change = _recent_trend(errors)

    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.plot(
        plot_steps,
        np.maximum(plot_errors, floor),
        color="#60a5fa",
        alpha=0.32,
        linewidth=0.8,
        label="Training MSE (raw)",
    )
    if smoothed.size:
        axis.plot(
            smooth_steps,
            np.maximum(smoothed, floor),
            color="#f97316",
            linewidth=2.2,
            label=f"Moving average ({smooth_window} iterations)",
        )

    axis.set_yscale("log")
    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Mean squared error (log scale; lower is better)")
    axis.set_title("Training Error — Convergence / Divergence View")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(loc="upper right")

    change_text = "" if recent_change is None else f"\nRecent change: {recent_change:+.1f}%"
    axis.text(
        0.02,
        0.04,
        f"Recent trend: {trend}{change_text}\nLatest MSE: {errors[-1]:.4g}",
        transform=axis.transAxes,
        color=trend_color,
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": trend_color,
            "alpha": 0.9,
        },
    )

    figure.tight_layout()
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    figure.savefig(temporary_path, dpi=160)
    plt.close(figure)
    temporary_path.replace(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot training MSE from a CAFlow log file.")
    parser.add_argument("log_file", type=Path, help="log.txt written by train.py")
    parser.add_argument("--output", type=Path, required=True, help="output PNG path")
    parser.add_argument("--smooth-window", type=int, default=50)
    args = parser.parse_args()

    steps, errors = read_training_errors(args.log_file)
    output_path = save_training_error_plot(steps, errors, args.output, args.smooth_window)
    print(f"Saved {len(errors)} training errors to {output_path}")


if __name__ == "__main__":
    main()
