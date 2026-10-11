"""Plot and save training curves from the CSV written by train.py.

Usage (from anywhere):

    python analysis/plot_training.py                       # asks for the log name (blank = training.csv)
    python analysis/plot_training.py --window 50
    python analysis/plot_training.py --csv run_a run_b --out analysis/compare
    python analysis/plot_training.py --show

Log names/paths are relative to ./logs (".csv" is added if missing).
With one CSV, each smoothed metric shows the raw values (faint) plus a moving
average. With several CSVs, only the moving averages are drawn, one line per
run, so the runs can be compared directly.

Outputs: one PNG per metric plus dashboard.png, written to --out.
Requires: numpy, matplotlib  (pip install matplotlib)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

# Headless by default: just write files. --show opens interactive windows.
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `src` is importable when run as a script

from src.logs import ask_log_path, log_path  # noqa: E402

DEFAULT_OUT = ROOT / "analysis" / "plots"

# (csv column, title, y-axis label, draw a moving average?)
METRICS = [
    ("score", "Score per episode", "Score", True),
    ("total_reward", "Total reward per episode", "Reward", True),
    ("steps", "Episode length", "Steps", True),
    ("mean_loss", "Mean training loss", "Loss", True),
    ("epsilon", "Exploration rate", "Epsilon", False),
    ("best_score", "Best score so far", "Score", False),
]


def load_run(path: Path) -> tuple[str, dict[str, np.ndarray]]:
    """Read a training CSV into {column: float array}. Blank cells become NaN."""
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise SystemExit(f"{path} has no data rows.")

    data = {
        key: np.array(
            [float(row[key]) if row[key] not in ("", None) else np.nan for row in rows]
        )
        for key in rows[0]
    }

    # Derived column: running maximum of the score.
    if "score" in data:
        data["best_score"] = np.maximum.accumulate(data["score"])

    return path.stem, data


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over up to `window` points, ignoring NaNs.

    The first `window - 1` points average over however many values exist so
    far, which matches the avg_score column logged by train.py.
    """
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)

    sums = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    counts = np.concatenate(([0.0], np.cumsum(valid)))

    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)

    total = sums[end] - sums[start]
    count = counts[end] - counts[start]

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def draw_metric(ax, runs, key, title, ylabel, smooth, window) -> None:
    multi = len(runs) > 1

    for name, data in runs:
        x = data["episode"]
        y = data[key]

        if smooth and not multi:
            (raw,) = ax.plot(x, y, alpha=0.25, linewidth=0.8, label="raw")
            ax.plot(
                x, moving_average(y, window),
                color=raw.get_color(), linewidth=2,
                label=f"{window}-episode average",
            )
        elif smooth:
            ax.plot(x, moving_average(y, window), linewidth=2, label=name)
        else:
            ax.plot(x, y, linewidth=2, label=name)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    if multi or smooth:
        ax.legend(loc="best", fontsize=8)


def print_summary(name: str, data: dict[str, np.ndarray], window: int) -> None:
    score = data["score"]
    episodes = data["episode"]
    tail = slice(-window, None)

    best = int(np.argmax(score))
    line = (
        f"{name}: {len(score)} episodes | "
        f"mean score (last {min(window, len(score))}): {score[tail].mean():.2f} | "
        f"best score: {int(score[best])} (episode {int(episodes[best])})"
    )

    if "steps" in data:
        line += f" | mean steps (last {min(window, len(score))}): {data['steps'][tail].mean():.1f}"

    print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RLSnake training curves.")
    parser.add_argument(
        "--csv", nargs="+", default=None,
        help="log names under ./logs (several = comparison mode); omit to be asked",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--window", type=int, default=100, help="moving-average window")
    parser.add_argument("--show", action="store_true", help="also open interactive windows")
    args = parser.parse_args()

    if args.window < 1:
        parser.error("--window must be at least 1")

    return args


def main() -> None:
    args = parse_args()

    paths = [log_path(n) for n in args.csv] if args.csv else [ask_log_path("training.csv")]

    for path in paths:
        if not path.exists():
            raise SystemExit(f"CSV not found: {path}")

    runs = [load_run(path) for path in paths]
    args.out.mkdir(parents=True, exist_ok=True)

    for name, data in runs:
        print_summary(name, data, args.window)

    # Only plot columns that actually exist in every CSV.
    metrics = [m for m in METRICS if all(m[0] in data for _, data in runs)]

    # ---------- One PNG per metric ----------
    for key, title, ylabel, smooth in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        draw_metric(ax, runs, key, title, ylabel, smooth, args.window)
        fig.tight_layout()

        path = args.out / f"{key}.png"
        fig.savefig(path, dpi=150)
        print(f"saved {path}")

        if not args.show:
            plt.close(fig)

    # ---------- Combined dashboard ----------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for ax, (key, title, ylabel, smooth) in zip(axes, metrics):
        draw_metric(ax, runs, key, title, ylabel, smooth, args.window)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    heading = f"Training summary: {runs[0][0]}" if len(runs) == 1 else "Run comparison"
    fig.suptitle(heading, fontsize=14)
    fig.tight_layout()

    dashboard = args.out / "dashboard.png"
    fig.savefig(dashboard, dpi=150)
    print(f"saved {dashboard}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
