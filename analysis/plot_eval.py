"""Plot and save evaluation results logged by eval.py.

Usage (from anywhere):

    python analysis/plot_eval.py                  # asks for the log name (blank = eval.csv)
    python analysis/plot_eval.py --csv my_eval    # name/path relative to ./logs
    python analysis/plot_eval.py --out analysis/plots --show

Rows are grouped by their `checkpoint` column, so one log file can compare
several checkpoints. Interrupted episodes are ignored. Outputs (to --out): the
dashboard as <log name>.png, plus eval_score.png, eval_score_distribution.png,
eval_steps.png and eval_endings.png in <log name>/.
Requires: numpy, matplotlib  (pip install matplotlib)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
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


def load_runs(path: Path) -> dict[str, dict]:
    """Return {checkpoint: {"score": array, "steps": array, "endings": Counter}}."""
    rows_by_checkpoint: dict[str, list[dict]] = {}

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["interrupted"] != "True":
                rows_by_checkpoint.setdefault(row["checkpoint"] or "unnamed", []).append(row)

    if not rows_by_checkpoint:
        raise SystemExit(f"{path} has no completed episodes.")

    return {
        name: {
            "score": np.array([float(r["score"]) for r in rows]),
            "steps": np.array([float(r["steps"]) for r in rows]),
            "endings": Counter(r["end_reason"] for r in rows),
        }
        for name, rows in rows_by_checkpoint.items()
    }


def draw_lines(ax, runs, key, title, ylabel) -> None:
    """Per-episode values, one line per checkpoint, with a dashed mean line."""
    for name, data in runs.items():
        y = data[key]
        x = np.arange(1, len(y) + 1)

        (line,) = ax.plot(
            x, y, linewidth=1.2, markersize=3,
            marker="o" if len(y) <= 50 else None,
            label=f"{name} (mean {y.mean():.1f})",
        )
        ax.axhline(y.mean(), color=line.get_color(), linestyle="--", alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def draw_box(ax, runs) -> None:
    """Score distribution per checkpoint (triangle = mean)."""
    ax.boxplot([data["score"] for data in runs.values()], showmeans=True)
    ax.set_xticks(range(1, len(runs) + 1), list(runs))
    ax.set_title("Score distribution")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.3, axis="y")


def draw_endings(ax, runs) -> None:
    """How episodes ended, as % of episodes per checkpoint."""
    names = list(runs)
    totals = np.array([sum(runs[n]["endings"].values()) for n in names], dtype=float)
    bottom = np.zeros(len(names))

    for reason in sorted({r for data in runs.values() for r in data["endings"]}):
        counts = np.array([runs[n]["endings"].get(reason, 0) for n in names], dtype=float)
        share = counts / totals * 100
        ax.bar(names, share, bottom=bottom, label=reason)
        bottom += share

    ax.set_title("How episodes ended")
    ax.set_ylabel("% of episodes")
    ax.legend(loc="best", fontsize=8)


PANELS = {
    "eval_score": lambda ax, runs: draw_lines(ax, runs, "score", "Score per episode", "Score"),
    "eval_score_distribution": draw_box,
    "eval_steps": lambda ax, runs: draw_lines(ax, runs, "steps", "Episode length", "Steps"),
    "eval_endings": draw_endings,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RLSnake evaluation results.")
    parser.add_argument(
        "--csv", default=None,
        help="log name under ./logs; omit to be asked",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--show", action="store_true", help="also open interactive windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    path = log_path(args.csv) if args.csv else ask_log_path("eval.csv")
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    runs = load_runs(path)
    panel_dir = args.out / path.stem
    panel_dir.mkdir(parents=True, exist_ok=True)

    for name, data in runs.items():
        score = data["score"]
        print(
            f"{name}: {len(score)} episodes | score mean {score.mean():.2f} "
            f"(std {score.std():.2f}, median {np.median(score):.1f}, best {int(score.max())}) | "
            f"mean steps {data['steps'].mean():.1f} | endings {dict(data['endings'])}"
        )

    # ---------- One PNG per panel ----------
    for filename, draw in PANELS.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        draw(ax, runs)
        fig.tight_layout()

        out = panel_dir / f"{filename}.png"
        fig.savefig(out, dpi=150)
        print(f"saved {out}")

        if not args.show:
            plt.close(fig)

    # ---------- Combined dashboard ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for ax, draw in zip(axes.flatten(), PANELS.values()):
        draw(ax, runs)

    fig.suptitle(f"Evaluation summary: {path.stem}", fontsize=14)
    fig.tight_layout()

    dashboard = args.out / f"{path.stem}.png"
    fig.savefig(dashboard, dpi=150)
    print(f"saved {dashboard}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
