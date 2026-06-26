#!/usr/bin/env python3
"""Figure 11-style small multiples: participant HR distributions by activity."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
HEATMAPS_DIR = SCRIPT_DIR.parent / "Heatmaps"
if str(HEATMAPS_DIR) not in sys.path:
    sys.path.insert(0, str(HEATMAPS_DIR))

from heatmap_shared import APPLE_WATCH_DATA_PREFIX, LOCAL_TZ, ensure_dir, list_participant_folders, participant_sort_key  # noqa: E402

from apple_watch_activity_boxplots import (  # noqa: E402
    display_class_label,
    load_participant_metric,
)

FIGURE_11_CLASS_ORDER = [
    "Cash-out",
    "ELA",
    "HW Rein./Study Hall",
    "History",
    "Homeroom",
    "Math",
    "Social Skills",
]

ACTIVITY_COLORS: Dict[str, str] = {
    "Cash-out": "#4C9A8A",
    "ELA": "#E07A5F",
    "HW Rein./Study Hall": "#5B7DB1",
    "History": "#C75B7A",
    "Homeroom": "#8FA63E",
    "Math": "#D4A017",
    "Social Skills": "#A67C52",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build participant-level heart-rate small multiples by classroom activity "
            "(Figure 11 style)."
        )
    )
    parser.add_argument(
        "--root",
        default=os.path.expanduser("~/Downloads/Exports"),
        help="Root folder containing participant folders with labeled HealthApp record CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots. Defaults to BoxPlots/Graphs.",
    )
    parser.add_argument(
        "--timezone",
        default=LOCAL_TZ,
        help="Timezone for naive timestamps.",
    )
    parser.add_argument(
        "--hr-min",
        type=float,
        default=40.0,
        help="Minimum valid heart-rate value.",
    )
    parser.add_argument(
        "--hr-max",
        type=float,
        default=180.0,
        help="Maximum valid heart-rate value.",
    )
    return parser.parse_args()


def load_all_heart_rate(root: Path, local_tz: str, hr_min: float, hr_max: float) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p_dir in list_participant_folders(root):
        metric_df = load_participant_metric(
            p_dir,
            "HeartRate",
            valid_min=hr_min,
            valid_max=hr_max,
            aggregate_by_day_class=False,
            local_tz=local_tz,
        )
        if metric_df.empty:
            print(f"[WARN] No usable heart-rate rows found for {p_dir.name}")
            continue
        print(f"[INFO] {p_dir.name}: {len(metric_df)} valid heart-rate rows")
        frames.append(metric_df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "value", "class", "participant"])

    combined = pd.concat(frames, ignore_index=True)
    combined["class_display"] = combined["class"].map(display_class_label)
    return combined


def participant_order(participants: List[str]) -> List[str]:
    return sorted(participants, key=participant_sort_key)


def figure_11_activity_order(combined_df: pd.DataFrame) -> List[str]:
    observed = set(combined_df["class"].dropna().astype(str).unique())
    preferred = [label for label in FIGURE_11_CLASS_ORDER if label in observed]
    remaining = sorted(observed.difference(preferred))
    return preferred + remaining


def plot_small_multiples(
    combined_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> None:
    if combined_df.empty:
        print("[WARN] No usable heart-rate rows found.")
        return

    activities = figure_11_activity_order(combined_df)
    if not activities:
        print("[WARN] No classroom activity labels found.")
        return

    participants = participant_order(combined_df["participant"].dropna().astype(str).unique().tolist())
    if not participants:
        print("[WARN] No participant labels found.")
        return

    ncols = 3
    nrows = (len(activities) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.6 * nrows),
        sharey=True,
        squeeze=False,
    )

    for idx, activity in enumerate(activities):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        activity_label = display_class_label(activity)
        activity_df = combined_df[combined_df["class"] == activity].copy()
        if activity_df.empty:
            ax.set_visible(False)
            continue

        color = ACTIVITY_COLORS.get(activity_label, "#888888")
        sns.boxplot(
            data=activity_df,
            x="participant",
            y="value",
            order=participants,
            color=color,
            showfliers=True,
            width=0.65,
            ax=ax,
        )

        ax.set_title(activity_label, fontsize=11, pad=8)
        ax.set_xlabel("Participant", fontsize=9)
        ax.set_ylabel("Heart Rate (bpm)" if col == 0 else "")
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(40, 160)
        ax.grid(axis="y", alpha=0.22)
        sns.despine(ax=ax)

    for idx in range(len(activities), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"{APPLE_WATCH_DATA_PREFIX} Heart Rate by Participant and Activity", fontsize=14, y=1.02)
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    out_path = output_dir / "heart_rate_participant_small_multiples.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else SCRIPT_DIR / "Graphs"
    )
    ensure_dir(out_dir)

    combined_df = load_all_heart_rate(
        root,
        local_tz=args.timezone,
        hr_min=args.hr_min,
        hr_max=args.hr_max,
    )
    plot_small_multiples(combined_df, output_dir=out_dir)


if __name__ == "__main__":
    main()
