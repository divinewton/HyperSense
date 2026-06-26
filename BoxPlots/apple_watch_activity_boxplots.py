#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
HEATMAPS_DIR = SCRIPT_DIR.parent / "Heatmaps"
if str(HEATMAPS_DIR) not in sys.path:
    sys.path.insert(0, str(HEATMAPS_DIR))

from heatmap_shared import (  # noqa: E402
    APPLE_WATCH_DATA_PREFIX,
    LOCAL_TZ,
    build_schedule_map,
    canonicalize_class_label,
    detect_columns,
    ensure_dir,
    list_participant_folders,
    normalize_timestamp,
    participant_code,
    safe_read_csv,
)


METRIC_CONFIGS = [
    {
        "type_token": "HeartRate",
        "metric_label": "Heart Rate",
        "metric_folder": "heart_rate",
        "value_label": "bpm",
        "valid_min": 40,
        "valid_max": 180,
        "aggregate_by_day_class": False,
    },
    {
        "type_token": "ActiveEnergyBurned",
        "metric_label": "Active Energy Burned",
        "metric_folder": "active_energy_burned",
        "value_label": "calories",
        "valid_min": 0,
        "valid_max": None,
        "aggregate_by_day_class": True,
    },
    {
        "type_token": "BasalEnergyBurned",
        "metric_label": "Basal Energy Burned",
        "metric_folder": "basal_energy_burned",
        "value_label": "calories",
        "valid_min": 0,
        "valid_max": None,
        "aggregate_by_day_class": True,
    },
    {
        "type_token": "AppleExerciseTime",
        "metric_label": "Apple Exercise Time",
        "metric_folder": "apple_exercise_time",
        "value_label": "minutes",
        "valid_min": 0,
        "valid_max": None,
        "aggregate_by_day_class": True,
    },
]

INVALID_CLASS_LABELS = {"", "DELETE", "NONE", "UNLABELED"}
DISPLAY_CLASS_LABELS = {
    "Homework Reinforcement/Study Hall": "HW Rein./Study Hall",
}
PAPER_CLASS_ORDER = [
    "Homeroom",
    "Math",
    "ELA",
    "Social Skills",
    "Cash-out",
    "HW Rein./Study Hall",
    "History",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Apple Watch boxplots by classroom activity."
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
        help="Timezone for naive timestamps. Use the export's local timezone if it is not Pacific.",
    )
    return parser.parse_args()


def participant_sort_key(name: str) -> int:
    return int("".join(ch for ch in name if ch.isdigit()) or 0)


def find_labeled_record_csvs(participant_dir: Path) -> List[Path]:
    labeled_record_dir = participant_dir / "HealthApp" / "Labeled" / "Record"
    if not labeled_record_dir.exists():
        return []
    return sorted(labeled_record_dir.rglob("*.csv"))


def load_participant_metric(
    participant_dir: Path,
    type_token: str,
    *,
    valid_min: Optional[float],
    valid_max: Optional[float],
    aggregate_by_day_class: bool,
    local_tz: str,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in find_labeled_record_csvs(participant_dir):
        df = safe_read_csv(csv_path)
        if df is None or df.empty:
            continue
        if "class" not in df.columns or "Type" not in df.columns:
            continue

        type_mask = df["Type"].astype(str).str.contains(type_token, case=False, na=False)
        df = df[type_mask].copy()
        if df.empty:
            continue

        ts_col, val_col = detect_columns(df)
        if ts_col is None or val_col is None:
            continue

        out = pd.DataFrame()
        out["timestamp"] = normalize_timestamp(df[ts_col], local_tz=local_tz)
        out["value"] = pd.to_numeric(df[val_col], errors="coerce")
        out["class"] = df["class"].astype(str).map(canonicalize_class_label)
        out["participant"] = participant_code(participant_dir.name)
        out = out.dropna(subset=["timestamp", "value", "class"])
        out = out[~out["class"].astype(str).str.upper().isin(INVALID_CLASS_LABELS)]

        if valid_min is not None:
            out = out[out["value"] >= valid_min]
        if valid_max is not None:
            out = out[out["value"] <= valid_max]

        if not out.empty:
            if aggregate_by_day_class:
                out["date"] = out["timestamp"].dt.date
                out = (
                    out.groupby(["date", "class", "participant"], as_index=False)["value"]
                    .sum()
                    .sort_values(["class", "participant", "date"])
                    .reset_index(drop=True)
                )
                out["timestamp"] = pd.NaT
            frames.append(out[["timestamp", "value", "class", "participant"]])

    if not frames:
        return pd.DataFrame(columns=["timestamp", "value", "class", "participant"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["class", "participant", "timestamp"]).reset_index(drop=True)
    return combined


def display_class_label(label: str) -> str:
    return DISPLAY_CLASS_LABELS.get(label, label)


def ordered_activity_labels(combined_df: pd.DataFrame, root: Path) -> List[str]:
    observed = combined_df["class"].dropna().astype(str).unique().tolist()
    observed_set = set(observed)

    preferred = [label for label in PAPER_CLASS_ORDER if label in observed_set]
    remaining = sorted(observed_set.difference(preferred))
    return preferred + remaining


def plot_metric_boxplot(
    combined_df: pd.DataFrame,
    *,
    root: Path,
    output_dir: Path,
    metric_label: str,
    metric_folder: str,
    value_label: str,
) -> None:
    if combined_df.empty:
        print(f"[WARN] No usable {metric_label.lower()} rows found.")
        return

    ensure_dir(output_dir)

    activities = ordered_activity_labels(combined_df, root)
    if not activities:
        print(f"[WARN] No classroom activity labels found for {metric_label.lower()}.")
        return

    display_order = [display_class_label(label) for label in activities]
    plot_df = combined_df.copy()
    plot_df["class_display"] = plot_df["class"].map(display_class_label)

    fig, ax = plt.subplots(figsize=(12, max(5.5, 0.8 * len(activities) + 2)))

    palette = sns.color_palette("Set3", len(activities))
    palette_map = dict(zip(display_order, palette))

    sns.boxplot(
        data=plot_df,
        x="value",
        y="class_display",
        hue="class_display",
        order=display_order,
        hue_order=display_order,
        palette=palette_map,
        orient="h",
        showfliers=True,
        width=0.65,
        dodge=False,
        legend=False,
        ax=ax,
    )

    ax.set_xlabel(value_label, fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="x", alpha=0.22)
    ax.set_title(f"{APPLE_WATCH_DATA_PREFIX} {metric_label} Distribution by Activity")
    sns.despine(ax=ax, left=False, bottom=False)

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    out_path = output_dir / f"{metric_folder}_distribution_by_activity.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def run_boxplot_suite(root: Path, output_dir: Path, local_tz: str) -> None:
    participant_dirs = list_participant_folders(root)
    if not participant_dirs:
        print(f"[ERROR] No participant folders found under {root}")
        return

    for config in METRIC_CONFIGS:
        all_data: List[pd.DataFrame] = []
        for p_dir in participant_dirs:
            metric_df = load_participant_metric(
                p_dir,
                config["type_token"],
                valid_min=config["valid_min"],
                valid_max=config["valid_max"],
                aggregate_by_day_class=config["aggregate_by_day_class"],
                local_tz=local_tz,
            )
            if metric_df.empty:
                continue
            print(
                f"[INFO] {p_dir.name}: {len(metric_df)} valid {config['metric_label'].lower()} rows"
            )
            all_data.append(metric_df)

        if not all_data:
            print(f"[WARN] No usable {config['metric_label'].lower()} rows found for any participant.")
            continue

        combined_df = pd.concat(all_data, ignore_index=True)
        plot_metric_boxplot(
            combined_df,
            root=root,
            output_dir=output_dir,
            metric_label=config["metric_label"],
            metric_folder=config["metric_folder"],
            value_label=config["value_label"],
        )


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parent / "Graphs"
    )
    ensure_dir(out_dir)
    run_boxplot_suite(root=root, output_dir=out_dir, local_tz=args.timezone)


if __name__ == "__main__":
    main()
