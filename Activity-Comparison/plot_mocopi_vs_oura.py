#!/usr/bin/env python3
"""Combined MOCOPI vs Oura activity comparison by classroom activity.

MOCOPI defaults to the right-wrist sensor (WristR) for wrist-to-ring comparability.
Units differ (acceleration magnitude vs MET), so panels keep separate x-axes.
Also regenerates the separate MOCOPI and Oura by-class figures.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
MOCOPI_COLOR = "#4C78A8"
OURA_COLOR = "#E37400"
PAPER_CLASS_ORDER = [
    "Homeroom",
    "Math",
    "ELA",
    "ELA/History",
    "History",
    "Social Skills",
    "Cash-out",
    "HW Rein./Study Hall",
    "Friday Funday",
]


def ordered_class_labels(labels: list[str]) -> list[str]:
    present = set(labels)
    ordered = [c for c in PAPER_CLASS_ORDER if c in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def run_device_scripts(args: argparse.Namespace) -> tuple[Path, Path]:
    mocopi_out = args.mocopi_output.expanduser().resolve()
    oura_out = args.oura_output.expanduser().resolve()
    mocopi_out.mkdir(parents=True, exist_ok=True)
    oura_out.mkdir(parents=True, exist_ok=True)

    mocopi_cmd = [
        sys.executable,
        str(REPO_ROOT / "Accelerometer-Patterns" / "plot_by_class_and_time.py"),
        "--output-dir",
        str(mocopi_out),
    ]
    if args.epoch_dir is not None:
        mocopi_cmd.extend(["--epoch-dir", str(args.epoch_dir)])
    mocopi_cmd.extend(["--sensor", args.sensor])

    oura_cmd = [
        sys.executable,
        str(REPO_ROOT / "Oura-Activity" / "plot_by_class_and_time.py"),
        "--output-dir",
        str(oura_out),
        "--ring-root",
        str(args.ring_root),
        "--schedule-root",
        str(args.schedule_root),
    ]

    print("[INFO] Generating MOCOPI figure...", flush=True)
    subprocess.run(mocopi_cmd, check=True)
    print("[INFO] Generating Oura figure...", flush=True)
    subprocess.run(oura_cmd, check=True)
    return mocopi_out, oura_out


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_combined_by_class(mocopi: pd.DataFrame, oura: pd.DataFrame, path: Path) -> None:
    classes = ordered_class_labels(
        sorted(set(mocopi["class_display"].astype(str)) | set(oura["class_display"].astype(str)))
    )
    mocopi_map = mocopi.set_index("class_display")
    oura_map = oura.set_index("class_display")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.8, 0.45 * len(classes) + 2.0)), sharey=True)
    y = np.arange(len(classes))

    for ax, data, color, title, xlabel in (
        (axes[0], mocopi_map, MOCOPI_COLOR, "MOCOPI right wrist", "Acceleration magnitude"),
        (axes[1], oura_map, OURA_COLOR, "Oura ring", "MET"),
    ):
        means = [float(data.loc[c, "mean"]) if c in data.index else np.nan for c in classes]
        ses = [float(data.loc[c, "se"]) if c in data.index else 0.0 for c in classes]
        ax.barh(
            y,
            means,
            xerr=ses,
            color=color,
            alpha=0.9,
            capsize=3,
            height=0.7,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            nm = [int(mocopi_map.loc[c, "n"]) if c in mocopi_map.index else 0 for c in classes]
            no = [int(oura_map.loc[c, "n"]) if c in oura_map.index else 0 for c in classes]
            ax.set_yticks(y)
            ax.set_yticklabels([f"{c}  (n={a}/{b})" for c, a, b in zip(classes, nm, no)])
            ax.invert_yaxis()

    fig.suptitle("Activity by classroom activity (MOCOPI right wrist vs Oura)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MOCOPI and Oura activity by class.")
    parser.add_argument("--epoch-dir", type=Path, default=None)
    parser.add_argument(
        "--sensor",
        default="WristR",
        help="MOCOPI sensor (default: WristR / right wrist).",
    )
    parser.add_argument(
        "--ring-root",
        type=Path,
        default=Path.home() / "Downloads" / "OuraRing",
    )
    parser.add_argument(
        "--schedule-root",
        type=Path,
        default=Path.home() / "Downloads" / "Apple Watch Export CSVs",
    )
    parser.add_argument(
        "--mocopi-output",
        type=Path,
        default=REPO_ROOT / "Accelerometer-Patterns" / "outputs",
    )
    parser.add_argument(
        "--oura-output",
        type=Path,
        default=REPO_ROOT / "Oura-Activity" / "outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--skip-regen",
        action="store_true",
        help="Reuse existing device summary CSVs instead of regenerating figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.skip_regen:
        mocopi_out = args.mocopi_output.expanduser().resolve()
        oura_out = args.oura_output.expanduser().resolve()
    else:
        mocopi_out, oura_out = run_device_scripts(args)

    try:
        mocopi_class = load_summary(mocopi_out / "summary_by_class.csv")
        oura_class = load_summary(oura_out / "summary_by_class.csv")
    except FileNotFoundError as exc:
        print(f"[Fatal Error] Missing summary CSV: {exc}")
        sys.exit(1)

    for stale in ("compare_by_5min.png",):
        old = out / stale
        if old.exists():
            old.unlink()

    save_combined_by_class(mocopi_class, oura_class, out / "compare_by_class.png")
    print(f"\n[DONE] Combined outputs in {out}")
    print(f"  Separate MOCOPI: {mocopi_out / 'intensity_by_class.png'}")
    print(f"  Separate Oura:   {oura_out / 'activity_by_class.png'}")


if __name__ == "__main__":
    main()
