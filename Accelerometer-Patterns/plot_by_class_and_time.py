#!/usr/bin/env python3
"""Two accelerometer figures with plain-language metric labels.

1) By classroom activity
2) By 30-minute clock intervals across the school day

Metric plotted = Intensity from epoch kinematics:
  Acc_Mag = sqrt(X^2 + Y^2 + Z^2) from the MOCOPI sensors
  Intensity = mean Acc_Mag in each 1-minute window
  Higher values = stronger / more abrupt acceleration (more vigorous movement)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY_CLASS_LABELS = {
    "Homework Reinforcement/Study Hall": "HW Rein./Study Hall",
}
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
INVALID_CLASS_LABELS = {"", "DELETE", "NONE", "UNLABELED", "NAN"}


def resolve_epoch_dir(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    candidates.extend(
        [
            Path.home() / "Downloads" / "epoch_kinematics",
            Path.home() / "Documents" / "epoch_kinematics",
            Path.home() / "Downloads" / "MOCOPI",
            Path.home() / "Documents" / "MOCOPI",
        ]
    )
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(
        "Could not find epoch kinematics directory. "
        "Pass --epoch-dir or place CSVs in ~/Downloads/epoch_kinematics."
    )


def display_class_label(label: object) -> str | None:
    raw = str(label).strip()
    if not raw or raw.upper() in INVALID_CLASS_LABELS:
        return None
    return DISPLAY_CLASS_LABELS.get(raw, raw)


def ordered_class_labels(labels: list[str]) -> list[str]:
    present = set(labels)
    ordered = [c for c in PAPER_CLASS_ORDER if c in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def load_epoch_kinematics(epoch_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(epoch_dir.glob("*_epoch_kinematics.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        for participant_dir in sorted(epoch_dir.glob("P*")):
            if not participant_dir.is_dir():
                continue
            path = participant_dir / f"{participant_dir.name}_epoch_kinematics.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            f"No *_epoch_kinematics.csv files found under {epoch_dir}."
        )

    df = pd.concat(frames, ignore_index=True)
    required = {"Sensor", "class", "Intensity", "Participant", "Epoch_1Min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Epoch files missing columns: {sorted(missing)}")

    df = df.copy()
    df["class_display"] = df["class"].map(display_class_label)
    df = df[df["class_display"].notna()].copy()
    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df["Epoch_1Min"] = pd.to_datetime(df["Epoch_1Min"], errors="coerce")
    df = df.dropna(subset=["Intensity", "Epoch_1Min"])
    return df


def bodywide_intensity(df: pd.DataFrame, sensor: str | None) -> pd.DataFrame:
    work = df if sensor is None else df[df["Sensor"] == sensor].copy()
    if work.empty:
        raise ValueError(f"No rows left after sensor filter ({sensor!r}).")
    keys = ["Participant", "Date", "Epoch_1Min", "class_display"]
    return work.groupby(keys, as_index=False, observed=True).agg(
        Intensity=("Intensity", "mean")
    )


def half_hour_range_label(start: pd.Timestamp) -> str:
    end = start + pd.Timedelta(minutes=30)
    return f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"


def add_half_hour_bin(minutes: pd.DataFrame) -> pd.DataFrame:
    out = minutes.copy()
    # Floor to :00 or :30 wall-clock bins (e.g. 8:45 -> 8:30).
    out["HalfHour"] = out["Epoch_1Min"].dt.floor("30min")
    out["HalfHourLabel"] = out["HalfHour"].map(half_hour_range_label)
    out["HalfHourSort"] = out["HalfHour"].dt.hour * 60 + out["HalfHour"].dt.minute
    return out


def cohort_means(person: pd.DataFrame, stratum: str, order: list[str]) -> pd.DataFrame:
    rows = []
    for level in order:
        values = person.loc[person[stratum].astype(str).eq(str(level)), "Intensity"]
        if values.empty:
            continue
        rows.append(
            {
                stratum: level,
                "mean": float(values.mean()),
                "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def save_by_class(by_class: pd.DataFrame, path: Path) -> None:
    if by_class.empty:
        print(f"[WARN] Skipping class plot; no data for {path.name}")
        return

    fig, ax = plt.subplots(figsize=(9.8, max(5.0, 0.5 * len(by_class) + 2.6)))
    y = np.arange(len(by_class))
    ax.barh(
        y,
        by_class["mean"],
        xerr=by_class["se"],
        color="#4C78A8",
        alpha=0.9,
        capsize=3,
        height=0.7,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{row.class_display}  (n={row.n})" for row in by_class.itertuples()]
    )
    ax.invert_yaxis()
    ax.set_xlabel("Acceleration magnitude")
    ax.set_title("Accelerometer intensity by classroom activity")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def save_by_half_hour(by_half: pd.DataFrame, path: Path) -> None:
    if by_half.empty:
        print(f"[WARN] Skipping 30-min plot; no data for {path.name}")
        return

    fig, ax = plt.subplots(figsize=(max(9.0, 0.7 * len(by_half) + 2), 5.4))
    x = np.arange(len(by_half))
    ax.bar(
        x,
        by_half["mean"],
        yerr=by_half["se"],
        color="#4C78A8",
        alpha=0.9,
        capsize=3,
        width=0.75,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row.HalfHourLabel}\n(n={row.n})" for row in by_half.itertuples()],
        rotation=35,
        ha="right",
    )
    ax.set_xlabel("Clock time (30-minute bins)")
    ax.set_ylabel("Acceleration magnitude")
    ax.set_title("Accelerometer intensity by 30-minute interval")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot accelerometer Intensity by class and by 30-minute intervals."
    )
    parser.add_argument("--epoch-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--sensor",
        default=None,
        help="Optional single sensor (e.g. Head). Default averages all sensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        epoch_dir = resolve_epoch_dir(args.epoch_dir)
        raw = load_epoch_kinematics(epoch_dir)
        minutes = add_half_hour_bin(bodywide_intensity(raw, args.sensor))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Fatal Error] {exc}")
        sys.exit(1)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    person_class = (
        minutes.groupby(["Participant", "class_display"], as_index=False, observed=True)
        .agg(Intensity=("Intensity", "mean"))
    )
    # One value per student per clock bin (same 8:30–9:00 across days is pooled).
    person_half = (
        minutes.groupby(["Participant", "HalfHourLabel"], as_index=False, observed=True)
        .agg(Intensity=("Intensity", "mean"), HalfHourSort=("HalfHourSort", "min"))
    )

    class_order = ordered_class_labels(
        person_class["class_display"].dropna().astype(str).unique().tolist()
    )
    half_order = (
        person_half[["HalfHourLabel", "HalfHourSort"]]
        .drop_duplicates()
        .sort_values("HalfHourSort")["HalfHourLabel"]
        .tolist()
    )

    by_class = cohort_means(person_class, "class_display", class_order)
    by_half = cohort_means(person_half, "HalfHourLabel", half_order)

    by_class.to_csv(out / "summary_by_class.csv", index=False)
    by_half.to_csv(out / "summary_by_30min.csv", index=False)

    for stale in (
        "intensity_class_x_hour_heatmap.png",
        "intensity_by_time_of_day.png",
        "intensity_by_class_and_time.png",
        "accelerometer_by_class_and_time.png",
        "summary_by_time_of_day.csv",
    ):
        old = out / stale
        if old.exists():
            old.unlink()

    save_by_class(by_class, out / "intensity_by_class.png")
    save_by_half_hour(by_half, out / "intensity_by_30min.png")

    print(f"\n[DONE] Outputs in {out}")


if __name__ == "__main__":
    main()
