#!/usr/bin/env python3
"""MOCOPI accelerometer intensity by classroom activity.

Metric: Intensity = mean Acc_Mag per 1-minute epoch, where
Acc_Mag = sqrt(X^2 + Y^2 + Z^2).

Default uses the right-wrist sensor (WristR) so plots are comparable to
wrist-worn Oura ring activity.
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
MOCOPI_COLOR = "#4C78A8"
DEFAULT_SENSOR = "WristR"
SENSOR_ALIASES = {
    "rightwrist": "WristR",
    "right_wrist": "WristR",
    "right-wrist": "WristR",
    "wristr": "WristR",
    "wrist_r": "WristR",
    "leftwrist": "WristL",
    "left_wrist": "WristL",
    "left-wrist": "WristL",
    "wristl": "WristL",
    "wrist_l": "WristL",
    "anklel": "AnkleL",
    "ankler": "AnkleR",
    "leftankle": "AnkleL",
    "rightankle": "AnkleR",
}


def normalize_sensor(sensor: str | None) -> str | None:
    if sensor is None:
        return None
    key = "".join(ch for ch in sensor.strip().lower() if ch.isalnum() or ch in {"_", "-"})
    if key in {"all", "bodywide", "body"}:
        return None
    return SENSOR_ALIASES.get(key, sensor.strip())


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
        raise FileNotFoundError(f"No *_epoch_kinematics.csv files found under {epoch_dir}.")

    df = pd.concat(frames, ignore_index=True)
    required = {"Sensor", "class", "Intensity", "Participant", "Epoch_1Min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Epoch files missing columns: {sorted(missing)}")

    df = df.copy()
    df["class_display"] = df["class"].map(display_class_label)
    df = df[df["class_display"].notna()].copy()
    for col in ("Intensity", "Variability", "Jerk"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Epoch_1Min"] = pd.to_datetime(df["Epoch_1Min"], errors="coerce")
    df = df.dropna(subset=["Intensity", "Epoch_1Min"])
    return df


def bodywide_intensity(df: pd.DataFrame, sensor: str | None) -> pd.DataFrame:
    """Right-wrist (or chosen sensor) minute-level Intensity, plus Variability/Jerk when present."""
    sensor = normalize_sensor(sensor)
    work = df if sensor is None else df[df["Sensor"] == sensor].copy()
    if work.empty:
        available = sorted(df["Sensor"].dropna().astype(str).unique())
        raise ValueError(
            f"No rows left after sensor filter ({sensor!r}). Available: {available}"
        )
    keys = ["Participant", "Date", "Epoch_1Min", "class_display"]
    agg = {"Intensity": ("Intensity", "mean")}
    if "Variability" in work.columns:
        agg["Variability"] = ("Variability", "mean")
    if "Jerk" in work.columns:
        agg["Jerk"] = ("Jerk", "mean")
    return work.groupby(keys, as_index=False, observed=True).agg(**{
        name: pd.NamedAgg(column=col, aggfunc=func) for name, (col, func) in agg.items()
    })


def cohort_means(person: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows = []
    for level in order:
        values = person.loc[person["class_display"].astype(str).eq(str(level)), "Intensity"]
        if values.empty:
            continue
        rows.append(
            {
                "class_display": level,
                "mean": float(values.mean()),
                "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def save_by_class(by_class: pd.DataFrame, path: Path, sensor_label: str) -> None:
    if by_class.empty:
        print(f"[WARN] Skipping class plot; no data for {path.name}")
        return
    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.5 * len(by_class) + 2.0)))
    y = np.arange(len(by_class))
    ax.barh(
        y,
        by_class["mean"],
        xerr=by_class["se"],
        color=MOCOPI_COLOR,
        alpha=0.9,
        capsize=3,
        height=0.7,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row.class_display}  (n={row.n})" for row in by_class.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel("Acceleration magnitude")
    ax.set_title(f"MOCOPI accelerometer by classroom activity ({sensor_label})")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MOCOPI Intensity by classroom activity.")
    parser.add_argument("--epoch-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--sensor",
        default=DEFAULT_SENSOR,
        help="Sensor placement to plot (default: WristR / right wrist). Use 'all' for body-wide mean.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensor = normalize_sensor(args.sensor)
    sensor_label = "all sensors" if sensor is None else (
        "right wrist" if sensor == "WristR" else sensor
    )
    try:
        epoch_dir = resolve_epoch_dir(args.epoch_dir)
        minutes = bodywide_intensity(load_epoch_kinematics(epoch_dir), sensor)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Fatal Error] {exc}")
        sys.exit(1)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    person_class = (
        minutes.groupby(["Participant", "class_display"], as_index=False, observed=True)
        .agg(Intensity=("Intensity", "mean"))
    )
    class_order = ordered_class_labels(
        person_class["class_display"].dropna().astype(str).unique().tolist()
    )
    by_class = cohort_means(person_class, class_order)
    by_class.to_csv(out / "summary_by_class.csv", index=False)

    for stale in (
        "intensity_by_5min.png",
        "intensity_by_30min.png",
        "summary_by_5min.csv",
        "summary_by_30min.csv",
        "intensity_class_x_hour_heatmap.png",
        "intensity_by_time_of_day.png",
        "intensity_by_class_and_time.png",
        "accelerometer_by_class_and_time.png",
    ):
        old = out / stale
        if old.exists():
            old.unlink()

    save_by_class(by_class, out / "intensity_by_class.png", sensor_label)
    print(f"\n[DONE] Outputs in {out} (sensor={sensor_label})")


if __name__ == "__main__":
    main()
