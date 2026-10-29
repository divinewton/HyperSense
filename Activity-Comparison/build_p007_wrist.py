#!/usr/bin/env python3
"""Rebuild P007 wrist minutes from raw MOCOPI serials.

The epoch file labeled P007 is a copy of P006. These two serials move like
wrists, but the recording does not say which one is the right wrist, so the
comparison uses their minute-level mean and can plot each serial on its own.
"""
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

P007_DATES = {
    "2025-02-24",
    "2025-02-25",
    "2025-02-26",
    "2025-02-27",
    "2025-02-28",
}
WRIST_SERIALS = ("0EA70", "14A53")
SERIAL_RE = re.compile(r"QM-SS1_([0-9A-Fa-f]+)")
INVALID_CLASS = {"", "DELETE", "NONE", "UNLABELED", "NAN"}
DISPLAY_CLASS = {
    "Homework Reinforcement/Study Hall": "HW Rein./Study Hall",
    "Study Hall": "HW Rein./Study Hall",
}


def _serial(name: str) -> str | None:
    match = SERIAL_RE.search(name)
    return match.group(1).upper() if match else None


def _display_class(label: str) -> str | None:
    raw = label.strip()
    if not raw or raw.upper() in INVALID_CLASS:
        return None
    return DISPLAY_CLASS.get(raw, raw)


def class_maps(epoch_dir: Path) -> tuple[dict[str, str], dict[tuple[int, str], str]]:
    """Clock labels from classmates on the same schedule (P006, then P008)."""
    exact: dict[str, str] = {}
    by_weekday: dict[tuple[int, str], str] = {}
    for pid in ("P006", "P008"):
        path = epoch_dir / f"{pid}_epoch_kinematics.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                label = _display_class(row.get("class", ""))
                stamp = row.get("Epoch_1Min", "")[:16]
                if label is None or len(stamp) < 16:
                    continue
                if pid == "P006":
                    exact.setdefault(stamp, label)
                else:
                    exact.setdefault(stamp, label)
                when = datetime.strptime(stamp, "%Y-%m-%d %H:%M")
                by_weekday.setdefault((when.weekday(), stamp[11:16]), label)
    return exact, by_weekday


def _accumulate_file(path: Path, dates: set[str]) -> dict[str, list[float]]:
    """minute -> [sum_mag, sum_sq, n, sum_jerk, n_jerk] for one recording segment."""
    samples: list[tuple[str, float, float, float, float]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        ts_i = header.index("Timestamp")
        x_i, y_i, z_i = [header.index(name) for name in ("Acceleration X", "Acceleration Y", "Acceleration Z")]
        for row in reader:
            if len(row) <= z_i:
                continue
            stamp = row[ts_i]
            if stamp[:10] not in dates:
                continue
            try:
                x, y, z = float(row[x_i]), float(row[y_i]), float(row[z_i])
            except ValueError:
                continue
            samples.append((stamp, x, y, z, math.sqrt(x * x + y * y + z * z)))
    samples.sort()
    buckets: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    previous: tuple[float, float, float] | None = None
    for stamp, x, y, z, mag in samples:
        minute = stamp[:16]
        bucket = buckets[minute]
        bucket[0] += mag
        bucket[1] += mag * mag
        bucket[2] += 1
        if previous is not None:
            dx, dy, dz = x - previous[0], y - previous[1], z - previous[2]
            bucket[3] += math.sqrt(dx * dx + dy * dy + dz * dz)
            bucket[4] += 1
        previous = (x, y, z)
    return buckets


def _merge_buckets(parts: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    for part in parts:
        for minute, values in part.items():
            dest = merged[minute]
            for i, value in enumerate(values):
                dest[i] += value
    return merged


def _metrics(bucket: list[float]) -> tuple[float, float, float] | None:
    sum_mag, sum_sq, n, sum_jerk, n_jerk = bucket
    if n < 2 or n_jerk < 1:
        return None
    mean = sum_mag / n
    variance = (sum_sq - (sum_mag * sum_mag) / n) / (n - 1)
    if variance < 0:
        variance = 0.0
    return mean, math.sqrt(variance), sum_jerk / n_jerk


def serial_minutes(root: Path, serial: str, dates: set[str]) -> dict[str, tuple[float, float, float]]:
    parts = []
    for path in root.rglob("*.csv"):
        if _serial(path.name) == serial.upper():
            parts.append(_accumulate_file(path, dates))
    merged = _merge_buckets(parts)
    out: dict[str, tuple[float, float, float]] = {}
    for minute, bucket in merged.items():
        metrics = _metrics(bucket)
        if metrics is not None:
            out[minute] = metrics
    return out


def _to_frame(
    minutes: dict[str, tuple[float, float, float]],
    exact: dict[str, str],
    by_weekday: dict[tuple[int, str], str],
    sensor: str,
) -> pd.DataFrame:
    rows = []
    for minute, (intensity, variability, jerk) in sorted(minutes.items()):
        label = exact.get(minute)
        if label is None:
            when = datetime.strptime(minute, "%Y-%m-%d %H:%M")
            label = by_weekday.get((when.weekday(), minute[11:16]))
        if label is None:
            continue
        rows.append(
            {
                "Participant": "P007",
                "Date": minute[:10],
                "Epoch_1Min": pd.Timestamp(minute),
                "class_display": label,
                "Intensity": intensity,
                "Variability": variability,
                "Jerk": jerk,
                "Sensor": sensor,
            }
        )
    return pd.DataFrame(rows)


def build_p007_wrist(
    raw_root: Path,
    epoch_dir: Path,
    dates: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return the mean of the two wrist serials, plus each serial on its own."""
    dates = P007_DATES if dates is None else dates
    exact, by_weekday = class_maps(epoch_dir)
    by_serial = {
        serial: serial_minutes(raw_root, serial, dates) for serial in WRIST_SERIALS
    }
    frames = {
        serial: _to_frame(minutes, exact, by_weekday, serial)
        for serial, minutes in by_serial.items()
    }
    pieces = [frame for frame in frames.values() if not frame.empty]
    columns = ["Participant", "Date", "Epoch_1Min", "class_display", "Intensity", "Variability", "Jerk", "Sensor"]
    if not pieces:
        return pd.DataFrame(columns=columns), frames
    stacked = pd.concat(pieces, ignore_index=True)
    mean = stacked.groupby(
        ["Participant", "Date", "Epoch_1Min", "class_display"], as_index=False
    ).agg(Intensity=("Intensity", "mean"), Variability=("Variability", "mean"), Jerk=("Jerk", "mean"))
    mean["Sensor"] = "WristMean"
    return mean, frames


def p007_as_epoch_rows(mean_minutes: pd.DataFrame) -> pd.DataFrame:
    """Convert rebuilt P007 wrist minutes into epoch_kinematics schema.

    Both unknown-side serials are stored as WristR and WristL so right-wrist
    and body-wide loaders pick them up. Values are identical (serial mean).
    """
    if mean_minutes.empty:
        return pd.DataFrame()
    rows = []
    for sensor in ("WristR", "WristL"):
        part = mean_minutes.copy()
        part["Sensor"] = sensor
        part["class"] = part["class_display"]
        part["Participant"] = "P007"
        part["Epoch_1Min"] = pd.to_datetime(part["Epoch_1Min"], errors="coerce")
        part["Date"] = pd.to_datetime(part["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        part["Hour"] = part["Epoch_1Min"].dt.hour
        part["Is_Active"] = (part["Intensity"] > 1.15).astype(int)
        rows.append(
            part[
                [
                    "Sensor",
                    "Date",
                    "class",
                    "Epoch_1Min",
                    "Intensity",
                    "Variability",
                    "Jerk",
                    "Is_Active",
                    "Participant",
                    "Hour",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def apply_p007_wrist_fix(
    epochs: pd.DataFrame,
    epoch_dir: Path,
    raw_root: Path | None = None,
) -> pd.DataFrame:
    """Drop the copied P007 epoch file and inject rebuilt wrist minutes."""
    raw_root = Path.home() / "Downloads" / "P007" if raw_root is None else raw_root
    work = epochs[epochs["Participant"].astype(str) != "P007"].copy()
    if not raw_root.is_dir():
        print(f"[WARN] P007 raw folder missing ({raw_root}); excluding copied P007 epochs")
        return work
    mean, _serials = build_p007_wrist(raw_root, epoch_dir)
    rebuilt = p007_as_epoch_rows(mean)
    if rebuilt.empty:
        print("[WARN] P007 wrist rebuild produced no minutes; excluding copied P007 epochs")
        return work
    print(
        f"[INFO] P007: replaced copied epoch file with rebuilt wrist minutes "
        f"({len(mean):,} minutes from serials {', '.join(WRIST_SERIALS)})"
    )
    return pd.concat([work, rebuilt], ignore_index=True)
