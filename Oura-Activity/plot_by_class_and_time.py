#!/usr/bin/env python3
"""Oura ring activity by classroom activity.

Metric: MET from OrDaLabeled ring_met_1_min (1 value per minute of the day).
Classroom labels come from the same schedule CSVs used for Apple Watch analyses.
Non-wear minutes (MET ≈ 0.1) are excluded.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_DIR = REPO_ROOT / "Coverage"
if str(COVERAGE_DIR) not in sys.path:
    sys.path.insert(0, str(COVERAGE_DIR))

from audit_binned_common import participants_dates  # noqa: E402


NONWEAR_MET_MAX = 0.15
OURA_COLOR = "#E37400"
LUNCH_START = time(11, 30)
LUNCH_END = time(13, 30)

DISPLAY_CLASS_LABELS = {
    "Homework Reinforcement/Study Hall": "HW Rein./Study Hall",
    "Study Hall": "HW Rein./Study Hall",
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
INVALID_CLASS_LABELS = {"", "DELETE", "NONE", "UNLABELED", "NAN", "LUNCH"}
CANONICAL_CLASS = {
    "homeroom": "Homeroom",
    "math": "Math",
    "ela": "ELA",
    "history": "History",
    "ela/history": "ELA/History",
    "history/ela": "ELA/History",
    "social skills": "Social Skills",
    "cash-out": "Cash-out",
    "cash out": "Cash-out",
    "hw rein./study hall": "HW Rein./Study Hall",
    "homework reinforcement/study hall": "HW Rein./Study Hall",
    "study hall": "HW Rein./Study Hall",
    "friday funday": "Friday Funday",
    "funday friday": "Friday Funday",
}


def pid_from_key(key: str) -> str:
    return f"P{int(key):03d}"


def canonical_class(label: object) -> str | None:
    raw = str(label).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return None
    if raw.upper() == "DELETE":
        return "DELETE"
    return CANONICAL_CLASS.get(re.sub(r"\s+", " ", raw.lower()), raw)


def display_class_label(label: object) -> str | None:
    raw = canonical_class(label)
    if raw is None or raw.upper() in INVALID_CLASS_LABELS:
        return None
    return DISPLAY_CLASS_LABELS.get(raw, raw)


def ordered_class_labels(labels: list[str]) -> list[str]:
    present = set(labels)
    ordered = [c for c in PAPER_CLASS_ORDER if c in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def load_schedule(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Class", "StartSec", "EndSec"])
    sched = pd.read_csv(path)
    if not {"Class", "TimeStart", "TimeEnd"}.issubset(sched.columns):
        return pd.DataFrame(columns=["Class", "StartSec", "EndSec"])
    sched = sched.copy()
    sched["Class"] = sched["Class"].fillna("").astype(str).str.strip()
    sched["StartTime"] = pd.to_datetime(sched["TimeStart"], format="%H:%M:%S", errors="coerce")
    sched["EndTime"] = pd.to_datetime(sched["TimeEnd"], format="%H:%M:%S", errors="coerce")
    sched = sched.dropna(subset=["StartTime", "EndTime"])
    sched["StartSec"] = (
        sched["StartTime"].dt.hour * 3600
        + sched["StartTime"].dt.minute * 60
        + sched["StartTime"].dt.second
    )
    sched["EndSec"] = (
        sched["EndTime"].dt.hour * 3600
        + sched["EndTime"].dt.minute * 60
        + sched["EndTime"].dt.second
    )
    return sched[["Class", "StartSec", "EndSec"]].reset_index(drop=True)


def schedule_paths(participant_key: str, root: Path) -> tuple[Path, Path, Path | None]:
    if participant_key in {"04", "05"}:
        friday = root / "schedData_P(04,05)_Fr.csv"
        other = root / "schedData_P(04,05)_M-Th.csv"
    else:
        friday = root / "schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv"
        other = root / "schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv"
    tuesday = root / "schedData_P(14,16)TU.csv" if participant_key in {"14", "16"} else None
    if tuesday is not None and not tuesday.exists():
        tuesday = None
    return friday, other, tuesday


def schedule_for_date(
    participant_key: str,
    date_str: str,
    sched_fri: pd.DataFrame,
    sched_oth: pd.DataFrame,
    sched_tu: pd.DataFrame | None,
) -> pd.DataFrame:
    weekday = pd.Timestamp(date_str).day_name()
    if weekday == "Friday":
        return sched_fri
    if weekday == "Tuesday" and participant_key in {"14", "16"} and sched_tu is not None and not sched_tu.empty:
        return sched_tu
    return sched_oth


def class_from_schedule(sec: int, sched: pd.DataFrame) -> str | None:
    if sched.empty:
        return None
    hits = sched[(sched["StartSec"] <= sec) & (sec < sched["EndSec"])].copy()
    if hits.empty:
        return None
    hits["duration"] = hits["EndSec"] - hits["StartSec"]
    named = hits[hits["Class"].str.upper() != "DELETE"]
    if not named.empty:
        return canonical_class(named.sort_values("duration").iloc[0]["Class"])
    clock = time(sec // 3600, (sec % 3600) // 60, sec % 60)
    if LUNCH_START <= clock < LUNCH_END:
        return "Lunch"
    return None


def expand_met_day(day: str, met_string: object) -> pd.DataFrame:
    raw = str(met_string).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return pd.DataFrame(columns=["timestamp", "MET"])
    parts = [p for p in raw.split(";") if p != ""]
    if len(parts) != 1440:
        return pd.DataFrame(columns=["timestamp", "MET"])
    mets = pd.to_numeric(pd.Series(parts), errors="coerce")
    base = pd.Timestamp(day)
    stamps = base + pd.to_timedelta(np.arange(1440), unit="m")
    out = pd.DataFrame({"timestamp": stamps, "MET": mets})
    return out.dropna(subset=["MET"])


def load_oura_activity(ring_root: Path, schedule_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key, dates in participants_dates.items():
        pid = pid_from_key(key)
        path = ring_root / pid / f"{pid}OrDaLabeled.csv"
        if not path.exists():
            print(f"[WARN] Missing Oura activity file: {path}")
            continue

        friday, other, tuesday = schedule_paths(key, schedule_root)
        sched_fri = load_schedule(friday)
        sched_oth = load_schedule(other)
        sched_tu = load_schedule(tuesday) if tuesday else None
        if sched_fri.empty and sched_oth.empty:
            print(f"[WARN] {pid}: no schedule CSVs found under {schedule_root}")
            continue

        daily = pd.read_csv(path)
        if "day" not in daily.columns:
            continue
        met_col = "ring_met_1_min" if "ring_met_1_min" in daily.columns else "met_1_min"
        if met_col not in daily.columns:
            print(f"[WARN] {pid}: no MET minute column in {path.name}")
            continue

        available_days = {str(d).strip() for d in daily["day"].dropna().tolist()}
        use_days = available_days & set(dates)
        if not use_days:
            # Fall back when OrDa days don't match the canonical study calendar.
            use_days = {
                d
                for d in available_days
                if pd.Timestamp(d).day_name() in {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
            }
            if use_days:
                print(
                    f"[WARN] {pid}: no overlap with study dates {sorted(dates)}; "
                    f"using OrDa weekdays {sorted(use_days)}"
                )

        day_frames: list[pd.DataFrame] = []
        for _, row in daily.iterrows():
            day = str(row["day"]).strip()
            if day not in use_days:
                continue
            minute_df = expand_met_day(day, row[met_col])
            if minute_df.empty:
                continue
            sched = schedule_for_date(key, day, sched_fri, sched_oth, sched_tu)
            secs = (
                minute_df["timestamp"].dt.hour * 3600
                + minute_df["timestamp"].dt.minute * 60
                + minute_df["timestamp"].dt.second
            )
            minute_df["class_raw"] = [class_from_schedule(int(s), sched) for s in secs]
            minute_df["class_display"] = minute_df["class_raw"].map(display_class_label)
            minute_df["Participant"] = pid
            minute_df["Date"] = day
            day_frames.append(minute_df)

        if not day_frames:
            print(f"[WARN] {pid}: no study-day MET rows after schedule filter")
            continue

        part = pd.concat(day_frames, ignore_index=True)
        part = part[part["MET"] > NONWEAR_MET_MAX].copy()
        part = part[part["class_display"].notna()].copy()
        if part.empty:
            print(f"[WARN] {pid}: no wear+scheduled MET rows")
            continue
        frames.append(part)
        print(f"[INFO] {pid}: {len(part):,} scheduled wear minutes")

    columns = ["Participant", "Date", "timestamp", "MET", "class_display"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def cohort_means(person: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows = []
    for level in order:
        values = person.loc[person["class_display"].astype(str).eq(str(level)), "MET"]
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


def save_by_class(by_class: pd.DataFrame, path: Path) -> None:
    if by_class.empty:
        print(f"[WARN] Skipping class plot; no data for {path.name}")
        return
    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.5 * len(by_class) + 2.0)))
    y = np.arange(len(by_class))
    ax.barh(
        y,
        by_class["mean"],
        xerr=by_class["se"],
        color=OURA_COLOR,
        alpha=0.9,
        capsize=3,
        height=0.7,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row.class_display}  (n={row.n})" for row in by_class.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel("MET")
    ax.set_title("Oura ring activity by classroom activity")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Oura MET activity by classroom activity.")
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
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ring_root = args.ring_root.expanduser().resolve()
    schedule_root = args.schedule_root.expanduser().resolve()
    if not ring_root.is_dir():
        print(f"[Fatal Error] Oura root not found: {ring_root}")
        sys.exit(1)
    if not schedule_root.is_dir():
        print(f"[Fatal Error] Schedule root not found: {schedule_root}")
        sys.exit(1)

    minutes = load_oura_activity(ring_root, schedule_root)
    if minutes.empty:
        print("[Fatal Error] No usable Oura activity minutes loaded.")
        sys.exit(1)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    person_class = (
        minutes.groupby(["Participant", "class_display"], as_index=False, observed=True)
        .agg(MET=("MET", "mean"))
    )
    class_order = ordered_class_labels(
        person_class["class_display"].dropna().astype(str).unique().tolist()
    )
    by_class = cohort_means(person_class, class_order)
    by_class.to_csv(out / "summary_by_class.csv", index=False)

    for stale in ("activity_by_5min.png", "summary_by_5min.csv"):
        old = out / stale
        if old.exists():
            old.unlink()

    save_by_class(by_class, out / "activity_by_class.png")
    print(f"\n[DONE] Outputs in {out}")


if __name__ == "__main__":
    main()
