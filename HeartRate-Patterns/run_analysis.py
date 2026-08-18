#!/usr/bin/env python3
"""Cohort mean heart rate by classroom context and time of day.

Watch HR comes from Apple Health exports. Ring HR comes from labeled Oura files.
Classroom labels come from the schedule CSVs used in Coverage/, plus the class
column on the Oura files.
"""
from __future__ import annotations

import argparse
import re
from datetime import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from Coverage.audit_binned_common import participants_dates

LOCAL_TZ = "US/Pacific"
HR_MIN = 40.0
HR_MAX = 180.0
WATCH_TYPE = "HKQuantityTypeIdentifierHeartRate"
LUNCH_START = time(11, 30)
LUNCH_END = time(13, 30)
MORNING_END = time(11, 55)
MIDDAY_END = time(13, 25)
WATCH_COLOR = "#1967d2"
RING_COLOR = "#e37400"

CONTEXT_ORDER = [
    "Academic classes",
    "Study Hall",
    "Homeroom",
    "Cash-out",
    "Social Skills",
    "Lunch",
    "Physical Education",
]
TIME_BLOCKS = ["Morning", "Midday", "Afternoon"]
DEVICES = [
    ("smartwatch", WATCH_COLOR, -0.16),
    ("smart-ring", RING_COLOR, 0.16),
]

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
    "hw rein./study hall": "Study Hall",
    "homework reinforcement/study hall": "Study Hall",
    "friday funday": "Friday Funday",
    "funday friday": "Friday Funday",
    "lunch": "Lunch",
    "commsci": "CommSci",
    "comm sci": "CommSci",
    "communication science": "CommSci",
}
PAPER_CONTEXT = {
    "Math": "Academic classes",
    "ELA": "Academic classes",
    "History": "Academic classes",
    "ELA/History": "Academic classes",
    "CommSci": "Academic classes",
    "Social Skills": "Social Skills",
    "Friday Funday": "Physical Education",
    "Lunch": "Lunch",
    "Cash-out": "Cash-out",
    "Homeroom": "Homeroom",
    "Study Hall": "Study Hall",
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


def paper_context(label: str | None) -> str | None:
    if label is None or label == "DELETE":
        return None
    return PAPER_CONTEXT.get(label, label)


def time_block(ts: pd.Timestamp) -> str:
    clock = ts.timetz().replace(tzinfo=None) if ts.tzinfo else ts.time()
    if clock < MORNING_END:
        return "Morning"
    if clock < MIDDAY_END:
        return "Midday"
    return "Afternoon"


def seconds_since_midnight(ts: pd.Timestamp) -> int:
    clock = ts.timetz().replace(tzinfo=None) if ts.tzinfo else ts.time()
    return clock.hour * 3600 + clock.minute * 60 + clock.second


def export_skiprows(path: Path) -> int:
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle):
            if "/@locale" in line:
                return index
    return 0


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
    sched["StartSec"] = sched["StartTime"].dt.hour * 3600 + sched["StartTime"].dt.minute * 60 + sched["StartTime"].dt.second
    sched["EndSec"] = sched["EndTime"].dt.hour * 3600 + sched["EndTime"].dt.minute * 60 + sched["EndTime"].dt.second
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


def load_watch_samples(apple_root: Path) -> pd.DataFrame:
    sample_frames: list[pd.DataFrame] = []
    usecols = ["/Record/@startDate", "/Record/@type", "/Record/@value", "/Record/@sourceName"]

    for key, dates in participants_dates.items():
        pid = pid_from_key(key)
        path = apple_root / f"{pid}export.csv"
        if not path.exists():
            print(f"[WARN] Missing watch export: {path}")
            continue
        print(f"[INFO] Reading watch HR {pid}", flush=True)
        raw = pd.read_csv(path, skiprows=export_skiprows(path), usecols=usecols, low_memory=False)
        hr = raw[raw["/Record/@type"].astype(str).eq(WATCH_TYPE)].copy()
        hr["timestamp"] = pd.to_datetime(hr["/Record/@startDate"], errors="coerce", utc=True).dt.tz_convert(LOCAL_TZ)
        hr["bpm"] = pd.to_numeric(hr["/Record/@value"], errors="coerce")
        hr["source"] = hr["/Record/@sourceName"].astype(str).str.replace("\u00a0", " ", regex=False)
        hr = hr.dropna(subset=["timestamp", "bpm"])
        hr["date"] = hr["timestamp"].dt.strftime("%Y-%m-%d")
        hr = hr[hr["date"].isin(dates)]
        watch = hr[hr["source"].str.contains("Apple Watch", case=False, na=False)]
        watch = watch[(watch["bpm"] >= HR_MIN) & (watch["bpm"] <= HR_MAX)]
        if watch.empty:
            print(f"[WARN] {pid}: no valid Apple Watch HR on study dates")
            continue

        friday, other, tuesday = schedule_paths(key, apple_root)
        sched_fri = load_schedule(friday)
        sched_oth = load_schedule(other)
        sched_tu = load_schedule(tuesday) if tuesday else None

        labeled_parts: list[pd.DataFrame] = []
        for date_str, day in watch.groupby("date"):
            sched = schedule_for_date(key, date_str, sched_fri, sched_oth, sched_tu)
            out = day[["timestamp", "bpm", "date"]].copy()
            out["participant"] = pid
            out["class_label"] = [class_from_schedule(seconds_since_midnight(ts), sched) for ts in out["timestamp"]]
            labeled_parts.append(out)
        labeled = pd.concat(labeled_parts, ignore_index=True)
        labeled = labeled.dropna(subset=["class_label"])
        labeled["context"] = labeled["class_label"].map(paper_context)
        labeled["time_block"] = labeled["timestamp"].map(time_block)
        labeled["device"] = "smartwatch"
        sample_frames.append(labeled)
        print(f"[INFO] {pid}: {len(labeled)} scheduled Apple Watch HR samples", flush=True)

    columns = ["participant", "timestamp", "bpm", "date", "class_label", "context", "time_block", "device"]
    return pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame(columns=columns)


def load_ring_samples(ring_root: Path) -> pd.DataFrame:
    file_rows: list[dict] = []
    sample_frames: list[pd.DataFrame] = []
    study_by_pid = {pid_from_key(key): dates for key, dates in participants_dates.items()}

    for path in sorted(ring_root.rglob("*OrHrLabeled*.csv")):
        folder_pid = path.parts[path.parts.index(ring_root.name) + 1] if ring_root.name in path.parts else path.parent.parent.name
        match = re.search(r"(P\d+)OrHrLabeled(\d{4}-\d{2}-\d{2})", path.name)
        if not match:
            continue
        file_pid, file_date = match.group(1), match.group(2)
        file_pid = f"P{int(re.search(r'(\d+)', file_pid).group(1)):03d}"
        folder_pid = f"P{int(re.search(r'(\d+)', folder_pid).group(1)):03d}" if re.search(r"\d+", folder_pid) else folder_pid
        file_rows.append(
            {
                "path": path,
                "folder_participant": folder_pid,
                "filename_participant": file_pid,
                "file_date": file_date,
                "ids_match": folder_pid == file_pid,
                "on_study": file_date in study_by_pid.get(file_pid, set()),
            }
        )

    pids_with_study = {row["filename_participant"] for row in file_rows if row["ids_match"] and row["on_study"]}
    for row in file_rows:
        if not row["ids_match"]:
            row["used"] = False
        elif row["on_study"] or row["filename_participant"] not in pids_with_study:
            row["used"] = True
        else:
            row["used"] = False

    for row in file_rows:
        if not row["used"]:
            continue
        df = pd.read_csv(row["path"])
        if "bpm" not in df.columns or "Time_In_ISO" not in df.columns:
            continue
        out = pd.DataFrame()
        out["timestamp"] = pd.to_datetime(df["Time_In_ISO"], errors="coerce", utc=True).dt.tz_convert(LOCAL_TZ)
        out["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
        out["quality"] = df["quality"].astype(str) if "quality" in df.columns else "unknown"
        if "class" not in df.columns:
            continue
        out["class_label"] = df["class"].map(canonical_class)
        out["participant"] = row["filename_participant"]
        out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
        out = out.dropna(subset=["timestamp", "bpm"])
        out = out[out["date"].eq(row["file_date"])]
        out = out[~out["quality"].str.lower().eq("bad")]
        out = out[(out["bpm"] >= HR_MIN) & (out["bpm"] <= HR_MAX)]
        lunch_mask = out["class_label"].eq("DELETE") | out["class_label"].isna()
        lunch_time = out["timestamp"].map(
            lambda ts: LUNCH_START <= (ts.timetz().replace(tzinfo=None) if ts.tzinfo else ts.time()) < LUNCH_END
        )
        out.loc[lunch_mask & lunch_time, "class_label"] = "Lunch"
        out = out.dropna(subset=["class_label"])
        out = out[out["class_label"] != "DELETE"]
        if out.empty:
            continue
        out["context"] = out["class_label"].map(paper_context)
        out["time_block"] = out["timestamp"].map(time_block)
        out["device"] = "smart-ring"
        sample_frames.append(out[["participant", "timestamp", "bpm", "date", "class_label", "context", "time_block", "device"]])

    columns = ["participant", "timestamp", "bpm", "date", "class_label", "context", "time_block", "device"]
    return pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame(columns=columns)


def participant_means(samples: pd.DataFrame, stratum: str) -> pd.DataFrame:
    rows = samples.dropna(subset=[stratum, "bpm"]).copy()
    return (
        rows.groupby(["device", "participant", "date", stratum], as_index=False)
        .agg(mean_hr=("bpm", "mean"))
        .groupby(["device", "participant", stratum], as_index=False)
        .agg(mean_hr=("mean_hr", "mean"))
    )


def cohort_means(person: pd.DataFrame, stratum: str) -> pd.DataFrame:
    rows_out = []
    for (device, level), group in person.groupby(["device", stratum], observed=False):
        values = group["mean_hr"].dropna()
        if values.empty:
            continue
        rows_out.append(
            {
                "device": device,
                stratum: level,
                "n_participants": int(values.size),
                "mean_hr": float(values.mean()),
            }
        )
    return pd.DataFrame(rows_out)


def device_n(cohort: pd.DataFrame, device: str, stratum: str, level: str) -> int:
    hit = cohort[cohort["device"].eq(device) & cohort[stratum].eq(level)]
    if hit.empty:
        return 0
    return int(hit.iloc[0]["n_participants"])


def mean_plot(ax, cohort: pd.DataFrame, stratum: str, levels: list[str], title: str) -> None:
    present = [level for level in levels if level in set(cohort[stratum])]
    for i, level in enumerate(present):
        for device, color, offset in DEVICES:
            hit = cohort[cohort["device"].eq(device) & cohort[stratum].eq(level)]
            if hit.empty:
                continue
            ax.scatter(hit.iloc[0]["mean_hr"], i + offset, color=color, s=88, zorder=3, edgecolors="white", linewidths=0.6)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(
        [f"{level}  (n={device_n(cohort, 'smartwatch', stratum, level)}/{device_n(cohort, 'smart-ring', stratum, level)})" for level in present]
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean heart rate (bpm)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)


def save_participant_context_figure(person: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.6), sharey=True, layout="constrained")
    rng = np.random.default_rng(1)
    panels = (("smartwatch", "Apple Watch", WATCH_COLOR), ("smart-ring", "Oura ring", RING_COLOR))
    for ax, (device, title, color) in zip(axes, panels):
        d = person[person["device"].eq(device)]
        for i, level in enumerate(CONTEXT_ORDER):
            g = d[d["context"].eq(level)].sort_values("participant")
            if g.empty:
                continue
            if len(g) == 1:
                ys = np.array([float(i)])
            else:
                ys = i + np.linspace(-0.22, 0.22, len(g)) + rng.normal(0, 0.008, len(g))
            ax.scatter(g["mean_hr"], ys, color=color, s=42, alpha=0.88, zorder=2, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("Mean heart rate (bpm)")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(range(len(CONTEXT_ORDER)))
    axes[0].set_yticklabels(CONTEXT_ORDER)
    axes[0].invert_yaxis()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def save_figure(cohort: pd.DataFrame, stratum: str, levels: list[str], title: str, figsize: tuple[float, float], path: Path) -> None:
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[5.2, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    mean_plot(ax, cohort, stratum, levels, title)
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=WATCH_COLOR, markeredgecolor="none", markersize=8, label="Apple Watch"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=RING_COLOR, markeredgecolor="none", markersize=8, label="Oura ring"),
        ],
        loc="center left",
        frameon=False,
        borderaxespad=0,
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cohort mean heart rate by classroom context and time of day.")
    parser.add_argument("--apple-root", type=Path, default=Path.home() / "Downloads" / "Apple Watch Export CSVs")
    parser.add_argument("--ring-root", type=Path, default=Path.home() / "Downloads" / "OuraRing")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    samples = pd.concat([load_watch_samples(args.apple_root), load_ring_samples(args.ring_root)], ignore_index=True, sort=False)
    scheduled = samples[samples["context"].notna()]
    person_context = participant_means(scheduled, "context")
    person_time = participant_means(scheduled, "time_block")
    cohort_context = cohort_means(person_context, "context")
    cohort_time = cohort_means(person_time, "time_block")

    cohort_context.to_csv(out / "cohort_descriptives_by_context.csv", index=False)
    cohort_time.to_csv(out / "cohort_descriptives_by_time_of_day.csv", index=False)
    save_participant_context_figure(person_context, out / "figure_participant_hr_by_context.png")
    save_figure(cohort_context, "context", CONTEXT_ORDER, "Heart rate by classroom context", (8.8, 5.4), out / "figure_1_hr_by_context.png")
    save_figure(cohort_time, "time_block", TIME_BLOCKS, "Heart rate by time of day", (8.8, 3.8), out / "figure_2_hr_by_time_of_day.png")
    print(f"\n[DONE] Outputs in {out}")


if __name__ == "__main__":
    main()
