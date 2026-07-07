from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from Coverage.audit_binned_common import (
    EXPORTS_DIR,
    build_schedule_bins_for_day,
    count_overlapping_bins,
    find_point_bin,
    parse_metric_timestamps,
    participants_dates,
)

LOCAL_TZ = "US/Pacific"

CLASSROOM_ORDER = [
    "Homeroom",
    "Mathematics",
    "English Language Arts",
    "History",
    "Physical Education",
    "Social Skills",
    "Cash-out",
    "Study Hall",
    "Lunch",
]

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIME_BLOCK_ORDER = ["Morning", "Midday", "Afternoon"]

METRIC_COLUMNS = ["sw_hr", "active_energy", "bmr", "logged_exercise"]

DISPLAY_METRIC_COLUMNS = [
    ("sw_hr", "HR"),
    ("active_energy", "Active Energy"),
    ("bmr", "BMR"),
    ("logged_exercise", "Logged Exercise"),
]

MORNING_END = time(11, 55)
MIDDAY_END = time(13, 25)
LUNCH_WINDOW_START = time(11, 30)
LUNCH_WINDOW_END = time(13, 30)

SCHEDULE_TO_DISPLAY = {
    "Homeroom": "Homeroom",
    "Math": "Mathematics",
    "ELA": "English Language Arts",
    "History": "History",
    "Social Skills": "Social Skills",
    "Cash-out": "Cash-out",
    "HW Rein./Study Hall": "Study Hall",
    "Friday Funday": "Physical Education",
    "ELA/History": "English Language Arts",
}

SMARTWATCH_METRICS = {
    "sw_hr": {
        "type_token": "HeartRate",
        "valid_min": 40,
        "valid_max": 180,
        "mode": "point",
    },
    "active_energy": {
        "type_token": "ActiveEnergyBurned",
        "valid_min": 0,
        "valid_max": None,
        "mode": "interval",
    },
    "bmr": {
        "type_token": "BasalEnergyBurned",
        "valid_min": 0,
        "valid_max": None,
        "mode": "interval",
    },
    "logged_exercise": {
        "type_token": "AppleExerciseTime",
        "valid_min": 0,
        "valid_max": None,
        "mode": "event",
    },
}


@dataclass
class ScheduledBin:
    participant: str
    date_str: str
    weekday: str
    time_block: str
    classroom: str
    bin_start_ns: int


@dataclass
class ParticipantCoverage:
    participant: str
    expected_bins: int = 0
    valid_counts: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in METRIC_COLUMNS})


@dataclass
class StratifierCoverage:
    section: str
    label: str
    expected_bins: int = 0
    valid_counts: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in METRIC_COLUMNS})


def participant_export_path(participant_key: str, root: str = EXPORTS_DIR) -> str:
    return os.path.join(root, f"P0{participant_key}export.csv")


def load_full_schedule(schedule_path: str) -> pd.DataFrame:
    # Read a schedule CSV and keep DELETE rows so lunch periods can be recovered later.
    if not os.path.exists(schedule_path):
        return pd.DataFrame(columns=["Class", "TimeStart", "TimeEnd", "StartSec", "EndSec"])

    sched = pd.read_csv(schedule_path)
    required_cols = {"Class", "TimeStart", "TimeEnd"}
    if not required_cols.issubset(sched.columns):
        return pd.DataFrame(columns=["Class", "TimeStart", "TimeEnd", "StartSec", "EndSec"])

    sched = sched.copy()
    sched["Class"] = sched["Class"].fillna("").astype(str).str.strip()
    sched["StartTime"] = pd.to_datetime(sched["TimeStart"], format="%H:%M:%S", errors="coerce")
    sched["EndTime"] = pd.to_datetime(sched["TimeEnd"], format="%H:%M:%S", errors="coerce")
    sched = sched.dropna(subset=["StartTime", "EndTime"]).copy()
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
    return sched


def schedule_paths_for_participant(participant_key: str, root: str = EXPORTS_DIR) -> Tuple[str, str, Optional[str]]:
    # Participants 04 and 05 share one schedule pair; the rest use the shared schedule pair.
    if participant_key in {"04", "05"}:
        fri_path = os.path.join(root, "schedData_P(04,05)_Fr.csv")
        oth_path = os.path.join(root, "schedData_P(04,05)_M-Th.csv")
    else:
        fri_path = os.path.join(root, "schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv")
        oth_path = os.path.join(root, "schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv")

    tu_path = None
    if participant_key in {"14", "16"}:
        candidate = os.path.join(root, "schedData_P(14,16)TU.csv")
        if os.path.exists(candidate):
            tu_path = candidate
    return fri_path, oth_path, tu_path


def schedule_for_date(
    participant_key: str,
    date_str: str,
    sched_fri: pd.DataFrame,
    sched_oth: pd.DataFrame,
    sched_tu: Optional[pd.DataFrame],
) -> pd.DataFrame:
    # Pick the correct weekday schedule for the current date.
    day_of_week = pd.to_datetime(date_str).strftime("%A")
    if day_of_week == "Friday":
        return sched_fri
    if day_of_week == "Tuesday" and participant_key in {"14", "16"} and sched_tu is not None and not sched_tu.empty:
        return sched_tu
    return sched_oth


def time_to_seconds(value: time) -> int:
    return value.hour * 3600 + value.minute * 60 + value.second


def time_block_for_seconds(seconds_since_midnight: int) -> str:
    # Map a bin start time to the Morning, Midday, or Afternoon stratifier row.
    morning_end = time_to_seconds(MORNING_END)
    midday_end = time_to_seconds(MIDDAY_END)
    if seconds_since_midnight < morning_end:
        return "Morning"
    if seconds_since_midnight < midday_end:
        return "Midday"
    return "Afternoon"


def display_class_for_schedule_row(class_name: str, start_sec: int) -> Optional[str]:
    # Convert a raw schedule label into a Table 6 classroom name, or None to skip the row.
    raw = str(class_name).strip()
    if not raw or raw.upper() == "DELETE":
        # Lunch is inferred from DELETE blocks inside the lunch window.
        start_time = time(start_sec // 3600, (start_sec % 3600) // 60, start_sec % 60)
        if LUNCH_WINDOW_START <= start_time < LUNCH_WINDOW_END:
            return "Lunch"
        return None
    return SCHEDULE_TO_DISPLAY.get(raw)


def build_scheduled_bins_for_day(
    participant_key: str,
    date_str: str,
    schedule_df: pd.DataFrame,
) -> List[ScheduledBin]:
    # Expand each class period into tagged 5-minute bins for one specific school day.
    if schedule_df.empty:
        return []

    base_day = pd.Timestamp(date_str).tz_localize(LOCAL_TZ)
    weekday = pd.to_datetime(date_str).strftime("%A")
    bins: List[ScheduledBin] = []

    for _, row in schedule_df.iterrows():
        classroom = display_class_for_schedule_row(row["Class"], int(row["StartSec"]))
        if classroom is None:
            continue

        start_dt = base_day + pd.Timedelta(seconds=int(row["StartSec"]))
        end_dt = base_day + pd.Timedelta(seconds=int(row["EndSec"]))
        if end_dt <= start_dt:
            continue

        current = start_dt
        while current < end_dt:
            next_edge = min(current + pd.Timedelta(minutes=5), end_dt)
            start_sec = int(row["StartSec"]) + int((current - start_dt).total_seconds())
            bins.append(
                ScheduledBin(
                    participant=participant_key,
                    date_str=date_str,
                    weekday=weekday,
                    time_block=time_block_for_seconds(start_sec),
                    classroom=classroom,
                    bin_start_ns=int(current.value),
                )
            )
            current = next_edge

    return bins


def detect_raw_export_skiprows(raw_path: str) -> int:
    # Detect the header start dynamically since the export files can contain metadata before the CSV header.
    skip_count = 0
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle):
            if "/@locale" in line:
                skip_count = index
                break
    return skip_count


def load_smartwatch_metric_df(
    participant_key: str,
    type_token: str,
    assigned_dates: Set[str],
    root: str = EXPORTS_DIR,
) -> pd.DataFrame:
    # Read one Apple Watch metric from a participant's raw export file.
    raw_path = participant_export_path(participant_key, root=root)
    if not os.path.exists(raw_path):
        return pd.DataFrame()

    skip_count = detect_raw_export_skiprows(raw_path)
    df = pd.read_csv(raw_path, skiprows=skip_count, low_memory=False)
    if "/Record/@type" not in df.columns or "/Record/@startDate" not in df.columns:
        return pd.DataFrame()

    metric_df = df[df["/Record/@type"].astype(str).str.contains(type_token, na=False, case=False)].copy()
    if metric_df.empty:
        return pd.DataFrame()

    metric_df = parse_metric_timestamps(metric_df)
    metric_df = metric_df.dropna(subset=["StartDT", "EndDT", "DateStr"])

    # Only analyze records that fall on dates assigned to the participant.
    return metric_df[metric_df["DateStr"].isin(assigned_dates)].copy()


def metric_value_is_valid(value: object, valid_min: Optional[float], valid_max: Optional[float]) -> bool:
    if pd.isna(value):
        return False
    if valid_min is not None and value < valid_min:
        return False
    if valid_max is not None and value > valid_max:
        return False
    return True


def valid_bins_for_smartwatch_metric(
    metric_df: pd.DataFrame,
    date_to_bins: Dict[str, Tuple[np.ndarray, np.ndarray]],
    valid_min: Optional[float],
    valid_max: Optional[float],
    mode: str,
) -> Set[Tuple[str, int]]:
    # Collect every scheduled 5-minute bin that contains a valid value for this metric.
    valid_bins: Set[Tuple[str, int]] = set()
    if metric_df.empty:
        return valid_bins

    for date_str, group in metric_df.groupby("DateStr"):
        if date_str not in date_to_bins:
            continue
        bin_starts_ns, bin_ends_ns = date_to_bins[date_str]
        if len(bin_starts_ns) == 0:
            continue

        for _, row in group.iterrows():
            if mode in {"point", "event"}:
                # Point and event metrics are assigned to the single bin containing the record start time.
                bin_idx = find_point_bin(row["StartDT"].value, bin_starts_ns, bin_ends_ns)
                if bin_idx is None:
                    continue
                if metric_value_is_valid(row["MetricValue"], valid_min, valid_max):
                    valid_bins.add((date_str, int(bin_starts_ns[bin_idx])))
                continue

            # Interval metrics can cover multiple bins, so collect every overlapping bin.
            overlapping_bins = count_overlapping_bins(
                row["StartDT"].value,
                row["EndDT"].value,
                bin_starts_ns,
                bin_ends_ns,
            )
            if overlapping_bins.size == 0:
                continue
            if not metric_value_is_valid(row["MetricValue"], valid_min, valid_max):
                continue
            for bin_idx in overlapping_bins:
                valid_bins.add((date_str, int(bin_starts_ns[bin_idx])))

    return valid_bins


def coverage_percent(valid_count: int, expected_count: int) -> float:
    if expected_count <= 0:
        return float("nan")
    return (valid_count / expected_count) * 100.0


def format_percent(value: float, latex: bool = False) -> str:
    if pd.isna(value):
        return "NA"
    suffix = r"\%" if latex else "%"
    return f"{value:.1f}{suffix}"


def format_mean_sd(values: Iterable[float]) -> str:
    clean = [float(v) for v in values if not pd.isna(v)]
    if not clean:
        return "NA"
    mean = float(np.mean(clean))
    sd = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    return f"{mean:.1f}±{sd:.1f}"


def format_range(values: Iterable[float], latex: bool = False) -> str:
    clean = [float(v) for v in values if not pd.isna(v)]
    if not clean:
        return "NA"
    suffix = r"\%" if latex else "%"
    return f"{min(clean):.1f}--{max(clean):.1f}{suffix}"


def init_stratifier_map(section: str, labels: Sequence[str]) -> Dict[Tuple[str, str], StratifierCoverage]:
    return {(section, label): StratifierCoverage(section=section, label=label) for label in labels}


def build_coverage_tables(root: str = EXPORTS_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    # Build the Table 6 stratifier rows and participant summary rows.
    classroom_map = init_stratifier_map("Classroom Context", CLASSROOM_ORDER)
    weekday_map = init_stratifier_map("Day of Week", WEEKDAY_ORDER)
    time_block_map = init_stratifier_map("Time of Day", TIME_BLOCK_ORDER)
    participant_map: Dict[str, ParticipantCoverage] = {}
    processed_participants = 0

    for participant_key, assigned_dates in participants_dates.items():
        raw_path = participant_export_path(participant_key, root=root)
        if not os.path.exists(raw_path):
            continue

        fri_path, oth_path, tu_path = schedule_paths_for_participant(participant_key, root=root)
        sched_fri = load_full_schedule(fri_path)
        sched_oth = load_full_schedule(oth_path)
        sched_tu = load_full_schedule(tu_path) if tu_path else pd.DataFrame()

        scheduled_bins: List[ScheduledBin] = []
        date_to_bins: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for date_str in sorted(assigned_dates):
            schedule_df = schedule_for_date(participant_key, date_str, sched_fri, sched_oth, sched_tu)
            scheduled_bins.extend(build_scheduled_bins_for_day(participant_key, date_str, schedule_df))
            date_to_bins[date_str] = build_schedule_bins_for_day(schedule_df, date_str)

        if not scheduled_bins:
            continue

        processed_participants += 1
        participant_map[participant_key] = ParticipantCoverage(participant=participant_key)

        metric_valid_bins: Dict[str, Set[Tuple[str, int]]] = {}
        for metric_key, config in SMARTWATCH_METRICS.items():
            metric_df = load_smartwatch_metric_df(
                participant_key,
                config["type_token"],
                assigned_dates,
                root=root,
            )
            metric_valid_bins[metric_key] = valid_bins_for_smartwatch_metric(
                metric_df,
                date_to_bins,
                config["valid_min"],
                config["valid_max"],
                config["mode"],
            )

        for scheduled_bin in scheduled_bins:
            bin_key = (scheduled_bin.date_str, scheduled_bin.bin_start_ns)

            classroom_row = classroom_map[("Classroom Context", scheduled_bin.classroom)]
            weekday_row = weekday_map[("Day of Week", scheduled_bin.weekday)]
            time_row = time_block_map[("Time of Day", scheduled_bin.time_block)]
            participant_row = participant_map[participant_key]

            for row in (classroom_row, weekday_row, time_row, participant_row):
                row.expected_bins += 1

            for metric_key, valid_set in metric_valid_bins.items():
                if bin_key not in valid_set:
                    continue
                classroom_row.valid_counts[metric_key] += 1
                weekday_row.valid_counts[metric_key] += 1
                time_row.valid_counts[metric_key] += 1
                participant_row.valid_counts[metric_key] += 1

    detail_rows: List[Dict[str, object]] = []
    for section, labels, mapping in (
        ("Classroom Context", CLASSROOM_ORDER, classroom_map),
        ("Day of Week", WEEKDAY_ORDER, weekday_map),
        ("Time of Day", TIME_BLOCK_ORDER, time_block_map),
    ):
        for label in labels:
            row = mapping[(section, label)]
            record = {
                "section": section,
                "stratifier": label,
                "expected_bins": row.expected_bins,
            }
            for metric_key, _ in DISPLAY_METRIC_COLUMNS:
                record[metric_key] = coverage_percent(row.valid_counts[metric_key], row.expected_bins)
            detail_rows.append(record)

    # Participant summary rows use each participant's overall coverage percentage.
    participant_percentages: Dict[str, List[float]] = {
        metric_key: [] for metric_key, _ in DISPLAY_METRIC_COLUMNS
    }
    for participant_row in participant_map.values():
        for metric_key, _ in DISPLAY_METRIC_COLUMNS:
            participant_percentages[metric_key].append(
                coverage_percent(participant_row.valid_counts[metric_key], participant_row.expected_bins)
            )

    summary_rows = [
        {
            "section": "Participant Summary",
            "stratifier": "Mean ± SD",
            "expected_bins": np.nan,
            **{
                metric_key: format_mean_sd(participant_percentages[metric_key])
                for metric_key, _ in DISPLAY_METRIC_COLUMNS
            },
        },
        {
            "section": "Participant Summary",
            "stratifier": "Range (Min -- Max)",
            "expected_bins": np.nan,
            **{
                metric_key: format_range(participant_percentages[metric_key], latex=False)
                for metric_key, _ in DISPLAY_METRIC_COLUMNS
            },
        },
    ]

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    meta = {
        "processed_participants": processed_participants,
        "participant_percentages": participant_percentages,
    }
    return detail_df, summary_df, meta


def render_latex_table(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    participant_percentages: Optional[Dict[str, List[float]]] = None,
) -> str:
    # Render a LaTeX table snippet for Paper Table 6 Apple Watch columns.
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("    \\centering")
    lines.append(
        "    \\caption{Apple Watch data coverage stratified by classroom context, day of the week, and time of day. Values represent the percentage of valid 5-minute bins out of the total expected bins for each given stratifier.}"
    )
    lines.append("    \\label{tab:coverage-stratification}")
    lines.append("    \\begin{tabular}{lccccc}")
    lines.append("        \\toprule")
    lines.append(
        "        & & \\multicolumn{4}{c}{\\textbf{Smartwatch Streams (\\%)}} \\\\"
    )
    lines.append("        \\cmidrule(r){3-6}")
    lines.append(
        "        \\textbf{Stratifier / Dimension} & \\textbf{Expected} & \\textbf{HR} & \\textbf{Active} & \\textbf{BMR} & \\textbf{Logged} \\\\"
    )
    lines.append("         & \\textbf{(Bins)} & & \\textbf{Energy} & & \\textbf{Exercise} \\\\")
    lines.append("        \\midrule")

    current_section = None
    for _, row in detail_df.iterrows():
        section = row["section"]
        if section != current_section:
            if current_section is not None:
                lines.append("        \\midrule")
            lines.append(f"        \\textbf{{{section}}} & & & & & \\\\")
            current_section = section

        expected = int(row["expected_bins"]) if not pd.isna(row["expected_bins"]) else 0
        values = [format_percent(row[metric_key], latex=True) for metric_key, _ in DISPLAY_METRIC_COLUMNS]
        lines.append(
            f"        {row['stratifier']} & {expected} & {values[0]} & {values[1]} & {values[2]} & {values[3]} \\\\"
        )

    lines.append("        \\midrule")
    lines.append("        \\textbf{Participant Summary} & & & & & \\\\")
    for _, row in summary_df.iterrows():
        if row["stratifier"] == "Mean ± SD":
            values = [str(row[metric_key]) for metric_key, _ in DISPLAY_METRIC_COLUMNS]
            lines.append(
                f"        Mean $\\pm$ SD & -- & {values[0]} & {values[1]} & {values[2]} & {values[3]} \\\\"
            )
        else:
            values = [
                format_range(participant_percentages[metric_key], latex=True)
                if participant_percentages is not None
                else str(row[metric_key])
                for metric_key, _ in DISPLAY_METRIC_COLUMNS
            ]
            lines.append(
                f"        Range (Min -- Max) & -- & {values[0]} & {values[1]} & {values[2]} & {values[3]} \\\\"
            )

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)
