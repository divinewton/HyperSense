#!/usr/bin/env python3
"""Print per-participant Apple Watch coverage across all smartwatch metrics."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Coverage.audit_binned_common import (  # noqa: E402
    EXPORTS_DIR,
    build_schedule_bins_for_day,
    count_overlapping_bins,
    find_point_bin,
    get_schedule_expected_bins,
    load_schedule,
    parse_metric_timestamps,
    participants_dates,
)
from Coverage.coverage_stratification_common import (  # noqa: E402
    SMARTWATCH_METRICS,
    detect_raw_export_skiprows,
    metric_value_is_valid,
    participant_export_path,
)

METRIC_OUTPUT_LABELS = {
    "sw_hr": "HR (%)",
    "active_energy": "Active Energy (%)",
    "bmr": "BMR (%)",
    "logged_exercise": "Logged Exerc. (%)",
}


@dataclass
class ParticipantResult:
    participant: str
    expected_bins: int = 0
    valid_counts: Dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in SMARTWATCH_METRICS}
    )


def schedule_paths_for_participant(participant_key: str, root: str) -> tuple[str, str]:
    if participant_key in {"04", "05"}:
        fri_path = os.path.join(root, "schedData_P(04,05)_Fr.csv")
        oth_path = os.path.join(root, "schedData_P(04,05)_M-Th.csv")
    else:
        fri_path = os.path.join(root, "schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv")
        oth_path = os.path.join(root, "schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv")
    return fri_path, oth_path


def expected_bins_for_participant(
    participant_key: str,
    assigned_dates: Set[str],
    root: str,
) -> int:
    fridays_count = sum(
        1 for date_str in assigned_dates if pd.to_datetime(date_str).strftime("%A") == "Friday"
    )
    other_days_count = len(assigned_dates) - fridays_count
    fri_path, oth_path = schedule_paths_for_participant(participant_key, root)
    return get_schedule_expected_bins(fri_path, fridays_count) + get_schedule_expected_bins(
        oth_path, other_days_count
    )


def load_all_smartwatch_records(
    participant_key: str,
    assigned_dates: Set[str],
    root: str,
) -> pd.DataFrame:
    raw_path = participant_export_path(participant_key, root=root)
    if not os.path.exists(raw_path):
        return pd.DataFrame()

    skip_count = detect_raw_export_skiprows(raw_path)
    df = pd.read_csv(raw_path, skiprows=skip_count, low_memory=False)
    if "/Record/@type" not in df.columns or "/Record/@startDate" not in df.columns:
        return pd.DataFrame()

    type_pattern = "|".join(config["type_token"] for config in SMARTWATCH_METRICS.values())
    metric_df = df[df["/Record/@type"].astype(str).str.contains(type_pattern, na=False, case=False)].copy()
    if metric_df.empty:
        return pd.DataFrame()

    metric_df = parse_metric_timestamps(metric_df)
    metric_df = metric_df.dropna(subset=["StartDT", "EndDT", "DateStr"])
    return metric_df[metric_df["DateStr"].isin(assigned_dates)].copy()


def count_valid_bins_for_metric(
    metric_df: pd.DataFrame,
    type_token: str,
    valid_min: Optional[float],
    valid_max: Optional[float],
    mode: str,
    date_to_bins: Dict[str, tuple],
) -> int:
    records = metric_df[
        metric_df["/Record/@type"].astype(str).str.contains(type_token, na=False, case=False)
    ]
    if records.empty:
        return 0

    valid_blocks: Set[int] = set()
    for date_str, group in records.groupby("DateStr"):
        if date_str not in date_to_bins:
            continue

        bin_starts_ns, bin_ends_ns = date_to_bins[date_str]
        if len(bin_starts_ns) == 0:
            continue

        for _, row in group.iterrows():
            if mode in {"point", "event"}:
                bin_idx = find_point_bin(row["StartDT"].value, bin_starts_ns, bin_ends_ns)
                if bin_idx is None:
                    continue
                if metric_value_is_valid(row["MetricValue"], valid_min, valid_max):
                    valid_blocks.add(int(bin_starts_ns[bin_idx]))
                continue

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
            valid_blocks.update(int(value) for value in bin_starts_ns[overlapping_bins])

    return len(valid_blocks)


def build_participant_results(root: str = EXPORTS_DIR) -> list[ParticipantResult]:
    results: list[ParticipantResult] = []

    for participant_key, assigned_dates in participants_dates.items():
        raw_path = participant_export_path(participant_key, root=root)
        if not os.path.exists(raw_path):
            continue

        expected_bins = expected_bins_for_participant(participant_key, assigned_dates, root)
        if expected_bins == 0:
            continue

        fri_path, oth_path = schedule_paths_for_participant(participant_key, root)
        sched_fri = load_schedule(fri_path)
        sched_oth = load_schedule(oth_path)

        date_to_bins: Dict[str, tuple] = {}
        for date_str in sorted(assigned_dates):
            day_of_week = pd.to_datetime(date_str).strftime("%A")
            current_sched = sched_fri if day_of_week == "Friday" else sched_oth
            if current_sched.empty:
                continue
            date_to_bins[date_str] = build_schedule_bins_for_day(current_sched, date_str)

        metric_df = load_all_smartwatch_records(participant_key, assigned_dates, root)
        participant_result = ParticipantResult(
            participant=participant_key,
            expected_bins=expected_bins,
        )

        for metric_key, config in SMARTWATCH_METRICS.items():
            valid_count = count_valid_bins_for_metric(
                metric_df,
                config["type_token"],
                config["valid_min"],
                config["valid_max"],
                config["mode"],
                date_to_bins,
            )
            participant_result.valid_counts[metric_key] = min(valid_count, expected_bins)

        results.append(participant_result)

    return results


def coverage_percent(valid_count: int, expected_count: int) -> float:
    if expected_count <= 0:
        return float("nan")
    return (valid_count / expected_count) * 100.0


def print_results(results: list[ParticipantResult]) -> None:
    if not results:
        print("No participant exports found.")
        return

    print("Participant Apple Watch Coverage")
    print("Each percentage is valid scheduled 5-minute bins out of that participant's expected bins.")
    print()

    for result in results:
        print(f"Participant P{result.participant}")
        print(f"  Expected Bins: {result.expected_bins:,d}")
        for metric_key, label in METRIC_OUTPUT_LABELS.items():
            valid_count = result.valid_counts[metric_key]
            percent = coverage_percent(valid_count, result.expected_bins)
            print(f"  {label}: {percent:.2f}%")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate per-participant expected schedule bins and Apple Watch coverage "
            "for heart rate, active energy, BMR, and logged exercise."
        )
    )
    parser.add_argument(
        "--root",
        default=EXPORTS_DIR,
        help="Root folder containing participant export CSVs and schedule files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = os.path.expanduser(args.root)
    results = build_participant_results(root=root)
    if not results:
        print(f"No participant exports found under {root}")
        sys.exit(1)
    print_results(results)


if __name__ == "__main__":
    main()
