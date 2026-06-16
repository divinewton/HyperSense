import os

import numpy as np
import pandas as pd

# Ensure all exported participant CSVs live in the user's Downloads/Exports folder
EXPORTS_DIR = os.path.expanduser("~/Downloads/Exports")

participants_dates = {
    "01": {"2025-02-03", "2025-02-04", "2025-02-05", "2025-02-06", "2025-02-07"},
    "02": {"2025-02-03", "2025-02-04", "2025-02-05"},
    "03": {"2025-02-03", "2025-02-04"},
    "04": {"2025-02-10", "2025-02-11", "2025-02-12", "2025-02-13", "2025-02-14"},
    "05": {"2025-02-10", "2025-02-11", "2025-02-12"},
    "06": {"2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28"},
    "07": {"2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28"},
    "08": {"2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28"},
    "09": {"2025-02-03", "2025-02-04", "2025-02-05", "2025-02-06", "2025-02-07"},
    "12": {"2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07"},
    "14": {"2025-03-25", "2025-03-26", "2025-03-27", "2025-03-31", "2025-04-01"},
    "16": {"2025-03-25", "2025-03-26", "2025-03-27", "2025-03-31", "2025-04-01"},
}


def time_to_seconds(t):
    # Normalize a time value to seconds since midnight so time-range checks are simplified.
    return t.hour * 3600 + t.minute * 60 + t.second


def load_schedule(schedule_path):
    # Read one of the saved class schedules and normalize it into a compact table.
    if not os.path.exists(schedule_path):
        return pd.DataFrame(columns=["Class", "TimeStart", "TimeEnd", "StartSec", "EndSec"])

    sched = pd.read_csv(schedule_path)
    required_cols = {"Class", "TimeStart", "TimeEnd"}
    if not required_cols.issubset(sched.columns):
        return pd.DataFrame(columns=["Class", "TimeStart", "TimeEnd", "StartSec", "EndSec"])

    sched = sched.copy()
    sched["Class"] = sched["Class"].fillna("").astype(str).str.strip()
    sched = sched[sched["Class"] != "DELETE"].copy()
    sched["StartTime"] = pd.to_datetime(sched["TimeStart"], format="%H:%M:%S", errors="coerce")
    sched["EndTime"] = pd.to_datetime(sched["TimeEnd"], format="%H:%M:%S", errors="coerce")
    sched = sched.dropna(subset=["StartTime", "EndTime"]).copy()
    sched["StartSec"] = sched["StartTime"].dt.hour * 3600 + sched["StartTime"].dt.minute * 60 + sched["StartTime"].dt.second
    sched["EndSec"] = sched["EndTime"].dt.hour * 3600 + sched["EndTime"].dt.minute * 60 + sched["EndTime"].dt.second
    return sched


def build_schedule_bins_for_day(schedule_df, day_str):
    # Expand each class period into concrete 5-minute bins for one specific school day.
    base_day = pd.Timestamp(day_str).tz_localize("US/Pacific")
    bin_starts = []
    bin_ends = []

    for _, row in schedule_df.iterrows():
        start_dt = base_day + pd.Timedelta(seconds=int(row["StartSec"]))
        end_dt = base_day + pd.Timedelta(seconds=int(row["EndSec"]))
        if end_dt <= start_dt:
            continue

        current = start_dt
        while current < end_dt:
            next_edge = min(current + pd.Timedelta(minutes=5), end_dt)
            bin_starts.append(current.value)
            bin_ends.append(next_edge.value)
            current = next_edge

    return np.array(bin_starts, dtype="int64"), np.array(bin_ends, dtype="int64")


def get_schedule_expected_bins(schedule_path, total_days):
    # Compute the number of 5-minute bins a schedule should produce for a given number of tracked days, excluding DELETE rows.
    if not os.path.exists(schedule_path) or total_days == 0:
        return 0

    sched = load_schedule(schedule_path)
    if sched.empty:
        return 0

    total_bins = 0
    for _, row in sched.iterrows():
        start_dt = pd.Timestamp("2000-01-01").tz_localize("US/Pacific") + pd.Timedelta(seconds=int(row["StartSec"]))
        end_dt = pd.Timestamp("2000-01-01").tz_localize("US/Pacific") + pd.Timedelta(seconds=int(row["EndSec"]))
        if end_dt <= start_dt:
            continue

        current = start_dt
        while current < end_dt:
            next_edge = min(current + pd.Timedelta(minutes=5), end_dt)
            total_bins += 1
            current = next_edge

    return total_bins * total_days


def parse_metric_timestamps(metric_df):
    # Parse timestamps as UTC first so mixed timezone exports normalize cleanly before converting to Pacific.
    metric_df = metric_df.copy()

    start_dt = pd.to_datetime(metric_df["/Record/@startDate"], errors="coerce", utc=True).dt.tz_convert("US/Pacific")
    if "/Record/@endDate" in metric_df.columns:
        end_dt = pd.to_datetime(metric_df["/Record/@endDate"], errors="coerce", utc=True).dt.tz_convert("US/Pacific")
    else:
        end_dt = start_dt.copy()

    missing_end = end_dt.isna() | (end_dt <= start_dt)
    end_dt.loc[missing_end] = start_dt.loc[missing_end] + pd.Timedelta(microseconds=1)

    metric_df["StartDT"] = start_dt
    metric_df["EndDT"] = end_dt
    metric_df["DateStr"] = metric_df["StartDT"].dt.strftime("%Y-%m-%d")

    if "/Record/@value" in metric_df.columns:
        metric_df["MetricValue"] = pd.to_numeric(metric_df["/Record/@value"], errors="coerce")
    else:
        metric_df["MetricValue"] = np.nan

    return metric_df


def count_overlapping_bins(record_start_ns, record_end_ns, bin_starts_ns, bin_ends_ns):
    # Return every schedule bin that overlaps a record interval using half-open interval math.
    return np.flatnonzero((record_start_ns < bin_ends_ns) & (record_end_ns > bin_starts_ns))


def find_point_bin(point_ns, bin_starts_ns, bin_ends_ns):
    # Place a point sample into exactly one bin using half-open bin boundaries.
    matches = np.flatnonzero((bin_starts_ns <= point_ns) & (point_ns < bin_ends_ns))
    if matches.size == 0:
        return None
    return int(matches[0])


def run_binned_audit(metric_label, type_token, valid_min=None, valid_max=None, mode="interval"):
    total_sample_expected_bins = 0
    total_sample_observed_bins = 0
    total_sample_valid_bins = 0
    processed_participants = 0

    for pNum, assigned_dates in participants_dates.items():
        # Skip participants whose exported raw file is missing.
        raw_path = os.path.join(EXPORTS_DIR, f"P0{pNum}export.csv")
        if not os.path.exists(raw_path):
            continue

        # Count how many dates are on Friday versus Monday-Thursday.
        fridays_count = sum(1 for d in assigned_dates if pd.to_datetime(d).strftime("%A") == "Friday")
        other_days_count = len(assigned_dates) - fridays_count

        # Participants 04 and 05 share one schedule pair; the rest use the shared schedule pair.
        if pNum in ["04", "05"]:
            fri_path = os.path.join(EXPORTS_DIR, "schedData_P(04,05)_Fr.csv")
            oth_path = os.path.join(EXPORTS_DIR, "schedData_P(04,05)_M-Th.csv")
        else:
            fri_path = os.path.join(EXPORTS_DIR, "schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv")
            oth_path = os.path.join(EXPORTS_DIR, "schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv")

        sched_fri = load_schedule(fri_path)
        sched_oth = load_schedule(oth_path)

        # Expected bins from the actual schedule bin layout, scaled by tracked Friday and non-Friday dates.
        user_expected_bins = get_schedule_expected_bins(fri_path, fridays_count) + get_schedule_expected_bins(oth_path, other_days_count)
        total_sample_expected_bins += user_expected_bins

        # Detect the header start dynamically since the export files can contain metadata before the CSV header.
        skip_count = 0
        with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if "/@locale" in line:
                    skip_count = i
                    break

        # Read the export after the metadata block.
        df = pd.read_csv(raw_path, skiprows=skip_count, low_memory=False)

        user_observed_bins = 0
        user_valid_bins = 0

        if "/Record/@type" in df.columns and "/Record/@startDate" in df.columns:
            # Keep only the requested record type from the export.
            metric_df = df[df["/Record/@type"].astype(str).str.contains(type_token, na=False, case=False)].copy()

            if not metric_df.empty:
                metric_df = parse_metric_timestamps(metric_df)
                metric_df = metric_df.dropna(subset=["StartDT", "EndDT", "DateStr"])

                # Only analyze records that fall on dates assigned to the participant.
                tracking_days_df = metric_df[metric_df["DateStr"].isin(assigned_dates)].copy()

                observed_blocks = set()
                valid_blocks = set()

                for date_str, group in tracking_days_df.groupby("DateStr"):
                    # Pick the correct weekday schedule for the current date.
                    day_of_week = pd.to_datetime(date_str).strftime("%A")
                    current_sched = sched_fri if day_of_week == "Friday" else sched_oth
                    if current_sched.empty:
                        continue

                    bin_starts_ns, bin_ends_ns = build_schedule_bins_for_day(current_sched, date_str)
                    if len(bin_starts_ns) == 0:
                        continue

                    for _, row in group.iterrows():
                        if mode == "point" or mode == "event":
                            # Point and event metrics are assigned to the single bin containing the record start time.
                            bin_idx = find_point_bin(row["StartDT"].value, bin_starts_ns, bin_ends_ns)
                            if bin_idx is None:
                                continue

                            bin_start_ns = int(bin_starts_ns[bin_idx])
                            observed_blocks.add(bin_start_ns)

                            metric_val = row["MetricValue"]
                            if pd.isna(metric_val):
                                continue
                            if valid_min is not None and metric_val < valid_min:
                                continue
                            if valid_max is not None and metric_val > valid_max:
                                continue
                            valid_blocks.add(bin_start_ns)
                            continue

                        # Interval metrics can cover multiple bins, so collect every overlapping bin.
                        start_ns = row["StartDT"].value
                        end_ns = row["EndDT"].value
                        overlapping_bins = count_overlapping_bins(start_ns, end_ns, bin_starts_ns, bin_ends_ns)
                        if overlapping_bins.size == 0:
                            continue

                        overlapping_bin_starts = bin_starts_ns[overlapping_bins].tolist()
                        observed_blocks.update(overlapping_bin_starts)

                        metric_val = row["MetricValue"]
                        if pd.isna(metric_val):
                            continue
                        if valid_min is not None and metric_val < valid_min:
                            continue
                        if valid_max is not None and metric_val > valid_max:
                            continue
                        valid_blocks.update(overlapping_bin_starts)

                # Clamp counts so a malformed export cannot report more bins than the schedule defines.
                user_observed_bins = min(len(observed_blocks), user_expected_bins)
                user_valid_bins = min(len(valid_blocks), user_observed_bins)

        total_sample_observed_bins += user_observed_bins
        total_sample_valid_bins += user_valid_bins
        processed_participants += 1

    # Final summary is shown as averages per processed participant, along with an overall coverage percentage for the entire sample.
    if processed_participants > 0:
        avg_expected = total_sample_expected_bins / processed_participants
        avg_observed = total_sample_observed_bins / processed_participants
        avg_valid = total_sample_valid_bins / processed_participants

        global_coverage_percentage = 0.0
        if total_sample_expected_bins > 0:
            global_coverage_percentage = (total_sample_valid_bins / total_sample_expected_bins) * 100

        prefix = f"{metric_label} " if metric_label else ""

        print(f"{prefix}Expected (5-Min Bins): {round(avg_expected):,d} bins")
        print(f"{prefix}Observed (5-Min Bins): {round(avg_observed):,d} bins")
        print(f"{prefix}Valid (5-Min Bins): {round(avg_valid):,d} bins")
        print(f"{prefix}Invalid (5-Min Bins): {round(max(avg_observed - avg_valid, 0)):,d} bins")
        print(f"{prefix}Coverage: {global_coverage_percentage:.2f}%")
