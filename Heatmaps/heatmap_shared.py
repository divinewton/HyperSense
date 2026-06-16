from __future__ import annotations

import os
import re
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


LOCAL_TZ = "US/Pacific"
# Default validity window for Apple Watch heart-rate samples.
VALID_HR_MIN = 40
VALID_HR_MAX = 180

DEFAULT_TIME_BINS: Sequence[Tuple[str, str]] = (
    ("08:30", "09:00"),
    ("09:00", "09:30"),
    ("09:30", "10:00"),
    ("10:00", "10:30"),
    ("10:30", "11:00"),
    ("11:00", "11:30"),
    ("11:30", "12:00"),
    ("12:00", "12:30"),
    ("12:30", "13:00"),
    ("13:00", "13:30"),
    ("13:30", "14:00"),
    ("14:00", "14:30"),
    ("14:30", "15:00"),
)

WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# Create an output directory tree if it does not already exist.
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Sort participant folders by their numeric id instead of lexicographic order.
def participant_sort_key(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


# Extract the numeric part of a participant name like `P014`.
def participant_numeric_id(name: str) -> Optional[int]:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


# Normalize participant names to the `P###` format used in the plots.
def participant_code(name: str) -> str:
    match = re.search(r"(\d+)", name)
    if not match:
        return name
    return f"P{int(match.group(1)):03d}"


# Find all participant directories under the export root.
def list_participant_folders(root: Path) -> List[Path]:
    if not root.exists():
        return []
    folders = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"P\d+", p.name)]
    return sorted(folders, key=lambda p: participant_sort_key(p.name))


# Read a CSV defensively and return `None` if the file is malformed.
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


# Raw Apple export files contain metadata before the actual CSV header.
def read_raw_export(path: Path) -> Optional[pd.DataFrame]:
    skip_count = None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for idx, line in enumerate(handle):
                if "/@locale" in line:
                    skip_count = idx
                    break
    except Exception:
        return None

    if skip_count is None:
        return None

    try:
        return pd.read_csv(path, skiprows=skip_count, low_memory=False)
    except Exception:
        return None


# Use the raw-export reader for `*export.csv`; otherwise read normally.
def safe_read_any_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.name.endswith("export.csv"):
        df = read_raw_export(path)
        if df is not None:
            return df
    return safe_read_csv(path)


# Collect all participant CSVs from labeled records, record folders, and raw exports.
def find_candidate_csvs(participant_dir: Path) -> List[Path]:
    candidates: List[Path] = []

    labeled_record = participant_dir / "HealthApp" / "Labeled" / "Record"
    if labeled_record.exists():
        candidates.extend(sorted(labeled_record.rglob("*.csv")))

    record_dir = participant_dir / "HealthApp" / "Record"
    if record_dir.exists():
        candidates.extend(sorted(record_dir.rglob("*.csv")))

    raw_id = participant_numeric_id(participant_dir.name)
    if raw_id is not None:
        for raw_export in sorted(participant_dir.parent.glob("P*export.csv")):
            if participant_numeric_id(raw_export.name) == raw_id:
                candidates.append(raw_export)

    seen = set()
    out = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


# Normalize timestamps into the local school timezone before aggregation.
def normalize_timestamp(series: pd.Series, local_tz: str) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", format="mixed")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(local_tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        ts = ts.dt.tz_convert(local_tz)
    return ts


# Detect the timestamp, value, and class columns across multiple Apple export formats.
def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    timestamp_candidates = [
        "StartDate",
        "CreationDate",
        "EndDate",
        "Time_In_PST",
        "time",
        "timestamp",
        "Timestamp",
        "date",
        "Date",
    ]
    value_candidates = [
        "bpm",
        "BPM",
        "Value",
        "value",
        "HeartRate",
        "heart_rate",
    ]
    class_candidates = ["class", "Class", "label", "Label", "activity", "Activity"]

    ts_col = next((c for c in timestamp_candidates if c in df.columns), None)
    val_col = next((c for c in value_candidates if c in df.columns), None)
    class_col = next((c for c in class_candidates if c in df.columns), None)

    if ts_col is None:
        ts_candidates = [
            c for c in df.columns
            if any(key in str(c).lower() for key in ("date", "time", "start", "end"))
        ]
        best_ts = None
        best_ts_score = 0
        for col in ts_candidates:
            score = pd.to_datetime(df[col], errors="coerce", format="mixed").notna().sum()
            if score > best_ts_score:
                best_ts = col
                best_ts_score = score
        if best_ts_score > 0:
            ts_col = best_ts

    if val_col is None:
        val_candidates = [
            c for c in df.columns
            if any(key in str(c).lower() for key in ("bpm", "value", "heart", "energy", "exercise"))
        ]
        best_val = None
        best_val_score = 0
        for col in val_candidates:
            score = pd.to_numeric(df[col], errors="coerce").notna().sum()
            if score > best_val_score:
                best_val = col
                best_val_score = score
        if best_val_score > 0:
            val_col = best_val

    return ts_col, val_col, class_col


# Clean up class labels so blank or placeholder values collapse to one bucket.
def canonicalize_class_labels(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    cleaned = cleaned.replace({"": "Unlabeled", "NONE": "Unlabeled", "None": "Unlabeled"})
    return cleaned.apply(canonicalize_class_label)


# Normalize class labels without merging ELA and History into one bucket.
# Only labels that are explicitly combined in the source data stay combined.
def canonicalize_class_label(label: object) -> str:
    raw = str(label).strip()
    if not raw:
        return "Unlabeled"

    key = raw.lower().replace("&", "/").replace(" and ", "/").replace(" ", "")
    if key in {"ela/history", "history/ela", "ela-history"}:
        return "ELA/History"
    if key == "fridayfunday":
        return "Friday Funday"
    if key == "delete":
        return "DELETE"
    if raw.upper() == "ELA":
        return "ELA"
    if raw.upper() == "HISTORY":
        return "History"
    return raw


# Parse schedule times from `HH:MM:SS` or `HH:MM` strings.
def parse_schedule_time(value: object) -> Optional[time]:
    try:
        parsed = pd.to_datetime(str(value).strip(), format="%H:%M:%S", errors="coerce")
        if pd.isna(parsed):
            parsed = pd.to_datetime(str(value).strip(), format="%H:%M", errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.time()
    except Exception:
        return None


# Extract participant ids from schedule filenames like `schedData_P(01,02)_M-TH.csv`.
def extract_participants_from_filename(fname: str) -> List[str]:
    match = re.search(r"P\(([^)]+)\)", fname)
    if not match:
        return []
    return [x.strip() for x in match.group(1).split(",") if x.strip()]


# Read a schedule CSV into `(start_time, end_time, class)` rows.
def parse_schedule_csv(filepath: Path) -> List[Tuple[time, time, str]]:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    rows: List[Tuple[time, time, str]] = []
    for _, row in df.iterrows():
        cls = str(row.get("Class", "")).strip()
        if not cls or cls.upper() == "DELETE":
            continue
        cls = canonicalize_class_label(cls)
        start = parse_schedule_time(row.get("TimeStart"))
        end = parse_schedule_time(row.get("TimeEnd"))
        if start is None or end is None:
            continue
        rows.append((start, end, cls))
    return rows


# Build a participant -> weekday -> class blocks lookup table from schedule CSVs.
def build_schedule_map(root: Path) -> dict[str, dict[str, List[Tuple[time, time, str]]]]:
    schedule_dirs = []
    for cand in (root / "Schedules", root):
        if cand.exists() and cand.is_dir():
            schedule_dirs.append(cand)

    mth_files: dict[str, Path] = {}
    fr_files: dict[str, Path] = {}
    tu_files: dict[str, Path] = {}

    for schedules_dir in schedule_dirs:
        for fname in os.listdir(schedules_dir):
            if not fname.endswith(".csv"):
                continue
            ids = extract_participants_from_filename(fname)
            if not ids:
                continue

            fpath = schedules_dir / fname
            upper = fname.upper()
            if upper.endswith("TU.CSV") or "_TU." in upper:
                for pid in ids:
                    tu_files[pid] = fpath
            elif "_FR" in upper:
                for pid in ids:
                    fr_files[pid] = fpath
            elif "_M-TH" in upper or "_MTH" in upper:
                for pid in ids:
                    mth_files[pid] = fpath

    all_pids = set(mth_files) | set(fr_files) | set(tu_files)
    schedule_map: dict[str, dict[str, List[Tuple[time, time, str]]]] = {}
    for pid in all_pids:
        mth_blocks = parse_schedule_csv(mth_files[pid]) if pid in mth_files else []
        fr_blocks = parse_schedule_csv(fr_files[pid]) if pid in fr_files else []
        tu_blocks = parse_schedule_csv(tu_files[pid]) if pid in tu_files else mth_blocks
        schedule_map[pid] = {
            "Monday": mth_blocks,
            "Tuesday": tu_blocks,
            "Wednesday": mth_blocks,
            "Thursday": mth_blocks,
            "Friday": fr_blocks,
        }
    return schedule_map


# Resolve the matching schedule for a participant folder name.
def get_participant_schedule(
    participant_name: str,
    schedule_map: dict[str, dict[str, List[Tuple[time, time, str]]]],
) -> Optional[dict[str, List[Tuple[time, time, str]]]]:
    num = participant_numeric_id(participant_name)
    if num is None:
        return None
    for candidate in [f"{num:02d}", f"{num:03d}", str(num)]:
        if candidate in schedule_map:
            return schedule_map[candidate]
    return None


# Convert a `datetime.time` into minutes since midnight for interval math.
def time_to_minutes(t: object) -> Optional[int]:
    if t is None:
        return None
    if hasattr(t, "hour") and hasattr(t, "minute"):
        return int(t.hour) * 60 + int(t.minute)
    return None


# Half-open time-bin membership test: `[start, end)`.
def time_bin_mask(series: pd.Series, start_t: time, end_t: time) -> pd.Series:
    return (series >= start_t) & (series < end_t)


# Generate a fixed list of contiguous time bins over the school-day window.
def build_time_bins(start_hhmm: str, end_hhmm: str, step_minutes: int) -> List[Tuple[time, time]]:
    start = datetime.strptime(start_hhmm, "%H:%M")
    end = datetime.strptime(end_hhmm, "%H:%M")
    bins: List[Tuple[time, time]] = []
    current = start
    while current < end:
        nxt = current + timedelta(minutes=step_minutes)
        bins.append((current.time(), nxt.time()))
        current = nxt
    return bins


# Convert time-bin tuples into display labels like `08:30-09:00`.
def overlap_bin_labels(bins: Sequence[Tuple[time, time]]) -> List[str]:
    return [f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}" for start, end in bins]


# Check whether a time bin overlaps any scheduled class block.
def bin_overlaps_blocks(
    bin_start: time,
    bin_end: time,
    class_blocks: Sequence[Tuple[time, time, str]],
) -> bool:
    start_min = time_to_minutes(bin_start)
    end_min = time_to_minutes(bin_end)
    if start_min is None or end_min is None:
        return False
    for start_t, end_t, _ in class_blocks:
        cs = time_to_minutes(start_t)
        ce = time_to_minutes(end_t)
        if cs is None or ce is None:
            continue
        if start_min < ce and end_min > cs:
            return True
    return False


# Gather the unique class labels that actually appear in the schedule files.
def class_labels_for_schedule(schedule_map: dict[str, dict[str, List[Tuple[time, time, str]]]]) -> List[str]:
    labels = set()
    for sched in schedule_map.values():
        for day_blocks in sched.values():
            for _, _, cls in day_blocks:
                cls_clean = canonicalize_class_label(cls)
                if cls_clean and cls_clean.upper() != "DELETE":
                    labels.add(cls_clean)
    return sorted(labels)


# Load one metric type for one participant and standardize timestamps, values, and labels.
def load_participant_metric(
    participant_dir: Path,
    type_token: str,
    valid_min: Optional[float] = None,
    valid_max: Optional[float] = None,
    local_tz: str = LOCAL_TZ,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for csv_path in find_candidate_csvs(participant_dir):
        df = safe_read_any_csv(csv_path)
        if df is None or df.empty:
            continue

        if "Type" in df.columns:
            df = df[df["Type"].astype(str).str.contains(type_token, case=False, na=False)].copy()
            if df.empty:
                continue

        ts_col, val_col, class_col = detect_columns(df)
        if ts_col is None or val_col is None:
            continue

        out = pd.DataFrame()
        out["timestamp"] = normalize_timestamp(df[ts_col], local_tz=local_tz)
        out["value"] = pd.to_numeric(df[val_col], errors="coerce")
        if class_col is not None:
            out["class"] = df[class_col].astype(str).str.strip()
        else:
            out["class"] = "Unlabeled"

        out = out.dropna(subset=["timestamp", "value"])
        if valid_min is not None:
            out = out[out["value"] >= valid_min]
        if valid_max is not None:
            out = out[out["value"] <= valid_max]

        if not out.empty:
            out["date"] = out["timestamp"].dt.date
            out["weekday"] = out["timestamp"].dt.day_name()
            out["time_obj"] = out["timestamp"].dt.time
            frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "value", "class", "date", "weekday", "time_obj"])

    combined = pd.concat(frames, ignore_index=True)
    combined["class"] = canonicalize_class_labels(combined["class"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


# Build text annotations for heatmap cells, leaving missing cells blank.
def format_annot(matrix: pd.DataFrame, percent: bool = False) -> pd.DataFrame:
    if matrix.empty:
        return matrix.copy()

    def _format_cell(x: object) -> str:
        if pd.isna(x):
            return ""
        return f"{x:.1f}"

    values = np.vectorize(_format_cell, otypes=[object])(matrix.to_numpy())
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)


# Convert raw counts into per-participant percentages.
def percent_by_participant(count_matrix: pd.DataFrame) -> pd.DataFrame:
    if count_matrix.empty:
        return count_matrix.copy()
    totals = count_matrix.sum(axis=0).replace(0, np.nan)
    return count_matrix.div(totals, axis=1) * 100.0


# Aggregate a list of values using either a mean or a sum, depending on the metric.
def reduce_values(values: List[float], mode: str) -> float:
    if not values:
        return np.nan
    if mode == "sum":
        return float(np.sum(values))
    return float(np.mean(values))


# Shared seaborn heatmap wrapper with the Aura-style formatting choices.
def plot_heatmap(
    matrix: pd.DataFrame,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "viridis_r",
    percent: bool = False,
    mask_zero: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    if matrix.empty:
        print(f"[WARN] No data available for {out_path.name}")
        return

    plot_df = matrix.replace(0, np.nan) if mask_zero else matrix.copy()
    annot = format_annot(plot_df, percent=percent)

    if figsize is None:
        figsize = (max(8, len(plot_df.columns) * 0.8), max(4, len(plot_df.index) * 0.5))

    plt.figure(figsize=figsize)
    plt.gcf().set_facecolor("white")
    ax = sns.heatmap(
        plot_df,
        cmap=cmap,
        linewidths=0.5,
        linecolor="gray",
        cbar=True,
        square=False,
        annot=annot,
        fmt="",
        mask=plot_df.isna(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.5, len(plot_df.columns) + 0.5)
    ax.set_ylim(len(plot_df.index) + 0.5, -0.5)
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {out_path}")


# Main driver used by each datatype wrapper. Produces both coverage and value plots.
def run_heatmap_suite(
    *,
    root: Path,
    output_dir: Path,
    type_token: str,
    metric_label: str,
    metric_folder: str,
    valid_min: Optional[float] = None,
    valid_max: Optional[float] = None,
    value_agg: str = "mean",
    local_tz: str = LOCAL_TZ,
) -> None:
    ensure_dir(output_dir)
    coverage_dir = output_dir / "Coverage"
    values_dir = output_dir / "Heatmaps"
    ensure_dir(coverage_dir)
    ensure_dir(values_dir)

    participant_dirs = list_participant_folders(root)
    if not participant_dirs:
        print(f"[ERROR] No participant folders found under {root}")
        return

    schedule_map = build_schedule_map(root)
    if not schedule_map:
        print("[WARN] No schedule CSVs found. Weekday and class coverage plots will be empty.")

    class_names = [
        c for c in class_labels_for_schedule(schedule_map)
        if c not in {"DELETE", "Friday Funday"}
    ]
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday"]
    weekday_label_map = {
        "Monday": "Mon",
        "Tuesday": "Tue",
        "Wednesday": "Wed",
        "Thursday": "Thu",
    }

    time_bins_5min = build_time_bins("08:30", "15:00", 5)
    time_bins_30min = build_time_bins("08:30", "15:00", 30)
    time_bin_labels_5min = overlap_bin_labels(time_bins_5min)
    time_bin_labels_30min = overlap_bin_labels(time_bins_30min)

    # Coverage plots hold percentages; value plots hold mean/sum summaries.
    weekday_coverage = pd.DataFrame(0.0, index=weekday_names, columns=[participant_code(p.name) for p in participant_dirs])
    class_coverage = pd.DataFrame(0.0, index=class_names, columns=[participant_code(p.name) for p in participant_dirs]) if class_names else pd.DataFrame()
    time_coverage = pd.DataFrame(0.0, index=time_bin_labels_30min, columns=[participant_code(p.name) for p in participant_dirs])
    weekday_values = pd.DataFrame(np.nan, index=weekday_names, columns=[participant_code(p.name) for p in participant_dirs])
    class_values = pd.DataFrame(np.nan, index=class_names, columns=[participant_code(p.name) for p in participant_dirs]) if class_names else pd.DataFrame()
    time_values = pd.DataFrame(np.nan, index=time_bin_labels_30min, columns=[participant_code(p.name) for p in participant_dirs])

    for p_dir in participant_dirs:
        participant = participant_code(p_dir.name)
        metric_df = load_participant_metric(
            p_dir,
            type_token=type_token,
            valid_min=valid_min,
            valid_max=valid_max,
            local_tz=local_tz,
        )
        if metric_df.empty:
            print(f"[WARN] No usable {metric_label.lower()} rows found for {p_dir.name}")
            continue

        print(f"[INFO] {p_dir.name}: {len(metric_df)} valid {metric_label.lower()} rows")

        p_schedule = get_participant_schedule(p_dir.name, schedule_map)

        weekday_bins: dict[str, dict[str, List[int]]] = {day: {} for day in weekday_names}
        weekday_value_samples: dict[str, List[float]] = {day: [] for day in weekday_names}
        class_expected: dict[str, set] = {cls: set() for cls in class_names}
        class_actual: dict[str, set] = {cls: set() for cls in class_names}
        class_value_samples: dict[str, List[float]] = {cls: [] for cls in class_names}
        time_5min_coverage = {label: [] for label in time_bin_labels_5min}
        time_value_samples = {label: [] for label in time_bin_labels_30min}

        for current_date, day_df in metric_df.groupby("date"):
            day_ts = pd.Timestamp(current_date)
            day_name = day_ts.day_name()
            if day_name not in weekday_names:
                continue

            # Track raw 5-minute coverage across the full school-day window.
            for (start_t, end_t), interval in zip(time_bins_5min, time_bin_labels_5min):
                bin_df = day_df[time_bin_mask(day_df["time_obj"], start_t, end_t)]
                time_5min_coverage[interval].append(1 if not bin_df.empty else 0)

            if p_schedule is None or day_name not in p_schedule:
                continue

            day_blocks = p_schedule[day_name]
            if not day_blocks:
                continue

            valid_day_df = day_df.copy()
            if valid_day_df.empty:
                continue

            # Only bins that fall inside scheduled class time count toward weekday coverage.
            weekday_bin_candidates = [
                (start_t, end_t, label)
                for (start_t, end_t), label in zip(time_bins_5min, time_bin_labels_5min)
                if bin_overlaps_blocks(start_t, end_t, day_blocks)
            ]

            day_weekday_values: List[float] = []

            for start_t, end_t, interval in weekday_bin_candidates:
                bin_df = valid_day_df[time_bin_mask(valid_day_df["time_obj"], start_t, end_t)]
                weekday_bins[day_name].setdefault(interval, []).append(1 if not bin_df.empty else 0)
                if not bin_df.empty:
                    day_weekday_values.extend(bin_df["value"].dropna().tolist())

            # Class coverage counts bins that overlap a scheduled class and contain the target class label.
            for start_t, end_t, cls in day_blocks:
                cls_clean = str(cls).strip()
                if cls_clean in {"", "DELETE", "Friday Funday"} or cls_clean not in class_expected:
                    continue
                for (bin_start, bin_end), interval in zip(time_bins_5min, time_bin_labels_5min):
                    if bin_overlaps_blocks(bin_start, bin_end, [(start_t, end_t, cls_clean)]):
                        class_expected[cls_clean].add((current_date, interval))
                        bin_df = valid_day_df[time_bin_mask(valid_day_df["time_obj"], bin_start, bin_end)]
                        if not bin_df.empty and (bin_df["class"] == cls_clean).any():
                            class_actual[cls_clean].add((current_date, interval))
                            class_value_samples[cls_clean].extend(
                                bin_df.loc[bin_df["class"] == cls_clean, "value"].dropna().tolist()
                            )

            if day_weekday_values:
                weekday_value_samples[day_name].extend(day_weekday_values)

            # Collect raw values for the broader 30-minute visualization bins.
            for (start_t, end_t), interval_30 in zip(time_bins_30min, time_bin_labels_30min):
                bin_df = day_df[time_bin_mask(day_df["time_obj"], start_t, end_t)]
                if not bin_df.empty:
                    time_value_samples[interval_30].extend(bin_df["value"].dropna().tolist())

        # Collapse per-day bin coverage into a single weekday summary per participant.
        for day_name in weekday_names:
            bins_for_this_day = []
            for _, coverage_list in weekday_bins.get(day_name, {}).items():
                if coverage_list:
                    bins_for_this_day.append((sum(coverage_list) / len(coverage_list)) * 100)
            if bins_for_this_day:
                weekday_coverage.loc[day_name, participant] = float(np.mean(bins_for_this_day))
            weekday_values.loc[day_name, participant] = reduce_values(weekday_value_samples[day_name], value_agg)

        # Convert the expected-vs-covered class bins into percent coverage and value summaries.
        for cls in class_names:
            total_bins = len(class_expected[cls])
            covered_bins = len(class_actual[cls])
            if total_bins > 0:
                class_coverage.loc[cls, participant] = (covered_bins / total_bins) * 100.0
            class_values.loc[cls, participant] = reduce_values(class_value_samples[cls], value_agg)

        time_5min_pct = {}
        for interval, coverage_list in time_5min_coverage.items():
            if coverage_list:
                time_5min_pct[interval] = (sum(coverage_list) / len(coverage_list)) * 100.0

        # Average the 5-minute coverage scores into the 30-minute display bins.
        for (start_t, end_t), interval_30 in zip(time_bins_30min, time_bin_labels_30min):
            bins_in_30 = []
            current_min = time_to_minutes(start_t)
            end_min = time_to_minutes(end_t)
            if current_min is None or end_min is None:
                continue
            while current_min < end_min:
                next_min = current_min + 5
                interval_5 = f"{current_min // 60:02d}:{current_min % 60:02d}-{next_min // 60:02d}:{next_min % 60:02d}"
                if interval_5 in time_5min_pct:
                    bins_in_30.append(time_5min_pct[interval_5])
                current_min = next_min
            if bins_in_30:
                time_coverage.loc[interval_30, participant] = float(np.mean(bins_in_30))
            time_values.loc[interval_30, participant] = reduce_values(time_value_samples[interval_30], value_agg)

    if not class_coverage.empty:
        class_order = class_coverage.sum(axis=1).sort_values(ascending=False).index.tolist()
        class_coverage = class_coverage.loc[class_order]
        class_values = class_values.loc[class_order]

    # Short weekday labels match the appearance of the Aura plots.
    weekday_coverage.index = [weekday_label_map.get(label, label) for label in weekday_coverage.index]
    weekday_values.index = [weekday_label_map.get(label, label) for label in weekday_values.index]

    weekday_coverage.to_csv(coverage_dir / f"{metric_folder}_coverage_by_weekday.csv")
    plot_heatmap(
        weekday_coverage,
        coverage_dir / f"{metric_folder}_coverage_by_weekday.png",
        title=f"{metric_label} Coverage by Weekday",
        xlabel="Participant",
        ylabel="Weekday",
        cmap="viridis_r",
        percent=True,
        mask_zero=True,
        vmin=0,
        vmax=100,
        figsize=(max(8, len(weekday_coverage.columns) * 0.8), 4.5),
    )

    if not class_coverage.empty:
        class_coverage.to_csv(coverage_dir / f"{metric_folder}_coverage_by_class.csv")
        plot_heatmap(
            class_coverage,
            coverage_dir / f"{metric_folder}_coverage_by_class.png",
            title=f"{metric_label} Coverage by Class",
            xlabel="Participant",
            ylabel="Class",
            cmap="viridis_r",
            percent=True,
            mask_zero=True,
            vmin=0,
            vmax=100,
            figsize=(max(8, len(class_coverage.columns) * 0.8), max(4, len(class_coverage.index) * 0.5)),
        )
    else:
        print(f"[WARN] No class schedule data available for {metric_label.lower()} class coverage.")

    time_coverage.to_csv(coverage_dir / f"{metric_folder}_coverage_by_time.csv")
    plot_heatmap(
        time_coverage,
        coverage_dir / f"{metric_folder}_coverage_by_time.png",
        title=f"{metric_label} Coverage by Time of Day",
        xlabel="Participant",
        ylabel="30-Minute Bin",
        cmap="viridis_r",
        percent=True,
        mask_zero=True,
        vmin=0,
        vmax=100,
        figsize=(max(8, len(time_coverage.columns) * 0.8), 7.5),
    )

    weekday_values.to_csv(values_dir / f"{metric_folder}_values_by_weekday.csv")
    plot_heatmap(
        weekday_values,
        values_dir / f"{metric_folder}_values_by_weekday.png",
        title=f"{metric_label} Values by Weekday",
        xlabel="Participant",
        ylabel="Weekday",
        cmap="viridis_r",
        percent=False,
        mask_zero=False,
        figsize=(max(8, len(weekday_values.columns) * 0.8), 4.5),
    )

    if not class_values.empty:
        class_values.to_csv(values_dir / f"{metric_folder}_values_by_class.csv")
        plot_heatmap(
            class_values,
            values_dir / f"{metric_folder}_values_by_class.png",
            title=f"{metric_label} Values by Class",
            xlabel="Participant",
            ylabel="Class",
            cmap="viridis_r",
            percent=False,
            mask_zero=False,
            figsize=(max(8, len(class_values.columns) * 0.8), max(4, len(class_values.index) * 0.5)),
        )

    time_values.to_csv(values_dir / f"{metric_folder}_values_by_time.csv")
    plot_heatmap(
        time_values,
        values_dir / f"{metric_folder}_values_by_time.png",
        title=f"{metric_label} Values by Time of Day",
        xlabel="Participant",
        ylabel="30-Minute Bin",
        cmap="viridis_r",
        percent=False,
        mask_zero=False,
        figsize=(max(8, len(time_values.columns) * 0.8), 7.5),
    )
