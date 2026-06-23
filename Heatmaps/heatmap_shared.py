from __future__ import annotations

import re
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LOCAL_TZ = "US/Pacific"

WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def participant_sort_key(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def participant_numeric_id(name: str) -> Optional[int]:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def participant_code(name: str) -> str:
    match = re.search(r"(\d+)", name)
    if not match:
        return name
    return f"P{int(match.group(1)):03d}"


def list_participant_folders(root: Path) -> List[Path]:
    if not root.exists():
        return []
    folders = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"P\d+", p.name)]
    return sorted(folders, key=lambda p: participant_sort_key(p.name))


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


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


def safe_read_any_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.name.endswith("export.csv"):
        df = read_raw_export(path)
        if df is not None:
            return df
    return safe_read_csv(path)


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

    seen: set[Path] = set()
    out: List[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def normalize_timestamp(series: pd.Series, local_tz: str) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", format="mixed")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(local_tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        ts = ts.dt.tz_convert(local_tz)
    return ts


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


def extract_participants_from_filename(fname: str) -> List[str]:
    match = re.search(r"P\(([^)]+)\)", fname)
    if not match:
        return []
    return [x.strip() for x in match.group(1).split(",") if x.strip()]


def parse_schedule_csv(filepath: Path) -> List[Tuple[time, time, str]]:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    rows: List[Tuple[time, time, str]] = []
    for _, row in df.iterrows():
        cls = canonicalize_class_label(row.get("Class", ""))
        if not cls or cls.upper() == "DELETE":
            continue
        start = parse_schedule_time(row.get("TimeStart"))
        end = parse_schedule_time(row.get("TimeEnd"))
        if start is None or end is None:
            continue
        rows.append((start, end, cls))
    return rows


def build_schedule_map(root: Path) -> Dict[str, Dict[str, List[Tuple[time, time, str]]]]:
    schedule_dirs = []
    for candidate in (root / "Schedules", root):
        if candidate.exists() and candidate.is_dir():
            schedule_dirs.append(candidate)

    mth_files: Dict[str, Path] = {}
    fr_files: Dict[str, Path] = {}
    tu_files: Dict[str, Path] = {}

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

    schedule_map: Dict[str, Dict[str, List[Tuple[time, time, str]]]] = {}
    for pid in set(mth_files) | set(fr_files) | set(tu_files):
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


def get_participant_schedule(
    participant_name: str,
    schedule_map: Dict[str, Dict[str, List[Tuple[time, time, str]]]],
) -> Optional[Dict[str, List[Tuple[time, time, str]]]]:
    num = participant_numeric_id(participant_name)
    if num is None:
        return None
    for candidate in [f"{num:02d}", f"{num:03d}", str(num)]:
        if candidate in schedule_map:
            return schedule_map[candidate]
    return None


def time_to_minutes(t: object) -> Optional[int]:
    if t is None:
        return None
    if hasattr(t, "hour") and hasattr(t, "minute"):
        return int(t.hour) * 60 + int(t.minute)
    return None


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


def class_labels_for_schedule(schedule_map: Dict[str, Dict[str, List[Tuple[time, time, str]]]]) -> List[str]:
    labels = set()
    for sched in schedule_map.values():
        for day_blocks in sched.values():
            for _, _, cls in day_blocks:
                cls_clean = canonicalize_class_label(cls)
                if cls_clean and cls_clean.upper() != "DELETE":
                    labels.add(cls_clean)
    return sorted(labels)


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
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

    ts_col = next((c for c in timestamp_candidates if c in df.columns), None)
    val_col = next((c for c in value_candidates if c in df.columns), None)

    if ts_col is None:
        ts_candidates = [
            c
            for c in df.columns
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
            c
            for c in df.columns
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

    return ts_col, val_col


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


def time_bin_mask(series: pd.Series, start_t: time, end_t: time) -> pd.Series:
    return (series >= start_t) & (series < end_t)


def overlap_bin_labels(bins: Sequence[Tuple[time, time]]) -> List[str]:
    return [f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}" for start, end in bins]


def aggregate_values(values: List[float], mode: str) -> float:
    if not values:
        return np.nan
    if mode == "sum":
        return float(np.sum(values))
    return float(np.mean(values))


def format_annot(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return matrix.copy()

    def _format_cell(x: object) -> str:
        if pd.isna(x):
            return ""
        value = float(x)
        if value.is_integer():
            return str(int(value))
        return f"{value:.1f}".rstrip("0").rstrip(".")

    values = np.vectorize(_format_cell, otypes=[object])(matrix.to_numpy())
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)


def plot_heatmap(
    matrix: pd.DataFrame,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "viridis_r",
    mask_zero: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    if matrix.empty:
        print(f"[WARN] No data available for {out_path.name}")
        return

    plot_df = matrix.replace(0, np.nan) if mask_zero else matrix.copy()
    annot = format_annot(plot_df)

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
    if title:
        ax.set_title(title)
    ax.set_xlim(-0.5, len(plot_df.columns) + 0.5)
    ax.set_ylim(len(plot_df.index) + 0.5, -0.5)
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {out_path}")


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

        ts_col, val_col = detect_columns(df)
        if ts_col is None or val_col is None:
            continue

        out = pd.DataFrame()
        out["timestamp"] = normalize_timestamp(df[ts_col], local_tz=local_tz)
        out["value"] = pd.to_numeric(df[val_col], errors="coerce")
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
        return pd.DataFrame(columns=["timestamp", "value", "date", "weekday", "time_obj"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def run_heatmap_suite(
    *,
    root: Path,
    output_dir: Path,
    type_token: str,
    metric_label: str,
    metric_folder: str,
    valid_min: Optional[float] = None,
    valid_max: Optional[float] = None,
    local_tz: str = LOCAL_TZ,
) -> None:
    ensure_dir(output_dir)

    # Find all participant folders and stop early if nothing is available.
    participant_dirs = list_participant_folders(root)
    if not participant_dirs:
        print(f"[ERROR] No participant folders found under {root}")
        return

    # Load classroom schedules so class and weekday coverage can stay schedule-aware.
    schedule_map = build_schedule_map(root)
    class_names = class_labels_for_schedule(schedule_map)

    # Build the fixed school-day bins used in the paper: 5-minute bins for coverage
    # and 30-minute bins for the time-of-day summary plot.
    time_bins_5min = build_time_bins("08:30", "15:00", 5)
    time_bins_30min = build_time_bins("08:30", "15:00", 30)
    time_bin_labels_5min = overlap_bin_labels(time_bins_5min)
    time_bin_labels_30min = overlap_bin_labels(time_bins_30min)

    # One row per output category and one column per participant.
    participant_labels = [participant_code(p.name) for p in participant_dirs]
    weekday_coverage = pd.DataFrame(0.0, index=WEEKDAY_ORDER, columns=participant_labels)
    class_coverage = (
        pd.DataFrame(0.0, index=class_names, columns=participant_labels)
        if class_names
        else pd.DataFrame()
    )
    time_coverage = pd.DataFrame(0.0, index=time_bin_labels_30min, columns=participant_labels)
    weekday_valid_points = pd.DataFrame(0.0, index=WEEKDAY_ORDER, columns=participant_labels)

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

        # Each participant is summarized day by day before we average across days.
        p_schedule = get_participant_schedule(p_dir.name, schedule_map)
        weekday_day_coverage = {day: [] for day in WEEKDAY_ORDER}
        weekday_day_valid_points = {day: 0 for day in WEEKDAY_ORDER}
        time_coverage_samples = {label: [] for label in time_bin_labels_5min}
        class_expected: Dict[str, set] = {cls: set() for cls in class_names}
        class_actual: Dict[str, set] = {cls: set() for cls in class_names}

        for current_date, day_df in metric_df.groupby("date"):
            day_name = pd.Timestamp(current_date).day_name()
            day_short = day_name[:3]
            if day_short not in WEEKDAY_ORDER:
                continue

            day_df = day_df.sort_values("timestamp")
            day_blocks = p_schedule[day_name] if p_schedule and day_name in p_schedule else []
            is_friday = day_name == "Friday"
            # Only count bins that overlap scheduled class time.
            weekday_bin_candidates = [
                (start_t, end_t)
                for start_t, end_t in time_bins_5min
                if day_blocks and bin_overlaps_blocks(start_t, end_t, day_blocks)
            ]

            day_coverage_flags: List[int] = []
            day_valid_bins = 0
            for (start_t, end_t), interval_5 in zip(time_bins_5min, time_bin_labels_5min):
                bin_df = day_df[time_bin_mask(day_df["time_obj"], start_t, end_t)]
                has_sample = not bin_df.empty
                if (start_t, end_t) in weekday_bin_candidates:
                    day_coverage_flags.append(1 if has_sample else 0)
                    if not is_friday:
                        time_coverage_samples[interval_5].append(1 if has_sample else 0)
                if has_sample and (start_t, end_t) in weekday_bin_candidates:
                    day_valid_bins += 1

            if day_coverage_flags:
                weekday_day_coverage[day_short].append(float(np.mean(day_coverage_flags)) * 100.0)
            weekday_day_valid_points[day_short] += day_valid_bins

        # Friday stays in the feasibility summaries, but not in the contextual class/time plots.
        if class_names and p_schedule is not None:
            for current_date, day_df in metric_df.groupby("date"):
                day_name = pd.Timestamp(current_date).day_name()
                if day_name == "Friday":
                    continue
                if day_name not in p_schedule:
                    continue
                day_blocks = p_schedule[day_name]
                if not day_blocks:
                    continue
                for cls in class_names:
                    cls_blocks = [(s, e, cls) for s, e, c in day_blocks if canonicalize_class_label(c) == cls]
                    if not cls_blocks:
                        continue
                    for (start_t, end_t), interval_5 in zip(time_bins_5min, time_bin_labels_5min):
                        if bin_overlaps_blocks(start_t, end_t, cls_blocks):
                            class_expected[cls].add((current_date, interval_5))
                            bin_df = day_df[time_bin_mask(day_df["time_obj"], start_t, end_t)]
                            if not bin_df.empty:
                                class_actual[cls].add((current_date, interval_5))

        for day_short in WEEKDAY_ORDER:
            if weekday_day_coverage[day_short]:
                weekday_coverage.loc[day_short, participant] = float(np.mean(weekday_day_coverage[day_short]))
            weekday_valid_points.loc[day_short, participant] = float(weekday_day_valid_points[day_short])

        time_pct = {
            interval: (sum(samples) / len(samples)) * 100.0
            for interval, samples in time_coverage_samples.items()
            if samples
        }
        for (start_30, end_30), interval_30 in zip(time_bins_30min, time_bin_labels_30min):
            bins_in_30 = []
            current_min = time_to_minutes(start_30)
            end_min = time_to_minutes(end_30)
            if current_min is None or end_min is None:
                continue
            while current_min < end_min:
                next_min = current_min + 5
                interval_5 = f"{current_min // 60:02d}:{current_min % 60:02d}-{next_min // 60:02d}:{next_min % 60:02d}"
                if interval_5 in time_pct:
                    bins_in_30.append(time_pct[interval_5])
                current_min = next_min
            if bins_in_30:
                time_coverage.loc[interval_30, participant] = float(np.mean(bins_in_30))

        for cls in class_names:
            total_bins = len(class_expected[cls])
            covered_bins = len(class_actual[cls])
            if total_bins > 0:
                class_coverage.loc[cls, participant] = (covered_bins / total_bins) * 100.0

    if not class_coverage.empty:
        # Remove classes that never have data in any participant before plotting.
        class_coverage = class_coverage.loc[(class_coverage != 0).any(axis=1)]
        class_order = class_coverage.sum(axis=1).sort_values(ascending=False).index.tolist()
        class_coverage = class_coverage.loc[class_order]

    # Write every graph directly into the datatype folder.
    weekday_coverage.to_csv(output_dir / f"{metric_folder}_coverage_by_weekday.csv")
    plot_heatmap(
        weekday_coverage,
        output_dir / f"{metric_folder}_coverage_by_weekday.png",
        title=f"{metric_label} Coverage by Weekday",
        xlabel="Participant",
        ylabel="Weekday",
        cmap="viridis_r",
        mask_zero=True,
        vmin=0,
        vmax=100,
        figsize=(max(8, len(weekday_coverage.columns) * 0.8), 4.5),
    )
    print(f"Caption: {metric_label} coverage by weekday. Each cell shows the percent of scheduled 5-minute bins with valid data for that participant and weekday. Friday is included in this feasibility summary.")

    if not class_coverage.empty:
        class_coverage.to_csv(output_dir / f"{metric_folder}_coverage_by_class.csv")
        plot_heatmap(
            class_coverage,
            output_dir / f"{metric_folder}_coverage_by_class.png",
            title=f"{metric_label} Coverage by Class",
            xlabel="Participant",
            ylabel="Class",
            cmap="viridis_r",
            mask_zero=True,
            vmin=0,
            vmax=100,
            figsize=(max(8, len(class_coverage.columns) * 0.8), max(4, len(class_coverage.index) * 0.5)),
        )
        print(f"Caption: {metric_label} coverage by class. Each cell shows the percent of scheduled 5-minute bins inside that class that contained valid data for that participant. Friday is excluded from this contextual comparison.")

    time_coverage.to_csv(output_dir / f"{metric_folder}_coverage_by_time.csv")
    plot_heatmap(
        time_coverage,
        output_dir / f"{metric_folder}_coverage_by_time.png",
        title=f"{metric_label} Coverage by Time of Day",
        xlabel="Participant",
        ylabel="30-Minute Bin",
        cmap="viridis_r",
        mask_zero=True,
        vmin=0,
        vmax=100,
        figsize=(max(8, len(time_coverage.columns) * 0.8), 7.5),
    )
    print(f"Caption: {metric_label} coverage by 30-minute interval. Each cell shows the mean percent coverage across the 5-minute bins inside that 30-minute window. Friday is excluded from this contextual comparison.")

    weekday_valid_points.to_csv(output_dir / f"{metric_folder}_valid_data_points_by_weekday.csv")
    plot_heatmap(
        weekday_valid_points,
        output_dir / f"{metric_folder}_valid_data_points_by_weekday.png",
        title=f"{metric_label} Valid Scheduled 5-Minute Bins by Weekday",
        xlabel="Participant",
        ylabel="Weekday",
        cmap="viridis_r",
        mask_zero=True,
        figsize=(max(8, len(weekday_valid_points.columns) * 0.8), 4.5),
    )
    print(f"Caption: {metric_label} valid scheduled 5-minute bins by weekday. Each cell shows the count of valid scheduled 5-minute bins available for that participant and weekday. Friday is included in this feasibility summary.")
