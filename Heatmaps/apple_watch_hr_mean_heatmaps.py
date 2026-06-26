#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heatmap_shared import APPLE_WATCH_DATA_PREFIX, LOCAL_TZ

TIME_BINS_30_MIN = [
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
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def participant_code(name: str) -> str:
    match = re.search(r"(\d+)", name)
    return f"P{int(match.group(1)):03d}" if match else name


def participant_sort_key(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def list_participant_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"P\d+", p.name)]
    return sorted(dirs, key=lambda p: participant_sort_key(p.name))


def normalize_class_label(label: object) -> Optional[str]:
    raw = str(label).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return None

    key = re.sub(r"\s+", " ", raw.lower())
    mapping = {
        "cash-out": "Cash-out",
        "cash out": "Cash-out",
        "commsci": "Commsci",
        "ela": "ELA",
        "ela/history": "ELA/History",
        "history/ela": "ELA/History",
        "funday friday": "Friday Funday",
        "friday funday": "Friday Funday",
        "history": "History",
        "homeroom": "Homeroom",
        "hw rein./study hall": "HW Rein./Study Hall",
        "homework reinforcement/study hall": "HW Rein./Study Hall",
        "math": "Math",
        "social skills": "Social Skills",
    }
    return mapping.get(key, raw)


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def pick_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["CreationDate", "StartDate", "EndDate", "Time_In_PST", "Time", "Date"]
    for col in candidates:
        if col in df.columns:
            return col

    best_col = None
    best_score = 0
    for col in df.columns:
        if not any(key in str(col).lower() for key in ("date", "time", "start", "end")):
            continue
        score = pd.to_datetime(df[col], errors="coerce").notna().sum()
        if score > best_score:
            best_col = col
            best_score = score
    return best_col


def to_local_timestamp(series: pd.Series, local_tz: str = LOCAL_TZ) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(local_tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        ts = ts.dt.tz_convert(local_tz)
    return ts


def find_heart_rate_files(participant_dir: Path) -> List[Path]:
    record_root = participant_dir / "HealthApp" / "Labeled" / "Record"
    if not record_root.exists():
        return []
    return sorted(p for p in record_root.rglob("*.csv") if p.name.endswith("_HeartRate.csv"))


def load_participant_heart_rate(participant_dir: Path, hr_min: float, hr_max: float) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for csv_path in find_heart_rate_files(participant_dir):
        df = safe_read_csv(csv_path)
        if df is None or df.empty:
            continue

        class_col = "class" if "class" in df.columns else ("Class" if "Class" in df.columns else None)
        value_col = "Value" if "Value" in df.columns else ("value" if "value" in df.columns else None)
        if class_col is None or value_col is None:
            continue

        ts_col = pick_timestamp_col(df)
        if ts_col is None:
            continue

        out = pd.DataFrame()
        out["class_label"] = df[class_col].map(normalize_class_label)
        out["timestamp_local"] = to_local_timestamp(df[ts_col], local_tz=LOCAL_TZ)
        out["bpm"] = pd.to_numeric(df[value_col], errors="coerce")
        out["participant"] = participant_code(participant_dir.name)
        out = out.dropna(subset=["class_label", "timestamp_local", "bpm"])
        out = out[(out["bpm"] >= hr_min) & (out["bpm"] <= hr_max)]
        if not out.empty:
            frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["participant", "class_label", "timestamp_local", "bpm"])

    return pd.concat(frames, ignore_index=True).sort_values("timestamp_local")


def time_bin_label(ts_local: pd.Series) -> pd.Series:
    times = ts_local.dt.time
    labels: List[str] = []
    for t in times:
        cur_min = t.hour * 60 + t.minute
        label = ""
        for start_str, end_str in TIME_BINS_30_MIN:
            sh, sm = map(int, start_str.split(":"))
            eh, em = map(int, end_str.split(":"))
            if sh * 60 + sm <= cur_min < eh * 60 + em:
                label = f"{start_str}-{end_str}"
                break
        labels.append(label)
    return pd.Series(labels, index=ts_local.index)


def plot_heatmap(
    matrix: pd.DataFrame,
    out_path: Path,
    title: str,
    ylabel: str,
    cbar_label: str,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    if matrix.empty:
        print(f"[WARN] No data available for {out_path.name}")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(matrix.columns) * 0.8), max(4, len(matrix.index) * 0.55)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        linewidths=0.5,
        linecolor="gray",
        cbar=True,
        annot=True,
        fmt=".1f",
        mask=matrix.isna(),
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = ax.collections[0].colorbar if ax.collections else None
    if colorbar is not None:
        colorbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("Participant")
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def build_mean_heatmaps(root: Path, output_dir: Path, hr_min: float, hr_max: float) -> None:
    ensure_dir(output_dir)
    participant_dirs = list_participant_dirs(root)
    if not participant_dirs:
        print(f"[ERROR] No participant folders found under {root}")
        return

    all_rows: List[pd.DataFrame] = []
    for participant_dir in participant_dirs:
        hr_df = load_participant_heart_rate(participant_dir, hr_min=hr_min, hr_max=hr_max)
        if hr_df.empty:
            print(f"[WARN] No usable heart-rate rows found for {participant_dir.name}")
            continue
        print(f"[INFO] {participant_dir.name}: {len(hr_df)} valid heart-rate rows")
        all_rows.append(hr_df)

    if not all_rows:
        print("[WARN] No usable heart-rate rows found across participants")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    class_agg = combined.groupby(["class_label", "participant"], as_index=False).agg(mean_hr_bpm=("bpm", "mean"))
    class_pivot = class_agg.pivot(index="class_label", columns="participant", values="mean_hr_bpm")
    class_pivot = class_pivot.sort_index()
    class_pivot = class_pivot.reindex(sorted(class_pivot.columns, key=participant_sort_key), axis=1)
    class_pivot.to_csv(output_dir / "heart_rate_mean_by_class.csv")
    plot_heatmap(
        class_pivot,
        output_dir / "heart_rate_mean_by_class.png",
        title=f"{APPLE_WATCH_DATA_PREFIX} Mean Heart Rate Heatmap by Class",
        ylabel="Class",
        cbar_label="Mean HR (bpm)",
    )

    time_rows = combined.copy()
    time_rows["time_bin"] = time_bin_label(time_rows["timestamp_local"])
    time_rows = time_rows[time_rows["time_bin"] != ""]
    if time_rows.empty:
        print("[WARN] No usable heart-rate rows within the configured time bins")
        return

    time_agg = time_rows.groupby(["time_bin", "participant"], as_index=False).agg(mean_hr_bpm=("bpm", "mean"))
    time_order = [f"{start}-{end}" for start, end in TIME_BINS_30_MIN]
    time_agg["time_bin"] = pd.Categorical(time_agg["time_bin"], categories=time_order, ordered=True)
    time_agg = time_agg.sort_values("time_bin")
    time_pivot = time_agg.pivot(index="time_bin", columns="participant", values="mean_hr_bpm")
    time_pivot = time_pivot.reindex(time_order)
    time_pivot = time_pivot.reindex(sorted(time_pivot.columns, key=participant_sort_key), axis=1)
    time_pivot.to_csv(output_dir / "heart_rate_mean_by_time_of_day.csv")
    plot_heatmap(
        time_pivot,
        output_dir / "heart_rate_mean_by_time_of_day.png",
        title=f"{APPLE_WATCH_DATA_PREFIX} Mean Heart Rate Heatmap by Time of Day (30-min bin)",
        ylabel="Time Interval (30-minute bins)",
        cbar_label="Mean HR (bpm)",
    )

    print(f"[DONE] Outputs in: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Apple Watch heart-rate mean heatmaps.")
    parser.add_argument(
        "--root",
        default=os.path.expanduser("~/Downloads/Exports"),
        help="Root folder containing participant folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots and summary CSVs.",
    )
    parser.add_argument("--hr-min", type=float, default=40.0, help="Minimum valid heart-rate value.")
    parser.add_argument("--hr-max", type=float, default=180.0, help="Maximum valid heart-rate value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parent / "Graphs" / "HeartRateMean"
    )
    build_mean_heatmaps(root=root, output_dir=out_dir, hr_min=args.hr_min, hr_max=args.hr_max)


if __name__ == "__main__":
    main()
