#!/usr/bin/env python3
"""Relationships between heart rate and movement.

Aligned 5-minute multimodal bins: Apple Watch mean HR and MOCOPI movement
intensity (mean across whatever sensors were recording in that minute),
labeled from the same school schedules used in HeartRate-Patterns/ and
Coverage/. Complete-case bins (both modalities present) are the unit of
analysis.

Nested mixed-effects models with a participant random intercept test whether:
  1. higher movement is associated with higher HR;
  2. classroom-context differences remain after controlling for movement;
  3. the HR-movement slope differs across classroom contexts.

Intensity is centered within participant so the slope is within-child, not
between-child. Classroom labels come only from the schedule maps below; lunch
is recovered from DELETE blocks in the lunch window, matching HeartRate-Patterns.
"""
from __future__ import annotations

import argparse
import re
import warnings
from datetime import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, pearsonr

from Coverage.audit_binned_common import participants_dates

LOCAL_TZ = "US/Pacific"
HR_MIN = 40.0
HR_MAX = 180.0
WATCH_TYPE = "HKQuantityTypeIdentifierHeartRate"
LUNCH_START = time(11, 30)
LUNCH_END = time(13, 30)
MORNING_END = time(11, 55)
MIDDAY_END = time(13, 25)
POINT_COLOR = "#1967d2"
LINE_COLOR = "#d93025"
REF_CONTEXT = "Academic classes"
REF_TIME = "Morning"
SENSOR_DEFAULT = "all"

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


def time_block_from_ts(ts: pd.Timestamp) -> str:
    clock = ts.timetz().replace(tzinfo=None) if ts.tzinfo else ts.time()
    if clock < MORNING_END:
        return "Morning"
    if clock < MIDDAY_END:
        return "Midday"
    return "Afternoon"


def seconds_since_midnight(ts: pd.Timestamp) -> int:
    clock = ts.timetz().replace(tzinfo=None) if ts.tzinfo else ts.time()
    return clock.hour * 3600 + clock.minute * 60 + clock.second


def as_pacific(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
    return parsed.dt.tz_convert(LOCAL_TZ)


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


def build_scheduled_bins(participant_key: str, date_str: str, sched: pd.DataFrame) -> pd.DataFrame:
    """Expand each labeled class period into 5-minute bins, then label from the schedule.

    Duplicate starts (overlapping schedule rows) keep the class_from_schedule
    label at that instant, matching HeartRate-Patterns sample labeling.
    """
    if sched.empty:
        return pd.DataFrame(columns=["participant", "date", "weekday", "bin_start", "bin_end", "context", "time_block"])

    base_day = pd.Timestamp(date_str).tz_localize(LOCAL_TZ)
    weekday = pd.Timestamp(date_str).day_name()
    by_start: dict[int, dict] = {}

    for _, row in sched.iterrows():
        start_dt = base_day + pd.Timedelta(seconds=int(row["StartSec"]))
        end_dt = base_day + pd.Timedelta(seconds=int(row["EndSec"]))
        if end_dt <= start_dt:
            continue
        current = start_dt
        while current < end_dt:
            next_edge = min(current + pd.Timedelta(minutes=5), end_dt)
            label = paper_context(class_from_schedule(seconds_since_midnight(current), sched))
            if label is not None:
                by_start[int(current.value)] = {
                    "participant": pid_from_key(participant_key),
                    "date": date_str,
                    "weekday": weekday,
                    "bin_start": current,
                    "bin_end": next_edge,
                    "context": label,
                    "time_block": time_block_from_ts(current),
                }
            current = next_edge

    if not by_start:
        return pd.DataFrame(columns=["participant", "date", "weekday", "bin_start", "bin_end", "context", "time_block"])
    out = pd.DataFrame(by_start.values()).sort_values("bin_start").reset_index(drop=True)
    return out


def load_all_bins(apple_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key, dates in participants_dates.items():
        friday, other, tuesday = schedule_paths(key, apple_root)
        sched_fri = load_schedule(friday)
        sched_oth = load_schedule(other)
        sched_tu = load_schedule(tuesday) if tuesday else None
        for date_str in sorted(dates):
            sched = schedule_for_date(key, date_str, sched_fri, sched_oth, sched_tu)
            frames.append(build_scheduled_bins(key, date_str, sched))
    columns = ["participant", "date", "weekday", "bin_start", "bin_end", "context", "time_block"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def load_watch_hr(apple_root: Path) -> pd.DataFrame:
    usecols = ["/Record/@startDate", "/Record/@type", "/Record/@value", "/Record/@sourceName"]
    frames: list[pd.DataFrame] = []
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
        out = watch[["timestamp", "bpm", "date"]].copy()
        out["participant"] = pid
        frames.append(out)
        print(f"[INFO] {pid}: {len(out)} valid Apple Watch HR samples on study dates", flush=True)
    columns = ["participant", "timestamp", "bpm", "date"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def load_movement_epochs(mocopi_root: Path, sensor: str) -> pd.DataFrame:
    """1-minute movement intensity on study dates.

    Default is the mean across every MOCOPI sensor recording in that minute.
    Hip-only would drop Physical Education for children whose hip channel is
    missing that period (e.g. P001 Friday Funday).
    """
    frames: list[pd.DataFrame] = []
    use_all = sensor.strip().lower() in {"", "all"}
    for key, dates in participants_dates.items():
        pid = pid_from_key(key)
        path = mocopi_root / f"{pid}_epoch_kinematics.csv"
        if not path.exists():
            print(f"[WARN] Missing epoch file: {path}")
            continue
        raw = pd.read_csv(path)
        needed = {"Sensor", "Date", "Epoch_1Min", "Intensity"}
        missing = needed - set(raw.columns)
        if missing:
            print(f"[WARN] {path.name} missing columns {sorted(missing)}")
            continue
        raw = raw.copy()
        if not use_all:
            raw = raw[raw["Sensor"].astype(str).eq(sensor)].copy()
        if raw.empty:
            print(f"[WARN] {pid}: no epochs for sensor={sensor}")
            continue
        raw["timestamp"] = as_pacific(raw["Epoch_1Min"])
        raw["intensity"] = pd.to_numeric(raw["Intensity"], errors="coerce")
        raw = raw.dropna(subset=["timestamp", "intensity"])
        raw["date"] = raw["timestamp"].dt.strftime("%Y-%m-%d")
        on_study = raw[raw["date"].isin(dates)].copy()
        extra = sorted(set(raw["date"]) - dates)
        missing_days = sorted(dates - set(raw["date"]))
        if on_study.empty:
            print(
                f"[INFO] {pid}: 0 movement minutes on study dates "
                f"(extra_days={extra or 'none'}; missing_study_days={missing_days or 'none'})",
                flush=True,
            )
            continue
        out = (
            on_study.groupby(["timestamp", "date"], as_index=False)
            .agg(intensity=("intensity", "mean"), n_sensors=("Sensor", "nunique"))
        )
        out["participant"] = pid
        print(
            f"[INFO] {pid}: {len(out)} movement minutes on study dates "
            f"(sensors_per_minute mean={out['n_sensors'].mean():.1f}; "
            f"extra_days={extra or 'none'}; missing_study_days={missing_days or 'none'})",
            flush=True,
        )
        frames.append(out[["participant", "timestamp", "intensity", "date"]])
    columns = ["participant", "timestamp", "intensity", "date"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def match_bin_index(timestamps: pd.Series, starts_ns: np.ndarray, ends_ns: np.ndarray) -> np.ndarray:
    t_ns = pd.DatetimeIndex(timestamps).asi8
    idx = np.searchsorted(starts_ns, t_ns, side="right") - 1
    out = np.full(len(t_ns), -1, dtype=np.int64)
    ok = idx >= 0
    if ok.any():
        chosen = idx[ok]
        still = (t_ns[ok] >= starts_ns[chosen]) & (t_ns[ok] < ends_ns[chosen])
        ok_idx = np.flatnonzero(ok)
        out[ok_idx[still]] = chosen[still]
    return out


def aggregate_into_bins(values: pd.DataFrame, bins: pd.DataFrame, value_col: str, count_name: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    grouped_values = values.groupby(["participant", "date"], sort=False)
    for (participant, date_str), day_bins in bins.groupby(["participant", "date"], sort=False):
        try:
            day_vals = grouped_values.get_group((participant, date_str))
        except KeyError:
            continue
        day_bins = day_bins.sort_values("bin_start").reset_index(drop=True)
        starts_ns = pd.DatetimeIndex(day_bins["bin_start"]).asi8
        ends_ns = pd.DatetimeIndex(day_bins["bin_end"]).asi8
        matched = match_bin_index(day_vals["timestamp"], starts_ns, ends_ns)
        usable = day_vals.loc[matched >= 0, [value_col]].copy()
        if usable.empty:
            continue
        usable["bin_row"] = matched[matched >= 0]
        stats = usable.groupby("bin_row")[value_col].agg(mean="mean", count="size").reset_index()
        part = day_bins.loc[stats["bin_row"], ["participant", "date", "bin_start"]].reset_index(drop=True)
        part[f"mean_{value_col}"] = stats["mean"].to_numpy()
        part[count_name] = stats["count"].to_numpy().astype(int)
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["participant", "date", "bin_start", f"mean_{value_col}", count_name])
    return pd.concat(rows, ignore_index=True)


def build_aligned_bins(bins: pd.DataFrame, hr: pd.DataFrame, movement: pd.DataFrame) -> pd.DataFrame:
    hr_bins = aggregate_into_bins(hr.rename(columns={"bpm": "hr"}), bins, "hr", "n_hr_samples")
    mv_bins = aggregate_into_bins(movement, bins, "intensity", "n_movement_minutes")
    merged = bins.merge(hr_bins, on=["participant", "date", "bin_start"], how="left")
    merged = merged.merge(mv_bins, on=["participant", "date", "bin_start"], how="left")
    merged["n_hr_samples"] = merged["n_hr_samples"].fillna(0).astype(int)
    merged["n_movement_minutes"] = merged["n_movement_minutes"].fillna(0).astype(int)
    return merged


def complete_case(merged: pd.DataFrame) -> pd.DataFrame:
    complete = merged[(merged["n_hr_samples"] > 0) & (merged["n_movement_minutes"] > 0)].copy()
    complete["intensity"] = complete["mean_intensity"]
    complete["intensity_c"] = complete["intensity"] - complete.groupby("participant")["intensity"].transform("mean")
    complete["context"] = pd.Categorical(
        complete["context"],
        categories=[c for c in CONTEXT_ORDER if c in set(complete["context"])],
        ordered=True,
    )
    complete["time_block"] = pd.Categorical(
        complete["time_block"],
        categories=[t for t in TIME_BLOCKS if t in set(complete["time_block"])],
        ordered=True,
    )
    return complete.reset_index(drop=True)


def n_by_context(complete: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for context in CONTEXT_ORDER:
        g = complete[complete["context"].astype(str).eq(context)]
        rows.append(
            {
                "context": context,
                "n_bins": int(len(g)),
                "n_participants": int(g["participant"].nunique()),
                "n_dates": int(g[["participant", "date"]].drop_duplicates().shape[0]),
                "mean_hr": float(g["mean_hr"].mean()) if len(g) else np.nan,
                "sd_hr": float(g["mean_hr"].std(ddof=1)) if len(g) > 1 else np.nan,
                "mean_intensity": float(g["intensity"].mean()) if len(g) else np.nan,
                "sd_intensity": float(g["intensity"].std(ddof=1)) if len(g) > 1 else np.nan,
                "mean_n_hr_samples": float(g["n_hr_samples"].mean()) if len(g) else np.nan,
                "mean_n_movement_minutes": float(g["n_movement_minutes"].mean()) if len(g) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def context_formula(interaction: bool) -> str:
    context = f"C(context, Treatment('{REF_CONTEXT}'))"
    time = f"C(time_block, Treatment('{REF_TIME}'))"
    if interaction:
        return f"mean_hr ~ intensity_c * {context} + {time}"
    return f"mean_hr ~ intensity_c + {context} + {time}"


def fit_mixedlm(data: pd.DataFrame, formula: str, reml: bool):
    model = smf.mixedlm(formula, data, groups=data["participant"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return model.fit(reml=reml, method=["lbfgs"], maxiter=400)
        except Exception:
            return model.fit(reml=reml, method="lbfgs", maxiter=400)


def result_table(result, model_name: str, formula: str, data: pd.DataFrame) -> pd.DataFrame:
    conf = result.conf_int()
    rows = []
    for term, coef in result.fe_params.items():
        rows.append(
            {
                "model": model_name,
                "formula": formula,
                "term": term,
                "coef": float(coef),
                "se": float(result.bse_fe[term]),
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
                "p_value": float(result.pvalues[term]),
                "n_bins": int(len(data)),
                "n_participants": int(data["participant"].nunique()),
            }
        )
    var_u = float(result.cov_re.iloc[0, 0]) if not result.cov_re.empty else np.nan
    var_e = float(result.scale)
    rows.append(
        {
            "model": model_name,
            "formula": formula,
            "term": "ICC",
            "coef": float(var_u / (var_u + var_e)) if np.isfinite(var_u) and (var_u + var_e) > 0 else np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "n_bins": int(len(data)),
            "n_participants": int(data["participant"].nunique()),
        }
    )
    return pd.DataFrame(rows)


def likelihood_ratio(restricted, full, comparison: str) -> dict:
    lr = 2.0 * (full.llf - restricted.llf)
    df_diff = int(len(full.fe_params) - len(restricted.fe_params))
    p_value = float(chi2.sf(lr, df_diff)) if df_diff > 0 else np.nan
    return {
        "comparison": comparison,
        "lr_stat": float(lr),
        "df": df_diff,
        "p_value": p_value,
        "restricted_llf": float(restricted.llf),
        "full_llf": float(full.llf),
    }


def context_slopes(result, data: pd.DataFrame) -> pd.DataFrame:
    """Within-person HR slope per unit intensity, by classroom context, from the interaction model."""
    params = result.fe_params
    cov = result.cov_params()
    intensity_term = "intensity_c"
    if intensity_term not in params.index:
        return pd.DataFrame(columns=["context", "slope", "se", "ci_low", "ci_high", "n_bins", "n_participants"])

    rows = []
    present = list(data["context"].cat.categories)
    for context in present:
        if context == REF_CONTEXT:
            terms = [intensity_term]
            weights = np.array([1.0])
        else:
            interact = f"intensity_c:C(context, Treatment('{REF_CONTEXT}'))[T.{context}]"
            if interact not in params.index:
                continue
            terms = [intensity_term, interact]
            weights = np.array([1.0, 1.0])
        beta = params.loc[terms].to_numpy(dtype=float)
        slope = float(weights @ beta)
        sub = cov.loc[terms, terms].to_numpy(dtype=float)
        var = float(weights @ sub @ weights)
        se = float(np.sqrt(var)) if var > 0 else np.nan
        g = data[data["context"].eq(context)]
        rows.append(
            {
                "context": context,
                "slope": slope,
                "se": se,
                "ci_low": slope - 1.96 * se if np.isfinite(se) else np.nan,
                "ci_high": slope + 1.96 * se if np.isfinite(se) else np.nan,
                "n_bins": int(len(g)),
                "n_participants": int(g["participant"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def context_contrasts(models: pd.DataFrame) -> dict[str, pd.Series]:
    prefix = f"C(context, Treatment('{REF_CONTEXT}'))[T."
    out: dict[str, pd.Series] = {}
    for _, row in models[models["model"].eq("model_2_context_time")].iterrows():
        term = str(row["term"])
        if prefix in term:
            out[term.split("[T.", 1)[1].rstrip("]")] = row
    return out


def build_results_table(n_table: pd.DataFrame, models: pd.DataFrame, slopes: pd.DataFrame) -> pd.DataFrame:
    """One paper table: sample, HR–movement slope by context, and movement-adjusted HR vs academic classes."""
    contrasts = context_contrasts(models)
    slope_by = {str(r["context"]): r for _, r in slopes.iterrows()}
    rows = []
    for _, row in n_table.iterrows():
        context = str(row["context"])
        rec = {
            "context": context,
            "n_bins": int(row["n_bins"]),
            "n_participants": int(row["n_participants"]),
            "mean_hr_bpm": np.nan if pd.isna(row["mean_hr"]) else round(float(row["mean_hr"]), 1),
            "mean_intensity_g": np.nan if pd.isna(row["mean_intensity"]) else round(float(row["mean_intensity"]), 2),
            "hr_movement_slope": np.nan,
            "slope_ci_low": np.nan,
            "slope_ci_high": np.nan,
            "adjusted_hr_vs_academic": np.nan,
            "adj_ci_low": np.nan,
            "adj_ci_high": np.nan,
            "adj_p": np.nan,
        }
        if context in slope_by:
            s = slope_by[context]
            rec.update(
                hr_movement_slope=round(float(s["slope"]), 1),
                slope_ci_low=round(float(s["ci_low"]), 1),
                slope_ci_high=round(float(s["ci_high"]), 1),
            )
        if context == REF_CONTEXT:
            rec["adjusted_hr_vs_academic"] = 0.0
        elif context in contrasts:
            c = contrasts[context]
            rec.update(
                adjusted_hr_vs_academic=round(float(c["coef"]), 1),
                adj_ci_low=round(float(c["ci_low"]), 1),
                adj_ci_high=round(float(c["ci_high"]), 1),
                adj_p=float(c["p_value"]),
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def _r_label(x: pd.Series, y: pd.Series) -> str:
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 3:
        return f"n={len(z)}"
    r, p = pearsonr(z["x"].to_numpy(float), z["y"].to_numpy(float))
    if p < 0.001:
        ptxt = "<.001"
    elif p >= 0.05:
        ptxt = "ns"
    else:
        ptxt = f"{p:.3f}".lstrip("0")
    return f"r={r:.2f} ({ptxt}), n={len(z)}"


def save_scatter_figure(complete: pd.DataFrame, path: Path) -> None:
    """Shared x-axis, r labeled on each panel. Weak relationships are drawn as clouds, not steep lines."""
    xmax = 8.0
    panels: list[tuple[str, pd.DataFrame]] = [("All bins", complete)]
    for context in CONTEXT_ORDER:
        g = complete[complete["context"].astype(str).eq(context)]
        if g.empty:
            continue
        panels.append((context, g))

    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.4, 3.55 * nrows), sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, g) in zip(axes, panels):
        shown = g[g["intensity"] <= xmax]
        clipped = int((g["intensity"] > xmax).sum())
        ax.scatter(shown["intensity"], shown["mean_hr"], s=12, alpha=0.32, color=POINT_COLOR, linewidths=0, zorder=2)
        if len(shown) >= 20:
            m, b = np.polyfit(shown["intensity"].to_numpy(float), shown["mean_hr"].to_numpy(float), 1)
            grid = np.linspace(0.0, xmax, 50)
            ax.plot(grid, m * grid + b, color=LINE_COLOR, lw=1.5, zorder=3)
        ax.set_xlim(0, xmax)
        ax.set_ylim(55, 165)
        extra = f", {clipped} off-scale" if clipped else ""
        ax.set_title(f"{title}\n{_r_label(shown['intensity'], shown['mean_hr'])}{extra}")
        ax.grid(alpha=0.25)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.supxlabel("Movement intensity (g)")
    fig.supylabel("Mean Apple Watch HR (bpm)")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HR–movement relationship in aligned 5-minute bins.")
    parser.add_argument("--apple-root", type=Path, default=Path.home() / "Downloads" / "Apple Watch Export CSVs")
    parser.add_argument("--mocopi-root", type=Path, default=Path.home() / "Downloads" / "epoch_kinematics")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--sensor", default=SENSOR_DEFAULT, help="MOCOPI sensor, or 'all' (default) to average every sensor present in each minute.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    bins = load_all_bins(args.apple_root)
    print(f"[INFO] Scheduled 5-minute bins: {len(bins)}", flush=True)
    hr = load_watch_hr(args.apple_root)
    movement = load_movement_epochs(args.mocopi_root, args.sensor)

    aligned = build_aligned_bins(bins, hr, movement)
    complete = complete_case(aligned)
    if complete.empty:
        raise SystemExit("No complete-case 5-minute bins. Check date overlap and file paths.")

    model_data = complete.copy()
    model_data["context"] = model_data["context"].cat.remove_unused_categories()
    model_data["time_block"] = model_data["time_block"].cat.remove_unused_categories()
    print(
        f"[INFO] Complete-case bins: {len(complete)} from {complete['participant'].nunique()} participants; "
        f"contexts={list(model_data['context'].cat.categories)}",
        flush=True,
    )

    f1 = "mean_hr ~ intensity_c"
    f2 = context_formula(interaction=False)
    f3 = context_formula(interaction=True)
    f0 = "mean_hr ~ 1"

    m0_ml = fit_mixedlm(model_data, f0, reml=False)
    m1_ml = fit_mixedlm(model_data, f1, reml=False)
    m2_ml = fit_mixedlm(model_data, f2, reml=False)
    m3_ml = fit_mixedlm(model_data, f3, reml=False)
    m2 = fit_mixedlm(model_data, f2, reml=True)
    m3 = fit_mixedlm(model_data, f3, reml=True)

    models = pd.concat(
        [
            result_table(m2, "model_2_context_time", f2, model_data),
            result_table(m3, "model_3_interaction", f3, model_data),
        ],
        ignore_index=True,
    )
    slopes = context_slopes(m3, model_data)
    n_table = n_by_context(complete)
    results = build_results_table(n_table, models, slopes)
    results.to_csv(out / "table_results.csv", index=False)
    save_scatter_figure(complete, out / "figure_hr_vs_movement.png")

    intensity = models[models["model"].eq("model_2_context_time") & models["term"].eq("intensity_c")].iloc[0]
    lrt_move = likelihood_ratio(m0_ml, m1_ml, "movement")
    lrt_ctx = likelihood_ratio(m1_ml, m2_ml, "context")
    lrt_int = likelihood_ratio(m2_ml, m3_ml, "interaction")
    print(
        f"[INFO] Movement effect (Model 2): {intensity['coef']:.1f} "
        f"({intensity['ci_low']:.1f} to {intensity['ci_high']:.1f}) bpm per 1 g, p={intensity['p_value']:.3g}"
    )
    print(
        f"[INFO] LRTs: movement p={lrt_move['p_value']:.3g}; "
        f"context after movement p={lrt_ctx['p_value']:.3g}; "
        f"movement×context p={lrt_int['p_value']:.3g}"
    )
    print(f"[DONE] Wrote {out / 'figure_hr_vs_movement.png'} and {out / 'table_results.csv'}")


if __name__ == "__main__":
    main()
