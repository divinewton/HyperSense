#!/usr/bin/env python3
"""Correlate Apple Watch active energy with Oura class and MOCOPI wrist motion.

Active Energy Burned (kcal) is summed into 5-minute bins on study dates, then
paired with:
  * Oura activity class (class_5_min, 1–5)
  * MOCOPI right-wrist magnitude / variability / jerk

P007 MOCOPI uses the shared wrist rebuild (copied epoch file is not used).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_DIR = REPO_ROOT / "Coverage"
if str(COVERAGE_DIR) not in sys.path:
    sys.path.insert(0, str(COVERAGE_DIR))

from audit_binned_common import participants_dates  # noqa: E402

BIN_MINUTES = 5
LOCAL_TZ = "US/Pacific"
ENERGY_TYPE = "HKQuantityTypeIdentifierActiveEnergyBurned"
MOCOPI_SENSOR = "WristR"

OURA_CLASS_TICKS = (1, 2, 3, 4, 5)
OURA_CLASS_LABELS = (
    "Rest",
    "Inactive",
    "Low activity",
    "Medium activity",
    "High activity",
)

MOCOPI_METRICS = (
    {"key": "magnitude", "column": "Intensity", "short": "Magnitude"},
    {"key": "acceleration", "column": "Variability", "short": "Accel. variability"},
    {"key": "jerk", "column": "Jerk", "short": "Jerk"},
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pid_from_key(key: str) -> str:
    return f"P{int(key):03d}"


def export_skiprows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for i, line in enumerate(handle):
            if "/@locale" in line:
                return i
    return 0


def floor_to_bin(series: pd.Series, minutes: int = BIN_MINUTES) -> pd.Series:
    return series.dt.floor(f"{minutes}min")


def corr_stats(x: pd.Series, y: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    n = int(len(x))
    if n < 3:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        "n": n,
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def _stats_text(stats: dict) -> str:
    if stats["n"] < 3 or np.isnan(stats["spearman_rho"]):
        return f"n={stats['n']}"
    return (
        f"n={stats['n']}\n"
        f"Spearman ρ={stats['spearman_rho']:.2f} (p={stats['spearman_p']:.3g})\n"
        f"Pearson r={stats['pearson_r']:.2f} (p={stats['pearson_p']:.3g})"
    )


def load_watch_active_energy(apple_root: Path) -> pd.DataFrame:
    """Sum Active Energy Burned (kcal) into 5-minute bins on study dates."""
    usecols = [
        "/Record/@startDate",
        "/Record/@endDate",
        "/Record/@type",
        "/Record/@value",
        "/Record/@sourceName",
    ]
    frames: list[pd.DataFrame] = []
    for key, dates in participants_dates.items():
        pid = pid_from_key(key)
        path = apple_root / f"{pid}export.csv"
        if not path.exists():
            print(f"[WARN] Missing watch export: {path}")
            continue
        print(f"[INFO] Reading active energy {pid}", flush=True)
        raw = pd.read_csv(
            path,
            skiprows=export_skiprows(path),
            usecols=usecols,
            low_memory=False,
        )
        energy = raw[raw["/Record/@type"].astype(str).eq(ENERGY_TYPE)].copy()
        energy["timestamp"] = pd.to_datetime(
            energy["/Record/@startDate"], errors="coerce", utc=True
        ).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        energy["kcal"] = pd.to_numeric(energy["/Record/@value"], errors="coerce")
        source = (
            energy["/Record/@sourceName"]
            .astype(str)
            .str.replace("\u00a0", " ", regex=False)
        )
        energy = energy[source.str.contains("Apple Watch", case=False, na=False)]
        energy = energy.dropna(subset=["timestamp", "kcal"])
        energy = energy[energy["kcal"] >= 0]
        energy["Date"] = energy["timestamp"].dt.strftime("%Y-%m-%d")
        energy = energy[energy["Date"].isin(dates)]
        if energy.empty:
            print(f"[WARN] {pid}: no Apple Watch active energy on study dates")
            continue
        energy["BinStart"] = floor_to_bin(energy["timestamp"])
        bins = (
            energy.groupby(["Date", "BinStart"], as_index=False)
            .agg(watch_kcal=("kcal", "sum"), n_records=("kcal", "size"))
        )
        bins["Participant"] = pid
        frames.append(bins)
        print(f"[INFO] {pid}: {len(bins):,} active-energy 5-min bins", flush=True)
    columns = ["Participant", "Date", "BinStart", "watch_kcal", "n_records"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def mocopi_5min_bins(minutes: pd.DataFrame) -> pd.DataFrame:
    work = minutes.copy()
    work["Epoch_1Min"] = pd.to_datetime(work["Epoch_1Min"], errors="coerce")
    work["BinStart"] = floor_to_bin(work["Epoch_1Min"])
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    agg = {m["column"]: (m["column"], "mean") for m in MOCOPI_METRICS}
    agg["n_mocopi_min"] = ("Intensity", "size")
    return work.groupby(["Participant", "Date", "BinStart"], as_index=False).agg(
        **{name: pd.NamedAgg(column=c, aggfunc=f) for name, (c, f) in agg.items()}
    )


def filter_study_dates(minutes: pd.DataFrame) -> pd.DataFrame:
    work = minutes.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    keep = []
    for pid, day in zip(work["Participant"].astype(str), work["Date"]):
        key = f"{int(pid[1:]):02d}"
        keep.append(day in participants_dates.get(key, set()))
    return work.loc[keep].copy()


def align(left: pd.DataFrame, right: pd.DataFrame, on_value_cols: list[str]) -> pd.DataFrame:
    a = left.copy()
    b = right.copy()
    a["Date"] = pd.to_datetime(a["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    b["Date"] = pd.to_datetime(b["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    a["BinStart"] = pd.to_datetime(a["BinStart"], errors="coerce")
    b["BinStart"] = pd.to_datetime(b["BinStart"], errors="coerce")
    merged = a.merge(b, on=["Participant", "Date", "BinStart"], how="inner")
    need = ["watch_kcal"] + on_value_cols
    return merged.dropna(subset=[c for c in need if c in merged.columns]).copy()


def save_scatter(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    y_categorical: bool = False,
) -> dict:
    stats = corr_stats(df[xcol], df[ycol])
    x = df[xcol].to_numpy(float)
    y = df[ycol].to_numpy(float)
    y_plot = y
    if y_categorical:
        y_plot = y + np.random.default_rng(42).uniform(-0.12, 0.12, size=len(y))

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    ax.scatter(x, y_plot, s=12, alpha=0.3, color="#4C78A8", edgecolors="none", zorder=3)
    if len(df) >= 2 and np.nanstd(x) > 0:
        coef = np.polyfit(x, y, 1)
        xline = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xline, np.polyval(coef, xline), color="0.15", linewidth=1.3, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_categorical:
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks(OURA_CLASS_TICKS)
        ax.set_yticklabels(OURA_CLASS_LABELS)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02,
        0.98,
        _stats_text(stats),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "0.85",
            "boxstyle": "round,pad=0.3",
        },
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}")
    return stats


def save_watch_panel(
    watch_oura: pd.DataFrame,
    watch_mocopi: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    # Watch vs Oura
    stats = corr_stats(watch_oura["watch_kcal"], watch_oura["oura_class"])
    y = watch_oura["oura_class"].to_numpy(float)
    axes[0].scatter(
        watch_oura["watch_kcal"],
        y + np.random.default_rng(1).uniform(-0.12, 0.12, size=len(y)),
        s=10,
        alpha=0.3,
        color="#E37400",
        edgecolors="none",
    )
    if len(watch_oura) >= 2:
        coef = np.polyfit(
            watch_oura["watch_kcal"].to_numpy(float),
            watch_oura["oura_class"].to_numpy(float),
            1,
        )
        xline = np.linspace(
            float(watch_oura["watch_kcal"].min()),
            float(watch_oura["watch_kcal"].max()),
            50,
        )
        axes[0].plot(xline, np.polyval(coef, xline), color="0.15", linewidth=1.1)
    axes[0].set_xlabel("Apple Watch active energy (kcal / 5 min)")
    axes[0].set_ylabel("Oura activity class")
    axes[0].set_ylim(0.5, 5.5)
    axes[0].set_yticks(OURA_CLASS_TICKS)
    axes[0].set_yticklabels(OURA_CLASS_LABELS)
    axes[0].set_title("vs Oura")
    axes[0].text(0.04, 0.96, _stats_text(stats), transform=axes[0].transAxes, va="top", fontsize=7.5)

    for ax, metric in zip(axes[1:], MOCOPI_METRICS):
        stats = corr_stats(watch_mocopi["watch_kcal"], watch_mocopi[metric["column"]])
        ax.scatter(
            watch_mocopi["watch_kcal"],
            watch_mocopi[metric["column"]],
            s=10,
            alpha=0.3,
            color="#4C78A8",
            edgecolors="none",
        )
        if len(watch_mocopi) >= 2:
            coef = np.polyfit(
                watch_mocopi["watch_kcal"].to_numpy(float),
                watch_mocopi[metric["column"]].to_numpy(float),
                1,
            )
            xline = np.linspace(
                float(watch_mocopi["watch_kcal"].min()),
                float(watch_mocopi["watch_kcal"].max()),
                50,
            )
            ax.plot(xline, np.polyval(coef, xline), color="0.15", linewidth=1.1)
        ax.set_xlabel("Apple Watch active energy (kcal / 5 min)")
        ax.set_ylabel(f"MOCOPI {metric['short']}")
        ax.set_title(f"vs {metric['short']}")
        ax.text(0.04, 0.96, _stats_text(stats), transform=ax.transAxes, va="top", fontsize=7.5)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Apple Watch active energy vs Oura activity class and MOCOPI wrist (5-min bins)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate Apple Watch active energy with Oura and MOCOPI."
    )
    parser.add_argument("--epoch-dir", type=Path, default=None)
    parser.add_argument(
        "--apple-root",
        type=Path,
        default=Path.home() / "Downloads" / "Apple Watch Export CSVs",
    )
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
    parser.add_argument("--sensor", default=MOCOPI_SENSOR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    mocopi_mod = load_module(
        "mocopi_plot_energy",
        REPO_ROOT / "Accelerometer-Patterns" / "plot_by_class_and_time.py",
    )
    oura_mod = load_module(
        "oura_plot_energy",
        REPO_ROOT / "Oura-Activity" / "plot_by_class_and_time.py",
    )

    watch = load_watch_active_energy(args.apple_root.expanduser().resolve())
    oura = oura_mod.load_oura_class_5min(
        args.ring_root.expanduser().resolve(),
        args.schedule_root.expanduser().resolve(),
        allow_missing_schedule=True,
    )
    epoch_dir = mocopi_mod.resolve_epoch_dir(args.epoch_dir)
    epochs = mocopi_mod.load_epoch_kinematics(epoch_dir)
    mocopi_minutes = filter_study_dates(
        mocopi_mod.bodywide_intensity(epochs, args.sensor)
    )
    mocopi = mocopi_5min_bins(mocopi_minutes)

    if watch.empty:
        print("[Fatal Error] No Apple Watch active-energy bins.")
        sys.exit(1)

    watch_oura = align(watch, oura, ["oura_class"])
    watch_mocopi = align(watch, mocopi, [m["column"] for m in MOCOPI_METRICS])
    if watch_oura.empty and watch_mocopi.empty:
        print("[Fatal Error] No overlapping bins with Oura or MOCOPI.")
        sys.exit(1)

    watch_oura.to_csv(out / "paired_watch_oura_5min.csv", index=False)
    watch_mocopi.to_csv(out / "paired_watch_mocopi_5min.csv", index=False)

    summary = []
    if not watch_oura.empty:
        stats = save_scatter(
            watch_oura,
            "watch_kcal",
            "oura_class",
            "Apple Watch active energy (kcal / 5 min)",
            "Oura activity class",
            "Apple Watch active energy vs Oura activity class",
            out / "scatter_watch_vs_oura.png",
            y_categorical=True,
        )
        summary.append(
            {
                "pair": "watch_vs_oura_class",
                "level": "paired_5min_bins",
                **stats,
                "n_participants": int(watch_oura["Participant"].nunique()),
            }
        )
        print(f"[INFO] Watch↔Oura ρ={stats['spearman_rho']:.3f} (n={stats['n']})")

    for metric in MOCOPI_METRICS:
        if watch_mocopi.empty or metric["column"] not in watch_mocopi.columns:
            continue
        stats = save_scatter(
            watch_mocopi,
            "watch_kcal",
            metric["column"],
            "Apple Watch active energy (kcal / 5 min)",
            f"MOCOPI wrist {metric['short']} (5-min mean)",
            f"Apple Watch active energy vs MOCOPI {metric['short']}",
            out / f"scatter_watch_vs_mocopi_{metric['key']}.png",
        )
        summary.append(
            {
                "pair": f"watch_vs_mocopi_{metric['key']}",
                "level": "paired_5min_bins",
                **stats,
                "n_participants": int(watch_mocopi["Participant"].nunique()),
            }
        )
        print(
            f"[INFO] Watch↔MOCOPI {metric['short']} ρ={stats['spearman_rho']:.3f} "
            f"(n={stats['n']})"
        )

    if not watch_oura.empty and not watch_mocopi.empty:
        save_watch_panel(watch_oura, watch_mocopi, out / "figure_watch_energy_correlations.png")

    pd.DataFrame(summary).to_csv(out / "watch_energy_correlation_summary.csv", index=False)
    print(f"\n[DONE] Outputs in {out}")


if __name__ == "__main__":
    main()
