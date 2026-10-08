#!/usr/bin/env python3
"""MOCOPI (right wrist) vs Oura ring correlation.

Both devices are averaged into aligned 5-minute bins first (so collection
windows match), then each participant is summarized to one point.

Three participant-level scatters (x = MOCOPI, y = Oura MET):
  1) Magnitude     = mean Acc Mag (Intensity)
  2) Acceleration  = Acc Mag variability (Variability)
  3) Jerk          = mean Jerk Mag
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
BIN_MINUTES = 5
MOCOPI_SENSOR = "WristR"
MIN_MINUTES_PER_BIN = 1

METRICS = (
    {
        "key": "magnitude",
        "column": "Intensity",
        "short": "Magnitude",
        "xlabel": "MOCOPI right-wrist magnitude (mean of paired 5-min bins)",
        "title": "Per-participant: magnitude vs Oura",
    },
    {
        "key": "acceleration",
        "column": "Variability",
        "short": "Acceleration variability",
        "xlabel": "MOCOPI right-wrist acceleration variability (mean of paired 5-min bins)",
        "title": "Per-participant: acceleration variability vs Oura",
    },
    {
        "key": "jerk",
        "column": "Jerk",
        "short": "Jerk",
        "xlabel": "MOCOPI right-wrist jerk (mean of paired 5-min bins)",
        "title": "Per-participant: jerk vs Oura",
    },
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def floor_to_bin(series: pd.Series, minutes: int = BIN_MINUTES) -> pd.Series:
    return series.dt.floor(f"{minutes}min")


def mocopi_5min_bins(minutes: pd.DataFrame) -> pd.DataFrame:
    work = minutes.copy()
    work["BinStart"] = floor_to_bin(work["Epoch_1Min"])
    agg = {
        "n_mocopi_min": ("Intensity", "size"),
        "class_display": (
            "class_display",
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0],
        ),
    }
    for metric in METRICS:
        col = metric["column"]
        if col not in work.columns:
            raise ValueError(f"MOCOPI minutes missing column {col!r}")
        agg[col] = (col, "mean")

    grouped = work.groupby(
        ["Participant", "Date", "BinStart"], as_index=False, observed=True
    ).agg(**{name: pd.NamedAgg(column=c, aggfunc=f) for name, (c, f) in agg.items()})
    return grouped[grouped["n_mocopi_min"] >= MIN_MINUTES_PER_BIN].copy()


def oura_5min_bins(minutes: pd.DataFrame) -> pd.DataFrame:
    work = minutes.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp", "MET"])
    work["BinStart"] = floor_to_bin(work["timestamp"])
    work["Date"] = work["timestamp"].dt.strftime("%Y-%m-%d")
    grouped = (
        work.groupby(["Participant", "Date", "BinStart"], as_index=False, observed=True)
        .agg(
            oura=("MET", "mean"),
            n_oura_min=("MET", "size"),
            class_display=(
                "class_display",
                lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0],
            ),
        )
    )
    return grouped[grouped["n_oura_min"] >= MIN_MINUTES_PER_BIN].copy()


def align_bins(mocopi: pd.DataFrame, oura: pd.DataFrame) -> pd.DataFrame:
    left = mocopi.copy()
    right = oura.copy()
    left["Date"] = pd.to_datetime(left["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    right["Date"] = pd.to_datetime(right["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = left.merge(
        right,
        on=["Participant", "Date", "BinStart"],
        how="inner",
        suffixes=("_mocopi", "_oura"),
    )
    if "class_display_mocopi" in merged.columns:
        merged["class_display"] = merged["class_display_mocopi"].fillna(
            merged.get("class_display_oura")
        )
    keep = ["Participant", "Date", "BinStart", "oura", "class_display"] + [
        m["column"] for m in METRICS
    ]
    keep = [c for c in keep if c in merged.columns]
    return merged.dropna(subset=["oura"] + [m["column"] for m in METRICS]).loc[:, keep].copy()


def corr_stats(x: pd.Series, y: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
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


def participant_means(paired: pd.DataFrame) -> pd.DataFrame:
    agg = {"oura": ("oura", "mean"), "n_bins": ("oura", "size")}
    for metric in METRICS:
        col = metric["column"]
        agg[col] = (col, "mean")
    return (
        paired.groupby("Participant", as_index=False)
        .agg(**{name: pd.NamedAgg(column=c, aggfunc=f) for name, (c, f) in agg.items()})
        .sort_values("Participant")
    )


def _stats_text(stats: dict) -> str:
    if stats["n"] < 3 or np.isnan(stats["spearman_rho"]):
        return f"n={stats['n']}"
    return (
        f"n={stats['n']}\n"
        f"Spearman ρ={stats['spearman_rho']:.2f} (p={stats['spearman_p']:.3g})\n"
        f"Pearson r={stats['pearson_r']:.2f} (p={stats['pearson_p']:.3g})"
    )


def save_participant_scatter(
    person: pd.DataFrame,
    xcol: str,
    xlabel: str,
    title: str,
    path: Path,
) -> dict:
    stats = corr_stats(person[xcol], person["oura"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(
        person[xcol],
        person["oura"],
        s=70,
        color="#4C78A8",
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )
    for row in person.itertuples():
        ax.annotate(
            row.Participant,
            (getattr(row, xcol), row.oura),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color="0.25",
        )

    if len(person) >= 2:
        coef = np.polyfit(person[xcol].to_numpy(), person["oura"].to_numpy(), 1)
        xline = np.linspace(person[xcol].min(), person[xcol].max(), 100)
        ax.plot(xline, np.polyval(coef, xline), color="0.2", linewidth=1.4, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Oura MET (mean of paired 5-min bins)")
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


def save_three_panel_participant(person: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    for ax, metric in zip(axes, METRICS):
        xcol = metric["column"]
        stats = corr_stats(person[xcol], person["oura"])
        ax.scatter(
            person[xcol],
            person["oura"],
            s=55,
            color="#4C78A8",
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        for row in person.itertuples():
            ax.annotate(
                row.Participant,
                (getattr(row, xcol), row.oura),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
                color="0.3",
            )
        if len(person) >= 2:
            coef = np.polyfit(person[xcol].to_numpy(), person["oura"].to_numpy(), 1)
            xline = np.linspace(person[xcol].min(), person[xcol].max(), 100)
            ax.plot(xline, np.polyval(coef, xline), color="0.2", linewidth=1.2, zorder=2)
        ax.set_xlabel(metric["short"])
        ax.set_title(metric["short"])
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
            fontsize=8,
            bbox={
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "0.85",
                "boxstyle": "round,pad=0.25",
            },
        )
    axes[0].set_ylabel("Oura MET (participant mean)")
    fig.suptitle(
        "MOCOPI right wrist vs Oura (per participant, from paired 5-min averages)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scatter MOCOPI right-wrist magnitude / acceleration variability / jerk vs Oura."
    )
    parser.add_argument("--epoch-dir", type=Path, default=None)
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
    parser.add_argument(
        "--sensor",
        default=MOCOPI_SENSOR,
        help="MOCOPI sensor to use (default: WristR).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    mocopi_mod = load_module(
        "mocopi_plot",
        REPO_ROOT / "Accelerometer-Patterns" / "plot_by_class_and_time.py",
    )
    oura_mod = load_module(
        "oura_plot",
        REPO_ROOT / "Oura-Activity" / "plot_by_class_and_time.py",
    )

    epoch_dir = mocopi_mod.resolve_epoch_dir(args.epoch_dir)
    mocopi_minutes = mocopi_mod.bodywide_intensity(
        mocopi_mod.load_epoch_kinematics(epoch_dir), args.sensor
    )
    oura_minutes = oura_mod.load_oura_activity(
        args.ring_root.expanduser().resolve(),
        args.schedule_root.expanduser().resolve(),
    )
    if mocopi_minutes.empty or oura_minutes.empty:
        print("[Fatal Error] Missing MOCOPI or Oura minutes.")
        sys.exit(1)

    for metric in METRICS:
        if metric["column"] not in mocopi_minutes.columns:
            print(f"[Fatal Error] Missing MOCOPI column {metric['column']!r}.")
            sys.exit(1)

    # Align devices on the same 5-minute windows, then collapse to one point per kid.
    paired = align_bins(mocopi_5min_bins(mocopi_minutes), oura_5min_bins(oura_minutes))
    if paired.empty:
        print("[Fatal Error] No overlapping 5-minute bins between MOCOPI and Oura.")
        sys.exit(1)

    person = participant_means(paired)
    paired.to_csv(out / "paired_5min_bins.csv", index=False)
    person.to_csv(out / "participant_means_from_paired_5min.csv", index=False)

    for stale in (
        "scatter_5min_bins.png",
        "scatter_by_participant.png",
        "scatter_5min_magnitude.png",
        "scatter_5min_acceleration.png",
        "scatter_5min_jerk.png",
    ):
        old = out / stale
        if old.exists():
            old.unlink()

    summary_rows = []
    for metric in METRICS:
        person_stats = save_participant_scatter(
            person,
            metric["column"],
            metric["xlabel"],
            metric["title"],
            out / f"scatter_participant_{metric['key']}.png",
        )
        summary_rows.append(
            {
                "metric": metric["key"],
                "mocopi_column": metric["column"],
                "level": "participant_means",
                **person_stats,
                "n_participants": len(person),
                "n_paired_5min_bins": len(paired),
            }
        )
        print(
            f"[INFO] {metric['short']}: participant ρ={person_stats['spearman_rho']:.3f} "
            f"(from {len(paired)} paired 5-min bins)"
        )

    save_three_panel_participant(person, out / "scatter_participant_magnitude_accel_jerk.png")
    pd.DataFrame(summary_rows).to_csv(out / "correlation_summary.csv", index=False)

    print(
        f"\n[DONE] {len(person)} participants "
        f"(averaged from {len(paired):,} paired 5-min bins)"
    )
    print(f"  Outputs in {out}")


if __name__ == "__main__":
    main()
