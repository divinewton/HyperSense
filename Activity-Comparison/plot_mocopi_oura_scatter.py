#!/usr/bin/env python3
"""MOCOPI (right wrist) vs Oura ring activity-class correlation.

MOCOPI minutes are averaged into 5-minute bins and paired with Oura's
class_5_min (already one value per 5 minutes). Primary figures use every
paired bin — not participant means.

Oura y-axis is the ordinal activity class 1–5 with official labels:
  1 Rest, 2 Inactive, 3 Low activity, 4 Medium activity, 5 High activity
Non-wear (0) is dropped. Study weeks follow the observation sheet.

P007 is rebuilt from raw serials 0EA70 and 14A53 (side unknown); the copied
epoch file is not used.

Three MOCOPI metrics vs Oura class:
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
Y_JITTER = 0.12
RNG = np.random.default_rng(42)

OURA_CLASS_TICKS = (1, 2, 3, 4, 5)
# Full Oura wording for standalone plots; short form for tight multi-panel figures.
OURA_CLASS_LABELS = (
    "Rest",
    "Inactive",
    "Low activity",
    "Medium activity",
    "High activity",
)
OURA_CLASS_LABELS_SHORT = (
    "Rest",
    "Inactive",
    "Low",
    "Medium",
    "High",
)

METRICS = (
    {
        "key": "magnitude",
        "column": "Intensity",
        "short": "Magnitude",
        "xlabel": "MOCOPI wrist magnitude (5-min mean)",
        "title": "5-min bins: magnitude vs Oura activity class",
    },
    {
        "key": "acceleration",
        "column": "Variability",
        "short": "Acceleration variability",
        "xlabel": "MOCOPI wrist acceleration variability (5-min mean)",
        "title": "5-min bins: acceleration variability vs Oura activity class",
    },
    {
        "key": "jerk",
        "column": "Jerk",
        "short": "Jerk",
        "xlabel": "MOCOPI wrist jerk (5-min mean)",
        "title": "5-min bins: jerk vs Oura activity class",
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
    work["Epoch_1Min"] = pd.to_datetime(work["Epoch_1Min"], errors="coerce")
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


def align_bins(mocopi: pd.DataFrame, oura: pd.DataFrame) -> pd.DataFrame:
    left = mocopi.copy()
    right = oura.copy()
    left["Date"] = pd.to_datetime(left["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    right["Date"] = pd.to_datetime(right["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    left["BinStart"] = pd.to_datetime(left["BinStart"], errors="coerce")
    right["BinStart"] = pd.to_datetime(right["BinStart"], errors="coerce")
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
    keep = ["Participant", "Date", "BinStart", "oura_class", "class_display"] + [
        m["column"] for m in METRICS
    ]
    keep = [c for c in keep if c in merged.columns]
    return (
        merged.dropna(subset=["oura_class"] + [m["column"] for m in METRICS])
        .loc[:, keep]
        .copy()
    )


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


def filter_study_dates(minutes: pd.DataFrame) -> pd.DataFrame:
    """Keep minutes whose calendar date is on that participant's observation week."""
    from audit_binned_common import participants_dates

    work = minutes.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    keep = []
    for pid, day in zip(work["Participant"].astype(str), work["Date"]):
        key = f"{int(pid[1:]):02d}"
        keep.append(day in participants_dates.get(key, set()))
    return work.loc[keep].copy()


def _stats_text(stats: dict) -> str:
    if stats["n"] < 3 or np.isnan(stats["spearman_rho"]):
        return f"n={stats['n']}"
    return (
        f"n={stats['n']}\n"
        f"Spearman ρ={stats['spearman_rho']:.2f} (p={stats['spearman_p']:.3g})\n"
        f"Pearson r={stats['pearson_r']:.2f} (p={stats['pearson_p']:.3g})"
    )


def _panel_stats(stats: dict) -> str:
    if stats["n"] < 3 or np.isnan(stats["spearman_rho"]):
        return f"n = {stats['n']}"
    return f"ρ = {stats['spearman_rho']:.2f}\np = {stats['spearman_p']:.3f}"


def _style_oura_yaxis(ax, short: bool = False) -> None:
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks(OURA_CLASS_TICKS)
    ax.set_yticklabels(OURA_CLASS_LABELS_SHORT if short else OURA_CLASS_LABELS)
    ax.set_ylabel("Oura activity class")


def _jitter_y(y: np.ndarray, seed_offset: int = 0) -> np.ndarray:
    local = np.random.default_rng(int(RNG.integers(0, 1_000_000)) + seed_offset)
    return y + local.uniform(-Y_JITTER, Y_JITTER, size=len(y))


def save_bin_scatter(
    paired: pd.DataFrame,
    xcol: str,
    xlabel: str,
    title: str,
    path: Path,
) -> dict:
    stats = corr_stats(paired[xcol], paired["oura_class"])
    x = paired[xcol].to_numpy(float)
    y = paired["oura_class"].to_numpy(float)
    y_plot = _jitter_y(y)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(
        x,
        y_plot,
        s=14,
        alpha=0.35,
        color="#4C78A8",
        edgecolors="none",
        zorder=3,
    )
    if len(paired) >= 2 and np.nanstd(x) > 0:
        coef = np.polyfit(x, y, 1)
        xline = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xline, np.polyval(coef, xline), color="0.15", linewidth=1.4, zorder=2)

    ax.set_xlabel(xlabel)
    _style_oura_yaxis(ax)
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


def save_three_panel_bins(paired: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    for i, (ax, metric) in enumerate(zip(axes, METRICS)):
        xcol = metric["column"]
        stats = corr_stats(paired[xcol], paired["oura_class"])
        x = paired[xcol].to_numpy(float)
        y = paired["oura_class"].to_numpy(float)
        ax.scatter(
            x,
            _jitter_y(y, seed_offset=i * 17),
            s=12,
            alpha=0.3,
            color="#4C78A8",
            edgecolors="none",
            zorder=3,
        )
        if len(paired) >= 2 and np.nanstd(x) > 0:
            coef = np.polyfit(x, y, 1)
            xline = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(xline, np.polyval(coef, xline), color="0.15", linewidth=1.2, zorder=2)
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
    _style_oura_yaxis(axes[0])
    for ax in axes[1:]:
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks(OURA_CLASS_TICKS)
    fig.suptitle(
        "MOCOPI wrist vs Oura activity class (all paired 5-min bins)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def save_paper_figures(paired: pd.DataFrame, within: pd.DataFrame, out: Path) -> None:
    """Double-column figures: all-bin association, and within-participant ρ."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    xlabels = {
        "Intensity": "Wrist acceleration magnitude",
        "Variability": "Wrist acceleration variability",
        "Jerk": "Wrist jerk",
    }
    letters = "ABC"

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharey=True)
    for i, (ax, metric, letter) in enumerate(zip(axes, METRICS, letters)):
        xcol = metric["column"]
        stats = corr_stats(paired[xcol], paired["oura_class"])
        x = paired[xcol].to_numpy(float)
        y = paired["oura_class"].to_numpy(float)
        ax.scatter(
            x,
            _jitter_y(y, seed_offset=i * 31),
            s=8,
            alpha=0.25,
            color="#4C78A8",
            edgecolors="none",
            zorder=3,
        )
        if len(paired) >= 2 and np.nanstd(x) > 0:
            coef = np.polyfit(x, y, 1)
            xline = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)
            ax.plot(xline, np.polyval(coef, xline), color="0.15", linewidth=1.0, zorder=2)
        ax.set_xlabel(xlabels[xcol])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.18,
            1.06,
            letter,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )
        ax.text(
            0.04,
            0.96,
            _panel_stats(stats),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            linespacing=1.25,
        )
    _style_oura_yaxis(axes[0], short=True)
    for ax in axes[1:]:
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks(OURA_CLASS_TICKS)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(out / "figure_wrist_vs_oura.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / "figure_wrist_vs_oura.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {out / 'figure_wrist_vs_oura.png'}")

    order = (
        within.loc[within["metric"].eq("magnitude")]
        .sort_values("spearman_rho", ascending=True)["Participant"]
        .tolist()
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True)
    for ax, metric, letter in zip(axes, METRICS, letters):
        part = within.loc[within["metric"].eq(metric["key"])].set_index("Participant").loc[order]
        y = np.arange(len(part))
        rho = part["spearman_rho"].to_numpy(float)
        ax.axvline(0, color="0.75", linewidth=0.6, zorder=1)
        median = float(np.nanmedian(rho))
        ax.axvline(median, color="#E45756", linewidth=0.8, linestyle="--", zorder=1)
        ax.hlines(y, 0, rho, color="#4C78A8", linewidth=1.2, zorder=2)
        ax.scatter(rho, y, s=22, color="#4C78A8", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(part.index, fontsize=7)
        ax.set_xlabel(r"Spearman $\rho$")
        ax.set_xlim(-0.2, 1.05)
        ax.text(
            0.98,
            0.06,
            "dashed = median",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#E45756",
            fontsize=6.5,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.28 if letter == "A" else -0.12,
            1.04,
            letter,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )
        ax.set_title(metric["short"], fontsize=9, pad=8)
    axes[0].set_ylabel("Participant")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "figure_within_participant_rho.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / "figure_within_participant_rho.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {out / 'figure_within_participant_rho.png'}")


def save_p007_sensor_check(
    paired_by_label: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    """All paired bins for each P007 wrist-serial choice (magnitude only)."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    colors = {"0EA70": "#E45756", "14A53": "#F2C14E", "mean": "#54A24B"}
    for label, paired in paired_by_label.items():
        p007 = paired.loc[paired["Participant"].eq("P007")]
        if p007.empty:
            continue
        stats = corr_stats(p007["Intensity"], p007["oura_class"])
        ax.scatter(
            p007["Intensity"],
            _jitter_y(p007["oura_class"].to_numpy(float)),
            s=18,
            alpha=0.45,
            color=colors.get(label, "0.2"),
            edgecolors="none",
            zorder=3,
            label=f"P007 {label} (ρ={stats['spearman_rho']:.2f}, n={stats['n']})",
        )
    ax.set_xlabel("MOCOPI wrist magnitude (5-min mean)")
    _style_oura_yaxis(ax)
    ax.set_title("P007 wrist serials (side unknown)")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def within_person_correlations(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, group in paired.groupby("Participant"):
        for metric in METRICS:
            stats = corr_stats(group[metric["column"]], group["oura_class"])
            rows.append(
                {
                    "Participant": pid,
                    "metric": metric["key"],
                    "n_bins": stats["n"],
                    "spearman_rho": stats["spearman_rho"],
                    "pearson_r": stats["pearson_r"],
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter MOCOPI right-wrist magnitude / acceleration variability / "
            "jerk vs Oura activity class (all paired 5-min bins)."
        )
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

    from build_p007_wrist import build_p007_wrist

    epoch_dir = mocopi_mod.resolve_epoch_dir(args.epoch_dir)
    raw_p007 = Path.home() / "Downloads" / "P007"
    # load_epoch_kinematics replaces the copied P007 file with rebuilt wrists.
    epochs = mocopi_mod.load_epoch_kinematics(epoch_dir, raw_p007=raw_p007)
    mocopi_minutes = filter_study_dates(mocopi_mod.bodywide_intensity(epochs, args.sensor))
    p007_mean, p007_serials = build_p007_wrist(raw_p007, epoch_dir)
    if p007_mean.empty:
        print("[Fatal Error] P007 wrist rebuild produced no school minutes.")
        sys.exit(1)
    # Cohort without P007, used only for the per-serial diagnostic overlays.
    base_minutes = mocopi_minutes[mocopi_minutes["Participant"].astype(str) != "P007"].copy()
    oura_bins = oura_mod.load_oura_class_5min(
        args.ring_root.expanduser().resolve(),
        args.schedule_root.expanduser().resolve(),
        allow_missing_schedule=True,
    )
    if mocopi_minutes.empty or oura_bins.empty:
        print("[Fatal Error] Missing MOCOPI or Oura class bins.")
        sys.exit(1)

    for metric in METRICS:
        if metric["column"] not in mocopi_minutes.columns:
            print(f"[Fatal Error] Missing MOCOPI column {metric['column']!r}.")
            sys.exit(1)

    paired = align_bins(mocopi_5min_bins(mocopi_minutes), oura_bins)
    if paired.empty:
        print("[Fatal Error] No overlapping 5-minute bins between MOCOPI and Oura.")
        sys.exit(1)

    paired.to_csv(out / "paired_5min_bins.csv", index=False)

    for stale in (
        "scatter_5min_bins.png",
        "scatter_by_participant.png",
        "scatter_5min_magnitude.png",
        "scatter_5min_acceleration.png",
        "scatter_5min_jerk.png",
        "scatter_participant_magnitude.png",
        "scatter_participant_acceleration.png",
        "scatter_participant_jerk.png",
        "scatter_participant_magnitude_accel_jerk.png",
        "participant_means_from_paired_5min.csv",
        "normalization_check.csv",
    ):
        old = out / stale
        if old.exists():
            old.unlink()

    summary_rows = []
    for metric in METRICS:
        bin_stats = save_bin_scatter(
            paired,
            metric["column"],
            metric["xlabel"],
            metric["title"],
            out / f"scatter_bins_{metric['key']}.png",
        )
        summary_rows.append(
            {
                "metric": metric["key"],
                "mocopi_column": metric["column"],
                "level": "paired_5min_bins",
                **bin_stats,
                "n_participants": int(paired["Participant"].nunique()),
                "n_paired_5min_bins": len(paired),
            }
        )
        print(
            f"[INFO] {metric['short']}: bin-level ρ={bin_stats['spearman_rho']:.3f} "
            f"(n={bin_stats['n']} paired 5-min bins)"
        )

    save_three_panel_bins(paired, out / "scatter_bins_magnitude_accel_jerk.png")
    pd.DataFrame(summary_rows).to_csv(out / "correlation_summary.csv", index=False)

    variants: dict[str, pd.DataFrame] = {}
    check_rows = []
    for label, extra in (("mean", p007_mean), *p007_serials.items()):
        minutes = pd.concat(
            [base_minutes, extra.drop(columns=["Sensor"], errors="ignore")],
            ignore_index=True,
        )
        paired_i = align_bins(mocopi_5min_bins(minutes), oura_bins)
        p007 = paired_i.loc[paired_i["Participant"].eq("P007")]
        if p007.empty:
            print(f"[WARN] P007 {label} has no paired bins")
            continue
        variants[label] = paired_i
        for metric in METRICS:
            stats = corr_stats(p007[metric["column"]], p007["oura_class"])
            check_rows.append(
                {
                    "sensor": label,
                    "metric": metric["key"],
                    "n_bins": stats["n"],
                    "spearman_rho": stats["spearman_rho"],
                    "pearson_r": stats["pearson_r"],
                    "mean_Intensity": float(p007["Intensity"].mean()),
                    "median_oura_class": float(p007["oura_class"].median()),
                }
            )
    if variants:
        save_p007_sensor_check(variants, out / "scatter_p007_wrist_serials.png")
        pd.DataFrame(check_rows).to_csv(out / "p007_wrist_serial_check.csv", index=False)

    bin_corr = within_person_correlations(paired)
    bin_corr.to_csv(out / "within_person_bin_correlation.csv", index=False)
    save_paper_figures(paired, bin_corr, out)

    print(
        f"\n[DONE] {paired['Participant'].nunique()} participants, "
        f"{len(paired):,} paired 5-min bins"
    )
    print(f"  Oura class counts:\n{paired['oura_class'].value_counts().sort_index().to_string()}")
    print(f"  Outputs in {out}")


if __name__ == "__main__":
    main()
