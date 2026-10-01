#!/usr/bin/env python3
"""Unit-free MOCOPI vs Oura comparison by classroom activity.

Absolute Acc Mag and MET are different physical quantities. This script compares
*relative* activity instead:

  relative = (mean in that class) / (that student's overall school-day mean)

So 1.0 = typical for that student, >1 = more active than their usual, <1 = quieter.
Both devices land on the same scale and can be plotted together.
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
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parent.parent
MOCOPI_COLOR = "#4C78A8"
OURA_COLOR = "#E37400"
PAPER_CLASS_ORDER = [
    "Homeroom",
    "Math",
    "ELA",
    "ELA/History",
    "History",
    "Social Skills",
    "Cash-out",
    "HW Rein./Study Hall",
    "Friday Funday",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def ordered_class_labels(labels: list[str]) -> list[str]:
    present = set(labels)
    ordered = [c for c in PAPER_CLASS_ORDER if c in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def relative_by_class(minutes: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """One relative value per participant x class (vs that participant's overall mean)."""
    person_overall = (
        minutes.groupby("Participant", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "overall"})
    )
    person_class = (
        minutes.groupby(["Participant", "class_display"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "class_mean"})
    )
    merged = person_class.merge(person_overall, on="Participant", how="inner")
    merged = merged[merged["overall"] > 0].copy()
    merged["relative"] = merged["class_mean"] / merged["overall"]
    return merged


def cohort_relative(person_rel: pd.DataFrame, class_order: list[str], min_n: int = 1) -> pd.DataFrame:
    rows = []
    for level in class_order:
        values = person_rel.loc[person_rel["class_display"].eq(level), "relative"]
        if len(values) < min_n:
            continue
        rows.append(
            {
                "class_display": level,
                "mean": float(values.mean()),
                "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def save_grouped_relative(
    mocopi: pd.DataFrame,
    oura: pd.DataFrame,
    path: Path,
    min_n: int,
) -> None:
    classes = ordered_class_labels(
        sorted(set(mocopi["class_display"]) | set(oura["class_display"]))
    )
    # Keep classes with enough students on BOTH devices for a fair read.
    keep = []
    for c in classes:
        nm = int(mocopi.loc[mocopi["class_display"].eq(c), "n"].iloc[0]) if c in set(mocopi["class_display"]) else 0
        no = int(oura.loc[oura["class_display"].eq(c), "n"].iloc[0]) if c in set(oura["class_display"]) else 0
        if nm >= min_n and no >= min_n:
            keep.append(c)
    if not keep:
        keep = classes

    mocopi_map = mocopi.set_index("class_display")
    oura_map = oura.set_index("class_display")
    x = np.arange(len(keep))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(9.0, 0.85 * len(keep) + 3), 5.2))
    m_means = [float(mocopi_map.loc[c, "mean"]) if c in mocopi_map.index else np.nan for c in keep]
    m_ses = [float(mocopi_map.loc[c, "se"]) if c in mocopi_map.index else 0.0 for c in keep]
    o_means = [float(oura_map.loc[c, "mean"]) if c in oura_map.index else np.nan for c in keep]
    o_ses = [float(oura_map.loc[c, "se"]) if c in oura_map.index else 0.0 for c in keep]
    m_ns = [int(mocopi_map.loc[c, "n"]) if c in mocopi_map.index else 0 for c in keep]
    o_ns = [int(oura_map.loc[c, "n"]) if c in oura_map.index else 0 for c in keep]

    ax.bar(
        x - width / 2,
        m_means,
        width=width,
        yerr=m_ses,
        color=MOCOPI_COLOR,
        alpha=0.9,
        capsize=3,
        label="MOCOPI",
        error_kw={"elinewidth": 1.1},
    )
    ax.bar(
        x + width / 2,
        o_means,
        width=width,
        yerr=o_ses,
        color=OURA_COLOR,
        alpha=0.9,
        capsize=3,
        label="Oura",
        error_kw={"elinewidth": 1.1},
    )
    ax.axhline(1.0, color="0.35", linewidth=1.0, linestyle="--", label="Student's typical level")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(n={a}/{b})" for c, a, b in zip(keep, m_ns, o_ns)],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Relative activity (class ÷ student's overall mean)")
    ax.set_title("MOCOPI vs Oura by class (same scale)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MOCOPI and Oura using relative (unit-free) activity by class."
    )
    parser.add_argument("--epoch-dir", type=Path, default=None)
    parser.add_argument("--sensor", default=None)
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
        "--min-n",
        type=int,
        default=3,
        help="Minimum students per device to include a class in the grouped plot (default: 3).",
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

    mocopi_rel = relative_by_class(mocopi_minutes, "Intensity")
    oura_rel = relative_by_class(oura_minutes, "MET")

    class_order = ordered_class_labels(
        sorted(set(mocopi_rel["class_display"]) | set(oura_rel["class_display"]))
    )
    mocopi_cohort = cohort_relative(mocopi_rel, class_order)
    oura_cohort = cohort_relative(oura_rel, class_order)

    mocopi_cohort.to_csv(out / "relative_by_class_mocopi.csv", index=False)
    oura_cohort.to_csv(out / "relative_by_class_oura.csv", index=False)

    # Rank agreement on shared classes with enough students.
    shared = sorted(
        set(mocopi_cohort.loc[mocopi_cohort["n"] >= args.min_n, "class_display"])
        & set(oura_cohort.loc[oura_cohort["n"] >= args.min_n, "class_display"])
    )
    if len(shared) >= 3:
        m = mocopi_cohort.set_index("class_display").loc[shared, "mean"]
        o = oura_cohort.set_index("class_display").loc[shared, "mean"]
        rho, p = spearmanr(m, o)
        pd.DataFrame(
            [{"n_classes": len(shared), "spearman_rho": rho, "p_value": p, "classes": ", ".join(shared)}]
        ).to_csv(out / "relative_rank_agreement.csv", index=False)
        print(
            f"[INFO] Spearman rank agreement on {len(shared)} classes "
            f"(n≥{args.min_n} each): ρ={rho:.3f}, p={p:.3f}"
        )

    save_grouped_relative(
        mocopi_cohort,
        oura_cohort,
        out / "compare_by_class_relative.png",
        min_n=args.min_n,
    )
    print(f"\n[DONE] Outputs in {out}")


if __name__ == "__main__":
    main()
