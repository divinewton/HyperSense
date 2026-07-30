#!/usr/bin/env python3
"""Calculate Apple Watch coverage using 10-minute schedule bins.

Uses the same methodology as the 5-minute scripts in Coverage/calculate_*.py,
which call run_binned_audit() from audit_binned_common.py. Only the bin size
changes from 5 to 10 minutes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Coverage.audit_binned_common import EXPORTS_DIR, compute_binned_audit  # noqa: E402

BIN_SIZE_MINUTES = 10

SMARTWATCH_METRICS = [
    {
        "label": "Smartwatch HR",
        "type_token": "HeartRate",
        "valid_min": 40,
        "valid_max": 180,
        "mode": "point",
    },
    {
        "label": "Smartwatch Active Energy Burned (Calories)",
        "type_token": "ActiveEnergyBurned",
        "valid_min": 0,
        "valid_max": None,
        "mode": "interval",
    },
    {
        "label": "Smartwatch Basal Metabolic Rate (BMR)",
        "type_token": "BasalEnergyBurned",
        "valid_min": 0,
        "valid_max": None,
        "mode": "interval",
    },
    {
        "label": "Smartwatch Logged Exercise Time",
        "type_token": "AppleExerciseTime",
        "valid_min": 0,
        "valid_max": None,
        "mode": "event",
    },
]


def print_compact_summary(results: dict, label: str, bin_size_minutes: int) -> None:
    expected = round(results["avg_expected_bins"])
    observed = round(results["avg_observed_bins"])
    valid = round(results["avg_valid_bins"])
    coverage = results["coverage_percentage"]
    print(
        f"{label} {bin_size_minutes}-min bins "
        f"{expected} {observed} {valid} {coverage:.2f}%"
    )


def print_detailed_summary(results: dict, label: str, bin_size_minutes: int) -> None:
    bin_label = f"{bin_size_minutes}-Min Bins"
    print(f"{label} Expected ({bin_label}): {round(results['avg_expected_bins']):,d} bins")
    print(f"{label} Observed ({bin_label}): {round(results['avg_observed_bins']):,d} bins")
    print(f"{label} Valid ({bin_label}): {round(results['avg_valid_bins']):,d} bins")
    print(
        f"{label} Invalid ({bin_label}): "
        f"{round(max(results['avg_observed_bins'] - results['avg_valid_bins'], 0)):,d} bins"
    )
    print(f"{label} Coverage: {results['coverage_percentage']:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Apple Watch coverage with 10-minute schedule bins for "
            "heart rate, active energy, BMR, and logged exercise."
        )
    )
    parser.add_argument(
        "--root",
        default=EXPORTS_DIR,
        help="Root folder containing participant export CSVs and schedule files.",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print the full expected/observed/valid/invalid breakdown for each metric.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = os.path.expanduser(args.root)

    any_processed = False
    for metric in SMARTWATCH_METRICS:
        results = compute_binned_audit(
            type_token=metric["type_token"],
            valid_min=metric["valid_min"],
            valid_max=metric["valid_max"],
            mode=metric["mode"],
            bin_size_minutes=BIN_SIZE_MINUTES,
            exports_dir=root,
        )
        if results["processed_participants"] == 0:
            continue

        any_processed = True
        if args.detailed:
            print_detailed_summary(results, metric["label"], BIN_SIZE_MINUTES)
            print()
        else:
            print_compact_summary(results, metric["label"], BIN_SIZE_MINUTES)

    if not any_processed:
        print(f"No participant exports found under {root}")
        sys.exit(1)


if __name__ == "__main__":
    main()
