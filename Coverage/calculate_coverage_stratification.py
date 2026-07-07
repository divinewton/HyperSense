#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Coverage.coverage_stratification_common import (  # noqa: E402
    DISPLAY_METRIC_COLUMNS,
    EXPORTS_DIR,
    build_coverage_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Apple Watch coverage stratified by classroom, weekday, and time of day."
    )
    parser.add_argument(
        "--root",
        default=EXPORTS_DIR,
        help="Root folder containing participant export CSVs and schedule files.",
    )
    return parser.parse_args()


def print_plain_results(detail_df: pd.DataFrame, meta: dict) -> None:
    # Print Table 6 stratifier rows as plain label/value lines.
    for _, row in detail_df.iterrows():
        label = f"{row['section']} {row['stratifier']}"
        print(f"{label} Expected {int(row['expected_bins'])}")
        for metric_key, metric_label in DISPLAY_METRIC_COLUMNS:
            value = row[metric_key]
            if pd.isna(value):
                continue
            print(f"{label} {metric_label} {value}")

    # Print participant summary mean±SD and min/max across the 12 participants.
    participant_percentages = meta.get("participant_percentages", {})
    for metric_key, metric_label in DISPLAY_METRIC_COLUMNS:
        values = [float(v) for v in participant_percentages.get(metric_key, []) if not pd.isna(v)]
        if not values:
            continue
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        print(f"Participant Summary Mean {metric_label} {mean} {sd}")
        print(f"Participant Summary Range {metric_label} {min(values)} {max(values)}")


def main() -> None:
    args = parse_args()
    root = os.path.expanduser(args.root)

    detail_df, _, meta = build_coverage_tables(root=root)
    if meta["processed_participants"] == 0:
        print(f"No participant exports found under {root}")
        sys.exit(1)

    print_plain_results(detail_df, meta)


if __name__ == "__main__":
    main()
