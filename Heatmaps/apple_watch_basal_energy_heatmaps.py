#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from heatmap_shared import ensure_dir, run_heatmap_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Apple Watch basal energy heatmaps.")
    parser.add_argument(
        "--root",
        default=os.path.expanduser("~/Downloads/Exports"),
        help="Root folder containing participant folders or raw export files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots and summary CSVs. Defaults to Heatmaps/Graphs/BasalEnergyBurned.",
    )
    parser.add_argument(
        "--timezone",
        default="US/Pacific",
        help="Timezone for naive timestamps. Use the export's local timezone if it is not Pacific.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(__file__).resolve().parent / "Graphs" / "BasalEnergyBurned"
    ensure_dir(out_dir)
    run_heatmap_suite(
        root=root,
        output_dir=out_dir,
        type_token="BasalEnergyBurned",
        metric_label="Basal Energy Burned",
        metric_folder="basal_energy_burned",
        valid_min=0,
        local_tz=args.timezone,
    )


if __name__ == "__main__":
    main()
