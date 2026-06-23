# HyperSense

This repository contains the data calculation scripts and heatmap generators used for the HyperSense research project.

## Input Data

All scripts expect participant exports in `~/Downloads/Exports`, with one folder per participant named like `P01`, `P02`, and so on.

For the heatmap scripts, the expected participant data layout is the Apple Health / HealthApp export structure, such as `P01/HealthApp/Labeled/Record/**/*.csv` or the raw export file `P01export.csv`.

The heatmap scripts default to `US/Pacific` for naive timestamps, but you can override that with `--timezone` if your export files were generated in a different local timezone.

## Output Layout

Heatmap outputs are written under `Heatmaps/Graphs/<Datatype>/`, and each datatype folder contains all figures and CSVs for that datatype.

## Heatmap Scripts

- `Heatmaps/apple_watch_hr_heatmaps.py` generates heart-rate coverage and valid scheduled-bin heatmaps.
- `Heatmaps/apple_watch_active_energy_heatmaps.py` generates active energy burned coverage and valid scheduled-bin heatmaps.
- `Heatmaps/apple_watch_basal_energy_heatmaps.py` generates basal energy burned coverage and valid scheduled-bin heatmaps.
- `Heatmaps/apple_watch_exercise_time_heatmaps.py` generates Apple exercise time coverage and valid scheduled-bin heatmaps.

Each of those scripts uses the same fixed weekday, class, and 30-minute time windows, the same participant naming conventions, and a valid-bin count summary by weekday.

## Calculation Scripts

- `calculate_heart_rate.py` calculates heart-rate expected, observed, valid, and invalid 5-minute bins using point-sample placement.
- `calculate_active_energy_burned.py` calculates active energy burned bins by mapping datapoint intervals onto schedule bins.
- `calculate_basal_energy_burned.py` calculates basal energy burned bins by mapping datapoint intervals onto schedule bins.
- `calculate_apple_exercise_time.py` calculates Apple Exercise Time bins using discrete event placement.
- `audit_binned_common.py` provides shared helpers used by the calculation scripts.

## How To Run

Run the heatmap scripts from the repository root:

```bash
python3 Heatmaps/apple_watch_hr_heatmaps.py --root ~/Downloads/Exports
python3 Heatmaps/apple_watch_active_energy_heatmaps.py --root ~/Downloads/Exports
python3 Heatmaps/apple_watch_basal_energy_heatmaps.py --root ~/Downloads/Exports
python3 Heatmaps/apple_watch_exercise_time_heatmaps.py --root ~/Downloads/Exports
```

Run the calculation scripts from the repository root:

```bash
python3 calculate_heart_rate.py
python3 calculate_active_energy_burned.py
python3 calculate_basal_energy_burned.py
python3 calculate_apple_exercise_time.py
```
