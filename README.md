# HyperSense

This repository contains the data calculation scripts and heatmap generators used for the HyperSense research project.

## Input Data

All scripts expect participant exports in `~/Downloads/Exports`, with one folder per participant named like `P01`, `P02`, and so on, plus any matching Apple Health schedule CSVs in `~/Downloads/Exports/Schedules` or directly in `~/Downloads/Exports`.

For the heatmap scripts, the expected participant data layout is the Apple Health / HealthApp export structure, such as `P01/HealthApp/Labeled/Record/**/*.csv` or the raw export file `P01export.csv`.

The heatmap scripts default to `US/Pacific` for naive timestamps, but you can override that with `--timezone` if your export files were generated in a different local timezone.

## Output Layout

Heatmap outputs are written under `Heatmaps/Graphs/<Datatype>/`, and each datatype folder contains two subfolders:

- `Coverage/` contains the coverage figures and CSVs that show how much data was collected in each weekday, class, and time-of-day bucket using schedule-aware bin coverage.
- `Heatmaps/` contains the actual-value figures and CSVs that show mean heart rate for the heart-rate script and total active energy, basal energy, or exercise time for the other scripts in the same weekday, class, and time-of-day buckets.

## Heatmap Scripts

- `Heatmaps/apple_watch_hr_heatmaps.py` generates heart-rate coverage and value heatmaps.
- `Heatmaps/apple_watch_active_energy_heatmaps.py` generates active energy burned coverage and value heatmaps.
- `Heatmaps/apple_watch_basal_energy_heatmaps.py` generates basal energy burned coverage and value heatmaps.
- `Heatmaps/apple_watch_exercise_time_heatmaps.py` generates Apple exercise time coverage and value heatmaps.

Each of those scripts uses the same schedule-aware binning rules, the same weekday and 30-minute time windows, and the same participant naming conventions, but the value heatmaps use mean aggregation for heart rate and sum aggregation for the energy and exercise metrics.

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
