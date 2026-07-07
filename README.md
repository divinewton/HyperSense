# HyperSense

This repository contains the data calculation scripts and heatmap generators used for the HyperSense research project.

## Input Data

All scripts expect participant exports in `~/Downloads/Exports`, with one folder per participant named like `P001`, `P002`, and so on.

Raw Apple Health export files should also be present at the exports root, such as `P001export.csv`, along with the schedule CSVs used to label classroom periods.

For the heatmap scripts, the expected participant data layout is the Apple Health / HealthApp export structure, such as `P001/HealthApp/Labeled/Record/**/*.csv` or the raw export file `P001export.csv`.

The heatmap scripts default to `US/Pacific` for naive timestamps, but you can override that with `--timezone` if your export files were generated in a different local timezone.

## Output Layout

Heatmap outputs are written under `Heatmaps/Graphs/<Datatype>/`, and each datatype folder contains all figures and CSVs for that datatype.

The coverage stratification script prints Table 6 values directly to the console instead of writing files.

## Heatmap Scripts

- `Heatmaps/apple_watch_hr_heatmaps.py` generates heart-rate coverage and valid scheduled-bin heatmaps.
- `Heatmaps/apple_watch_active_energy_heatmaps.py` generates active energy burned coverage and valid scheduled-bin heatmaps.
- `Heatmaps/apple_watch_basal_energy_heatmaps.py` generates basal energy burned coverage and valid scheduled-bin heatmaps.
- `Heatmaps/apple_watch_exercise_time_heatmaps.py` generates Apple exercise time coverage and valid scheduled-bin heatmaps.

Each of those scripts uses the same fixed weekday, class, and 30-minute time windows, the same participant naming conventions, and a valid-bin count summary by weekday.

## Boxplot Scripts

- `BoxPlots/apple_watch_activity_boxplots.py` generates one horizontal boxplot per Apple Watch datatype, saving the PNGs directly in `BoxPlots/Graphs/`.
- `BoxPlots/apple_watch_hr_participant_small_multiples.py` generates a Figure 11-style heart-rate small-multiples grid: one panel per classroom activity with participant-level boxplots, saving to `BoxPlots/Graphs/heart_rate_participant_small_multiples.png`.

## Calculation Scripts

- `Coverage/calculate_heart_rate.py` calculates heart-rate expected, observed, valid, and invalid 5-minute bins using point-sample placement.
- `Coverage/calculate_active_energy_burned.py` calculates active energy burned bins by mapping datapoint intervals onto schedule bins.
- `Coverage/calculate_basal_energy_burned.py` calculates basal energy burned bins by mapping datapoint intervals onto schedule bins.
- `Coverage/calculate_apple_exercise_time.py` calculates Apple Exercise Time bins using discrete event placement.
- `Coverage/audit_binned_common.py` provides shared helpers used by the calculation scripts.

## Coverage Stratification Script

- `Coverage/calculate_coverage_stratification.py` calculates Paper Table 6 Apple Watch coverage values stratified by classroom context, day of week, and time of day.
- `Coverage/coverage_stratification_common.py` provides the shared schedule-binning and coverage logic used by the stratification script.

The stratification script reports, for each row category:

- Expected 5-minute bins from the school schedule
- Apple Watch HR coverage (40–180 bpm)
- Active Energy coverage
- BMR coverage
- Logged Exercise coverage

It also prints participant summary rows with mean ± SD and min–max across the 12 participants. Stratifier rows pool valid and expected bins across all participants; the participant summary rows average each participant's overall coverage percentage.

## How To Run

Run the heatmap scripts from the repository root:

```bash
python3 Heatmaps/apple_watch_hr_heatmaps.py --root ~/Downloads/Exports
python3 Heatmaps/apple_watch_active_energy_heatmaps.py --root ~/Downloads/Exports
python3 Heatmaps/apple_watch_basal_energy_heatmaps.py --root ~/Downloads/Exports
python3 Heatmaps/apple_watch_exercise_time_heatmaps.py --root ~/Downloads/Exports
```

Run the boxplot scripts from the repository root:

```bash
python3 BoxPlots/apple_watch_activity_boxplots.py --root ~/Downloads/Exports
python3 BoxPlots/apple_watch_hr_participant_small_multiples.py --root ~/Downloads/Exports
```

Run the calculation scripts from the repository root:

```bash
PYTHONPATH=. python3 Coverage/calculate_heart_rate.py
PYTHONPATH=. python3 Coverage/calculate_active_energy_burned.py
PYTHONPATH=. python3 Coverage/calculate_basal_energy_burned.py
PYTHONPATH=. python3 Coverage/calculate_apple_exercise_time.py
```

Run the Table 6 coverage stratification script from the repository root:

```bash
python3 Coverage/calculate_coverage_stratification.py --root ~/Downloads/Exports
```
