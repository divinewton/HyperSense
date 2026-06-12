# HyperSense

Data calculation scripts for the HyperSense research project.

## What’s Included

All scripts read participant exports from `~/Downloads/Exports` and use the same participant/date schedule data and 5-minute binning logic.

- `calculate_heart_rate.py` - Calculates heart-rate expected, observed, and valid 5-minute bins for tracked days.
- `calculate_active_energy_burned.py` - Calculates active energy burned bins using the same schedule-based filtering.
- `calculate_basal_energy_burned.py` - Calculates basal energy burned bins using the same schedule-based filtering.
- `calculate_apple_exercise_time.py` - Calculates Apple Exercise Time bins using the same schedule-based filtering.
- `audit_binned_common.py` - Shared helper used by the calculation scripts.

## Data Layout

Place the exported Health data and schedule CSVs in:

`~/Downloads/Exports`

## How To Run

Run any script directly with Python from the repository root:

```bash
python3 calculate_heart_rate.py
python3 calculate_active_energy_burned.py
python3 calculate_basal_energy_burned.py
python3 calculate_apple_exercise_time.py
```
