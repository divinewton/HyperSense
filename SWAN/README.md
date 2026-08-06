# SWAN wearable analysis

One script produces the concise paper-ready analysis. It uses the private
crosswalk in `SWAN/inputs/` and writes only the outputs listed below.

```bash
MPLCONFIGDIR=/tmp/hypersense-mpl python3 SWAN/run_analysis.py \
  --apple-root "/Users/divinewton/Downloads/Apple Watch Export CSVs" \
  --swan-workbook "/Users/divinewton/Downloads/Swan (Responses).xlsx" \
  --mocopi-root "/Users/divinewton/Downloads/epoch_kinematics"
```

Outputs: seven focused figures: five analytic figures (Apple Watch heart rate,
Apple Watch steps, head sensor, rater agreement, and a sensor-by-sensor
acceleration profile) plus two appendix figures (data coverage and classroom
context). It also writes
participant coverage, primary correlations, head-sensor replication, two
supplement tables, and the participant-level analysis dataset.
All analyses are exploratory and participant-level (n=12).

To additionally generate six optional ways of visualizing the focused
head-sensor result (rank plots, an effect forest plot, leave-one-out
sensitivity, class-colored plots, a head-only heatmap, and an all-sensor
MOCOPI heatmap), run:

```bash
MPLCONFIGDIR=/tmp/hypersense-mpl python3 SWAN/run_analysis.py --extra-figures-only \
  --apple-root "/Users/divinewton/Downloads/Apple Watch Export CSVs" \
  --swan-workbook "/Users/divinewton/Downloads/Swan (Responses).xlsx" \
  --mocopi-root "/Users/divinewton/Downloads/epoch_kinematics"
```

`--extra-figures-only` skips the slow Apple Watch import and writes only these
optional exploratory figures.
