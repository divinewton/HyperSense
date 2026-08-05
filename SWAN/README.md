# SWAN wearable analysis

One script produces the concise paper-ready analysis. It uses the private
crosswalk in `SWAN/inputs/` and writes only the outputs listed below.

```bash
MPLCONFIGDIR=/tmp/hypersense-mpl python3 SWAN/run_analysis.py \
  --apple-root "/Users/divinewton/Downloads/Apple Watch Export CSVs" \
  --swan-workbook "/Users/divinewton/Downloads/Swan (Responses).xlsx" \
  --mocopi-root "/Users/divinewton/Downloads/epoch_kinematics"
```

Outputs: six focused figures: four main figures (Apple Watch, head sensor,
rater agreement, and a sensor-by-sensor acceleration profile) plus two
appendix figures (data coverage and classroom context). It also writes
participant coverage, primary correlations, head-sensor replication, two
supplement tables, captions, and a short results summary.
All analyses are exploratory and participant-level (n=12).
