import os
import pandas as pd

MOCOPI_DIR = "/Users/divinewton/Downloads/MOCOPI"
all_epochs = []

# Loop through and grab every generated kinematics file
for folder in os.listdir(MOCOPI_DIR):
    folder_path = os.path.join(MOCOPI_DIR, folder)
    if os.path.isdir(folder_path):
        features_file = os.path.join(folder_path, f"{folder}_epoch_kinematics.csv")
        if os.path.exists(features_file):
            all_epochs.append(pd.read_csv(features_file))

# Combine into one master cohort dataframe
cohort_df = pd.concat(all_epochs, ignore_index=True)

# 1. Summarize by Sensor Placement
print("\n==================================================")
print(" GLOBAL COHORT SUMMARY: BY SENSOR PLACEMENT")
print("==================================================")
sensor_summary = cohort_df.groupby('Sensor').agg(
    Intensity=('Intensity', 'mean'),
    Variability=('Variability', 'mean'),
    Jerk=('Jerk', 'mean'),
    Active_Pct=('Is_Active', lambda x: x.mean() * 100)
).round(2)
print(sensor_summary)

# 2. Summarize by Classroom Context
print("\n==================================================")
print(" GLOBAL COHORT SUMMARY: BY CLASSROOM CONTEXT")
print("==================================================")
context_summary = cohort_df.groupby('class').agg(
    Intensity=('Intensity', 'mean'),
    Variability=('Variability', 'mean'),
    Jerk=('Jerk', 'mean'),
    Active_Pct=('Is_Active', lambda x: x.mean() * 100)
).round(2)
print(context_summary)

# 3. Summarize Global Cross-Body Correlation Matrix
print("\n==================================================")
print(" GLOBAL COHORT SUMMARY: CROSS-BODY CORRELATIONS")
print("==================================================")
pivot_df = cohort_df.pivot_table(index=['Participant', 'Date', 'class', 'Epoch_1Min'], columns='Sensor', values='Intensity').dropna()
print(pivot_df.corr(method='pearson').round(3))