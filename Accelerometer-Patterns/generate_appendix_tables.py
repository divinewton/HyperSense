import os
import sys
from pathlib import Path

import pandas as pd

# --- CONFIG ---
EPOCH_DIR = "/Users/divinewton/Downloads/epoch_kinematics"

SENSORS = ["AnkleL", "AnkleR", "Head", "Hip", "WristL", "WristR"]

# P007's on-disk epoch file is a copy of P006; rebuild wrist minutes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_by_class_and_time import load_epoch_kinematics

df = load_epoch_kinematics(Path(EPOCH_DIR))
print(f"Loaded epoch kinematics from {EPOCH_DIR} (P007 wrist rebuild applied)\n")

# --- STEP 2: Add day of week from Date ---
df["Date"] = pd.to_datetime(df["Date"])
df["Day_of_Week"] = df["Date"].dt.day_name()

# --- STEP 3: Print one table per sensor ---
# Each cell = Active Ratio (%) = % of 1-min epochs where Intensity > 1.15
for sensor in SENSORS:
    sensor_df = df[df["Sensor"] == sensor]

    table = sensor_df.pivot_table(
        index="class",           # rows = classroom context
        columns="Day_of_Week",   # columns = Mon, Tue, Wed, Thu, Fri
        values="Is_Active",      # 1 = active epoch, 0 = not active
        aggfunc=lambda x: round(x.mean() * 100, 2),
    )

    print(f"{'=' * 60}")
    print(f"SENSOR: {sensor}  (Active Ratio %)")
    print(f"{'=' * 60}")
    print(table.to_string())
    print()
