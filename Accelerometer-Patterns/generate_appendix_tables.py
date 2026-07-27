import os
import pandas as pd

# --- CONFIG ---
EPOCH_DIR = "/Users/divinewton/Downloads/epoch_kinematics"

SENSORS = ["AnkleL", "AnkleR", "Head", "Hip", "WristL", "WristR"]

# --- STEP 1: Load every *_epoch_kinematics.csv in the folder ---
all_epochs = []
for filename in os.listdir(EPOCH_DIR):
    if filename.endswith("_epoch_kinematics.csv"):
        all_epochs.append(pd.read_csv(os.path.join(EPOCH_DIR, filename)))

if not all_epochs:
    print(f"No epoch files found in: {EPOCH_DIR}")
    raise SystemExit(1)

df = pd.concat(all_epochs, ignore_index=True)
print(f"Loaded {len(all_epochs)} participant files from {EPOCH_DIR}\n")

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
