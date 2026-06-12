import pandas as pd
import pytz
import os
from collections import defaultdict
from datetime import datetime, timezone, date

def convert_date_format(date_str):
    try:
        dt = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None
    
def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000

def get_day_of_week(date_obj):
    return date_obj.strftime("%A")

def filter_dates_for_participant(df, participant_id, date_column):
    allowed_dates = participants_dates.get(participant_id, set())
    converted_dates = df[date_column].astype(str).apply(convert_date_format)
    filtered_df = df[converted_dates.isin(allowed_dates)].reset_index(drop=True)
    
    return filtered_df


participants_dates = {
    "01": {"2025-02-03", "2025-02-04", "2025-02-05", "2025-02-06", "2025-02-07"},
    "02": {"2025-02-03", "2025-02-04", "2025-02-05"},
    "03": {"2025-02-03", "2025-02-04"},
    "04": {"2025-02-10", "2025-02-11", "2025-02-12", "2025-02-13", "2025-02-14"},
    "05": {"2025-02-10", "2025-02-11", "2025-02-12"},
    "06": {"2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28"},
    "07": {"2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28"},
    "08": {"2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28"},
    "09": {"2025-02-03", "2025-02-04", "2025-02-05", "2025-02-06", "2025-02-07"},
    "12": {"2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07"},
    "14": {"2025-03-25", "2025-03-26", "2025-03-27", "2025-03-31", "2025-04-01"},
    "16": {"2025-03-25", "2025-03-26", "2025-03-27", "2025-03-31", "2025-04-01"},
}

EXPORTS_DIR = os.path.expanduser("~/Downloads/Exports")
SCHEDULES_DIR = EXPORTS_DIR

# Prompt for participant number
pNum = input("Enter the participant number: ")

# Load schedule data
if pNum in ["04", "05"]:
    scheduleDataFri = pd.read_csv(os.path.join(SCHEDULES_DIR, "schedData_P(04,05)_Fr.csv"))
    scheduleDataOth = pd.read_csv(os.path.join(SCHEDULES_DIR, "schedData_P(04,05)_M-Th.csv"))
else:
    scheduleDataFri = pd.read_csv(os.path.join(SCHEDULES_DIR, "schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv"))
    scheduleDataOth = pd.read_csv(os.path.join(SCHEDULES_DIR, "schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv"))

# Gathering parent paths
participant_dir = os.path.join(EXPORTS_DIR, f"P0{pNum}")
healthapp_dir = os.path.join(participant_dir, "HealthApp")
rawParentPath = os.path.join(EXPORTS_DIR, f"P0{pNum}export.csv")
labeledActivityParentPath = os.path.join(healthapp_dir, "Labeled", "ActivitySummary")
labeledRecordParentPath = os.path.join(healthapp_dir, "Labeled", "ActivitySummary")

# Ensure the Record directory exists
recordDir = os.path.join(healthapp_dir, "Labeled", "Record")
os.makedirs(recordDir, exist_ok=True)

skip_count = 0
with open(rawParentPath, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        if '/@locale' in line:
            skip_count = i
            break

majorDF = pd.read_csv(rawParentPath, skiprows=skip_count, low_memory=False)

# Drop uneccessary colummns
majorDF.drop(columns=["/@locale"], errors='ignore', inplace=True)
majorDF.drop(columns=[col for col in majorDF.columns if col.startswith("/Me/")], inplace=True)
majorDF.drop(columns=[col for col in majorDF.columns if col.startswith("/Workout/")], inplace=True)

# Making DF for activity data
activityCols = [col for col in majorDF.columns if col.startswith("/ActivitySummary/")]
activityDF = majorDF[activityCols]

# Making DF for record data
recordCols = [col for col in majorDF.columns if col.startswith("/Record/")]
recordDF = majorDF[recordCols]

# Delete empty rows
activityDF = activityDF.dropna(how='all')
recordDF = recordDF.dropna(how='all')

# Reset index
activityDF = activityDF.reset_index(drop=True)
recordDF = recordDF.reset_index(drop=True)

# Keep the record date filtering out here so it always runs
recordDF = filter_dates_for_participant(recordDF, pNum, "/Record/@startDate")

if not activityDF.empty and "/ActivitySummary/@dateComponents" in activityDF.columns:
    activityDF = filter_dates_for_participant(activityDF, pNum, "/ActivitySummary/@dateComponents")

    # renaming and removing unnecessary columns
    activityDF = activityDF.rename(columns={'/ActivitySummary/@activeEnergyBurned': 'ActiveEnergyBurned'})
    activityDF.drop(columns=["/ActivitySummary/@activeEnergyBurnedGoal"], errors='ignore', inplace=True)
    activityDF = activityDF.rename(columns={'/ActivitySummary/@activeEnergyBurnedUnit': 'ActiveEnergyBurnedUnit'})
    activityDF = activityDF.rename(columns={'/ActivitySummary/@appleExerciseTime': 'ExerciseTime'})
    activityDF.drop(columns=["/ActivitySummary/@appleExerciseTimeGoal"], errors='ignore', inplace=True)
    activityDF = activityDF.rename(columns={'/ActivitySummary/@appleMoveTime': 'MoveTime'})
    activityDF.drop(columns=["/ActivitySummary/@appleMoveTimeGoal"], errors='ignore', inplace=True)
    activityDF = activityDF.rename(columns={'/ActivitySummary/@appleStandHours': 'StandHurs'})
    activityDF.drop(columns=["/ActivitySummary/@appleStandHoursGoal"], errors='ignore', inplace=True)
    activityDF = activityDF.rename(columns={'/ActivitySummary/@dateComponents': 'date'})
    
    # Save the activity file only if it exists
    activityDF.to_csv(os.path.join(healthapp_dir, "Labeled", f"P0{pNum}ActivityLabeled.csv"), index=False)
else:
    print(f"⚠️ Note: No valid activity summary tracking data found for Participant P0{pNum}. Skipping activity file generation.")

recordDF = recordDF.rename(columns={'/Record/@creationDate': 'CreationDate'})
recordDF.drop(columns=["/Record/@device"], inplace=True)
recordDF = recordDF.rename(columns={'/Record/@endDate': 'EndDate'})
recordDF.drop(columns=["/Record/@sourceName"], inplace=True)
recordDF.drop(columns=["/Record/@sourceVersion"], inplace=True)
recordDF = recordDF.rename(columns={'/Record/@startDate': 'StartDate'})
recordDF = recordDF.rename(columns={'/Record/@type': 'Type'})
recordDF = recordDF.rename(columns={'/Record/@unit': 'Unit'})
recordDF = recordDF.rename(columns={'/Record/@value': 'Value'})
recordDF = recordDF.rename(columns={'/Record/#id': 'ID'})
recordDF['Type'] = recordDF['Type'].str.replace("HKQuantityTypeIdentifier", '', regex=False)

# Saving activity DF
activityDF.to_csv(os.path.join(healthapp_dir, "Labeled", f"P0{pNum}ActivityLabeled.csv"), index=False)

# Adding columns to record DF
zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
recordDF.insert(0, 'class', "NONE")
recordDF.insert(1, 'Time_In_PST', zero_time)
recordDF.insert(2, 'time', 0.0)

# Creating list of the different data frames
recordDF = recordDF.sort_values(by='StartDate').reset_index(drop=True)
prevDate = convert_date_format(recordDF.iloc[0]['StartDate'])
start_idx = 0
dfList = []


#Separating based on date
for idx, row in recordDF.iterrows():
    currDate = convert_date_format(row['StartDate'])
    if currDate != prevDate:
        dfList.append(recordDF.iloc[start_idx:idx].copy())
        start_idx = idx
        prevDate = currDate
dfList.append(recordDF.iloc[start_idx:].copy())

#adding time columns
for i, df in enumerate(dfList):
    dt = pd.to_datetime(df['StartDate'], errors='coerce')
    dt = dt.dt.tz_convert('US/Pacific')
    df['Time_In_PST'] = dt.dt.time
    df['time'] = dt.dt.floor('s').astype('int64') // 10**9
    dfList[i] = df

for dataFrame in dfList:
    day_of_week = get_day_of_week(datetime.fromtimestamp(dataFrame.iloc[0]['time']))
    schedule = scheduleDataFri if day_of_week == 'Friday' else scheduleDataOth

    schedule = schedule.copy()
    schedule['TimeStart'] = pd.to_datetime(schedule['TimeStart'], format="%H:%M:%S").dt.time
    schedule['TimeEnd']   = pd.to_datetime(schedule['TimeEnd'],   format="%H:%M:%S").dt.time

    schedule['TimeStart_sec'] = schedule['TimeStart'].apply(time_to_seconds)
    schedule['TimeEnd_sec']   = schedule['TimeEnd'].apply(time_to_seconds)

    time_values_sec = dataFrame['Time_In_PST'].apply(time_to_seconds)

    intervals = pd.IntervalIndex.from_arrays(
    schedule['TimeStart_sec'],
    schedule['TimeEnd_sec'],
    closed='right'
    )

    import numpy as np
    matched_class = np.full(len(time_values_sec), None, dtype=object)
    for i, interval in enumerate(intervals):
        mask = (interval.left < time_values_sec) & (time_values_sec <= interval.right)
        matched_class = np.where(mask, schedule.iloc[i]['Class'], matched_class)

    dataFrame['class'] = matched_class

# Creates date directories
for df in dfList:
    date = convert_date_format(df['StartDate'].iloc[0])
    recordDir = os.path.join(healthapp_dir, "Labeled", "Record", date)
    os.makedirs(recordDir, exist_ok=True)

#Splitting data frames up by data type
dfListTypes = []

# Splitting data frames up by data type cleanly and fast
existing = defaultdict(list)

for df in dfList:
    date = convert_date_format(df['StartDate'].iloc[0])
    # Convert dataframe directly to a list of dicts for maximum loop speed
    for row_dict in df.to_dict('records'):
        t = row_dict['Type']
        key = (date, t)
        existing[key].append(row_dict)

# Now store the final DataFrames all at once (this skips the slow pd.concat loop)
dfListTypes = [pd.DataFrame(rows) for rows in existing.values()]

# Creating list of paths to save to
csvPathList = []

for df in dfListTypes:
    date = convert_date_format(df['StartDate'].iloc[0])
    type = df['Type'].iloc[0]
    csvPathList.append(os.path.join(healthapp_dir, "Labeled", "Record", date, f"P0{pNum}HealthAppRecord{date}_{type}.csv"))

for i in range(len(dfListTypes)):
    dataFrame = dfListTypes[i].copy()
    dataFrame.loc[:, 'class'] = dataFrame['class'].str.strip()
    dataFrame = dataFrame[dataFrame['class'] != 'DELETE'].reset_index(drop=True)
    dataFrame.to_csv(csvPathList[i], index=False)
    dfListTypes[i] = dataFrame