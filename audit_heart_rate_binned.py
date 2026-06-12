import os
import pandas as pd
import numpy as np

# Esure all exported participant CSVs live in the user's Downloads/Exports folder
EXPORTS_DIR = os.path.expanduser("~/Downloads/Exports")

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

def get_schedule_expected_bins(schedule_path, total_days):
    # Compute the number of 5-minute bins a schedule should produce for a given number of tracked days, excluding DELETE rows
    if not os.path.exists(schedule_path) or total_days == 0:
        return 0
    sched = pd.read_csv(schedule_path)
    sched = sched[sched['Class'].str.strip() != 'DELETE'].copy()
    
    total_bins = 0
    for _, row in sched.iterrows():
        start = pd.to_datetime(row['TimeStart'], format="%H:%M:%S")
        end = pd.to_datetime(row['TimeEnd'], format="%H:%M:%S")
        minutes = (end - start).total_seconds() / 60.0
        total_bins += int(np.ceil(minutes / 5.0))
    return total_bins * total_days

def time_to_seconds(t):
    # Normalize a time value to seconds since midnight so time-range checks are simplified
    return t.hour * 3600 + t.minute * 60 + t.second

# Counters across all processed participants
total_sample_expected_bins = 0
total_sample_observed_bins = 0
total_sample_valid_bins = 0
processed_participants = 0

for pNum, assigned_dates in participants_dates.items():
    # Skip participants whose exported raw file is missing
    raw_path = os.path.join(EXPORTS_DIR, f"P0{pNum}export.csv")
    if not os.path.exists(raw_path):
        continue

    # Count how many dates are on Friday versus Monday-Thursday
    fridays_count = sum(1 for d in assigned_dates if pd.to_datetime(d).strftime("%A") == 'Friday')
    other_days_count = len(assigned_dates) - fridays_count
    
    # Participants 04 and 05 share one schedule pair; the rest use the shared schedule pair
    if pNum in ["04", "05"]:
        fri_path = os.path.join(EXPORTS_DIR, "schedData_P(04,05)_Fr.csv")
        oth_path = os.path.join(EXPORTS_DIR, "schedData_P(04,05)_M-Th.csv")
    else:
        fri_path = os.path.join(EXPORTS_DIR, "schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv")
        oth_path = os.path.join(EXPORTS_DIR, "schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv")

    # Expected bins from the schedule definition, scaled by the number of tracked Friday and non-Friday dates for this participant
    user_expected_bins = get_schedule_expected_bins(fri_path, fridays_count) + get_schedule_expected_bins(oth_path, other_days_count)
    total_sample_expected_bins += user_expected_bins

    # Load the schedule tables once
    sched_fri = pd.read_csv(fri_path) if os.path.exists(fri_path) else pd.DataFrame()
    sched_oth = pd.read_csv(oth_path) if os.path.exists(oth_path) else pd.DataFrame()

    # Detect the header start dynamically since the export files can contain metadata before the CSV header
    skip_count = 0
    with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if '/@locale' in line:
                skip_count = i
                break

    # Read the export after the metadata block
    df = pd.read_csv(raw_path, skiprows=skip_count, low_memory=False)
    
    if "/Record/@type" in df.columns and "/Record/@startDate" in df.columns:
        # Keep only heart-rate records from the export
        hr_df = df[df["/Record/@type"].str.contains("HeartRate", na=False, case=False)].copy()
        
        if not hr_df.empty:
            # Parse timestamps and convert them to the target timezone
            hr_df['ParsedDT'] = pd.to_datetime(hr_df["/Record/@startDate"], errors='coerce')
            hr_df = hr_df.dropna(subset=['ParsedDT'])
            
            try:
                hr_df['ParsedDT'] = hr_df['ParsedDT'].dt.tz_convert('US/Pacific')
            except TypeError:
                # Localize exports as UTC before converting to Pacific time
                hr_df['ParsedDT'] = hr_df['ParsedDT'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
            
            # Get helper columns used for date filtering and 5-minute binning
            hr_df['DateStr'] = hr_df['ParsedDT'].dt.strftime('%Y-%m-%d')
            hr_df['5MinBlock'] = hr_df['ParsedDT'].dt.floor('5min')
            hr_df['TimeSec'] = hr_df['ParsedDT'].dt.time.apply(time_to_seconds)
            
            # Only analyze records that fall on dates assigned to the participant
            tracking_days_df = hr_df[hr_df['DateStr'].isin(assigned_dates)].copy()
            
            observed_blocks = set() # tracks any 5-min interval that has at least one heart-rate record during class time
            valid_blocks = set() # observed blocks that are numeric and fall inside expected range
            
            for date_str, group in tracking_days_df.groupby('DateStr'):
                # Pick the correct weekday schedule for the current date
                day_of_week = pd.to_datetime(date_str).strftime("%A")
                current_sched = sched_fri if day_of_week == 'Friday' else sched_oth
                current_sched = current_sched[current_sched['Class'].str.strip() != 'DELETE']
                
                if current_sched.empty:
                    continue
                
                # Convert class start/end times once per day
                start_secs = pd.to_datetime(current_sched['TimeStart'], format="%H:%M:%S").dt.time.apply(time_to_seconds).values
                end_secs = pd.to_datetime(current_sched['TimeEnd'], format="%H:%M:%S").dt.time.apply(time_to_seconds).values
                
                for _, row in group.iterrows():
                    row_sec = row['TimeSec']
                    
                    # Count only records that occur during a scheduled class window
                    if any((start <= row_sec <= end) for start, end in zip(start_secs, end_secs)):
                        observed_blocks.add(row['5MinBlock'])
                        
                        # A block is considered valid only if the value can be converted to a number and falls in the expected HR range
                        hr_val = pd.to_numeric(row.get('/Record/@value'), errors='coerce')
                        if pd.notna(hr_val) and 40 <= hr_val <= 180:
                            valid_blocks.add(row['5MinBlock'])
            
            # Prevent a participant from reporting more observed/valid blocks than the schedule allows
            user_observed_bins = len(observed_blocks)
            if user_observed_bins > user_expected_bins:
                user_observed_bins = user_expected_bins
                
            user_valid_bins = len(valid_blocks)
            if user_valid_bins > user_observed_bins:
                user_valid_bins = user_observed_bins

            total_sample_observed_bins += user_observed_bins
            total_sample_valid_bins += user_valid_bins
            processed_participants += 1

# Final summary is shown as averages per processed participant, along with an overall coverage percentage for the entire sample
if processed_participants > 0:
    avg_expected = total_sample_expected_bins / processed_participants
    avg_observed = total_sample_observed_bins / processed_participants
    avg_valid    = total_sample_valid_bins / processed_participants
    
    global_coverage_percentage = (total_sample_valid_bins / total_sample_expected_bins) * 100
    
    print(f"Expected (5-Min Bins): {avg_expected:,.1f} bins")
    print(f"Observed (5-Min Bins): {avg_observed:,.1f} bins")
    print(f"Valid (5-Min Bins): {avg_valid:,.1f} bins")
    print(f"Coverage: {global_coverage_percentage:.2f}%")