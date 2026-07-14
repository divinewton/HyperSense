import os
import re
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def clean_and_combine_raw_data(base_directory, participant_id):
    """
    [Phase 1 - Jaime's Logic]
    Dynamically scans the directory for YYYY-MM-DD subfolders, maps sensors 
    from raw file names, and merges everything into a chronological master CSV.
    """
    print(f"\n[{participant_id} - Step 1/3] Combining raw daily files...")
    
    # Dynamically find any subfolders matching the YYYY-MM-DD format
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    dates = [d for d in os.listdir(base_directory) if date_pattern.match(d) and os.path.isdir(os.path.join(base_directory, d))]
    dates.sort()
    
    if not dates:
        print(f"      [Error] No date subfolders (YYYY-MM-DD) found inside {base_directory}.")
        return None
        
    print(f"      Found date subfolders: {dates}")
    combined_list = []

    for date in dates:
        day_folder = os.path.join(base_directory, date)
        for file in os.listdir(day_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(day_folder, file)
                
                # Jaime's exact naming parser to find the sensor placement
                parts = file.split("Mocopi")
                if len(parts) > 1:
                    sensor = parts[1].split("Device")[0]
                else:
                    sensor = "Unknown"
                    
                try:
                    temp_df = pd.read_csv(file_path)
                    temp_df['Sensor'] = sensor
                    temp_df['Date'] = date
                    combined_list.append(temp_df)
                except Exception as e:
                    print(f"      Error reading {file}: {e}")

    if not combined_list:
        print(f"      [Error] No raw CSV files found inside the date folders for {participant_id}.")
        return None

    combined = pd.concat(combined_list, ignore_index=True)
    
    # Jaime's timeseries sorting logic
    combined = combined.sort_values(['Sensor', 'Date', 'Time_In_PST']).reset_index(drop=True)
    
    # Jaime's exact raw Jerk and Acceleration Magnitude calculations
    print(f"[{participant_id} - Step 2/3] Computing raw kinematics (Acc Magnitude, Jerk)...")
    combined['Jerk_X'] = combined.groupby(['Sensor', 'Date'])['Acceleration X'].diff()
    combined['Jerk_Y'] = combined.groupby(['Sensor', 'Date'])['Acceleration Y'].diff()
    combined['Jerk_Z'] = combined.groupby(['Sensor', 'Date'])['Acceleration Z'].diff()

    combined['Jerk_Mag'] = np.sqrt(
        combined['Jerk_X']**2 +
        combined['Jerk_Y']**2 +
        combined['Jerk_Z']**2
    )

    combined['Acc_Mag'] = np.sqrt(
        combined['Acceleration X']**2 +
        combined['Acceleration Y']**2 +
        combined['Acceleration Z']**2
    )

    # Clean up temporary component columns to save runtime memory
    combined = combined.drop(columns=['Jerk_X', 'Jerk_Y', 'Jerk_Z'])
    
    # Save master raw combined file
    output_combined_path = os.path.join(base_directory, f"{participant_id}_dates_combined.csv")
    combined.to_csv(output_combined_path, index=False)
    print(f"      Master combined dataset saved to: {output_combined_path}")
    return combined

def generate_epoch_features(combined_df, base_directory, participant_id, movement_threshold=1.15):
    """
    [Phase 2 - Your Epoch Logic]
    Aggregates the sub-second signals into clean 1-minute windowed averages,
    variabilities, jerk profiles, and active-vs-sedentary categorizations.
    """
    print(f"[{participant_id} - Step 3/3] Epoching raw data to 1-minute blocks...")
    
    # Generate timestamp epochs - using format='mixed' or explicit warning suppression to keep console output clean
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Time_In_PST'], errors='coerce')
    combined_df['Epoch_1Min'] = combined_df['Timestamp'].dt.floor('1min')
    
    # Calculate window metrics (including standard deviation for variability)
    epoch_df = combined_df.groupby(['Sensor', 'Date', 'class', 'Epoch_1Min']).agg(
        Intensity=('Acc_Mag', 'mean'),
        Variability=('Acc_Mag', 'std'),  # Movement variability
        Jerk=('Jerk_Mag', 'mean')        # Windowed average Jerk
    ).reset_index()
    
    # Classify active vs low-movement window ratios
    epoch_df['Is_Active'] = (epoch_df['Intensity'] > movement_threshold).astype(int)
    epoch_df['Participant'] = participant_id
    
    # Extract Hour of Day for Time of Day stratification
    epoch_df['Hour'] = epoch_df['Epoch_1Min'].dt.hour
    epoch_df['Time_of_Day'] = pd.cut(
        epoch_df['Hour'],
        bins=[0, 11, 14, 24],
        labels=['Morning', 'Midday', 'Afternoon'],
        right=False
    )
    
    # Save the windowed features CSV
    output_features_path = os.path.join(base_directory, f"{participant_id}_epoch_kinematics.csv")
    epoch_df.to_csv(output_features_path, index=False)
    print(f"      Completed kinematic features saved to: {output_features_path}")
    return epoch_df

def print_results(epoch_df, participant_id):
    """
    Generates command-line printouts of comparisons and sensor correlations.
    """
    print(f"\n========================================================")
    print(f" RESULTS ANALYSIS FOR {participant_id}")
    print(f"========================================================")
    
    dimensions = {
        'Sensor Placement': 'Sensor',
        'Classroom Context': 'class',
        'Time of Day': 'Time_of_Day'
    }
    
    for label, col in dimensions.items():
        print(f"\n--- Stratification by {label} ---")
        summary = epoch_df.groupby(col).agg(
            Mean_Intensity=('Intensity', 'mean'),
            SD_Intensity=('Intensity', 'std'),
            Mean_Variability=('Variability', 'mean'),
            SD_Variability=('Variability', 'std'),
            Mean_Jerk=('Jerk', 'mean'),
            SD_Jerk=('Jerk', 'std'),
            Active_Ratio=('Is_Active', 'mean')
        ).reset_index()
        
        summary['Active_Ratio_Pct'] = (summary['Active_Ratio'] * 100).round(2)
        print(summary.to_string(index=False))

    # Calculate and output Cross-Body Correlations
    print(f"\n--- Cross-Body Sensor Correlations ({participant_id}) ---")
    pivot_df = epoch_df.pivot_table(
        index=['Participant', 'Date', 'class', 'Epoch_1Min'],
        columns='Sensor',
        values='Intensity'
    ).dropna()
    
    if pivot_df.empty:
        print("Warning: Insufficient overlapping epochs to run sensor correlations.")
        return
        
    corr_matrix = pivot_df.corr(method='pearson')
    print("\nPearson Correlation Matrix (r):")
    print(corr_matrix.round(3))
    
    cols = corr_matrix.columns
    print("\nPairwise Cross-Body Correlations (p-values):")
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s1, s2 = cols[i], cols[j]
            r_val, p_val = pearsonr(pivot_df[s1], pivot_df[s2])
            print(f"  {s1} vs {s2}: r = {r_val:.3f} (p = {p_val:.3e})")

def main():
    parser = argparse.ArgumentParser(description="Run complete MOCOPI Processing and Feature Analysis pipeline.")
    parser.add_argument(
        "--participant", 
        type=str, 
        required=True, 
        help="Specific participant ID to process (e.g. 'P002', 'P014') or use 'all' to run sequentially."
    )
    args = parser.parse_args()
    
    # Try multiple common directories where your raw folders might be stored
    possible_paths = [
        "/Users/divinewton/Downloads/MOCOPI",
        "/Users/divinewton/Documents/MOCOPI"
    ]
    
    MOCOPI_DIR = None
    for path in possible_paths:
        if os.path.exists(path):
            MOCOPI_DIR = path
            break
            
    if not MOCOPI_DIR:
        print("[Fatal Error] Could not find 'MOCOPI' folder in Downloads or Documents.")
        return
        
    print(f"Using MOCOPI directory: {MOCOPI_DIR}")
    
    # Identify target participant(s)
    if args.participant.lower() == 'all':
        participants = [d for d in os.listdir(MOCOPI_DIR) if re.match(r'^P\d+$', d)]
        participants.sort()
        print(f"Found {len(participants)} participant directories to process: {participants}")
    else:
        participants = [args.participant]
        
    for p_id in participants:
        folder_path = os.path.join(MOCOPI_DIR, p_id)
        if not os.path.exists(folder_path):
            print(f"\n[Warning] Participant folder does not exist at: '{folder_path}'. Skipping.")
            continue
            
        print(f"\n========================================================")
        print(f" RUNNING FULL PIPELINE FOR PARTICIPANT: {p_id}")
        print(f"========================================================")
        
        # Step 1 & 2: Clean and merge raw timeseries (Jaime's pipeline)
        combined_data = clean_and_combine_raw_data(folder_path, p_id)
        
        if combined_data is not None:
            # Step 3: Run windowed feature aggregation (Your pipeline)
            epoch_data = generate_epoch_features(combined_data, folder_path, p_id)
            # Output results
            print_results(epoch_data, p_id)

if __name__ == "__main__":
    main()