#!/usr/bin/env python3
"""
Minilamp Data Preprocessing Script
Processes minilamp data from folders 28-60, handling PWM_Percent as a dynamic parameter
for period detection (PWM changes create separate periods, but PWM is not in output columns)
"""

import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_minilamp_dataset(config_number, minilamp_path):
    """Load and combine all part files for minilamp"""
    print(f"Processing Minilamp in Configuration {config_number}...")
    
    # Find all part files
    part_files = glob.glob(os.path.join(minilamp_path, "guest_data_*_part*_refined.csv"))
    part_files.sort()
    print(f"Found {len(part_files)} part files")
    
    # Load and combine all part files
    datasets = []
    for file_path in tqdm(part_files, desc="Loading all part files"):
        try:
            df = pd.read_csv(file_path)
            datasets.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not datasets:
        print(f"No valid datasets found for Configuration {config_number}")
        return None
    
    # Combine all datasets
    combined_data = pd.concat(datasets, ignore_index=True)
    print(f"  Combined dataset size: {len(combined_data)} rows")
    
    # Filter out first 4.5 seconds (unstable signal)
    if len(combined_data) > 0:
        first_timestamp = combined_data["Timestamp (s)"].min()
        filtered_data = combined_data[combined_data["Timestamp (s)"] >= first_timestamp + 4.5]
        print(f"  After filtering first 4.5 seconds: {len(filtered_data)} rows")
        print(f"  Timestamp range: {filtered_data['Timestamp (s)'].min():.2f} to {filtered_data['Timestamp (s)'].max():.2f} seconds")
    
    return filtered_data

def map_state(running_state):
    """Map Running_State to new state system (use original value directly)"""
    # Simply return the original Running_State value
    # Running_State 0 → State 0 (Initialization)
    # Running_State 1 → State 1 (Advertising)
    # Running_State 2 → State 2 (Operation)
    # Running_State 3 → State 3 (Sleep)
    return running_state

def process_minilamp_data_chronological(config_number, minilamp_path):
    """
    Process minilamp data chronologically, handling PWM_Percent as dynamic parameter.
    PWM_Percent changes create separate periods, but PWM is not included in output columns.
    Output columns: Number, Guest ID, State, Duration, Current, Voltage, Total Energy
    """
    # Load dataset
    minilamp_data = load_minilamp_dataset(config_number, minilamp_path)
    if minilamp_data is None or len(minilamp_data) == 0:
        return []
    
    # Sort by timestamp to ensure chronological order
    minilamp_data = minilamp_data.sort_values("Timestamp (s)").reset_index(drop=True)
    
    # Extract Guest_ID (should be 10 for all minilamps)
    guest_id = minilamp_data["Guest_ID"].iloc[0]
    print(f"Processing {len(minilamp_data)} samples...")
    print(f"Guest ID: {guest_id}")
    
    # Create state mapping
    minilamp_data["New_State"] = minilamp_data["Running_State"].apply(map_state)
    
    # For dynamic PWM handling, create a combined key: Running_State + PWM_Percent
    # This will split periods when PWM changes
    minilamp_data["Period_Key"] = (minilamp_data["Running_State"].astype(str) + "_" + 
                                    minilamp_data["PWM_Percent"].astype(str))
    
    # Find period boundaries (where Period_Key changes)
    minilamp_data["Period_Change"] = minilamp_data["Period_Key"] != minilamp_data["Period_Key"].shift(1)
    minilamp_data.loc[0, "Period_Change"] = True  # First row is always a new period
    
    # Get period boundaries
    period_starts = minilamp_data[minilamp_data["Period_Change"]].index.tolist()
    period_ends = period_starts[1:] + [len(minilamp_data)]
    
    print(f"Found {len(period_starts)} periods (including PWM changes)")
    
    # Process each period independently
    final_rows = []
    for start_idx, end_idx in tqdm(list(zip(period_starts, period_ends)), desc="Processing periods"):
        period_data = minilamp_data.iloc[start_idx:end_idx].copy()
        period_data = period_data.sort_values("Timestamp (s)").reset_index(drop=True)
        
        running_state = period_data["Running_State"].iloc[0]
        state = map_state(running_state)
        # PWM is used for period detection but not included in output
        # pwm_percent = period_data["PWM_Percent"].iloc[0]
        
        # Duration and averages for this period
        duration = period_data["Timestamp (s)"].iloc[-1] - period_data["Timestamp (s)"].iloc[0]
        avg_current = period_data["Current (uA)"].mean()
        avg_voltage = period_data["Voltage (mV)"].mean()
        total_energy = avg_current * avg_voltage * duration / 1000  # Energy in microjoules
        
        # Create row for this period (excluding TxPower, AdvInterval, ConnInterval, PWM_Percent)
        final_rows.append({
            "Number": config_number,
            "Guest ID": guest_id,
            "State": state,
            "Duration": duration,
            "Current": avg_current,
            "Voltage": avg_voltage,
            "Total Energy": total_energy
        })
    
    print(f"Created {len(final_rows)} rows for Configuration {config_number}")
    return final_rows


def main():
    """Main function"""
    # Configuration parameters
    BASE_DATASET_PATH = r"F:\Codes\Collect-datasets\PPK2-Visulization-Current-SuperIoT-V6\Dataset-all\Refined original dataset"
    
    # Process configurations 28-60 (minilamp data)
    CONFIGS_TO_PROCESS = list(range(28, 61))  # 28 to 60 inclusive
    OUTPUT_FILENAME = "processed_minilamp_data.csv"
    
    print("=== Minilamp Data Preprocessing ===")
    print(f"Base dataset path: {BASE_DATASET_PATH}")
    print(f"Configurations to process: {CONFIGS_TO_PROCESS[0]} to {CONFIGS_TO_PROCESS[-1]}")
    print(f"Output: {OUTPUT_FILENAME}")
    print()
    
    # Verify base path exists
    if not os.path.exists(BASE_DATASET_PATH):
        print(f"ERROR: Base dataset path does not exist: {BASE_DATASET_PATH}")
        print("Please update BASE_DATASET_PATH in the script to point to your dataset location.")
        return
    
    all_rows = []
    
    # Process each configuration
    for config_number in CONFIGS_TO_PROCESS:
        config_path = os.path.join(BASE_DATASET_PATH, str(config_number))
        
        # Check if configuration folder exists
        if not os.path.exists(config_path):
            print(f"Configuration {config_number} not found at {config_path}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing Configuration {config_number}")
        print(f"{'='*60}")
        
        # Look for Minilamp-refined folder
        minilamp_path = os.path.join(config_path, "Minilamp-refined")
        
        if not os.path.exists(minilamp_path):
            print(f"Minilamp-refined folder not found in {config_path}, skipping...")
            continue
        
        try:
            minilamp_rows = process_minilamp_data_chronological(config_number, minilamp_path)
            if minilamp_rows:
                all_rows.extend(minilamp_rows)
                print(f"  Configuration {config_number}: {len(minilamp_rows)} rows processed")
        except Exception as e:
            print(f"  Error processing Configuration {config_number}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save all results
    if not all_rows:
        print("\nNo data processed!")
        return
    
    try:
        # Create final DataFrame
        final_df = pd.DataFrame(all_rows)
        
        # Save to CSV
        final_df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to {OUTPUT_FILENAME}")
        print(f"Total rows: {len(final_df)}")
        print(f"Total configurations processed: {final_df['Number'].nunique()}")
        
        # Display summary
        print("\n=== Summary by Configuration ===")
        summary = final_df.groupby("Number").size()
        for config_num, count in summary.items():
            print(f"  Configuration {config_num}: {count} rows")
        
        # Summary by State
        print("\n=== Summary by State ===")
        state_summary = final_df.groupby("State").agg({
            "Duration": "sum",
            "Current": "mean"
        })
        print(state_summary)
        
        # Verify Guest ID
        unique_guest_ids = final_df["Guest ID"].unique()
        print(f"\n=== Guest IDs ===")
        print(f"Unique Guest IDs: {unique_guest_ids}")
        if len(unique_guest_ids) == 1 and unique_guest_ids[0] == 10:
            print("✓ All data has Guest ID = 10 (as expected for minilamp)")
        else:
            print("⚠ WARNING: Found unexpected Guest IDs!")
        
        # Verify duration totals
        total_duration = final_df["Duration"].sum()
        print(f"\nTotal duration across all data: {total_duration:.2f} seconds")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

