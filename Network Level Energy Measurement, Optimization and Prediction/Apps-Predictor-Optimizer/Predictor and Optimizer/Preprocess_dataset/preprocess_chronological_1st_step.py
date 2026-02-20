#!/usr/bin/env python3
"""
Chronological Data Preprocessing Script
Processes data in chronological order and identifies actual state periods
"""

import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_dataset(config_number, node_folder, node_path):
    """Load and combine all part files for a specific node"""
    print(f"Processing {node_folder} in Configuration {config_number}...")
    
    # Find all part files
    part_files = glob.glob(os.path.join(node_path, "guest_data_*_part*_refined.csv"))
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
        print(f"No valid datasets found for {node_folder}")
        return None
    
    # Combine all datasets
    combined_data = pd.concat(datasets, ignore_index=True)
    print(f"  Combined dataset size: {len(combined_data)} rows")
    
    # Filter out first 9 seconds (unstable signal)
    if len(combined_data) > 0:
        first_timestamp = combined_data["Timestamp (s)"].min()
        filtered_data = combined_data[combined_data["Timestamp (s)"] >= first_timestamp + 4.5]
        print(f"  After filtering first 4.5 seconds: {len(filtered_data)} rows")
        print(f"  Timestamp range: {filtered_data['Timestamp (s)'].min():.2f} to {filtered_data['Timestamp (s)'].max():.2f} seconds")
    
    return filtered_data

def map_state(running_state, comm_mode):
    """Map Running_State and Comm_Mode to new state system"""
    if running_state == 0:
        return 1  # Ignore Comm_Mode
    elif running_state == 1:
        return 2  # Ignore Comm_Mode
    elif running_state == 2:
        return 3  # Comm_Mode will be handled separately
    elif running_state == 3:
        return 4  # Ignore Comm_Mode
    else:
        return 0

def find_communication_chunks(period_data, mask):
    """Find distinct communication chunks within a period"""
    if not mask.any():
        return []
    
    # Get the communication data
    comm_data = period_data[mask].copy()
    comm_data = comm_data.sort_values("Timestamp (s)").reset_index(drop=True)
    
    # Find gaps in timestamps to identify distinct chunks
    if len(comm_data) <= 1:
        # Single sample or no data
        if len(comm_data) == 1:
            return [{
                "start_time": comm_data["Timestamp (s)"].iloc[0],
                "end_time": comm_data["Timestamp (s)"].iloc[0],
                "avg_current": comm_data["Current (uA)"].iloc[0]
            }]
        else:
            return []
    
    # Calculate time differences between consecutive samples
    time_diffs = comm_data["Timestamp (s)"].diff()
    
    # Find gaps (where time difference > threshold, e.g., 0.1 seconds)
    gap_threshold = 0.1  # 0.1 second gap indicates separate chunks
    gaps = time_diffs > gap_threshold
    
    # Get indices where gaps occur
    gap_indices = comm_data[gaps].index.tolist()
    
    # Create chunks based on gaps
    chunks = []
    start_idx = 0
    
    for gap_idx in gap_indices:
        # End current chunk at gap
        chunk_data = comm_data.iloc[start_idx:gap_idx]
        if len(chunk_data) > 0:
            chunks.append({
                "start_time": chunk_data["Timestamp (s)"].iloc[0],
                "end_time": chunk_data["Timestamp (s)"].iloc[-1],
                "avg_current": chunk_data["Current (uA)"].mean()
            })
        start_idx = gap_idx
    
    # Add the last chunk
    if start_idx < len(comm_data):
        chunk_data = comm_data.iloc[start_idx:]
        if len(chunk_data) > 0:
            chunks.append({
                "start_time": chunk_data["Timestamp (s)"].iloc[0],
                "end_time": chunk_data["Timestamp (s)"].iloc[-1],
                "avg_current": chunk_data["Current (uA)"].mean()
            })
    
    return chunks

def process_node_data_chronological(config_number, node_folder, node_path):
    """Process node data chronologically to identify actual state periods using vectorized operations"""
    # Load dataset
    guest_data = load_dataset(config_number, node_folder, node_path)
    if guest_data is None or len(guest_data) == 0:
        return []
    
    # Sort by timestamp to ensure chronological order
    guest_data = guest_data.sort_values("Timestamp (s)").reset_index(drop=True)
    
    # Extract Guest_ID from the data
    if "Guest_ID" in guest_data.columns:
        guest_id = guest_data["Guest_ID"].iloc[0]  # Get the Guest_ID from the first row
    else:
        print(f"Warning: Guest_ID column not found in {node_folder}, using default value 1")
        guest_id = 1
    
    print(f"Processing {len(guest_data)} samples using vectorized operations...")
    print(f"Guest ID: {guest_id}")
    
    # Create state mapping vectorized
    guest_data["New_State"] = guest_data["Running_State"].apply(lambda x: map_state(x, 0))
    
    # Create period keys vectorized - process each Running_State as one period
    guest_data["Period_Key"] = guest_data["Running_State"]
    
    # Find period boundaries (where Period_Key changes)
    guest_data["Period_Change"] = guest_data["Period_Key"] != guest_data["Period_Key"].shift(1)
    guest_data.loc[0, "Period_Change"] = True  # First row is always a new period
    
    # Get period boundaries
    period_starts = guest_data[guest_data["Period_Change"]].index.tolist()
    period_ends = period_starts[1:] + [len(guest_data)]
    
    print(f"Found {len(period_starts)} periods")
    
    # Process each period independently (no combining), while not splitting State 3 by Comm_Mode
    final_rows = []
    for start_idx, end_idx in tqdm(list(zip(period_starts, period_ends)), desc="Processing periods"):
        period_data = guest_data.iloc[start_idx:end_idx].copy()
        period_data = period_data.sort_values("Timestamp (s)").reset_index(drop=True)
        running_state = period_data["Period_Key"].iloc[0]
        state = map_state(running_state, 0)
        
        # Duration and averages for this period
        duration = period_data["Timestamp (s)"].iloc[-1] - period_data["Timestamp (s)"].iloc[0]
        avg_current = period_data["Current (uA)"].mean()
        avg_voltage = period_data["Voltage (mV)"].mean()
        avg_tx_power = period_data["Tx_Power"].mean()
        avg_adv_interval = period_data["Adv_Interval"].mean()
        avg_conn_interval = period_data["Conn_Interval"].mean()
        total_energy = avg_current * avg_voltage * duration / 1000
        
        if state == 3:
            # Split State 3 by Comm_Mode
            print(f"\n  Period {start_idx}-{end_idx}: State 3, Period duration={duration:.2f}s")
            
            # Process each Comm_Mode separately
            for comm_mode_val in sorted(period_data["Comm_Mode"].unique()):
                comm_mode_data = period_data[period_data["Comm_Mode"] == comm_mode_val]
                
                if len(comm_mode_data) == 0:
                    continue
                
                # Calculate duration for this Comm_Mode
                comm_mode_duration = comm_mode_data["Timestamp (s)"].iloc[-1] - comm_mode_data["Timestamp (s)"].iloc[0]
                
                # Detect VLC/BLE chunks within this Comm_Mode
                vlc_mask = comm_mode_data["VLC_Comm_Vol"] > 0
                ble_mask = comm_mode_data["BLE_Comm_Vol"] > 0
                vlc_chunks = find_communication_chunks(comm_mode_data, vlc_mask)
                ble_chunks = find_communication_chunks(comm_mode_data, ble_mask)
                
                # Idle within this Comm_Mode
                idle_mask = (comm_mode_data["VLC_Comm_Vol"] == 0) & (comm_mode_data["BLE_Comm_Vol"] == 0)
                idle_data = comm_mode_data[idle_mask]
                
                # Debug output
                print(f"    Comm_Mode {comm_mode_val}:")
                print(f"      VLC: {len(vlc_chunks)} chunks, {vlc_mask.sum()} samples")
                for i, chunk in enumerate(vlc_chunks):
                    print(f"        Chunk {i+1}: {chunk['start_time']:.2f}s to {chunk['end_time']:.2f}s, duration={chunk['end_time']-chunk['start_time']:.2f}s")
                print(f"      BLE: {len(ble_chunks)} chunks, {ble_mask.sum()} samples")
                for i, chunk in enumerate(ble_chunks):
                    print(f"        Chunk {i+1}: {chunk['start_time']:.2f}s to {chunk['end_time']:.2f}s, duration={chunk['end_time']-chunk['start_time']:.2f}s")
                print(f"      Idle: {len(idle_data)} samples")
                
                # VLC row for this Comm_Mode
                if len(vlc_chunks) > 0:
                    vlc_duration = sum(c["end_time"] - c["start_time"] for c in vlc_chunks)
                    # Calculate average current from actual VLC samples
                    vlc_samples = comm_mode_data[vlc_mask]
                    vlc_avg_current = vlc_samples["Current (uA)"].mean()
                    vlc_payload = len(vlc_chunks) * 24
                    vlc_energy = vlc_avg_current * avg_voltage * vlc_duration / 1000
                    final_rows.append({
                        "Number": config_number,
                        "Guest ID": guest_id,
                        "State": state,
                        "Duration": vlc_duration,
                        "Current": vlc_avg_current,
                        "Voltage": avg_voltage,
                        "TxPower": avg_tx_power,
                        "AdvInterval": avg_adv_interval,
                        "ConnInterval": avg_conn_interval,
                        "Comm_Mode": comm_mode_val,
                        "VLC Payload": vlc_payload,
                        "BLE Payload": 0,
                        "Total Energy": vlc_energy
                    })
                
                # BLE row for this Comm_Mode
                if len(ble_chunks) > 0:
                    ble_duration = sum(c["end_time"] - c["start_time"] for c in ble_chunks)
                    # Calculate average current from actual BLE samples
                    ble_samples = comm_mode_data[ble_mask]
                    ble_avg_current = ble_samples["Current (uA)"].mean()
                    ble_payload = len(ble_chunks) * 31
                    ble_energy = ble_avg_current * avg_voltage * ble_duration / 1000
                    final_rows.append({
                        "Number": config_number,
                        "Guest ID": guest_id,
                        "State": state,
                        "Duration": ble_duration,
                        "Current": ble_avg_current,
                        "Voltage": avg_voltage,
                        "TxPower": avg_tx_power,
                        "AdvInterval": avg_adv_interval,
                        "ConnInterval": avg_conn_interval,
                        "Comm_Mode": comm_mode_val,
                        "VLC Payload": 0,
                        "BLE Payload": ble_payload,
                        "Total Energy": ble_energy
                    })
                
                # Idle row for this Comm_Mode - calculate as remaining duration after VLC and BLE
                if len(idle_data) > 0:
                    # Calculate VLC and BLE total durations for this Comm_Mode
                    vlc_total_duration = sum(c["end_time"] - c["start_time"] for c in vlc_chunks) if len(vlc_chunks) > 0 else 0
                    ble_total_duration = sum(c["end_time"] - c["start_time"] for c in ble_chunks) if len(ble_chunks) > 0 else 0
                    
                    # Idle duration = Comm_Mode duration - VLC duration - BLE duration
                    idle_duration = comm_mode_duration - vlc_total_duration - ble_total_duration
                    
                    if idle_duration > 0:  # Only create idle row if there's actual idle time
                        idle_avg_current = idle_data["Current (uA)"].mean()
                        idle_energy = idle_avg_current * avg_voltage * idle_duration / 1000
                        final_rows.append({
                            "Number": config_number,
                            "Guest ID": guest_id,
                            "State": state,
                            "Duration": idle_duration,
                            "Current": idle_avg_current,
                            "Voltage": avg_voltage,
                            "TxPower": avg_tx_power,
                            "AdvInterval": avg_adv_interval,
                            "ConnInterval": avg_conn_interval,
                            "Comm_Mode": comm_mode_val,
                            "VLC Payload": 0,
                            "BLE Payload": 0,
                            "Total Energy": idle_energy
                        })
        else:
            # Non-State-3: single row for this period
            comm_mode = period_data["Comm_Mode"].mode().iloc[0] if len(period_data) > 0 else 0
            final_rows.append({
                "Number": config_number,
                "Guest ID": guest_id,
                "State": state,
                "Duration": duration,
                "Current": avg_current,
                "Voltage": avg_voltage,
                "TxPower": avg_tx_power,
                "AdvInterval": avg_adv_interval,
                "ConnInterval": avg_conn_interval,
                "Comm_Mode": comm_mode,
                "VLC Payload": 0,
                "BLE Payload": 0,
                "Total Energy": total_energy
            })
    
    return final_rows


def main():
    """Main function"""
    # Configuration parameters
    # Dataset base path - UPDATE THIS TO YOUR DATASET LOCATION
    BASE_DATASET_PATH = r"F:\Codes\Collect-datasets\PPK2-Visulization-Current-SuperIoT-V6\Dataset-all\Refined original dataset"
    
    # Alternative: Use relative path (commented out)
    # BASE_DATASET_PATH = "Dataset"
    
    # CONFIGS_TO_PROCESS = [1, 2, 3]  # Choose how many folders to process, like [1,2] or [1,2,3...]
    CONFIGS_TO_PROCESS = list(range(1, 60))  # Process all 27 configurations (1 to 27)
    OUTPUT_FILENAME = "processed_data.csv"
    
    print("=== Chronological Data Preprocessing ===")
    print(f"Base dataset path: {BASE_DATASET_PATH}")
    print(f"Configurations to process: {CONFIGS_TO_PROCESS}")
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
        # config_number comes from folder name (e.g., Dataset/1 -> Number=1)
        config_path = os.path.join(BASE_DATASET_PATH, str(config_number))
        
        # Check if configuration folder exists
        if not os.path.exists(config_path):
            print(f"Configuration {config_number} not found at {config_path}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing Configuration {config_number}")
        print(f"{'='*60}")
        
        # Find all node folders in this configuration
        node_folders = [d for d in os.listdir(config_path) 
                       if os.path.isdir(os.path.join(config_path, d)) and 'Node' in d]
        node_folders.sort()
        
        if not node_folders:
            print(f"No node folders found in {config_path}")
            continue
        
        print(f"Found {len(node_folders)} node folders: {node_folders}")
        
        # Process each node
        for node_folder in tqdm(node_folders, desc=f"Config {config_number} nodes"):
            node_path = os.path.join(config_path, node_folder)
            
            try:
                node_rows = process_node_data_chronological(config_number, node_folder, node_path)
                if node_rows:
                    all_rows.extend(node_rows)
                    print(f"  {node_folder}: {len(node_rows)} rows processed")
            except Exception as e:
                print(f"  Error processing {node_folder}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save all results
    if not all_rows:
        print("\nNo data processed!")
        return
    
    try:
        # Create final DataFrame
        final_df = pd.DataFrame(all_rows)
        
        # Force Comm_Mode=0 for State 1
        final_df.loc[final_df["State"] == 1, "Comm_Mode"] = 0
        
        # Save to CSV
        final_df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to {OUTPUT_FILENAME}")
        print(f"Total rows: {len(final_df)}")
        print(f"Total configurations processed: {len(CONFIGS_TO_PROCESS)}")
        
        # Display summary
        print("\n=== Summary by Configuration ===")
        summary = final_df.groupby("Number").size()
        for config_num, count in summary.items():
            print(f"  Configuration {config_num}: {count} rows")
        
        # Verify duration totals
        total_duration = final_df["Duration"].sum()
        print(f"\nTotal duration across all data: {total_duration:.2f} seconds")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()