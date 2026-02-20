"""
Calculate Average Time per Byte for VLC and BLE Transmissions

This script analyzes processed_data.csv to determine the average transmission
time per byte for both VLC and BLE communications.
"""

import pandas as pd
import numpy as np

print("="*80)
print("CALCULATE AVERAGE TIME PER BYTE FOR VLC AND BLE TRANSMISSIONS")
print("="*80)

# Read the data
print("\n1. Reading processed_data.csv...")
df = pd.read_csv('processed_data.csv')
print(f"   Total rows: {len(df)}")
print(f"   Columns: {df.columns.tolist()}")

# Filter data with VLC transmission
print("\n2. Analyzing VLC Transmissions...")
vlc_data = df[df['VLC Payload'] > 0].copy()
print(f"   Rows with VLC transmission: {len(vlc_data)}")

if len(vlc_data) > 0:
    # Calculate time per byte for each VLC transmission
    vlc_data['Time_per_VLC_Byte'] = vlc_data['Duration'] / vlc_data['VLC Payload']
    
    # Statistics
    avg_vlc_time = vlc_data['Time_per_VLC_Byte'].mean()
    median_vlc_time = vlc_data['Time_per_VLC_Byte'].median()
    std_vlc_time = vlc_data['Time_per_VLC_Byte'].std()
    min_vlc_time = vlc_data['Time_per_VLC_Byte'].min()
    max_vlc_time = vlc_data['Time_per_VLC_Byte'].max()
    
    print(f"\n   VLC Time per Byte Statistics (ms/byte):")
    print(f"     Mean:     {avg_vlc_time:.4f} ms/byte")
    print(f"     Median:   {median_vlc_time:.4f} ms/byte")
    print(f"     Std Dev:  {std_vlc_time:.4f} ms/byte")
    print(f"     Min:      {min_vlc_time:.4f} ms/byte")
    print(f"     Max:      {max_vlc_time:.4f} ms/byte")
    
    # Show distribution by State
    print(f"\n   VLC Transmission by State:")
    for state in sorted(vlc_data['State'].unique()):
        state_vlc = vlc_data[vlc_data['State'] == state]
        state_avg = state_vlc['Time_per_VLC_Byte'].mean()
        print(f"     State {state}: {len(state_vlc)} rows, avg {state_avg:.4f} ms/byte")
    
    # Show sample data
    print(f"\n   Sample VLC Transmissions:")
    sample_cols = ['State', 'Duration', 'VLC Payload', 'Time_per_VLC_Byte']
    print(vlc_data[sample_cols].head(10).to_string(index=False))
else:
    print("   No VLC transmissions found in dataset")

# Filter data with BLE transmission
print("\n3. Analyzing BLE Transmissions...")
ble_data = df[df['BLE Payload'] > 0].copy()
print(f"   Rows with BLE transmission: {len(ble_data)}")

if len(ble_data) > 0:
    # Calculate time per byte for each BLE transmission
    ble_data['Time_per_BLE_Byte'] = ble_data['Duration'] / ble_data['BLE Payload']
    
    # Statistics
    avg_ble_time = ble_data['Time_per_BLE_Byte'].mean()
    median_ble_time = ble_data['Time_per_BLE_Byte'].median()
    std_ble_time = ble_data['Time_per_BLE_Byte'].std()
    min_ble_time = ble_data['Time_per_BLE_Byte'].min()
    max_ble_time = ble_data['Time_per_BLE_Byte'].max()
    
    print(f"\n   BLE Time per Byte Statistics (ms/byte):")
    print(f"     Mean:     {avg_ble_time:.4f} ms/byte")
    print(f"     Median:   {median_ble_time:.4f} ms/byte")
    print(f"     Std Dev:  {std_ble_time:.4f} ms/byte")
    print(f"     Min:      {min_ble_time:.4f} ms/byte")
    print(f"     Max:      {max_ble_time:.4f} ms/byte")
    
    # Show distribution by State
    print(f"\n   BLE Transmission by State:")
    for state in sorted(ble_data['State'].unique()):
        state_ble = ble_data[ble_data['State'] == state]
        state_avg = state_ble['Time_per_BLE_Byte'].mean()
        print(f"     State {state}: {len(state_ble)} rows, avg {state_avg:.4f} ms/byte")
    
    # Show sample data
    print(f"\n   Sample BLE Transmissions:")
    sample_cols = ['State', 'Duration', 'BLE Payload', 'Time_per_BLE_Byte']
    print(ble_data[sample_cols].head(10).to_string(index=False))
else:
    print("   No BLE transmissions found in dataset")

# Check for rows with both VLC and BLE
print("\n4. Analyzing Simultaneous VLC and BLE Transmissions...")
both_data = df[(df['VLC Payload'] > 0) & (df['BLE Payload'] > 0)]
print(f"   Rows with both VLC and BLE: {len(both_data)}")

if len(both_data) > 0:
    print("\n   WARNING: Found rows with both VLC and BLE payload!")
    print("   These rows need special handling as duration is shared.")
    print(both_data[['State', 'Duration', 'VLC Payload', 'BLE Payload']].head(10).to_string(index=False))

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if len(vlc_data) > 0:
    print(f"\n[OK] VLC Transmission Analysis:")
    print(f"  - Average time per byte: {avg_vlc_time:.4f} ms/byte")
    print(f"  - Median time per byte: {median_vlc_time:.4f} ms/byte")
    print(f"  - Recommended default: {avg_vlc_time:.3f} ms/byte")
    print(f"  - Total VLC transmission rows: {len(vlc_data)}")
else:
    print(f"\n[X] No VLC transmissions found")

if len(ble_data) > 0:
    print(f"\n[OK] BLE Transmission Analysis:")
    print(f"  - Average time per byte: {avg_ble_time:.4f} ms/byte")
    print(f"  - Median time per byte: {median_ble_time:.4f} ms/byte")
    print(f"  - Recommended default: {avg_ble_time:.3f} ms/byte")
    print(f"  - Total BLE transmission rows: {len(ble_data)}")
else:
    print(f"\n[X] No BLE transmissions found")

print("\n" + "="*80)
print("RECOMMENDATIONS FOR predict_app.py")
print("="*80)

if len(vlc_data) > 0 and len(ble_data) > 0:
    print(f"\nUpdate the default values in predict_app.py:")
    print(f"  - time_per_vlc_byte: {avg_vlc_time:.3f} ms/byte (currently 2.0)")
    print(f"  - time_per_ble_byte: {avg_ble_time:.3f} ms/byte (currently 4.0)")
    print(f"\nThese values are based on actual transmission data from {len(vlc_data)} VLC")
    print(f"and {len(ble_data)} BLE transmission records in the dataset.")
    print(f"\nNote: The current defaults (2.0 and 4.0 ms/byte) are much higher than")
    print(f"the measured values. Consider updating to match real data or keeping")
    print(f"higher values for conservative energy estimates.")

print("\n" + "="*80)

