"""
Fix Comm_Mode in processed_data.csv:
Set Comm_Mode=2 when VLC Payload > 0 OR BLE Payload > 0

This ensures that data transmission (VLC or BLE) requires BLE connection (Comm_Mode=2).
"""
import pandas as pd
import shutil
from datetime import datetime

print("="*80)
print("FIX COMM_MODE FOR NON-ZERO PAYLOAD DATA")
print("="*80)

# Read the original data
print("\n1. Reading processed_data.csv...")
df = pd.read_csv('processed_data.csv')
print(f"   Total rows: {len(df)}")

# Create backup with timestamp
backup_filename = f'processed_data_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
print(f"\n2. Creating backup: {backup_filename}")
shutil.copy('processed_data.csv', backup_filename)
print(f"   ✓ Backup created")

# Show current state
print("\n3. Analyzing current data...")
print("\n   Current Comm_Mode distribution:")
print(df['Comm_Mode'].value_counts().sort_index())

# Identify rows that need fixing
mask_nonzero_payload = (df['VLC Payload'] > 0) | (df['BLE Payload'] > 0)
rows_to_fix = mask_nonzero_payload & (df['Comm_Mode'] != 2)

print(f"\n   Rows with non-zero payload: {mask_nonzero_payload.sum()}")
print(f"   Rows needing fix (non-zero payload AND Comm_Mode != 2): {rows_to_fix.sum()}")

# Show breakdown by state
if rows_to_fix.sum() > 0:
    print("\n   Breakdown by State:")
    for state in sorted(df[rows_to_fix]['State'].unique()):
        state_count = ((df['State'] == state) & rows_to_fix).sum()
        print(f"     State {state}: {state_count} rows")
    
    print("\n   Sample of rows to be fixed:")
    sample_df = df[rows_to_fix][['State', 'VLC Payload', 'BLE Payload', 'Comm_Mode', 'Current']].head(5)
    print(sample_df.to_string(index=False))

# Apply the fix
print(f"\n4. Applying fix: Setting Comm_Mode=2 for {rows_to_fix.sum()} rows...")
df.loc[mask_nonzero_payload, 'Comm_Mode'] = 2
print("   ✓ Fix applied")

# Verify the fix
print("\n5. Verification:")
verify_mask = (df['VLC Payload'] > 0) | (df['BLE Payload'] > 0)
comm_mode_not_two = df[verify_mask]['Comm_Mode'].ne(2).sum()

if comm_mode_not_two == 0:
    print("   ✓ All rows with non-zero payload now have Comm_Mode=2")
else:
    print(f"   ✗ WARNING: {comm_mode_not_two} rows still have Comm_Mode != 2")

print("\n   Updated Comm_Mode distribution:")
print(df['Comm_Mode'].value_counts().sort_index())

# Show final statistics
print("\n6. Final Statistics:")
print("\n   Zero Payload (Pure Comm_Mode data):")
zero_payload = df[(df['VLC Payload'] == 0) & (df['BLE Payload'] == 0)]
print(f"     Total rows: {len(zero_payload)}")
print(f"     Comm_Mode distribution:")
print(zero_payload['Comm_Mode'].value_counts().sort_index())

print("\n   Non-Zero Payload (Now all Comm_Mode=2):")
nonzero_payload = df[(df['VLC Payload'] > 0) | (df['BLE Payload'] > 0)]
print(f"     Total rows: {len(nonzero_payload)}")
print(f"     Comm_Mode distribution:")
print(nonzero_payload['Comm_Mode'].value_counts().sort_index())

# Save the fixed data
print("\n7. Saving fixed data to processed_data.csv...")
df.to_csv('processed_data.csv', index=False)
print("   ✓ Saved")

print("\n   Final Comm_Mode distribution:")
print(df['Comm_Mode'].value_counts().sort_index())

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Set Comm_Mode=2 for non-zero payload ({rows_to_fix.sum()} rows)")
print(f"✓ Backup saved as: {backup_filename}")
print(f"✓ Updated data saved to: processed_data.csv")
print("\nFinal Comm_Mode logic:")
print("  - When VLC Payload > 0 OR BLE Payload > 0 → Comm_Mode=2 (Connected)")
print("  - This reflects reality: data transmission requires BLE connection")
print("\nRationale:")
print("  - VLC and BLE transmissions occur when device is connected (Comm_Mode=2)")
print("  - Zero-payload cases can have any Comm_Mode (0, 1, or 2)")
print("  - The model will learn that transmission always happens in connected state")
print("\nNext steps:")
print("  1. Retrain the model: python train_model.py --cross-validation")
print("  2. The model will now correctly associate payloads with Comm_Mode=2")
print("  3. Re-run predictions with updated model")
print("="*80)

