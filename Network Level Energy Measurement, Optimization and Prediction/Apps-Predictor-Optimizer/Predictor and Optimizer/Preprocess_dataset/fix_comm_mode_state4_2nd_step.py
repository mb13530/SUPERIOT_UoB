"""
Fix Comm_Mode for State 4
Sets Comm_Mode=0 for all State=4 rows in processed_data.csv

State 4 (Deep Sleep) should always have Comm_Mode=0 (no communication)
"""

import pandas as pd
import shutil
from datetime import datetime

def fix_comm_mode_state4(input_file='processed_data.csv', create_backup=True):
    """
    Fix Comm_Mode values for State 4
    
    Args:
        input_file: Path to processed_data.csv
        create_backup: If True, creates backup before modifying
    """
    print("="*80)
    print("FIX COMM_MODE FOR STATE 4")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} rows")
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found!")
        return
    
    # Check current state
    print("\n" + "-"*80)
    print("BEFORE FIXING:")
    print("-"*80)
    
    state4_rows = df[df['State'] == 4]
    print(f"Total State=4 rows: {len(state4_rows)}")
    
    if len(state4_rows) > 0:
        comm_mode_counts = state4_rows['Comm_Mode'].value_counts().sort_index()
        print(f"\nComm_Mode distribution in State=4:")
        for mode, count in comm_mode_counts.items():
            print(f"  Comm_Mode={mode}: {count} rows")
        
        # Count how many need fixing
        needs_fix = len(state4_rows[state4_rows['Comm_Mode'] != 0])
        print(f"\n⚠️  Rows needing fix (Comm_Mode != 0): {needs_fix}")
    else:
        print("No State=4 rows found in dataset.")
        return
    
    # Create backup if requested
    if create_backup and needs_fix > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{input_file}.backup_{timestamp}"
        print(f"\n📦 Creating backup: {backup_file}")
        shutil.copy2(input_file, backup_file)
        print(f"✓ Backup created")
    
    # Fix the data
    if needs_fix > 0:
        print(f"\n🔧 Fixing {needs_fix} rows...")
        df.loc[df['State'] == 4, 'Comm_Mode'] = 0
        print("✓ Fixed: Set Comm_Mode=0 for all State=4 rows")
    else:
        print("\n✓ No fixes needed - all State=4 rows already have Comm_Mode=0")
        return
    
    # Verify fix
    print("\n" + "-"*80)
    print("AFTER FIXING:")
    print("-"*80)
    
    state4_rows_fixed = df[df['State'] == 4]
    comm_mode_counts_fixed = state4_rows_fixed['Comm_Mode'].value_counts().sort_index()
    print(f"Comm_Mode distribution in State=4:")
    for mode, count in comm_mode_counts_fixed.items():
        print(f"  Comm_Mode={mode}: {count} rows")
    
    non_zero = len(state4_rows_fixed[state4_rows_fixed['Comm_Mode'] != 0])
    if non_zero == 0:
        print(f"\n✅ All State=4 rows now have Comm_Mode=0")
    else:
        print(f"\n⚠️  Warning: {non_zero} State=4 rows still have Comm_Mode != 0")
    
    # Save fixed data
    print(f"\n💾 Saving fixed data to: {input_file}")
    df.to_csv(input_file, index=False)
    print("✓ Data saved successfully")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total rows: {len(df)}")
    print(f"State=4 rows: {len(state4_rows_fixed)}")
    print(f"Rows fixed: {needs_fix}")
    print(f"✅ All State=4 rows now have Comm_Mode=0")
    print("="*80)


def verify_all_states(input_file='processed_data.csv'):
    """
    Verify Comm_Mode distribution across all states
    """
    print("\n" + "="*80)
    print("VERIFICATION: COMM_MODE DISTRIBUTION BY STATE")
    print("="*80)
    
    df = pd.read_csv(input_file)
    
    for state in sorted(df['State'].unique()):
        state_data = df[df['State'] == state]
        comm_counts = state_data['Comm_Mode'].value_counts().sort_index()
        
        print(f"\nState {state} ({len(state_data)} rows):")
        for mode, count in comm_counts.items():
            pct = (count / len(state_data)) * 100
            print(f"  Comm_Mode={mode}: {count:5d} rows ({pct:5.1f}%)")


if __name__ == "__main__":
    # Fix Comm_Mode for State 4
    fix_comm_mode_state4(
        input_file='processed_data.csv',
        create_backup=True
    )
    
    # Verify all states
    verify_all_states('processed_data.csv')
    
    print("\n✅ Done!")

