#!/usr/bin/env python3
"""
Enhanced plot_saved_raw.py with PPK2-Python.py exact canvas reproduction.

NEW FEATURES:
- Handles multiple Excel-split CSV files automatically
- Exact guest app canvas reproduction (colors, style, background)
- Statistical binning (PPK2-Python.py dark blue curve)
- Min/Max envelope plotting
- Automatic file detection and merging
- Same visual appearance as guest app canvas
- IR/BLE duration rectangles: LIGHT RED rectangles for IR (running_mode=2), LIGHT GREEN rectangles for BLE (running_mode=1)
- Outlier filtering: Automatically detects and removes extremely high current spikes (10x+ surrounding values)
- Advanced IR/BLE detection: Analyzes current patterns to refine communication period timing

Usage:
  python plot_saved_raw_enhanced.py --file measurement.csv --mode guest_app
  python plot_saved_raw_enhanced.py --file measurement_part1.csv --mode guest_app  # Auto-finds all parts
  python plot_saved_raw_enhanced.py --file measurement.csv --mode envelope --bins 500
  python plot_saved_raw_enhanced.py --file measurement.csv --mode linear  # Original method
  python plot_saved_raw_enhanced.py --file measurement.csv --no-filter  # Disable outlier filtering
  python plot_saved_raw_enhanced.py --file measurement.csv --outlier-factor 5 --outlier-window 100  # Custom filter settings
  python plot_saved_raw_enhanced.py --file measurement.csv --advanced-detection  # Use current pattern analysis
  python plot_saved_raw_enhanced.py --file measurement.csv --simple-detection    # Use parameter-based detection (default)
"""

import argparse
import os
import csv
import sys
import math
from typing import List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Prefer an interactive backend; fallback to Agg if none available
def _use_interactive_backend() -> bool:
    for candidate in ("TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg", "WXAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return True
        except Exception:
            continue
    matplotlib.use("Agg", force=True)
    return False


_HAS_INTERACTIVE = _use_interactive_backend()

# ----- Enhanced Configuration -----
HARDCODED_USE = True
HARDCODED_FILE = "latest"  # "latest" finds most recent guest_data_*.csv automatically
HARDCODED_MODE = "guest_app"  # "guest_app", "linear", "statistical", "envelope"
HARDCODED_BINS = None  # Use adaptive binning (like guest app)
HARDCODED_SAVE_PNG = "guest_app_canvas_plot.png"  # Default: Always save PNG
HARDCODED_SAVE_PDF = "guest_app_canvas_plot.pdf"  # Default: Always save PDF
HARDCODED_SAVE_SVG = ""  # Default: No SVG (empty string disables)
SAVE_DIR = "Plot_saved_raw"

# Running State Visualization Configuration
# Note: Running state timeline is now controlled by --plot-running-state / --no-plot-running-state arguments
# Default: Enabled (use --no-plot-running-state to disable)
RUNNING_STATE_HEIGHT_RATIO = 0.15  # Height ratio for running state subplot (15% of plot height)

# GUEST APP CANVAS COLORS (exact match from PPK2-Python.py)
GUEST_APP_COLORS = {
    'background': 'white',  # White background (setBackground('w'))
    'grid': (0, 0, 0, 0.3),  # Grid lines (showGrid alpha=0.3)
    'light_blue': (33, 150, 243),  # Light blue envelope (RGB 33, 150, 243)
    'dark_blue': 'b',  # Dark blue mean curve (pen='b')
    'text': 'black',  # Black text
    'envelope_alpha': 60,  # Envelope transparency (alpha=60)
    'line_width': 1,  # Line thickness (width=1)
}


class DataAccumulator:
    """PPK2-Python.py statistical binning methods for smooth curves"""
    
    @staticmethod
    def bin_min_max_mean(times: np.ndarray, values: np.ndarray, t0: float, t1: float, bins: int):
        """
        Statistical binning method from PPK2-Python.py
        Creates smooth min/max envelope and mean curve (dark blue curve method)
        """
        if bins <= 0 or t1 <= t0 or times.size == 0:
            return (np.array([], dtype=np.float64),) * 5
        
        # Clip range to available data
        t0 = max(t0, float(times[0]))
        t1 = min(t1, float(times[-1]))
        if t1 <= t0:
            return (np.array([], dtype=np.float64),) * 5
        
        # Create uniform time bins
        edges = np.linspace(t0, t1, bins + 1)
        centers = (edges[:-1] + edges[1:]) * 0.5
        idx = np.searchsorted(times, edges)
        
        # Initialize statistics arrays
        mins = np.full(bins, np.nan)
        maxs = np.full(bins, np.nan)
        sums = np.zeros(bins, dtype=np.float64)
        counts = np.zeros(bins, dtype=np.int64)
        
        # Calculate statistics for each bin
        left = idx[0]
        for b in range(bins):
            right = idx[b + 1]
            if right > left:
                seg = values[left:right]
                if seg.size:
                    mins[b] = float(np.nanmin(seg))
                    maxs[b] = float(np.nanmax(seg))
                    sums[b] = float(np.nansum(seg))
                    counts[b] = int(np.count_nonzero(~np.isnan(seg)))
            left = right
        
        # Calculate means
        means = np.full(bins, np.nan)
        valid = counts > 0
        means[valid] = sums[valid] / counts[valid]
        
        return centers[valid], mins[valid], maxs[valid], means[valid], counts[valid]
    
    @staticmethod
    def mean_per_bin(times: np.ndarray, values: np.ndarray, target_hz: float):
        """
        Frequency-based binning for smooth mean curve
        """
        if target_hz <= 0 or times.size == 0:
            return times.copy(), values.copy()
        
        t0, t1 = float(times[0]), float(times[-1])
        dur = max(1e-9, t1 - t0)
        bins = int(max(1, math.floor(dur * target_hz)))
        
        centers, _, _, means, valid_counts = DataAccumulator.bin_min_max_mean(times, values, t0, t1, bins)
        if centers.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        
        return centers, means


def find_all_csv_parts(base_path: str) -> List[str]:
    """
    Find all CSV parts for multi-file Excel splits with smart guest app filename detection.
    
    Examples:
    - measurement.csv → [measurement.csv] (single file)
    - measurement_part1.csv → [measurement_part1.csv, measurement_part2.csv, ...] (multiple parts)
    - guest_data_20241203_143022.csv → [guest_data_20241203_143022.csv] (guest app single)
    - guest_data_20241203_143022_part1.csv → [guest_data_20241203_143022_part1.csv, ...] (guest app multi)
    - "latest" → finds most recent guest_data_*.csv file
    """
    # Special case: "latest" finds most recent guest app file
    if base_path.lower() == "latest":
        return find_latest_guest_files()
    
    if not os.path.exists(base_path):
        # Try to find part files
        base_name = base_path.rsplit('.', 1)[0]
        part1_path = f"{base_name}_part1.csv"
        
        if os.path.exists(part1_path):
            print(f"🔍 Auto-detected multi-part files starting with: {part1_path}")
            # Find all parts
            parts = []
            part_num = 1
            while True:
                part_path = f"{base_name}_part{part_num}.csv"
                if os.path.exists(part_path):
                    parts.append(part_path)
                    part_num += 1
                else:
                    break
            return parts
        else:
            # Try guest app filename pattern search
            guest_files = find_guest_app_files(base_path)
            if guest_files:
                return guest_files
            else:
                raise FileNotFoundError(f"CSV file not found: {base_path}\n"
                                      f"Tried: {base_path}, {part1_path}, guest_data_* patterns")
    else:
        # Single file exists
        return [base_path]


def find_latest_guest_files() -> List[str]:
    """Find the most recent guest_data_*.csv file(s) in current directory"""
    import glob
    
    # Find all guest app files
    pattern = "guest_data_*.csv"
    all_files = glob.glob(pattern)
    
    if not all_files:
        raise FileNotFoundError("No guest_data_*.csv files found in current directory")
    
    # Group files by base name (without part suffix)
    file_groups = {}
    for file_path in all_files:
        # Remove .csv extension
        name_without_ext = file_path.rsplit('.', 1)[0]
        
        # Check if it's already a part file
        if '_part' in name_without_ext:
            # Extract base name (remove _partN suffix)
            base_name = name_without_ext.rsplit('_part', 1)[0]
        else:
            base_name = name_without_ext
            
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(file_path)
    
    # Sort groups by base name (timestamp) to get most recent
    latest_base_name = sorted(file_groups.keys(), reverse=True)[0]
    latest_files = sorted(file_groups[latest_base_name])  # Sort parts in order
    
    print(f"🔍 Latest guest file group: {latest_base_name} ({len(latest_files)} file(s))")
    
    # Check if we have multiple parts
    if len(latest_files) > 1:
        print(f"📁 Multi-part detected: {[os.path.basename(f) for f in latest_files]}")
        return latest_files
    
    # Single file - check if it has additional parts
    latest_base = latest_files[0]
    base_name = latest_base.rsplit('.', 1)[0]
    
    # If it's not already a part file, look for part1
    if '_part' not in base_name:
        part1_path = f"{base_name}_part1.csv"
        
        if os.path.exists(part1_path):
            # Multi-part file
            parts = []
            part_num = 1
            while True:
                part_path = f"{base_name}_part{part_num}.csv"
                if os.path.exists(part_path):
                    parts.append(part_path)
                    part_num += 1
                else:
                    break
            print(f"📁 Found {len(parts)} parts of latest guest data")
            return parts
    
    # Single file (either no parts found, or already a part file)
    print(f"📄 Single file: {os.path.basename(latest_base)}")
    return [latest_base]


def find_guest_app_files(partial_name: str) -> List[str]:
    """
    Find guest app files by partial name matching.
    
    Examples:
    - "guest_data_20241203" → finds guest_data_20241203_*.csv
    - "20241203_143022" → finds guest_data_20241203_143022.csv
    """
    import glob
    
    # Try different patterns
    patterns = [
        f"{partial_name}*.csv",
        f"guest_data_{partial_name}*.csv",
        f"*{partial_name}*.csv"
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            # Sort to get consistent ordering
            files.sort()
            base_file = files[0]
            
            print(f"🔍 Found guest file by pattern '{pattern}': {base_file}")
            
            # Check for parts
            base_name = base_file.rsplit('.', 1)[0]
            part1_path = f"{base_name}_part1.csv"
            
            if os.path.exists(part1_path):
                parts = []
                part_num = 1
                while True:
                    part_path = f"{base_name}_part{part_num}.csv"
                    if os.path.exists(part_path):
                        parts.append(part_path)
                        part_num += 1
                    else:
                        break
                return parts
            else:
                return [base_file]
    
    return []


def read_time_current_multiple(csv_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read and combine multiple CSV files (for Excel-split data).
    Returns combined time, current, running_state, running_mode, and comm_mode arrays in chronological order.
    """
    all_times = []
    all_currents = []
    all_running_states = []
    all_running_modes = []
    all_comm_modes = []
    
    print(f"📊 Loading {len(csv_paths)} CSV file(s)...")
    
    for i, csv_path in enumerate(csv_paths):
        print(f"   📄 Loading part {i + 1}/{len(csv_paths)}: {os.path.basename(csv_path)}")
        
        times, currents, running_states, running_modes, comm_modes = read_time_current_single(csv_path)
        all_times.extend(times.tolist())
        all_currents.extend(currents.tolist())
        all_running_states.extend(running_states.tolist())
        all_running_modes.extend(running_modes.tolist())
        all_comm_modes.extend(comm_modes.tolist())
        
        print(f"      ✅ Loaded {len(times):,} samples")
    
    # Convert to arrays and sort by time (ensure chronological order)
    combined_times = np.array(all_times, dtype=np.float64)
    combined_currents = np.array(all_currents, dtype=np.float64)
    combined_running_states = np.array(all_running_states, dtype=np.int32)
    combined_running_modes = np.array(all_running_modes, dtype=np.int32)
    combined_comm_modes = np.array(all_comm_modes, dtype=np.int32)
    
    # Sort by time to handle any out-of-order data
    sort_indices = np.argsort(combined_times)
    combined_times = combined_times[sort_indices]
    combined_currents = combined_currents[sort_indices]
    combined_running_states = combined_running_states[sort_indices]
    combined_running_modes = combined_running_modes[sort_indices]
    combined_comm_modes = combined_comm_modes[sort_indices]
    
    print(f"✅ Combined total: {len(combined_times):,} samples")
    print(f"⏱️  Time range: {combined_times[0]:.6f}s to {combined_times[-1]:.6f}s")
    print(f"⚡ Current range: {combined_currents.min():.2f} to {combined_currents.max():.2f} μA")
    
    return combined_times, combined_currents, combined_running_states, combined_running_modes, combined_comm_modes


def read_time_current_single(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read Time (s), Current (uA), Running_State, Running Mode, and Comm_Mode columns from a single CSV with headers.
    Returns: (times, currents, running_states, running_modes, comm_modes)
    """
    time_col = None
    current_col = None
    running_state_col = None
    running_mode_col = None
    comm_mode_col = None
    times: List[float] = []
    currents: List[float] = []
    running_states: List[int] = []
    running_modes: List[int] = []
    comm_modes: List[int] = []

    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV is empty: {csv_path}")

        # Normalize header names
        norm = [h.strip().lower() for h in header]

        for idx, name in enumerate(norm):
            if name in ("timestamp (s)", "time (s)", "time", "t(s)", "host_timestamp (s)"):
                time_col = idx
            if name in ("current (ua)", "current (μa)", "current", "i(ua)"):
                current_col = idx
            if name in ("running_state", "running state"):
                running_state_col = idx
            if name in ("running mode", "running_mode"):
                running_mode_col = idx
            if name in ("comm_mode", "comm mode"):
                comm_mode_col = idx

        if time_col is None or current_col is None:
            raise ValueError(f"CSV must contain time and current columns: {csv_path}")

        for row in reader:
            if not row or len(row) <= max(time_col, current_col):
                continue
            try:
                t = float(row[time_col])
                i = float(row[current_col])
                # Get running state if available, default to 0
                rs = 0
                if running_state_col is not None and running_state_col < len(row):
                    try:
                        rs = int(float(row[running_state_col]))
                    except ValueError:
                        rs = 0
                # Get running mode if available, default to 0
                rm = 0
                if running_mode_col is not None and running_mode_col < len(row):
                    try:
                        rm = int(float(row[running_mode_col]))
                    except ValueError:
                        rm = 0
                # Get comm mode if available, default to 0
                cm = 0
                if comm_mode_col is not None and comm_mode_col < len(row):
                    try:
                        cm = int(float(row[comm_mode_col]))
                    except ValueError:
                        cm = 0
            except ValueError:
                continue
            times.append(t)
            currents.append(i)
            running_states.append(rs)
            running_modes.append(rm)
            comm_modes.append(cm)

    if not times:
        raise ValueError(f"No numeric samples found in CSV: {csv_path}")

    return (np.asarray(times, dtype=np.float64),
            np.asarray(currents, dtype=np.float64),
            np.asarray(running_states, dtype=np.int32),
            np.asarray(running_modes, dtype=np.int32),
            np.asarray(comm_modes, dtype=np.int32))


def read_node_parameters(csv_path: str) -> dict:
    """Read node parameters (Guest_ID, Tx_Power, Adv_Interval, Conn_Interval) from CSV.
    Returns a dictionary with parameter values from the first data row.
    """
    params = {
        'guest_id': 'N/A',
        'tx_power': 'N/A',
        'adv_interval': 'N/A',
        'conn_interval': 'N/A'
    }
    
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return params
        
        # Normalize header names
        norm = [h.strip().lower() for h in header]
        
        # Find column indices
        guest_id_col = None
        tx_power_col = None
        adv_interval_col = None
        conn_interval_col = None
        
        for idx, name in enumerate(norm):
            if name in ("guest_id", "node_id", "node id"):
                guest_id_col = idx
            if name in ("tx_power", "tx power", "tx_power (dbm)"):
                tx_power_col = idx
            if name in ("adv_interval", "advertising_interval", "advertising interval"):
                adv_interval_col = idx
            if name in ("conn_interval", "connection_interval", "connection interval"):
                conn_interval_col = idx
        
        # Read first data row to get parameter values
        try:
            first_row = next(reader)
            if guest_id_col is not None and guest_id_col < len(first_row):
                try:
                    params['guest_id'] = str(int(float(first_row[guest_id_col])))
                except ValueError:
                    pass
            if tx_power_col is not None and tx_power_col < len(first_row):
                try:
                    params['tx_power'] = f"{int(float(first_row[tx_power_col]))} dBm"
                except ValueError:
                    pass
            if adv_interval_col is not None and adv_interval_col < len(first_row):
                try:
                    params['adv_interval'] = f"{int(float(first_row[adv_interval_col]))} ms"
                except ValueError:
                    pass
            if conn_interval_col is not None and conn_interval_col < len(first_row):
                try:
                    params['conn_interval'] = f"{int(float(first_row[conn_interval_col]))} ms"
                except ValueError:
                    pass
        except StopIteration:
            pass
    
    return params


def read_running_state_for_wakeup(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read Time (s), Current (uA), and Running_State columns specifically for wake-up detection."""
    time_col = None
    current_col = None
    running_state_col = None
    times: List[float] = []
    currents: List[float] = []
    running_states: List[int] = []

    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV is empty: {csv_path}")

        # Normalize header names
        norm = [h.strip().lower() for h in header]

        for idx, name in enumerate(norm):
            if name in ("timestamp (s)", "time (s)", "time", "t(s)", "host_timestamp (s)"):
                time_col = idx
            if name in ("current (ua)", "current (μa)", "current", "i(ua)"):
                current_col = idx
            if name in ("running_state", "running state"):
                running_state_col = idx

        if time_col is None or current_col is None or running_state_col is None:
            raise ValueError(f"CSV must contain time, current, and Running_State columns: {csv_path}")

        for row in reader:
            if not row or len(row) <= max(time_col, current_col, running_state_col):
                continue
            try:
                t = float(row[time_col])
                i = float(row[current_col])
                rs = int(float(row[running_state_col]))
            except ValueError:
                continue
            times.append(t)
            currents.append(i)
            running_states.append(rs)

    if not times:
        raise ValueError(f"No numeric samples found in CSV: {csv_path}")

    return (np.asarray(times, dtype=np.float64),
            np.asarray(currents, dtype=np.float64),
            np.asarray(running_states, dtype=np.int32))


def read_time_current(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Smart CSV reader that handles both single files and multi-part Excel splits.
    Automatically detects and combines multiple parts.
    Returns time, current, running_state, running_mode, and comm_mode arrays.
    """
    csv_parts = find_all_csv_parts(csv_path)
    
    if len(csv_parts) == 1:
        print(f"📄 Single file mode: {os.path.basename(csv_parts[0])}")
        return read_time_current_single(csv_parts[0])
    else:
        print(f"📁 Multi-file mode: Found {len(csv_parts)} parts")
        return read_time_current_multiple(csv_parts)


def filter_extreme_outliers(times, currents, modes=None, running_states=None, running_modes=None, comm_modes=None, outlier_factor=30, window_size=50):
    """
    Filter out extremely high current values that are significantly higher than surrounding values.

    Args:
        times: Time array
        currents: Current array
        modes: Running mode array (optional)
        running_states: Running state array (optional)
        running_modes: Running mode array (optional, separate from modes)
        comm_modes: Communication mode array (optional)
        outlier_factor: Factor for outlier detection (default: 10x surrounding values)
        window_size: Window size for calculating surrounding average (default: 50 samples)

    Returns:
        Tuple of (filtered_times, filtered_currents, filtered_modes, filtered_running_states, filtered_running_modes, filtered_comm_modes, outlier_info)
    """
    if len(currents) == 0:
        return times, currents, modes, running_states, running_modes, comm_modes, {"removed": 0, "total": 0}

    # Convert to numpy arrays for efficient processing
    times_arr = np.array(times)
    currents_arr = np.array(currents)
    modes_arr = np.array(modes) if modes is not None else None
    running_states_arr = np.array(running_states) if running_states is not None else None
    running_modes_arr = np.array(running_modes) if running_modes is not None else None
    comm_modes_arr = np.array(comm_modes) if comm_modes is not None else None

    # Fast statistical outlier detection (no rolling window needed)

    import time as time_module
    start_time = time_module.perf_counter()

    print(f"⚡ FAST OUTLIER FILTER: Processing {len(currents_arr):,} samples (factor: {outlier_factor}x)")

    # Calculate median and standard deviation efficiently
    sorted_currents = np.sort(currents_arr)
    n = len(sorted_currents)

    # Calculate median
    if n % 2 == 0:
        median = (sorted_currents[n // 2 - 1] + sorted_currents[n // 2]) / 2
    else:
        median = sorted_currents[n // 2]

    # Calculate standard deviation
    std_dev = np.std(currents_arr)

    # Set threshold - outliers are values much higher than median + factor*std
    threshold = median + (outlier_factor * std_dev)

    # Filter outliers in single vectorized operation
    valid_mask = currents_arr <= threshold

    # Apply filter
    filtered_times = times_arr[valid_mask]
    filtered_currents = currents_arr[valid_mask]
    filtered_modes = modes_arr[valid_mask] if modes_arr is not None else None
    filtered_running_states = running_states_arr[valid_mask] if running_states_arr is not None else None
    filtered_running_modes = running_modes_arr[valid_mask] if running_modes_arr is not None else None
    filtered_comm_modes = comm_modes_arr[valid_mask] if comm_modes_arr is not None else None

    # Statistics
    removed_count = np.sum(~valid_mask)
    total_count = len(currents_arr)
    kept_count = len(filtered_currents)

    elapsed_time = time_module.perf_counter() - start_time

    outlier_info = {
        "removed": removed_count,
        "total": total_count,
        "kept": kept_count,
        "percentage": (removed_count / total_count * 100) if total_count > 0 else 0,
        "processing_time": elapsed_time
    }

    if removed_count > 0:
        print(f"   🚫 Removed {removed_count:,} outliers ({outlier_info['percentage']:.2f}%) in {elapsed_time:.3f}s")
        print(f"   ✅ Kept {kept_count:,} samples")
    else:
        print(f"   ✅ No outliers detected in {elapsed_time:.3f}s")

    return filtered_times, filtered_currents, filtered_modes, filtered_running_states, filtered_running_modes, filtered_comm_modes, outlier_info


def detect_vlc_ble_from_current_patterns(times, currents, modes):
    """
    Smart IR/BLE detection:
    1. First detect IR/BLE regions from current patterns
    2. Compare with parameter periods to see if adjustments are needed
    3. Move forward/backward or extend slightly as needed

    Returns:
        tuple: (refined_vlc_periods, refined_ble_periods, detection_confidence)
    """
    print("🔍 SMART DETECTION: First detecting regions from current, then comparing with parameters...")

    # Convert to numpy arrays for efficient processing
    times = np.array(times)
    currents = np.array(currents)
    modes = np.array(modes) if modes is not None else np.zeros_like(times)

    # Calculate current statistics for baseline detection
    baseline_current = np.median(currents)
    current_std = np.std(currents)

    print(f"📊 Current Analysis: Baseline={baseline_current:.1f}μA, Std={current_std:.1f}μA")

    # STEP 1: Detect IR/BLE regions from current patterns (independent detection)
    detected_vlc_regions = detect_high_activity_regions(times, currents, baseline_current, current_std, "IR")
    detected_ble_regions = detect_high_activity_regions(times, currents, baseline_current, current_std, "BLE")

    print(
        f"🔍 Current-based Detection: {len(detected_vlc_regions)} IR regions, {len(detected_ble_regions)} BLE regions")

    # STEP 2: Get parameter-based periods for comparison
    param_vlc_periods = find_continuous_periods(times, modes == 2) if len(modes) > 0 else []
    param_ble_periods = find_continuous_periods(times, modes == 1) if len(modes) > 0 else []

    print(f"📋 Parameter Periods: {len(param_vlc_periods)} IR, {len(param_ble_periods)} BLE from running_mode")

    # STEP 3: Compare and adjust - move forward/backward or extend as needed
    refined_vlc_periods = compare_and_adjust_periods(detected_vlc_regions, param_vlc_periods, "IR")
    refined_ble_periods = compare_and_adjust_periods(detected_ble_regions, param_ble_periods, "BLE")

    # Calculate confidence based on agreement between detection methods
    confidence = calculate_detection_confidence(detected_vlc_regions + detected_ble_regions,
                                                param_vlc_periods + param_ble_periods,
                                                refined_vlc_periods + refined_ble_periods)

    return refined_vlc_periods, refined_ble_periods, confidence


def detect_high_activity_regions(times, currents, baseline, std, comm_type):
    """
    ULTRA-SIMPLE IR/BLE detection: Find ANY regions where current is higher than baseline.

    Very sensitive to catch all possible IR/BLE activity.
    """
    regions = []

    if len(currents) < 10:  # Need enough samples
        return regions

    print(f"🔍 ULTRA-SIMPLE {comm_type} DETECTION: Finding ALL regions above baseline activity...")
    print(f"📊 Baseline: {baseline:.1f}μA, Std: {std:.1f}μA")

    # Very simple threshold - just above baseline
    if comm_type == "IR":
        threshold = baseline + 0.3 * std  # Very low threshold for IR
        min_duration_ms = 5.0  # 5ms minimum
    else:
        threshold = baseline + 0.2 * std  # Even lower for BLE
        min_duration_ms = 1.0  # 1ms minimum

    min_duration_time = min_duration_ms / 1000.0  # Convert to seconds

    print(
        f"🎯 {comm_type} Detection: Threshold={threshold:.1f}μA (baseline + {threshold - baseline:.1f}μA), Min duration={min_duration_ms}ms")

    # Find ALL regions above threshold
    above_threshold = currents > threshold

    # Find continuous regions
    all_regions = find_continuous_regions_from_mask(times, above_threshold, min_duration_time)

    print(f"🔍 Found {len(all_regions)} potential {comm_type} regions above threshold")

    # Accept ALL regions that meet minimum duration (very permissive)
    for start_time, end_time in all_regions:
        duration_ms = (end_time - start_time) * 1000

        # Get region statistics
        start_idx = np.searchsorted(times, start_time)
        end_idx = np.searchsorted(times, end_time)
        region_avg = np.mean(currents[start_idx:end_idx])
        region_max = np.max(currents[start_idx:end_idx])

        regions.append((start_time, end_time))

        print(f"✅ {comm_type} Region: {start_time:.3f}s-{end_time:.3f}s (Duration: {duration_ms:.1f}ms)")
        print(
            f"   📊 Current: avg={region_avg:.0f}μA, max={region_max:.0f}μA, above baseline by {region_avg - baseline:.0f}μA")

    print(f"🎯 TOTAL {comm_type} regions detected: {len(regions)}")

    return regions


def find_continuous_regions_from_mask(times, mask, min_duration):
    """
    Find continuous regions from a boolean mask.
    """
    regions = []

    # Find continuous regions
    edges = np.diff(mask.astype(int))
    starts = np.where(edges == 1)[0] + 1  # +1 because diff shifts indices
    ends = np.where(edges == -1)[0] + 1

    # Handle edge cases
    if len(starts) == 0 and len(ends) == 0:
        return regions

    if len(starts) > 0 and (len(ends) == 0 or starts[0] < ends[0]):
        # Signal starts high
        if mask[0]:
            starts = np.concatenate([[0], starts])

    if len(ends) > 0 and (len(starts) == 0 or ends[-1] > starts[-1]):
        # Signal ends high
        if mask[-1]:
            ends = np.concatenate([ends, [len(times) - 1]])

    # Match starts and ends
    for i in range(min(len(starts), len(ends))):
        start_time = times[starts[i]]
        end_time = times[ends[i]]
        duration = end_time - start_time

        if duration >= min_duration:
            regions.append((start_time, end_time))

    return regions


def compare_and_adjust_periods(detected_regions, param_periods, comm_type):
    """
    Compare detected regions with parameter periods and make adjustments:
    - Move forward/backward if needed
    - Extend slightly if needed
    - Keep parameter periods if they match well
    """
    if len(param_periods) == 0:
        print(f"📋 {comm_type}: No parameter periods - using detected regions")
        return detected_regions

    if len(detected_regions) == 0:
        print(f"📋 {comm_type}: No detected regions - keeping parameter periods")
        return param_periods

    print(
        f"🔧 {comm_type} ADJUSTMENT: Comparing {len(detected_regions)} detected regions with {len(param_periods)} parameter periods")

    adjusted_periods = []
    max_shift = 0.100  # Maximum 100ms shift
    max_extend = 0.050  # Maximum 50ms extension

    for param_start, param_end in param_periods:
        # Find the best matching detected region
        best_detected = find_best_matching_region(param_start, param_end, detected_regions)

        if best_detected is None:
            # No matching detected region - keep parameter period
            adjusted_periods.append((param_start, param_end))
            print(f"📋 {comm_type}: No match found - keeping parameter period {param_start:.3f}s-{param_end:.3f}s")
            continue

        detected_start, detected_end = best_detected

        # Calculate potential adjustments
        new_start = param_start
        new_end = param_end

        # Check if we should move forward/backward (within max_shift)
        start_diff = detected_start - param_start
        end_diff = detected_end - param_end

        if abs(start_diff) <= max_shift:
            new_start = detected_start

        if abs(end_diff) <= max_shift:
            new_end = detected_end

        # Check if we should extend (within max_extend)
        if detected_start < param_start and (param_start - detected_start) <= max_extend:
            new_start = detected_start

        if detected_end > param_end and (detected_end - param_end) <= max_extend:
            new_end = detected_end

        # Only apply adjustment if it makes sense
        if new_end > new_start:
            total_shift = abs(new_start - param_start) + abs(new_end - param_end)
            if total_shift > 0.001:  # Only report significant changes
                print(
                    f"🔧 {comm_type} Adjusted: {param_start:.3f}s-{param_end:.3f}s → {new_start:.3f}s-{new_end:.3f}s (shift: {total_shift * 1000:.1f}ms)")
            adjusted_periods.append((new_start, new_end))
        else:
            # Keep original if adjustment doesn't make sense
            adjusted_periods.append((param_start, param_end))

    return adjusted_periods


def find_best_matching_region(param_start, param_end, detected_regions):
    """
    Find the detected region that best matches the parameter period.
    """
    if len(detected_regions) == 0:
        return None

    param_center = (param_start + param_end) / 2
    param_duration = param_end - param_start

    best_region = None
    best_score = float('inf')

    for detected_start, detected_end in detected_regions:
        detected_center = (detected_start + detected_end) / 2
        detected_duration = detected_end - detected_start

        # Score based on center distance and duration similarity
        center_distance = abs(detected_center - param_center)
        duration_ratio = abs(detected_duration - param_duration) / max(param_duration, 0.001)

        # Combined score (lower is better)
        score = center_distance + duration_ratio * 0.1

        # Only consider if regions have reasonable overlap
        overlap_start = max(param_start, detected_start)
        overlap_end = min(param_end, detected_end)
        overlap = max(0, overlap_end - overlap_start)
        overlap_ratio = overlap / param_duration

        if overlap_ratio > 0.3 and score < best_score:  # At least 30% overlap
            best_score = score
            best_region = (detected_start, detected_end)

    return best_region


def calculate_detection_confidence(detected_regions, param_periods, refined_periods):
    """
    Calculate confidence based on agreement between detection methods.
    """
    if len(refined_periods) == 0:
        return 0.0

    # Simple confidence based on how much we had to adjust
    total_param_time = sum(end - start for start, end in param_periods)
    total_refined_time = sum(end - start for start, end in refined_periods)

    if total_param_time == 0:
        return 0.5  # Medium confidence if no parameters

    # Higher confidence if less adjustment was needed
    time_change_ratio = abs(total_refined_time - total_param_time) / total_param_time
    confidence = max(0.0, 1.0 - time_change_ratio)

    return min(1.0, confidence)


def refine_period_boundaries(param_periods, times, currents, baseline, std, comm_type):
    """
    Refine the start/end boundaries of existing parameter-based periods using current analysis.

    This only makes small timing adjustments (typically <50ms) to align period boundaries
    with actual current transitions, without splitting or completely re-detecting periods.
    """
    if len(param_periods) == 0:
        return param_periods

    refined_periods = []
    max_adjustment = 0.050  # Maximum 50ms adjustment

    print(
        f"🔧 BOUNDARY REFINEMENT: Adjusting {len(param_periods)} {comm_type} period boundaries (max ±{max_adjustment * 1000:.0f}ms)")

    for param_start, param_end in param_periods:
        # Find indices for this period
        start_idx = np.searchsorted(times, param_start)
        end_idx = np.searchsorted(times, param_end)

        if start_idx >= len(times) or end_idx >= len(times) or start_idx >= end_idx:
            # Keep original if indices are invalid
            refined_periods.append((param_start, param_end))
            continue

        # Look for better start boundary (within max_adjustment window)
        refined_start = find_transition_boundary(times, currents, start_idx, baseline, std,
                                                 param_start, max_adjustment, "start", comm_type)

        # Look for better end boundary (within max_adjustment window)
        refined_end = find_transition_boundary(times, currents, end_idx, baseline, std,
                                               param_end, max_adjustment, "end", comm_type)

        # Only use refined boundaries if they make sense
        if refined_end > refined_start:
            time_shift = abs(refined_start - param_start) + abs(refined_end - param_end)
            if time_shift > 0.001:  # Only report significant shifts > 1ms
                print(
                    f"🔧 {comm_type} Refined: {param_start:.3f}s-{param_end:.3f}s → {refined_start:.3f}s-{refined_end:.3f}s (shift: {time_shift * 1000:.1f}ms)")
            refined_periods.append((refined_start, refined_end))
        else:
            # Keep original if refined boundaries don't make sense
            refined_periods.append((param_start, param_end))

    return refined_periods


def find_transition_boundary(times, currents, center_idx, baseline, std, param_time, max_adjustment, boundary_type,
                             comm_type):
    """
    Find a better transition boundary by looking for current changes near the parameter boundary.
    """
    # Define search window (limited by max_adjustment)
    search_samples = int(max_adjustment * 100000)  # Assume ~100kHz sampling
    search_samples = min(search_samples, 50)  # Cap at reasonable number

    start_search = max(0, center_idx - search_samples)
    end_search = min(len(times), center_idx + search_samples)

    if end_search <= start_search:
        return param_time

    # Look for the steepest current change in the search window
    search_currents = currents[start_search:end_search]
    search_times = times[start_search:end_search]

    if len(search_currents) < 3:
        return param_time

    # Calculate current gradient (rate of change)
    gradient = np.gradient(search_currents)

    if boundary_type == "start":
        # For start boundary, look for rising edge (positive gradient)
        best_idx = np.argmax(gradient) if comm_type == "IR" else np.argmax(np.abs(gradient))
    else:
        # For end boundary, look for falling edge (negative gradient)
        best_idx = np.argmin(gradient) if comm_type == "IR" else np.argmax(np.abs(gradient))

    refined_time = search_times[best_idx]

    # Only return refined time if it's within max_adjustment
    if abs(refined_time - param_time) <= max_adjustment:
        return refined_time
    else:
        return param_time


def detect_vlc_patterns(times, currents, baseline, std):
    """
    Detect IR communication patterns in current data.

    IR characteristics:
    - High-frequency modulation (LED switching)
    - Sustained high current periods during transmission
    - Sharp transitions at start/end of communication
    """
    vlc_periods = []

    # IR detection: Look for sustained high-current periods with high-frequency activity
    window_size = min(100, len(currents) // 10)  # Adaptive window size
    high_current_threshold = baseline + 1.5 * std

    print(f"🔴 IR Detection: Threshold={high_current_threshold:.1f}μA, Window={window_size}")

    i = 0
    while i < len(currents) - window_size:
        window_currents = currents[i:i + window_size]
        window_times = times[i:i + window_size]

        # Check for IR signature: high average current + high variance (LED modulation)
        avg_current = np.mean(window_currents)
        current_variance = np.var(window_currents)

        if avg_current > high_current_threshold and current_variance > std ** 2:
            # Found potential IR start - find the end
            vlc_start = window_times[0]

            # Extend forward to find IR end
            j = i + window_size
            while j < len(currents) - window_size:
                next_window = currents[j:j + window_size]
                if np.mean(next_window) < high_current_threshold:
                    break
                j += window_size // 2

            vlc_end = times[min(j, len(times) - 1)]

            if vlc_end - vlc_start > 0.001:  # Minimum 1ms duration
                vlc_periods.append((vlc_start, vlc_end))
                print(
                    f"🔴 IR Period: {vlc_start:.3f}s - {vlc_end:.3f}s (Duration: {(vlc_end - vlc_start) * 1000:.1f}ms)")

            i = j
        else:
            i += window_size // 4

    return vlc_periods


def detect_ble_patterns(times, currents, baseline, std):
    """
    Detect BLE communication patterns in current data.

    BLE characteristics:
    - Periodic advertising intervals (typically 20ms-10s)
    - Short burst transmissions (few ms)
    - Lower power than IR but higher than idle
    """
    ble_periods = []

    # BLE detection: Look for periodic current spikes
    ble_threshold = baseline + 0.8 * std  # Lower threshold than IR
    min_spike_duration = 0.002  # 2ms minimum
    max_spike_duration = 0.050  # 50ms maximum

    print(
        f"🟢 BLE Detection: Threshold={ble_threshold:.1f}μA, Spike Duration={min_spike_duration * 1000:.0f}-{max_spike_duration * 1000:.0f}ms")

    # Find current spikes that match BLE characteristics
    above_threshold = currents > ble_threshold

    # Find rising and falling edges
    edges = np.diff(above_threshold.astype(int))
    rising_edges = np.where(edges == 1)[0]
    falling_edges = np.where(edges == -1)[0]

    # Match rising and falling edges to find BLE bursts
    for rise_idx in rising_edges:
        # Find corresponding falling edge
        fall_candidates = falling_edges[falling_edges > rise_idx]
        if len(fall_candidates) == 0:
            continue

        fall_idx = fall_candidates[0]

        spike_start = times[rise_idx]
        spike_end = times[fall_idx]
        spike_duration = spike_end - spike_start

        # Check if duration matches BLE characteristics
        if min_spike_duration <= spike_duration <= max_spike_duration:
            avg_spike_current = np.mean(currents[rise_idx:fall_idx])
            if avg_spike_current > ble_threshold:
                ble_periods.append((spike_start, spike_end))
                print(
                    f"🟢 BLE Burst: {spike_start:.3f}s - {spike_end:.3f}s (Duration: {spike_duration * 1000:.1f}ms, Avg: {avg_spike_current:.1f}μA)")

    return ble_periods


def refine_periods_with_parameters(detected_periods, times, parameter_mask, comm_type):
    """
    Refine detected periods using parameter information to improve accuracy.

    This corrects small timing mismatches between current patterns and parameter timestamps.
    """
    if len(detected_periods) == 0 or len(parameter_mask) == 0:
        return detected_periods

    print(f"🔧 REFINEMENT: Adjusting {len(detected_periods)} {comm_type} periods using parameter hints...")

    refined_periods = []

    # Find parameter-indicated periods
    param_edges = np.diff(parameter_mask.astype(int))
    param_starts = times[np.where(param_edges == 1)[0]]
    param_ends = times[np.where(param_edges == -1)[0]]

    # Match parameter periods with detected periods
    for detected_start, detected_end in detected_periods:
        # Find closest parameter period
        best_param_start = detected_start
        best_param_end = detected_end

        # Look for parameter start near detected start
        if len(param_starts) > 0:
            start_diffs = np.abs(param_starts - detected_start)
            closest_start_idx = np.argmin(start_diffs)
            if start_diffs[closest_start_idx] < 0.1:  # Within 100ms
                best_param_start = param_starts[closest_start_idx]

        # Look for parameter end near detected end
        if len(param_ends) > 0:
            end_diffs = np.abs(param_ends - detected_end)
            closest_end_idx = np.argmin(end_diffs)
            if end_diffs[closest_end_idx] < 0.1:  # Within 100ms
                best_param_end = param_ends[closest_end_idx]

        # Only use refined period if it makes sense
        if best_param_end > best_param_start:
            time_shift = abs(best_param_start - detected_start) + abs(best_param_end - detected_end)
            if time_shift > 0.001:  # Only report significant shifts
                print(
                    f"🔧 {comm_type} Refined: {detected_start:.3f}s-{detected_end:.3f}s → {best_param_start:.3f}s-{best_param_end:.3f}s (shift: {time_shift * 1000:.1f}ms)")
            refined_periods.append((best_param_start, best_param_end))
        else:
            refined_periods.append((detected_start, detected_end))

    return refined_periods


def add_vlc_ble_markers_simple(times, currents, modes, ax):
    """
    Simple IR/BLE markers based on running_mode parameters (original method).
    """
    if modes is None or len(modes) == 0:
        return

    # Get the current y-axis limits to create full-height rectangles
    ylim = ax.get_ylim()
    y_min, y_max = ylim[0], ylim[1]

    # Find continuous periods for BLE (mode = 1)
    ble_periods = find_continuous_periods(times, modes == 1)
    if ble_periods:
        for start_time, end_time in ble_periods:
            # Create light green rectangle for BLE duration
            ax.axvspan(start_time, end_time, alpha=0.3, color='lightgreen',
                       zorder=1, label='BLE Transmission' if start_time == ble_periods[0][0] else "")
        print(f"🟢 BLE Duration: {len(ble_periods)} periods marked with light green rectangles")

    # Find continuous periods for IR (mode = 2)
    vlc_periods = find_continuous_periods(times, modes == 2)
    if vlc_periods:
        for start_time, end_time in vlc_periods:
            # Create light red rectangle for IR duration
            ax.axvspan(start_time, end_time, alpha=0.3, color='lightcoral',
                       zorder=1, label='IR Transmission' if start_time == vlc_periods[0][0] else "")
        print(f"🔴 IR Duration: {len(vlc_periods)} periods marked with light red rectangles")


def calculate_refined_boundaries(times, currents, modes, baseline):
    """
    Calculate refined IR/BLE boundaries ONCE from combined dataset.

    Args:
        times: Combined time array
        currents: Combined current array
        modes: Combined Running_Mode array
        baseline: Calculated baseline current

    Returns:
        List of boundary adjustments with refined timing
    """
    print("🔍 ANALYZING IR/BLE BOUNDARIES...")

    # Find original IR/BLE periods from combined data
    vlc_periods = find_continuous_periods(times, modes == 2)
    ble_periods = find_continuous_periods(times, modes == 1)

    print(f"  📊 Found {len(vlc_periods)} IR + {len(ble_periods)} BLE periods in combined data")

    # Track boundary adjustments
    boundary_adjustments = []

    # Refine IR periods
    for original_start, original_end in vlc_periods:
        refined_start, refined_end = find_first_last_peaks(times, currents, original_start, original_end, baseline,
                                                           "IR")
        boundary_adjustments.append({
            'type': 'IR',
            'original_start': original_start,
            'original_end': original_end,
            'refined_start': refined_start,
            'refined_end': refined_end
        })

    # Refine BLE periods
    for original_start, original_end in ble_periods:
        refined_start, refined_end = find_first_last_peaks(times, currents, original_start, original_end, baseline,
                                                           "BLE")
        boundary_adjustments.append({
            'type': 'BLE',
            'original_start': original_start,
            'original_end': original_end,
            'refined_start': refined_start,
            'refined_end': refined_end
        })

    print(f"✅ CALCULATED {len(boundary_adjustments)} BOUNDARY ADJUSTMENTS")
    return boundary_adjustments


def save_refined_dataset_with_boundaries(boundary_adjustments, input_file, enable_wakeup_detection=False,
                                         global_wakeup_results=None, enable_advanced_detection=False,
                                         enable_wakeup_refinement=True):
    """
    Apply pre-calculated boundary adjustments to individual CSV file.
    Optionally apply wake-up detection changes with cross-file continuity.
    Optionally refine wakeup transition points (3→0) to exact current jump locations.

    Args:
        boundary_adjustments: Pre-calculated boundary refinements
        input_file: Individual CSV file path
        enable_wakeup_detection: Whether to apply wake-up detection changes
        global_wakeup_results: Pre-calculated wake-up results for this file
        enable_advanced_detection: Whether advanced detection is enabled
        enable_wakeup_refinement: Whether to refine wakeup transition points (default: True)

    Returns:
        bool: True if wake-up was detected in this file, False otherwise
    """
    try:
        import pandas as pd
        from pathlib import Path

        # Generate output filename
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_refined{input_path.suffix}"

        # Load individual CSV
        df = pd.read_csv(input_file)
        refined_df = df.copy()

        # Apply boundary adjustments to Running_Mode column
        adjustments_applied = 0
        for adj in boundary_adjustments:
            # Check if this file contains the time range for this adjustment
            file_min_time = df['Timestamp (s)'].min()
            file_max_time = df['Timestamp (s)'].max()

            # Skip adjustments outside this file's time range
            if adj['refined_end'] < file_min_time or adj['refined_start'] > file_max_time:
                continue

            # Reset original period to idle (mode 0)
            original_mask = (df['Timestamp (s)'] >= adj['original_start']) & (
                    df['Timestamp (s)'] <= adj['original_end'])
            refined_df.loc[original_mask, 'Running_Mode'] = 0

            # Set refined period to correct mode
            refined_mask = (df['Timestamp (s)'] >= adj['refined_start']) & (df['Timestamp (s)'] <= adj['refined_end'])
            mode_value = 2 if adj['type'] == 'IR' else 1
            refined_df.loc[refined_mask, 'Running_Mode'] = mode_value

            adjustments_applied += 1

        # Apply wake-up detection if enabled
        if enable_wakeup_detection and global_wakeup_results is not None:
            print(f"    🌐 Applying global wake-up results to {input_file}")

            # Get the pre-calculated wake-up results for this file
            corrected_states = global_wakeup_results['corrected_states']
            wakeup_states = global_wakeup_results['wakeup_states']
            running_state_changes = global_wakeup_results['running_state_changes']
            wakeup_state_changes = global_wakeup_results['wakeup_state_changes']

            print(
                f"    📊 Global wake-up results: {running_state_changes} Running_State changes, {wakeup_state_changes} Wakeup_State changes")

            # Apply the pre-calculated results to the refined dataframe
            if 'Running_State' in refined_df.columns:
                # Ensure Wakeup_State column exists
                if 'Wakeup_State' not in refined_df.columns:
                    refined_df['Wakeup_State'] = 0  # Initialize all to 0

                if running_state_changes > 0 or wakeup_state_changes > 0:
                    # Apply the corrected states directly
                    refined_df['Running_State'] = corrected_states
                    refined_df['Wakeup_State'] = wakeup_states

                    # Find timing info for logging
                    if running_state_changes > 0:
                        # Read times for this file to get timing info
                        individual_times, _, original_states = read_running_state_for_wakeup(input_file)

                        # Find first and last changed indices
                        changed_indices = np.where(original_states != corrected_states)[0]
                        if len(changed_indices) > 0:
                            first_wakeup_time = individual_times[changed_indices[0]]
                            last_wakeup_time = individual_times[changed_indices[-1]]
                            wakeup_duration = last_wakeup_time - first_wakeup_time

                            print(f"      ⚡ GLOBAL UPDATE: Applied pre-calculated wake-up results")
                            print(
                                f"      🚨 Wake-up period: {first_wakeup_time:.3f}s - {last_wakeup_time:.3f}s ({wakeup_duration:.3f}s duration)")

                print(f"    🔄 Global wake-up detection results:")
                print(f"      📊 Running_State changes (3→0): {running_state_changes}")
                print(f"      📊 Wakeup_State changes (0→1): {wakeup_state_changes}")

                if running_state_changes == 0 and wakeup_state_changes == 0:
                    print(f"      ℹ️  No wake-up changes needed for this file")
            else:
                print(f"    ❌ ERROR: Required column 'Running_State' not found in CSV!")
                running_state_changes = wakeup_state_changes = 0
        elif enable_wakeup_detection:
            print(f"    ℹ️  No global wake-up results available for {input_file}")
            running_state_changes = wakeup_state_changes = 0

        # Apply wakeup transition refinement if enabled and Running_State column exists
        if enable_wakeup_refinement and 'Running_State' in refined_df.columns and 'Current (µA)' in refined_df.columns:
            # Extract arrays for refinement
            times_array = refined_df['Timestamp (s)'].values if 'Timestamp (s)' in refined_df.columns else refined_df['Time (s)'].values
            currents_array = refined_df['Current (µA)'].values
            running_states_array = refined_df['Running_State'].values
            
            # Apply refinement (verbose=False for cleaner output when processing multiple files)
            refined_states_array = refine_wakeup_transition_points(times_array, currents_array, running_states_array, search_window_seconds=5.0, verbose=False)
            
            # Update the dataframe with refined states
            refined_df['Running_State'] = refined_states_array
            
            # Count refinements for logging
            changes = np.sum(running_states_array != refined_states_array)
            if changes > 0:
                print(f"    ✅ Refined {changes} wakeup transition samples")
        elif enable_wakeup_refinement:
            print(f"    ℹ️  Wakeup refinement skipped (missing required columns)")
        
        # Apply IR/BLE communication voltage assignments (after finishing advanced detection or wake-up detection)
        if enable_advanced_detection or enable_wakeup_detection:
            print(f"    🔧 Applying IR/BLE communication voltage assignments...")

            # Ensure IR_Comm_Vol and BLE_Comm_Vol columns exist
            if 'IR_Comm_Vol' not in refined_df.columns:
                refined_df['IR_Comm_Vol'] = 0  # Initialize to 0
            if 'BLE_Comm_Vol' not in refined_df.columns:
                refined_df['BLE_Comm_Vol'] = 0  # Initialize to 0

            # Check if required columns exist
            if 'Running_State' in refined_df.columns and 'Running_Mode' in refined_df.columns:
                # Force IR_Comm_Vol=24 when Running_State=2 and Running_Mode=2 (IR communication)
                vlc_mask = (refined_df['Running_State'] == 2) & (refined_df['Running_Mode'] == 2)
                vlc_count = vlc_mask.sum()
                if vlc_count > 0:
                    refined_df.loc[vlc_mask, 'IR_Comm_Vol'] = 24
                    print(
                        f"      📡 IR: Set IR_Comm_Vol=24 for {vlc_count} samples (Running_State=2 & Running_Mode=2)")

                # Force BLE_Comm_Vol=31 when Running_State=2 and Running_Mode=1 (BLE communication)
                ble_mask = (refined_df['Running_State'] == 2) & (refined_df['Running_Mode'] == 1)
                ble_count = ble_mask.sum()
                if ble_count > 0:
                    refined_df.loc[ble_mask, 'BLE_Comm_Vol'] = 31
                    print(
                        f"      📶 BLE: Set BLE_Comm_Vol=31 for {ble_count} samples (Running_State=2 & Running_Mode=1)")

                if vlc_count == 0 and ble_count == 0:
                    print(f"      ℹ️  No IR/BLE communication periods found (Running_State=2)")
            else:
                print(
                    f"    ❌ ERROR: Required columns 'Running_State' or 'Running_Mode' not found for IR/BLE voltage assignment!")
        else:
            print(f"    ℹ️  IR/BLE voltage assignment skipped (requires --advanced-detection or --wakeup-detection)")

        # Save refined dataset
        refined_df.to_csv(output_file, index=False)

        print(f"    ✅ SAVED: {output_file.name} ({adjustments_applied} adjustments applied)")

        # Return whether wake-up was detected in this file
        if enable_wakeup_detection:
            # Check if any wake-up changes were made (regardless of global state)
            file_wakeup_detected = (running_state_changes > 0 or wakeup_state_changes > 0)
            return file_wakeup_detected
        else:
            return False

    except Exception as e:
        print(f"    ❌ Error processing {input_file}: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_refined_dataset(times, currents, modes, baseline, input_file):
    """
    Save refined dataset with adjusted IR/BLE timing based on boundary detection.

    Args:
        times: Time array
        currents: Current array
        modes: Running_Mode array
        baseline: Calculated baseline current
        input_file: Original CSV file path
    """
    try:
        import pandas as pd
        from pathlib import Path

        print("\n💾 SAVING REFINED DATASET...")

        # Generate output filename
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_refined{input_path.suffix}"

        print(f"📂 Loading original dataset: {input_file}")

        # Load original CSV
        df = pd.read_csv(input_file)

        # Create refined copy
        refined_df = df.copy()

        # Find original IR/BLE periods
        vlc_periods = find_continuous_periods(times, modes == 2)
        ble_periods = find_continuous_periods(times, modes == 1)

        print(f"🔍 Processing {len(vlc_periods)} IR + {len(ble_periods)} BLE periods...")

        # Track boundary adjustments
        boundary_adjustments = []

        # Refine IR periods
        for original_start, original_end in vlc_periods:
            refined_start, refined_end = find_first_last_peaks(times, currents, original_start, original_end, baseline,
                                                               "IR")
            boundary_adjustments.append({
                'type': 'IR',
                'original_start': original_start,
                'original_end': original_end,
                'refined_start': refined_start,
                'refined_end': refined_end
            })

        # Refine BLE periods
        for original_start, original_end in ble_periods:
            refined_start, refined_end = find_first_last_peaks(times, currents, original_start, original_end, baseline,
                                                               "BLE")
            boundary_adjustments.append({
                'type': 'BLE',
                'original_start': original_start,
                'original_end': original_end,
                'refined_start': refined_start,
                'refined_end': refined_end
            })

        # Apply boundary adjustments to Running_Mode column
        for adj in boundary_adjustments:
            # Reset original period to idle (mode 0)
            original_mask = (df['Timestamp (s)'] >= adj['original_start']) & (
                    df['Timestamp (s)'] <= adj['original_end'])
            refined_df.loc[original_mask, 'Running_Mode'] = 0

            # Set refined period to correct mode
            refined_mask = (df['Timestamp (s)'] >= adj['refined_start']) & (df['Timestamp (s)'] <= adj['refined_end'])
            mode_value = 2 if adj['type'] == 'IR' else 1
            refined_df.loc[refined_mask, 'Running_Mode'] = mode_value

            print(
                f"  ✅ {adj['type']}: {adj['original_start']:.3f}s-{adj['original_end']:.3f}s → {adj['refined_start']:.3f}s-{adj['refined_end']:.3f}s")

        # Save refined dataset
        refined_df.to_csv(output_file, index=False)

        print(f"✅ REFINED DATASET SAVED: {output_file}")
        print(f"📊 Preserved: Current, Timestamp, Voltage, Power (unchanged)")
        print(f"🔧 Adjusted: Running_Mode timing to match electrical boundaries")

    except Exception as e:
        print(f"❌ Error saving refined dataset: {e}")


def add_vlc_ble_markers_advanced(times, currents, modes, ax, baseline_start=9.0, baseline_end=10.0,
                                 vlc_threshold_mult=2.0, vlc_min_samples=250,
                                 ble_threshold_mult=2.5, ble_min_samples=300, search_window=2.0):
    """
    USER'S SIMPLE METHOD:
    1. Detect main IR/BLE durations from original parameter data
    2. Look 2 seconds before/after each period to extend if needed
    """
    if len(times) == 0 or len(currents) == 0:
        return

    print("🎯 USER'S SIMPLE METHOD: Start with parameter data, then extend ±2 seconds...")

    # Convert to numpy arrays
    times = np.array(times)
    currents = np.array(currents)
    modes = np.array(modes) if modes is not None else np.zeros_like(times)

    # STEP 1: Get main durations from original parameter data
    param_vlc_periods = find_continuous_periods(times, modes == 2)  # running_mode = 2 (IR)
    param_ble_periods = find_continuous_periods(times, modes == 1)  # running_mode = 1 (BLE)

    print(
        f"📋 ORIGINAL DATA: {len(param_vlc_periods)} IR periods, {len(param_ble_periods)} BLE periods from parameters")

    # STEP 2: Calculate baseline from configured window
    baseline_start_time = baseline_start
    baseline_end_time = baseline_end

    baseline_start_idx = np.searchsorted(times, baseline_start_time)
    baseline_end_idx = np.searchsorted(times, baseline_end_time)

    if baseline_end_idx > baseline_start_idx:
        baseline_currents = currents[baseline_start_idx:baseline_end_idx]
        baseline = np.median(baseline_currents)
        print(f"📊 Baseline calculated from 2-3 seconds: {baseline:.0f}μA (from {len(baseline_currents)} samples)")
    else:
        baseline = np.median(currents[:1000])  # Fallback to first 1000 samples
        print(f"📊 Baseline fallback (no data at 2-3s): {baseline:.0f}μA")

    threshold_2x = baseline * 2.0  # 2x baseline threshold
    print(f"🎯 Detection threshold (2x baseline): {threshold_2x:.0f}μA")

    # REDESIGNED DETECTION: Find first and last high current peaks within parameter periods
    # IR/BLE has multiple peaks: 11mA, 2.5mA, 10.5mA, 2.6mA...
    # Beginning boundary = first high peak (11mA), End boundary = last high peak (10.5mA)

    extended_vlc_periods = []
    extended_ble_periods = []

    print(
        f"🔧 PEAK-BASED DETECTION: Finding first/last high current peaks in {len(param_vlc_periods)} IR + {len(param_ble_periods)} BLE periods...")

    # Process IR periods
    for start_time, end_time in param_vlc_periods:
        refined_start, refined_end = find_first_last_peaks(times, currents, start_time, end_time, baseline, "IR",
                                                           vlc_threshold_mult, vlc_min_samples, ble_threshold_mult,
                                                           ble_min_samples, search_window)
        extended_vlc_periods.append((refined_start, refined_end))

    # Process BLE periods
    for start_time, end_time in param_ble_periods:
        refined_start, refined_end = find_first_last_peaks(times, currents, start_time, end_time, baseline, "BLE",
                                                           vlc_threshold_mult, vlc_min_samples, ble_threshold_mult,
                                                           ble_min_samples, search_window)
        extended_ble_periods.append((refined_start, refined_end))

    print(f"✅ REFINED PERIODS: {len(extended_vlc_periods)} IR periods, {len(extended_ble_periods)} BLE periods")

    # Add BLE markers (light green)
    if extended_ble_periods:
        for start_time, end_time in extended_ble_periods:
            duration_ms = (end_time - start_time) * 1000
            print(f"🟢 BLE: {start_time:.3f}s-{end_time:.3f}s ({duration_ms:.1f}ms)")
            ax.axvspan(start_time, end_time, alpha=0.3, color='lightgreen',
                       zorder=1, label='BLE Transmission' if start_time == extended_ble_periods[0][0] else "")

    # Add IR markers (light red)
    if extended_vlc_periods:
        for start_time, end_time in extended_vlc_periods:
            duration_ms = (end_time - start_time) * 1000
            print(f"🔴 IR: {start_time:.3f}s-{end_time:.3f}s ({duration_ms:.1f}ms)")
            ax.axvspan(start_time, end_time, alpha=0.3, color='lightcoral',
                       zorder=1, label='IR Transmission' if start_time == extended_vlc_periods[0][0] else "")

    # Store periods for zoom callback
    ax._vlc_periods = extended_vlc_periods
    ax._ble_periods = extended_ble_periods

    # Store data for potential refined dataset saving
    ax._times = times
    ax._currents = currents
    ax._modes = modes
    ax._baseline = baseline
    
    # Don't add the old "Peak-Based Detection" legend - it's replaced by node parameters


def find_first_last_peaks(times, currents, start_time, end_time, baseline, comm_type,
                          vlc_threshold_mult=2.0, vlc_min_samples=250,
                          ble_threshold_mult=2.5, ble_min_samples=300, search_window=2.0):
    """
    IMPROVED 3-STEP BOUNDARY REFINEMENT:
    1. Use original IR/BLE duration from parameter data (start_time, end_time)
    2. Extend ±3 seconds around original duration for boundary search
    3. Find first/last threshold crossings as refined boundaries

    Args:
        times: Time array
        currents: Current array (mean current - dark blue line)
        start_time, end_time: Original parameter period boundaries from STEP 1
        baseline: Fixed baseline from 9-10 seconds (e.g., 2.5mA)
        comm_type: "IR" or "BLE" for logging

    Returns:
        (refined_start, refined_end): Boundaries based on first/last threshold crossings
    """
    print(f"🔍 STEP 1: Original {comm_type} duration from parameter data: {start_time:.3f}s-{end_time:.3f}s")

    # STEP 2: Extend ±configured seconds around original duration for boundary search
    search_start_time = max(0, start_time - search_window)  # Don't go before time 0
    search_end_time = end_time + search_window

    print(
        f"🔍 STEP 2: Extended search window: {search_start_time:.3f}s-{search_end_time:.3f}s (±{search_window}s extension)")

    search_start_idx = np.searchsorted(times, search_start_time)
    search_end_idx = min(len(times), np.searchsorted(times, search_end_time))

    if search_end_idx <= search_start_idx:
        print(f"  📋 {comm_type} unchanged: insufficient data in search window")
        return start_time, end_time

    # Get extended data (±2 seconds around parameter period)
    search_times = times[search_start_idx:search_end_idx]
    search_currents = currents[search_start_idx:search_end_idx]

    print(f"  🔍 Search window: {search_start_time:.3f}s-{search_end_time:.3f}s ({len(search_currents)} samples)")

    # Define IR/BLE detection criteria: 4x baseline for 1500+ consecutive samples
    # But IR and BLE might have different current levels - let's analyze the parameter period first
    param_start_idx = np.searchsorted(times, start_time)
    param_end_idx = np.searchsorted(times, end_time)

    if param_end_idx > param_start_idx:
        param_currents = currents[param_start_idx:param_end_idx]
        param_max = np.max(param_currents)
        param_avg = np.mean(param_currents)
        param_min = np.min(param_currents)

        print(f"  📊 Parameter period analysis:")
        print(f"      Baseline: {baseline:.1f}mA")
        print(
            f"      {comm_type} period current - Min: {param_min:.1f}mA, Avg: {param_avg:.1f}mA, Max: {param_max:.1f}mA")
        print(f"      Max/{comm_type} vs Baseline ratio: {param_max / baseline:.1f}x")

    # Use configured thresholds for IR vs BLE
    if comm_type == "IR":
        high_threshold = baseline * vlc_threshold_mult
        min_window_size = vlc_min_samples
        print(
            f"  🎯 IR Detection threshold: {high_threshold:.1f}mA ({vlc_threshold_mult}x baseline), Min window: {min_window_size} samples")
    else:  # BLE
        high_threshold = baseline * ble_threshold_mult
        min_window_size = ble_min_samples
        print(
            f"  🎯 BLE Detection threshold: {high_threshold:.1f}mA ({ble_threshold_mult}x baseline), Min window: {min_window_size} samples")

    # STEP 3: Find first/last threshold crossings as refined boundaries
    print(f"🔍 STEP 3: Finding first/last threshold crossings with {high_threshold:.1f}μA threshold")

    # Find all samples above threshold
    high_current_mask = search_currents > high_threshold
    high_samples_count = np.sum(high_current_mask)
    total_samples = len(search_currents)

    print(
        f"  📊 Found {high_samples_count}/{total_samples} samples ({high_samples_count / total_samples * 100:.1f}%) above threshold")

    if not np.any(high_current_mask):
        print(f"  📋 No samples above threshold, keeping original boundaries")
        return start_time, end_time

    # Find indices of all high current samples
    high_indices = np.where(high_current_mask)[0]

    # Find first valid sustained activity (beginning boundary)
    refined_start = start_time
    found_start = False

    for i in range(len(high_indices) - min_window_size + 1):
        # Check if we have min_window_size consecutive samples
        consecutive_count = 1
        for j in range(i + 1, min(i + min_window_size, len(high_indices))):
            if high_indices[j] == high_indices[j - 1] + 1:
                consecutive_count += 1
            else:
                break

        if consecutive_count >= min_window_size:
            # Found valid sustained activity - this is the beginning boundary
            refined_start = search_times[high_indices[i]]
            print(f"  🎯 First sustained activity: {refined_start:.3f}s ({consecutive_count} consecutive samples)")
            found_start = True
            break

    # Find last valid sustained activity (end boundary)
    refined_end = end_time
    found_end = False

    for i in range(len(high_indices) - min_window_size, -1, -1):
        # Check if we have min_window_size consecutive samples
        consecutive_count = 1
        for j in range(i + 1, min(i + min_window_size, len(high_indices))):
            if j < len(high_indices) and high_indices[j] == high_indices[j - 1] + 1:
                consecutive_count += 1
            else:
                break

        if consecutive_count >= min_window_size:
            # Found valid sustained activity - this is the end boundary
            last_sample_idx = min(i + consecutive_count - 1, len(high_indices) - 1)
            refined_end = search_times[high_indices[last_sample_idx]]
            print(f"  🎯 Last sustained activity: {refined_end:.3f}s ({consecutive_count} consecutive samples)")
            found_end = True
            break

    # If no sustained activity found, use first and last individual threshold crossings
    if not found_start or not found_end:
        if not found_start:
            refined_start = search_times[high_indices[0]]
            print(f"  📍 Using first threshold crossing: {refined_start:.3f}s")
        if not found_end:
            refined_end = search_times[high_indices[-1]]
            print(f"  📍 Using last threshold crossing: {refined_end:.3f}s")

    # Calculate and display adjustments
    start_adjustment = refined_start - start_time
    end_adjustment = refined_end - end_time
    total_adjustment = abs(start_adjustment) + abs(end_adjustment)

    print(f"  ✅ {comm_type} REFINED: {refined_start:.3f}s-{refined_end:.3f}s")
    print(f"     Start adjustment: {start_adjustment:+.3f}s, End adjustment: {end_adjustment:+.3f}s")
    print(f"     Total adjustment: {total_adjustment * 1000:.1f}ms")

    return refined_start, refined_end


def extend_period_with_fixed_baseline(times, currents, start_time, end_time, baseline, threshold_2x, extend_seconds,
                                      comm_type):
    """
    USER'S CORRECT METHOD with FIXED BASELINE:
    Use the baseline calculated from 9-10 seconds to find true start/end

    Args:
        times: Time array
        currents: Current array
        start_time, end_time: Original period boundaries
        baseline: Fixed baseline calculated from 9-10 seconds
        threshold_2x: 2x baseline threshold
        extend_seconds: How many seconds to look before/after (e.g., 2.0)
        comm_type: "IR" or "BLE" for logging

    Returns:
        (extended_start, extended_end): New period boundaries
    """
    print(f"🔧 CONSERVATIVE refinement of {comm_type} period {start_time:.3f}s-{end_time:.3f}s...")
    print(f"  📊 Baseline: {baseline:.0f}μA, Looking for small timing adjustments only")

    # STEP 1: Look backwards to find TRUE START
    # Only look in a small window before the parameter start
    look_back_time = max(0, start_time - extend_seconds)  # Don't go before time 0
    look_back_idx = np.searchsorted(times, look_back_time)
    start_idx = np.searchsorted(times, start_time)

    extended_start = start_time  # Default: keep original start

    if look_back_idx < start_idx:
        look_back_currents = currents[look_back_idx:start_idx]
        look_back_times = times[look_back_idx:start_idx]

        # GRADIENT-BASED BOUNDARY DETECTION: Find where current starts rising rapidly
        # Calculate the rate of change (gradient) to find the steepest rise

        if len(look_back_currents) > 10:  # Need enough samples for gradient
            # Calculate gradient (rate of change) over a small window
            window_size = 5
            gradients = []
            gradient_indices = []

            for i in range(window_size, len(look_back_currents) - window_size):
                # Calculate slope over window_size samples
                y1 = np.mean(look_back_currents[i - window_size:i])
                y2 = np.mean(look_back_currents[i:i + window_size])
                dt = look_back_times[i + window_size - 1] - look_back_times[i - window_size]

                if dt > 0:
                    gradient = (y2 - y1) / dt  # μA/second
                    gradients.append(gradient)
                    gradient_indices.append(i)

            if gradients:
                gradients = np.array(gradients)
                gradient_indices = np.array(gradient_indices)

                # Find the steepest positive gradient (fastest rise)
                max_gradient_idx = np.argmax(gradients)
                steepest_rise_rate = gradients[max_gradient_idx]

                # Only accept if the rise is significant (> 1000 μA/second)
                if steepest_rise_rate > 1000:
                    rise_sample_idx = gradient_indices[max_gradient_idx]
                    extended_start = look_back_times[rise_sample_idx]
                    back_extension = start_time - extended_start

                    print(
                        f"  🔙 Found steepest rise: {extended_start:.3f}s (extended backward by {back_extension * 1000:.1f}ms)")
                    print(
                        f"      Rise rate: {steepest_rise_rate:.0f}μA/s, Current: {look_back_currents[rise_sample_idx]:.0f}μA")
                else:
                    print(f"  📋 Start unchanged (max rise rate {steepest_rise_rate:.0f}μA/s too slow)")
            else:
                print(f"  📋 Start unchanged (cannot calculate gradients)")
        else:
            print(f"  📋 Start unchanged (insufficient samples for gradient analysis)")

    # STEP 2: Look forwards to find TRUE END
    # Only look in a small window after the parameter end
    look_forward_time = end_time + extend_seconds
    end_idx = np.searchsorted(times, end_time)
    look_forward_idx = min(len(times), np.searchsorted(times, look_forward_time))

    extended_end = end_time  # Default: keep original end

    if end_idx < look_forward_idx:
        look_forward_currents = currents[end_idx:look_forward_idx]
        look_forward_times = times[end_idx:look_forward_idx]

        # GRADIENT-BASED BOUNDARY DETECTION: Find where current starts dropping rapidly
        # Calculate the rate of change (gradient) to find the steepest drop

        if len(look_forward_currents) > 10:  # Need enough samples for gradient
            # Calculate gradient (rate of change) over a small window
            window_size = 5
            gradients = []
            gradient_indices = []

            for i in range(window_size, len(look_forward_currents) - window_size):
                # Calculate slope over window_size samples
                y1 = np.mean(look_forward_currents[i - window_size:i])
                y2 = np.mean(look_forward_currents[i:i + window_size])
                dt = look_forward_times[i + window_size - 1] - look_forward_times[i - window_size]

                if dt > 0:
                    gradient = (y2 - y1) / dt  # μA/second
                    gradients.append(gradient)
                    gradient_indices.append(i)

            if gradients:
                gradients = np.array(gradients)
                gradient_indices = np.array(gradient_indices)

                # Find the steepest negative gradient (fastest drop)
                min_gradient_idx = np.argmin(gradients)
                steepest_drop_rate = gradients[min_gradient_idx]

                # Only accept if the drop is significant (< -1000 μA/second)
                if steepest_drop_rate < -1000:
                    drop_sample_idx = gradient_indices[min_gradient_idx]
                    extended_end = look_forward_times[drop_sample_idx]
                    forward_extension = extended_end - end_time

                    print(
                        f"  🔜 Found steepest drop: {extended_end:.3f}s (extended forward by {forward_extension * 1000:.1f}ms)")
                    print(
                        f"      Drop rate: {steepest_drop_rate:.0f}μA/s, Current: {look_forward_currents[drop_sample_idx]:.0f}μA")
                else:
                    print(f"  📋 End unchanged (max drop rate {steepest_drop_rate:.0f}μA/s too slow)")
            else:
                print(f"  📋 End unchanged (cannot calculate gradients)")
        else:
            print(f"  📋 End unchanged (insufficient samples for gradient analysis)")

    total_extension = (start_time - extended_start) + (extended_end - end_time)
    if total_extension > 0.001:  # Report if extended by > 1ms
        print(
            f"  ✅ {comm_type} FINAL: {extended_start:.3f}s-{extended_end:.3f}s (total extension: {total_extension:.3f}s)")
    else:
        print(f"  📋 {comm_type} unchanged: {extended_start:.3f}s-{extended_end:.3f}s (no 2x baseline activity)")

    return extended_start, extended_end


def refine_wakeup_transition_points(times, currents, running_states, search_window_seconds=5.0, verbose=True):
    """
    Refine wakeup transition points (3→0) by finding the exact current jump location.
    
    When running_state changes from 3 to 0, the timestamp might be slightly off.
    This function searches ±search_window_seconds around each transition to find
    the exact point where current jumps significantly (e.g., 750µA → 1800µA).
    
    Args:
        times: Time array
        currents: Current array (µA)
        running_states: Running state array
        search_window_seconds: Time window to search before/after transition in seconds (default: 5.0)
        verbose: Whether to print detailed output (default: True)
    
    Returns:
        refined_running_states: Adjusted running states with accurate transition points
    """
    if verbose:
        print(f"\n🔍 Refining wakeup transition points (3→0)...")
        print(f"   Search window: ±{search_window_seconds}s around each transition")
    
    # Copy to avoid modifying original
    refined_states = running_states.copy()
    
    # Find all transitions from 3 to 0
    state_changes = np.diff(running_states.astype(int))
    # Where it changes from 3 to 0: state_changes == -3
    transition_indices = np.where(state_changes == -3)[0] + 1  # +1 because diff shifts by 1
    
    if len(transition_indices) == 0:
        if verbose:
            print(f"   No 3→0 transitions found")
        return refined_states
    
    if verbose:
        print(f"   Found {len(transition_indices)} transition(s) from 3→0")
    
    refinements = 0
    for trans_idx in transition_indices:
        # Define search window based on TIME, not samples
        trans_time = times[trans_idx]
        search_start_time = trans_time - search_window_seconds
        search_end_time = trans_time + search_window_seconds
        
        # Find sample indices corresponding to these times
        search_start = np.searchsorted(times, search_start_time, side='left')
        search_end = np.searchsorted(times, search_end_time, side='right')
        
        # Clamp to valid range
        search_start = max(0, search_start)
        search_end = min(len(currents), search_end)
        
        if search_end - search_start < 2:
            continue  # Need at least 2 samples
        
        # Get currents in search window
        window_currents = currents[search_start:search_end]
        
        # Find the largest jump (derivative)
        current_diffs = np.diff(window_currents)
        max_jump_idx = np.argmax(current_diffs)  # Largest positive jump
        
        # Position of the jump in the original array
        actual_transition_idx = search_start + max_jump_idx + 1  # +1 because diff shifts
        
        # Validate the jump
        jump_size = current_diffs[max_jump_idx]
        before_current = window_currents[max_jump_idx]
        after_current = window_currents[max_jump_idx + 1]
        adjustment = actual_transition_idx - trans_idx
        time_shift = times[actual_transition_idx] - times[trans_idx] if adjustment != 0 else 0
        
        # CONSERVATIVE REFINEMENT: Only adjust if ALL conditions are met:
        # 1. Jump is significant (>800µA, not just noise)
        # 2. Before current is sleep-like (<1000µA)
        # 3. After current is wakeup-like (>1500µA)
        # 4. Time shift is meaningful (>0.01s = 10ms)
        # 5. Adjustment is not too far (within ±2s of original)
        
        if (jump_size > 800 and                    # Significant jump threshold
            before_current < 1000 and              # Before is sleep-like
            after_current > 1500 and               # After is wakeup-like
            abs(time_shift) > 0.01 and            # Shift is meaningful (>10ms)
            abs(time_shift) < 2.0 and             # Not too far from original
            adjustment != 0):                      # Actually needs adjustment
            
            # Shift the transition: set states before actual_transition_idx to 3
            if adjustment < 0:
                # Transition should be earlier
                refined_states[actual_transition_idx:trans_idx] = 3
            else:
                # Transition should be later
                refined_states[trans_idx:actual_transition_idx] = 3
            
            refinements += 1
            if verbose:
                print(f"   ✅ Refined transition at t={times[trans_idx]:.3f}s:")
                print(f"      Original index: {trans_idx}, Adjusted to: {actual_transition_idx} (shift: {adjustment} samples, {time_shift:.6f}s)")
                print(f"      Current jump: {before_current:.0f}µA → {after_current:.0f}µA (+{jump_size:.0f}µA)")
                print(f"      New transition time: t={times[actual_transition_idx]:.6f}s")
                print(f"      ✓ Validated: jump>{800}µA, before<{1000}µA, after>{1500}µA")
        elif jump_size > 500 and verbose:
            # Log why we rejected this potential refinement
            reasons = []
            if jump_size <= 800:
                reasons.append(f"jump too small ({jump_size:.0f}µA ≤ 800µA)")
            if before_current >= 1000:
                reasons.append(f"before not sleep-like ({before_current:.0f}µA ≥ 1000µA)")
            if after_current <= 1500:
                reasons.append(f"after not wakeup-like ({after_current:.0f}µA ≤ 1500µA)")
            if abs(time_shift) <= 0.01:
                reasons.append(f"shift too small ({abs(time_shift)*1000:.1f}ms ≤ 10ms)")
            if abs(time_shift) >= 2.0:
                reasons.append(f"shift too large ({abs(time_shift):.3f}s ≥ 2s)")
            
            if reasons:
                print(f"   ℹ️  Skipped potential refinement at t={times[trans_idx]:.3f}s: {', '.join(reasons)}")
    
    if verbose:
        if refinements > 0:
            print(f"   🎯 Total refinements: {refinements}")
        else:
            print(f"   ℹ️ No significant refinements needed")
    
    return refined_states


def extend_period_by_seconds(times, currents, start_time, end_time, activity_threshold, extend_seconds, comm_type):
    """
    USER'S CORRECT METHOD:
    1. Calculate baseline from 9-10 seconds of the measurement
    2. Find start/end where current becomes >2x baseline

    Args:
        times: Time array
        currents: Current array
        start_time, end_time: Original period boundaries
        activity_threshold: Not used - we calculate baseline dynamically
        extend_seconds: How many seconds to look before/after (e.g., 2.0)
        comm_type: "IR" or "BLE" for logging

    Returns:
        (extended_start, extended_end): New period boundaries
    """
    print(f"🔧 Extending {comm_type} period {start_time:.3f}s-{end_time:.3f}s using baseline method...")

    # STEP 1: Calculate baseline from 9-10 seconds of the measurement
    baseline_start_time = 9.0  # 9th second
    baseline_end_time = 10.0  # 10th second

    baseline_start_idx = np.searchsorted(times, baseline_start_time)
    baseline_end_idx = np.searchsorted(times, baseline_end_time)

    if baseline_end_idx > baseline_start_idx:
        before_baseline_currents = currents[baseline_start_idx:baseline_end_idx]
        before_baseline = np.median(before_baseline_currents)
    else:
        before_baseline = np.median(currents[:100])  # Fallback to first 100 samples

    # Use the single baseline from 9-10 seconds
    baseline = before_baseline
    threshold_2x = baseline * 2.0  # 2x baseline threshold

    print(f"  📊 Baseline (9-10s): {baseline:.0f}μA")
    print(f"  🎯 Threshold (2x baseline): {threshold_2x:.0f}μA")

    # STEP 3: Look backwards to find TRUE START (first time current >2x baseline)
    look_back_time = start_time - extend_seconds
    look_back_idx = np.searchsorted(times, look_back_time)
    start_idx = np.searchsorted(times, start_time)

    extended_start = start_time  # Default: no extension

    if look_back_idx < start_idx:
        look_back_currents = currents[look_back_idx:start_idx]
        above_threshold = look_back_currents > threshold_2x

        if np.any(above_threshold):
            # Find FIRST point above 2x baseline
            first_above_idx = np.argmax(above_threshold)  # First True
            extended_start = times[look_back_idx + first_above_idx]
            back_extension = start_time - extended_start
            print(f"  🔙 TRUE START found: {extended_start:.3f}s (extended backward by {back_extension:.3f}s)")
            print(f"      Current at start: {look_back_currents[first_above_idx]:.0f}μA > {threshold_2x:.0f}μA")

    # STEP 4: Look forwards to find TRUE END (last time current >2x baseline)
    look_forward_time = end_time + extend_seconds
    end_idx = np.searchsorted(times, end_time)
    look_forward_idx = np.searchsorted(times, look_forward_time)

    extended_end = end_time  # Default: no extension

    if end_idx < look_forward_idx and look_forward_idx <= len(times):
        look_forward_currents = currents[end_idx:look_forward_idx]
        above_threshold = look_forward_currents > threshold_2x

        if np.any(above_threshold):
            # Find LAST point above 2x baseline
            last_above_idx = len(above_threshold) - 1 - np.argmax(above_threshold[::-1])  # Last True
            extended_end = times[end_idx + last_above_idx]
            forward_extension = extended_end - end_time
            print(f"  🔜 TRUE END found: {extended_end:.3f}s (extended forward by {forward_extension:.3f}s)")
            print(f"      Current at end: {look_forward_currents[last_above_idx]:.0f}μA > {threshold_2x:.0f}μA")

    total_extension = (start_time - extended_start) + (extended_end - end_time)
    if total_extension > 0.001:  # Report if extended by > 1ms
        print(
            f"  ✅ {comm_type} FINAL: {extended_start:.3f}s-{extended_end:.3f}s (total extension: {total_extension:.3f}s)")
    else:
        print(f"  📋 {comm_type} unchanged: {extended_start:.3f}s-{extended_end:.3f}s (no 2x baseline activity)")

    return extended_start, extended_end


# Global variable to control detection method (set by command line argument)
USE_ADVANCED_DETECTION = None  # Will be set by argument parsing


def add_vlc_ble_markers(times, currents, modes, ax):
    """
    Add IR/BLE markers using either simple or advanced detection method.
    """
    if USE_ADVANCED_DETECTION:
        add_vlc_ble_markers_advanced(times, currents, modes, ax, BASELINE_START_TIME, BASELINE_END_TIME,
                                     IR_THRESHOLD_MULTIPLIER, IR_MIN_SAMPLES,
                                     BLE_THRESHOLD_MULTIPLIER, BLE_MIN_SAMPLES, SEARCH_WINDOW_SECONDS)
    else:
        add_vlc_ble_markers_simple(times, currents, modes, ax)


def find_continuous_periods(times, mask):
    """
    Find continuous time periods where mask is True.
    Returns list of (start_time, end_time) tuples.
    """
    if not np.any(mask):
        return []

    periods = []
    in_period = False
    start_time = None

    for i, (time, is_active) in enumerate(zip(times, mask)):
        if is_active and not in_period:
            # Start of new period
            start_time = time
            in_period = True
        elif not is_active and in_period:
            # End of current period
            end_time = times[i - 1] if i > 0 else time
            periods.append((start_time, end_time))
            in_period = False

    # Handle case where period extends to the end
    if in_period and start_time is not None:
        periods.append((start_time, times[-1]))

    return periods


def plot_running_state_timeline(times, running_states, running_modes, comm_modes, ax_state):
    """
    Plot Running_State changes as a timeline with |<--s0-->|<--s1-->| format.
    Also displays Running_Mode information inside the state bars.
    Shows BLE Connected overlay when Running_State=2 AND Comm_Mode=2.
    
    Args:
        times: numpy array of timestamps
        running_states: numpy array of Running_State values (used for bar colors/states)
        running_modes: numpy array of Running_Mode values (displayed inside bars)
        comm_modes: numpy array of Comm_Mode values (for BLE Connected detection)
        ax_state: matplotlib axis for running state subplot
    
    This creates a visual timeline showing when the device is in different running states.
    """
    if running_states is None or len(running_states) == 0:
        ax_state.text(0.5, 0.5, 'No Running_State data available', 
                     ha='center', va='center', transform=ax_state.transAxes)
        ax_state.set_xlim(times[0], times[-1])
        ax_state.set_ylim(0, 1)
        return
    
    # Detect state changes and create periods
    # Store both Running_State and Running_Mode for each period
    state_periods = []
    current_state = None
    current_mode = None
    start_time = None
    start_idx = 0
    
    for i, (t, state, mode) in enumerate(zip(times, running_states, running_modes)):
        if state != current_state:
            # State changed
            if current_state is not None and start_time is not None:
                # Get the most common mode in this period
                mode_in_period = running_modes[start_idx:i]
                if len(mode_in_period) > 0:
                    most_common_mode = int(np.bincount(mode_in_period).argmax())
                else:
                    most_common_mode = 0
                # Save previous period with (start, end, state, mode)
                state_periods.append((start_time, times[i-1] if i > 0 else t, current_state, most_common_mode))
            # Start new period
            current_state = state
            start_time = t
            start_idx = i
    
    # Add final period
    if current_state is not None and start_time is not None:
        mode_in_period = running_modes[start_idx:]
        if len(mode_in_period) > 0:
            most_common_mode = int(np.bincount(mode_in_period).argmax())
        else:
            most_common_mode = 0
        state_periods.append((start_time, times[-1], current_state, most_common_mode))
    
    if len(state_periods) == 0:
        ax_state.text(0.5, 0.5, 'No state changes detected', 
                     ha='center', va='center', transform=ax_state.transAxes)
        ax_state.set_xlim(times[0], times[-1])
        ax_state.set_ylim(0, 1)
        return
    
    # Color map for different running states (distinct, non-overlapping colors)
    state_colors = {
        0: '#FFB300',  # Amber - Running state 0 (wakeup/active)
        1: '#1976D2',  # Dark Blue - Running state 1 
        2: '#7B1FA2',  # Purple - Running state 2
        3: '#757575',  # Gray - Running state 3 (sleep/idle)
    }
    
    # Plot timeline bars
    y_pos = 0.5
    bar_height = 0.6
    
    for start_t, end_t, state, mode in state_periods:
        duration = end_t - start_t
        if duration <= 0:
            continue
            
        # Get color for this state (based on Running_State)
        color = state_colors.get(int(state), '#FFC107')  # Default yellow for unknown states
        
        # Draw horizontal bar for Running_State
        ax_state.barh(y_pos, duration, height=bar_height, left=start_t, 
                     color=color, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Add label with Running_State in the middle of the bar
        mid_time = start_t + duration / 2
        label_text = f"s{int(state)}"
        
        # Only show label if bar is wide enough
        time_range = times[-1] - times[0]
        min_width_for_label = time_range * 0.01  # Show label if bar is at least 1% of total width
        
        if duration >= min_width_for_label:
            ax_state.text(mid_time, y_pos, label_text, 
                         ha='center', va='center', fontsize=12, fontweight='bold', color='white',
                         zorder=4)  # Above hatching (zorder=3)
        
        # Add state transition markers |
        ax_state.axvline(start_t, color='black', linewidth=1.5, alpha=0.8, ymin=0.2, ymax=0.8)
    
    # Add final marker at the end
    if len(state_periods) > 0:
        ax_state.axvline(state_periods[-1][1], color='black', linewidth=1.5, alpha=0.8, ymin=0.2, ymax=0.8)
    
    # OVERLAY: Add BLE/IR markers as transparent overlays (same as canvas)
    # Find continuous periods for BLE (Running_Mode = 1)
    ble_periods = find_continuous_periods(times, running_modes == 1)
    ble_count = 0
    if ble_periods:
        ble_count = len(ble_periods)
        for start_time, end_time in ble_periods:
            # Create light green transparent overlay for BLE duration (same as canvas)
            ax_state.axvspan(start_time, end_time, alpha=0.35, color='lightgreen',
                           zorder=2, edgecolor='darkgreen', linewidth=1)
        print(f"🟢 BLE OVERLAY: {ble_count} periods marked with light green (same as canvas)")
    
    # Find continuous periods for IR (Running_Mode = 2)
    vlc_periods = find_continuous_periods(times, running_modes == 2)
    vlc_count = 0
    if vlc_periods:
        vlc_count = len(vlc_periods)
        for start_time, end_time in vlc_periods:
            # Create light red transparent overlay for IR duration (same as canvas)
            ax_state.axvspan(start_time, end_time, alpha=0.35, color='lightcoral',
                           zorder=2, edgecolor='darkred', linewidth=1)
        print(f"🔴 IR OVERLAY: {vlc_count} periods marked with light red (same as canvas)")
    
    # NEW: Find periods for BLE Connected (Running_State = 2 AND Comm_Mode = 2)
    # This shows when BLE is actually connected during Running_State=2
    ble_connected_periods = find_continuous_periods(times, (running_states == 2) & (comm_modes == 2))
    ble_connected_count = 0
    if ble_connected_periods:
        ble_connected_count = len(ble_connected_periods)
        for start_time, end_time in ble_connected_periods:
            # Create hatching pattern overlay on top of s2 bars
            # Match the exact height of the state bars: y_pos=0.5, bar_height=0.6
            # So ymin = 0.5 - 0.3 = 0.2, ymax = 0.5 + 0.3 = 0.8
            ax_state.axvspan(start_time, end_time, ymin=0.2, ymax=0.8, facecolor='none',
                           zorder=3, edgecolor='black', linewidth=0, hatch='////')  # Black diagonal lines only
        print(f"🔵 BLE CONNECTED OVERLAY: {ble_connected_count} periods marked with diagonal hatching (Running_State=2 & Comm_Mode=2)")
    
    # Formatting
    ax_state.set_xlim(times[0], times[-1])
    ax_state.set_ylim(0, 1)
    ax_state.set_ylabel('Running\nState', fontsize=14, rotation=0, ha='right', va='center')
    ax_state.set_yticks([])
    ax_state.grid(True, axis='x', alpha=0.3)
    ax_state.set_xlabel('Time (s)', fontsize=16)
    ax_state.tick_params(labelsize=14)
    
    # Add legend with both state colors and BLE/IR overlays - ALL IN ONE ROW
    from matplotlib.patches import Patch
    legend_elements = []
    
    # Add ALL state colors to legend (in order: 0, 1, 2, 3)
    state_names = {
        0: 'State 0',
        1: 'State 1',
        2: 'State 2',
        3: 'State 3',
    }
    # Always show all 4 states in order
    for state in [0, 1, 2, 3]:
        legend_elements.append(Patch(facecolor=state_colors[state], edgecolor='black', alpha=0.8,
                                    label=state_names[state]))
    
    # Add BLE/IR overlays to legend if they exist
    if ble_count > 0:
        legend_elements.append(Patch(facecolor='lightgreen', edgecolor='darkgreen', alpha=0.35,
                                    label='BLE'))
    if vlc_count > 0:
        legend_elements.append(Patch(facecolor='lightcoral', edgecolor='darkred', alpha=0.35,
                                    label='IR'))
    
    # Add BLE Connected overlay to legend if it exists (after State 3)
    # Show purple State 2 rectangle with black hatching on top
    if ble_connected_count > 0:
        legend_elements.append(Patch(facecolor='#7B1FA2', edgecolor='black', linewidth=0.5, alpha=0.8, hatch='////',
                                    label='BLE Connected'))
    
    if legend_elements:
        # Display all legend items in ONE ROW ABOVE the plot
        ax_state.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 1.05, 1, 0.2), 
                       fontsize=11, ncol=len(legend_elements), mode='expand', borderaxespad=0, frameon=True)
    
    print(f"📊 RUNNING STATE TIMELINE: Plotted {len(state_periods)} state periods")


def plot_linear_mode(times, currents, ax, modes=None, color='blue'):
    """Original linear interpolation method with IR/BLE markers"""
    ax.plot(times, currents, color=color, lw=1.2, rasterized=False, label='Linear Interpolation')

    # Add IR/BLE markers
    add_vlc_ble_markers(times, currents, modes, ax)

    return ax


def plot_statistical_mode(times, currents, ax, modes=None, bins=1000, color='darkblue'):
    """PPK2-Python.py dark blue curve method - statistical binning with IR/BLE markers"""
    t0, t1 = float(times[0]), float(times[-1])
    
    # Use statistical binning to create smooth mean curve
    centers, _, _, means, _ = DataAccumulator.bin_min_max_mean(times, currents, t0, t1, bins)
    
    if centers.size > 0:
        # Plot the dark blue mean curve (like PPK2-Python.py)
        ax.plot(centers, means, color=color, lw=1.5, rasterized=False, 
                label=f'Statistical Mean (Dark Blue Curve) - {bins} bins')
        print(f"✅ Statistical binning: {len(times):,} samples → {len(centers):,} smooth points")
    else:
        # Fallback to original data
        ax.plot(times, currents, color=color, lw=1.2, rasterized=False, 
                label='Statistical Mean (fallback)')
    
    # Add IR/BLE markers on top of the statistical curve
    add_vlc_ble_markers(times, currents, modes, ax)

    return ax


def plot_envelope_mode(times, currents, ax, modes=None, bins=1000):
    """Full PPK2-Python.py envelope method - min/max + mean with IR/BLE markers"""
    t0, t1 = float(times[0]), float(times[-1])
    
    # Use statistical binning to create envelope
    centers, mins, maxs, means, _ = DataAccumulator.bin_min_max_mean(times, currents, t0, t1, bins)
    
    if centers.size > 0:
        # Plot light blue envelope (min/max)
        ax.plot(centers, mins, color='#2196f3', lw=1, alpha=0.7, label='Min Envelope')
        ax.plot(centers, maxs, color='#2196f3', lw=1, alpha=0.7, label='Max Envelope')
        
        # Fill between min and max (like PPK2-Python.py)
        ax.fill_between(centers, mins, maxs, color='#2196f3', alpha=0.2, label='Current Range')
        
        # Dark blue mean curve (the main signal)
        ax.plot(centers, means, color='darkblue', lw=1.5, rasterized=False, 
                label=f'Mean Current (Dark Blue) - {bins} bins')
        
        print(f"✅ Envelope mode: {len(times):,} samples → {len(centers):,} envelope points")
    else:
        # Fallback to original data
        ax.plot(times, currents, color='darkblue', lw=1.2, rasterized=False, 
                label='Envelope (fallback)')
    
    # Add IR/BLE markers on top of the envelope
    add_vlc_ble_markers(times, currents, modes, ax)

    return ax


def plot_guest_app_mode(times, currents, ax, modes=None, bins=None, time_window=None, zoom_detail=False,
                        zoom_window=0.01):
    """
    EXACT GUEST APP CANVAS REPRODUCTION
    Matches PPK2-Python.py canvas appearance exactly:
    - White background (setBackground('w'))
    - Light blue envelope RGB(33, 150, 243) with alpha=60
    - Dark blue mean curve (pen='b')
    - RAW MODE for 100kHz data (no binning - direct plot like canvas)
    - ZOOM-AWARE adaptive binning for lower resolution data
    """
    colors = GUEST_APP_COLORS
    t0, t1 = float(times[0]), float(times[-1])
    data_duration = t1 - t0
    sample_rate = len(times) / data_duration if data_duration > 0 else 1
    
    # SPECIAL HANDLING for HIGH-RESOLUTION DATA: Use guest app's approach
    if sample_rate >= 80000:  # 80kHz+ = true 100kHz raw data
        print(f"🚀 100kHz RAW DATA: {len(times):,} samples at {sample_rate:.0f}Hz")
        
        # Guest app uses SHORT time windows for 100kHz: [0.010, 0.100, 1.0, 3.0, 10.0]
        # Simulate a "zoomed" view by using a shorter time window
        total_duration = data_duration
        
        # Choose time window - either user-specified or auto-detect
        if time_window is not None:
            # User specified time window via --time-window parameter (for zoomed detail view)
            window_duration = min(time_window, total_duration)
            start_time = times[0] + max(0, (total_duration - window_duration) * 0.5)
            print(f"🎯 USER-SPECIFIED TIME WINDOW: {window_duration:.3f}s (zoomed detail view)")
        else:
            # DEFAULT: Show FULL dataset with ZOOM-RESPONSIVE binning
            window_duration = total_duration
            start_time = times[0]
            print(f"📊 FULL DATASET VIEW: {window_duration:.3f}s (zoom-responsive like guest app)")
            print(f"🔍 ZOOM BEHAVIOR: Matplotlib zoom will trigger higher detail automatically")
        
        end_time = start_time + window_duration
        
        # Filter data to the time window (simulate guest app zoom)
        mask = (times >= start_time) & (times <= end_time)
        window_times = times[mask]
        window_currents = currents[mask]
        
        if len(window_times) == 0:
            # Fallback to full data
            window_times, window_currents = times, currents
            window_duration = total_duration
            
        print(f"📊 TIME WINDOW SIMULATION: {window_duration:.3f}s window ({len(window_times):,} samples)")
        print(f"🔍 This simulates guest app's zoomed view for 100kHz data")
        
        # Now use the same binning approach as guest app for this time window
        t0, t1 = float(window_times[0]), float(window_times[-1])
        
        # Calculate bins with ZOOM-RESPONSIVE strategy
        fig = ax.get_figure()
        width_inches = fig.get_size_inches()[0]
        dpi = fig.get_dpi()
        plot_width_pixels = int(width_inches * dpi * 0.8)
        
        # ZOOM-RESPONSIVE: Use much higher bin density for 100kHz data
        # This allows matplotlib zoom to reveal fine details
        base_bins = max(300, plot_width_pixels)
        
        # ZOOM DETAIL MODE: Simulate guest app's zoom behavior
        if zoom_detail:
            # Guest app zoom: same bins (~800) but TINY time window
            # Simulate extreme zoom: 0.01s window with 800 bins = 0.0000125s per bin
            # At 100kHz, that's ~1.25 samples per bin = nearly raw data!
            
            if window_duration > zoom_window * 2:  # Large window - force tiny zoom window
                zoom_window_duration = zoom_window  # User-specified zoom window
                # Take a representative section from the middle
                zoom_start = start_time + (window_duration - zoom_window_duration) * 0.5
                zoom_end = zoom_start + zoom_window_duration
                
                # Filter to zoom window
                zoom_mask = (window_times >= zoom_start) & (window_times <= zoom_end)
                zoom_times = window_times[zoom_mask]
                zoom_currents = window_currents[zoom_mask]
                
                if len(zoom_times) > 0:
                    window_times, window_currents = zoom_times, zoom_currents
                    window_duration = zoom_window_duration
                    t0, t1 = float(window_times[0]), float(window_times[-1])
                    print(f"🔍 EXTREME ZOOM: {zoom_window_duration:.3f}s window ({len(zoom_times):,} samples)")
            
            # Use standard guest app bins for the tiny window
            bins = base_bins  # ~800 bins like guest app
            samples_per_bin = len(window_times) / bins if bins > 0 else 1
            print(f"🎯 GUEST APP ZOOM SIMULATION: {bins} bins, {samples_per_bin:.1f} samples/bin")
            
        elif sample_rate >= 80000:
            # High resolution mode for 100kHz data
            bins = min(base_bins * 2, len(window_times) // 50)  # 2x density
            bins = max(bins, base_bins)  # At least base bins
            print(f"🚀 HIGH-RES MODE: {bins} bins (100kHz optimized)")
        else:
            bins = base_bins
            print(f"🎨 STANDARD BINNING: {bins} bins")
        
        print(f"📊 Window: {window_duration:.3f}s, Samples: {len(window_times):,}, Bins: {bins}")
        
        # ADAPTIVE PLOTTING: If very few samples per bin, plot raw data instead
        samples_per_bin = len(window_times) / bins if bins > 0 else 1
        
        if samples_per_bin <= 3 and len(window_times) <= 10000:  # Very high zoom - plot raw data
            print(f"🎯 RAW DATA MODE: {samples_per_bin:.1f} samples/bin ≤ 3 → plotting individual points")
            
            # Plot raw data directly (like guest app at maximum zoom)
            ax.plot(window_times, window_currents, 
                    color=colors['dark_blue'], 
                    linewidth=0.8,
                    alpha=0.9, 
                    marker='.',
                    markersize=1,
                    zorder=2, 
                    label='Raw Current (Individual Points)')
            
            print(f"🔍 RAW PLOT: {len(window_times):,} individual data points (maximum detail)")

            # Add IR/BLE markers for raw data mode
            if modes is not None and len(modes) == len(times):
                window_modes = modes[mask] if 'mask' in locals() else modes
                add_vlc_ble_markers(window_times, window_currents, window_modes, ax)

            return ax
        
        # STANDARD BINNING: Use guest app's statistical approach
        print(f"📊 BINNING MODE: {samples_per_bin:.1f} samples/bin → using statistical binning")
        centers, mins, maxs, means, _ = DataAccumulator.bin_min_max_mean(window_times, window_currents, t0, t1, bins)
        
        # Plot the 4-layer system just like guest app
        if centers.size > 0:
            # Convert RGB to matplotlib format (0-1 range)
            light_blue_rgb = tuple(c / 255.0 for c in colors['light_blue'])
            envelope_alpha = colors['envelope_alpha'] / 255.0
            
            # 1. Light blue MIN curve (detailed signal information)
            ax.plot(centers, mins, 
                    color=light_blue_rgb, 
                    linewidth=colors['line_width'],
                    alpha=1.0, zorder=1)
            
            # 2. Light blue MAX curve (detailed signal information)  
            ax.plot(centers, maxs, 
                    color=light_blue_rgb, 
                    linewidth=colors['line_width'],
                    alpha=1.0, zorder=1)
            
            # 3. Fill between min and max (envelope background)
            ax.fill_between(centers, mins, maxs, 
                           color=light_blue_rgb, 
                           alpha=envelope_alpha, 
                           zorder=0, label='Current Range')
            
            # 4. Blue mean curve (main signal) - EXACT guest app style
            ax.plot(centers, means, 
                    color=colors['dark_blue'],
                    linewidth=colors['line_width'],
                    zorder=2, label='Mean Current')
            
            print(f"🎯 100kHz PLOT: {len(window_times):,} samples → {len(centers):,} canvas points")
            print(f"📊 EXACT guest app reproduction: 4-layer plot with time window simulation")
            
            # Add IR/BLE markers for the windowed data
            if modes is not None and len(modes) == len(times):
                window_modes = modes[mask] if 'mask' in locals() else modes
                add_vlc_ble_markers(window_times, window_currents, window_modes, ax)

        return ax  # Skip the standard binning mode
    
    # STANDARD MODE: Use adaptive binning for lower resolution data  
    print(f"📊 BINNING MODE: {sample_rate:.0f}Hz data - using adaptive binning")
    
    # Calculate bins with ZOOM AWARENESS like guest app
    if bins is None:
        fig = ax.get_figure()
        width_inches = fig.get_size_inches()[0]
        dpi = fig.get_dpi()
        plot_width_pixels = int(width_inches * dpi * 0.8)  # Account for margins
        
        # Calculate adaptive bins based on both plot width AND data density
        base_bins = max(300, plot_width_pixels)
        
        # If we have high-resolution data, use more bins to show detail
        if sample_rate > 5000:  # High sample rate data (like 10kHz+)
            # Use more bins for high-res data, but cap at reasonable limit
            detail_multiplier = min(3.0, sample_rate / 5000)
            bins = int(base_bins * detail_multiplier)
        else:
            bins = base_bins
            
        # Cap at reasonable maximum for performance
        bins = min(bins, 5000)
        
        print(f"🎨 ZOOM-AWARE BINNING: {plot_width_pixels}px plot, {sample_rate:.0f}Hz data → {bins} bins")
    
    # Use EXACT same statistical binning as guest app
    centers, mins, maxs, means, _ = DataAccumulator.bin_min_max_mean(times, currents, t0, t1, bins)
    
    if centers.size > 0:
        # Convert RGB to matplotlib format (0-1 range)
        light_blue_rgb = tuple(c / 255.0 for c in colors['light_blue'])
        envelope_alpha = colors['envelope_alpha'] / 255.0  # Convert alpha to 0-1 range
        
        # Light blue envelope - EXACT guest app implementation
        # Guest app plots THREE separate curves: curve_min, curve_max, curve_mean + fill
        
        # 1. Light blue MIN curve (detailed signal information)
        ax.plot(centers, mins, 
                color=light_blue_rgb, 
                linewidth=colors['line_width'],
                alpha=1.0, zorder=1)
        
        # 2. Light blue MAX curve (detailed signal information)  
        ax.plot(centers, maxs, 
                color=light_blue_rgb, 
                linewidth=colors['line_width'],
                alpha=1.0, zorder=1)
        
        # 3. Fill between min and max (envelope background)
        ax.fill_between(centers, mins, maxs, 
                       color=light_blue_rgb, 
                       alpha=envelope_alpha, 
                       zorder=0, label='Current Range')
        
        # 4. Blue mean curve (main signal) - EXACT guest app style
        ax.plot(centers, means, 
                color=colors['dark_blue'],  # 'b' = blue
                linewidth=colors['line_width'],
                zorder=2, label='Mean Current')
        
        print(f"🎨 GUEST APP MODE: {len(times):,} samples → {len(centers):,} canvas points")
        print(f"📊 PLOTTING 4 LAYERS: min curve + max curve + fill + mean curve (like guest app)")
        print(f"🎯 EXACT PPK2-Python.py reproduction with HIGH DETAIL + SMOOTH curves")
    else:
        # Fallback to original data with guest app colors
        ax.plot(times, currents, 
                color=colors['dark_blue'], 
                linewidth=colors['line_width'], 
                label='Current (fallback)')
    
    # Add IR/BLE markers for standard binning mode
    # Use global parameters defined in main()
    global BASELINE_START_TIME, BASELINE_END_TIME, IR_THRESHOLD_MULTIPLIER, IR_MIN_SAMPLES
    global BLE_THRESHOLD_MULTIPLIER, BLE_MIN_SAMPLES, SEARCH_WINDOW_SECONDS
    add_vlc_ble_markers(times, currents, modes, ax)

    return ax


# ================================================================================================
# 🎯 DETECTION PARAMETERS - IMPORTED FROM detection_parameters.py
# ================================================================================================

try:
    from detection_parameters import (
        BASELINE_START_TIME, BASELINE_END_TIME,
        IR_THRESHOLD_MULTIPLIER, IR_MIN_SAMPLES,
        BLE_THRESHOLD_MULTIPLIER, BLE_MIN_SAMPLES,
        SEARCH_WINDOW_SECONDS, DEFAULT_OUTLIER_FACTOR, DEFAULT_OUTLIER_WINDOW
    )

    print("📥 Using parameters from detection_parameters.py")
except ImportError:
    # Fallback built-in parameters
    BASELINE_START_TIME = 9.0  # Start of baseline window (seconds)
    BASELINE_END_TIME = 10.0  # End of baseline window (seconds)
    IR_THRESHOLD_MULTIPLIER = 2.5  # IR threshold = baseline × this value
    IR_MIN_SAMPLES = 400  # Minimum consecutive samples for IR (4.0ms @ 100kHz)
    BLE_THRESHOLD_MULTIPLIER = 3.0  # BLE threshold = baseline × this value
    BLE_MIN_SAMPLES = 500  # Minimum consecutive samples for BLE (5.0ms @ 100kHz)
    SEARCH_WINDOW_SECONDS = 1.0  # ±seconds around original parameter periods
    DEFAULT_OUTLIER_FACTOR = 20.0  # Outlier detection factor (20x median) - Increased threshold
    DEFAULT_OUTLIER_WINDOW = 50  # Window size for outlier detection
    print("⚙️  Using fallback built-in parameters")


# ================================================================================================

def get_detection_parameters():
    """Get current detection parameters for use by other scripts"""
    return {
        'baseline_start': BASELINE_START_TIME,
        'baseline_end': BASELINE_END_TIME,
        'vlc_threshold_mult': IR_THRESHOLD_MULTIPLIER,
        'vlc_min_samples': IR_MIN_SAMPLES,
        'ble_threshold_mult': BLE_THRESHOLD_MULTIPLIER,
        'ble_min_samples': BLE_MIN_SAMPLES,
        'search_window': SEARCH_WINDOW_SECONDS,
        'outlier_factor': DEFAULT_OUTLIER_FACTOR
    }


def process_dataset_with_detection(times, currents, modes):
    """
    Process a dataset with IR/BLE boundary detection using current parameters.

    Args:
        times: List of timestamps
        currents: List of current values
        modes: List of mode values

    Returns:
        Dictionary with refined boundaries for IR and BLE periods
    """
    # Apply outlier filtering
    times_filtered, currents_filtered, modes_filtered = filter_extreme_outliers(
        times, currents, modes,
        outlier_factor=DEFAULT_OUTLIER_FACTOR,
        window_size=DEFAULT_OUTLIER_WINDOW
    )

    # Calculate baseline
    baseline_currents = []
    for i, t in enumerate(times_filtered):
        if BASELINE_START_TIME <= t <= BASELINE_END_TIME:
            baseline_currents.append(currents_filtered[i])

    if baseline_currents:
        baseline_currents_sorted = sorted(baseline_currents)
        n = len(baseline_currents_sorted)
        if n % 2 == 0:
            baseline = (baseline_currents_sorted[n // 2 - 1] + baseline_currents_sorted[n // 2]) / 2
        else:
            baseline = baseline_currents_sorted[n // 2]
    else:
        # Use built-in median calculation for compatibility
        currents_sorted = sorted(currents_filtered)
        n = len(currents_sorted)
        if n % 2 == 0:
            baseline = (currents_sorted[n // 2 - 1] + currents_sorted[n // 2]) / 2
        else:
            baseline = currents_sorted[n // 2]

    # Find IR and BLE periods with validation
    vlc_periods = []
    ble_periods = []

    current_mode = None
    period_start = None

    for i, mode in enumerate(modes_filtered):
        if mode != current_mode:
            # End previous period
            if current_mode is not None and period_start is not None:
                period_end = times_filtered[i - 1] if i > 0 else times_filtered[i]

                # Validate IR periods - check if current is actually higher than baseline*2
                if current_mode == 2:  # IR
                    # Get current values in this period
                    period_currents = []
                    for j, t in enumerate(times_filtered):
                        if period_start <= t <= period_end:
                            period_currents.append(currents_filtered[j])

                    # Check if ANY current in the period exceeds baseline*2
                    vlc_threshold = baseline * 2.0
                    has_high_current = any(c > vlc_threshold for c in period_currents)

                    if has_high_current:
                        vlc_periods.append((period_start, period_end))
                        print(
                            f"✅ IR period {period_start:.3f}s-{period_end:.3f}s: Valid (max current {max(period_currents):.1f}mA > {vlc_threshold:.1f}mA)")
                    else:
                        print(
                            f"❌ IR period {period_start:.3f}s-{period_end:.3f}s: Invalid (max current {max(period_currents):.1f}mA ≤ {vlc_threshold:.1f}mA) - Converting to mode 0")
                        # Convert this period to mode 0 in the filtered data
                        for j, t in enumerate(times_filtered):
                            if period_start <= t <= period_end:
                                modes_filtered[j] = 0

                elif current_mode == 1:  # BLE
                    ble_periods.append((period_start, period_end))

            # Start new period
            current_mode = mode
            period_start = times_filtered[i]

    # Handle last period
    if current_mode is not None and period_start is not None:
        period_end = times_filtered[-1]

        # Validate IR periods - check if current is actually higher than baseline*2
        if current_mode == 2:  # IR
            # Get current values in this period
            period_currents = []
            for j, t in enumerate(times_filtered):
                if period_start <= t <= period_end:
                    period_currents.append(currents_filtered[j])

            # Check if ANY current in the period exceeds baseline*2
            vlc_threshold = baseline * 2.0
            has_high_current = any(c > vlc_threshold for c in period_currents)

            if has_high_current:
                vlc_periods.append((period_start, period_end))
                print(
                    f"✅ IR period {period_start:.3f}s-{period_end:.3f}s: Valid (max current {max(period_currents):.1f}mA > {vlc_threshold:.1f}mA)")
            else:
                print(
                    f"❌ IR period {period_start:.3f}s-{period_end:.3f}s: Invalid (max current {max(period_currents):.1f}mA ≤ {vlc_threshold:.1f}mA) - Converting to mode 0")
                # Convert this period to mode 0 in the filtered data
                for j, t in enumerate(times_filtered):
                    if period_start <= t <= period_end:
                        modes_filtered[j] = 0

        elif current_mode == 1:  # BLE
            ble_periods.append((period_start, period_end))

    # Refine boundaries
    refined_boundaries = []

    # Process IR periods
    for start_time, end_time in vlc_periods:
        refined_start, refined_end = find_first_last_peaks(
            times_filtered, currents_filtered, start_time, end_time, baseline, "IR",
            baseline_start=BASELINE_START_TIME, baseline_end=BASELINE_END_TIME,
            vlc_threshold_mult=IR_THRESHOLD_MULTIPLIER, vlc_min_samples=IR_MIN_SAMPLES,
            ble_threshold_mult=BLE_THRESHOLD_MULTIPLIER, ble_min_samples=BLE_MIN_SAMPLES,
            search_window=SEARCH_WINDOW_SECONDS
        )
        refined_boundaries.append({
            'mode': 2,
            'comm_type': 'IR',
            'original_start': start_time,
            'original_end': end_time,
            'refined_start': refined_start,
            'refined_end': refined_end
        })

    # Process BLE periods
    for start_time, end_time in ble_periods:
        refined_start, refined_end = find_first_last_peaks(
            times_filtered, currents_filtered, start_time, end_time, baseline, "BLE",
            baseline_start=BASELINE_START_TIME, baseline_end=BASELINE_END_TIME,
            vlc_threshold_mult=IR_THRESHOLD_MULTIPLIER, vlc_min_samples=IR_MIN_SAMPLES,
            ble_threshold_mult=BLE_THRESHOLD_MULTIPLIER, ble_min_samples=BLE_MIN_SAMPLES,
            search_window=SEARCH_WINDOW_SECONDS
        )
        refined_boundaries.append({
            'mode': 1,
            'comm_type': 'BLE',
            'original_start': start_time,
            'original_end': end_time,
            'refined_start': refined_start,
            'refined_end': refined_end
        })

    return {
        'boundaries': refined_boundaries,
        'baseline': baseline,
        'filtered_data': (times_filtered, currents_filtered, modes_filtered)
    }


def apply_wakeup_detection(times, currents, modes):
    """
    Apply wake-up detection algorithm to detect when node wakes up from sleep state.
    Based on the algorithm from PPK2-Python.py

    Args:
        times: Array of timestamps
        currents: Array of current values (in μA)
        modes: Array of running modes (running_states)

    Returns:
        tuple: (corrected_modes, wakeup_states)
    """
    print("🔍 APPLYING WAKE-UP DETECTION...")

    # Wake-up detection parameters (from PPK2-Python.py)
    BASELINE_COLLECTION_TIME = 4.0  # seconds
    WAKEUP_ABSOLUTE_DIFF = 600  # μA
    WAKEUP_COUNT_REQUIRED = 100  # consecutive samples (1ms @ 100kHz)

    print(f"📊 Data info: {len(times)} samples, time range {times[0]:.3f}s - {times[-1]:.3f}s")
    print(f"📊 Current range: {currents.min():.1f}μA - {currents.max():.1f}μA")
    print(f"📊 Running states: {np.unique(modes, return_counts=True)}")

    # Create copies to modify
    corrected_modes = np.array(modes, copy=True)
    wakeup_states = np.zeros_like(modes)  # Initialize all to 0

    # Find sleep periods where running_state=3 (only condition needed)
    sleep_mask = (corrected_modes == 3)

    if not np.any(sleep_mask):
        print("ℹ️  No sleep periods found (running_state=3)")
        return corrected_modes, wakeup_states

    # Find continuous sleep periods
    sleep_periods = []
    in_sleep = False
    sleep_start = None

    for i, is_sleep in enumerate(sleep_mask):
        if is_sleep and not in_sleep:
            # Start of sleep period
            sleep_start = i
            in_sleep = True
        elif not is_sleep and in_sleep:
            # End of sleep period
            sleep_periods.append((sleep_start, i - 1))
            in_sleep = False

    # Handle case where sleep period extends to end of data
    if in_sleep:
        sleep_periods.append((sleep_start, len(sleep_mask) - 1))

    print(f"🛌 Found {len(sleep_periods)} sleep periods")

    wakeup_detections = 0

    for sleep_start_idx, sleep_end_idx in sleep_periods:
        sleep_start_time = times[sleep_start_idx]
        sleep_end_time = times[sleep_end_idx]

        print(f"🔍 Analyzing sleep period: {sleep_start_time:.3f}s - {sleep_end_time:.3f}s")

        # Calculate baseline from the entire sleep period (more robust)
        sleep_mask_period = (times >= sleep_start_time) & (times <= sleep_end_time)
        sleep_currents = currents[sleep_mask_period]

        if len(sleep_currents) == 0:
            print(f"⚠️  No sleep samples found for period")
            continue

        # Use median of sleep period as baseline (more robust than mean of first 2 seconds)
        baseline = np.median(sleep_currents)
        wakeup_threshold = baseline + WAKEUP_ABSOLUTE_DIFF

        # If most currents in this "sleep" period are already above threshold,
        # this is likely a wake-up that wasn't detected by the original system
        high_current_ratio = np.sum(sleep_currents > wakeup_threshold) / len(sleep_currents)
        if high_current_ratio > 0.5:  # More than 50% of samples are above wake-up threshold
            print(f"🚨 IMMEDIATE WAKE-UP DETECTED: {high_current_ratio * 100:.1f}% of 'sleep' period above threshold")
            print(f"📊 Sleep period baseline: {baseline:.1f}μA, Threshold: {wakeup_threshold:.1f}μA")
            print(f"📊 Sleep period current range: {sleep_currents.min():.1f}μA - {sleep_currents.max():.1f}μA")

            # Mark entire period as awake
            corrected_modes[sleep_start_idx:sleep_end_idx + 1] = 0  # Change running_state from 3 to 0
            wakeup_states[sleep_start_idx:sleep_end_idx + 1] = 1  # Set wakeup_state to 1

            wakeup_detections += 1
            continue

        print(f"📊 Baseline: {baseline:.1f}μA, Threshold: {wakeup_threshold:.1f}μA")

        # Skip first 5 seconds of sleep period (wake-up usually occurs after 10 seconds)
        SLEEP_SKIP_SECONDS = 5.0
        detection_start_time = sleep_start_time + SLEEP_SKIP_SECONDS

        # Check if we have enough data after skipping
        if detection_start_time >= sleep_end_time:
            print(
                f"⏭️  Skipping sleep period (too short: {sleep_end_time - sleep_start_time:.1f}s < {SLEEP_SKIP_SECONDS}s)")
            continue

        print(f"⏭️  Skipping first {SLEEP_SKIP_SECONDS}s of sleep period, analyzing from {detection_start_time:.3f}s")

        # Look for wake-up in the sleep period (after skipping first 5 seconds)
        detection_mask = (times >= detection_start_time) & (times <= sleep_end_time)
        detection_indices = np.where(detection_mask)[0]

        if len(detection_indices) == 0:
            continue

        # WINDOW-BASED WAKE-UP DETECTION: Use sliding window mean instead of individual samples
        WINDOW_SIZE = WAKEUP_COUNT_REQUIRED  # Use same window size as consecutive count requirement
        wakeup_detected = False
        wakeup_start_idx = None

        print(f"🔍 Using sliding window approach: {WINDOW_SIZE} samples window, threshold={wakeup_threshold:.1f}μA")

        # Slide window through detection period
        for i in range(len(detection_indices) - WINDOW_SIZE + 1):
            window_start_idx = detection_indices[i]
            window_end_idx = detection_indices[i + WINDOW_SIZE - 1]
            
            # Calculate mean current in this window
            window_currents = currents[window_start_idx:window_end_idx + 1]
            window_mean = np.mean(window_currents)
            
            # Check if window mean exceeds threshold
            if window_mean > wakeup_threshold:
                # Wake-up detected!
                wakeup_time = times[window_start_idx]
                print(f"🚨 WAKE-UP DETECTED at {wakeup_time:.3f}s")
                print(f"   📊 Window mean: {window_mean:.1f}μA > threshold: {wakeup_threshold:.1f}μA")
                print(f"   📊 Window range: {window_currents.min():.1f}μA - {window_currents.max():.1f}μA")

                # Update states from wake-up point to end of sleep period
                corrected_modes[window_start_idx:sleep_end_idx + 1] = 0  # Change running_state from 3 to 0
                wakeup_states[window_start_idx:sleep_end_idx + 1] = 1  # Set wakeup_state to 1

                wakeup_detections += 1
                wakeup_detected = True
                break

        if not wakeup_detected:
            print(f"ℹ️  No wake-up detected in this sleep period")

    print(f"✅ WAKE-UP DETECTION COMPLETE: {wakeup_detections} wake-ups detected")

    # After individual sleep period analysis, implement global wake-up propagation
    # Once wake-up is detected anywhere, all subsequent Running_State=3 should become 0
    if wakeup_detections > 0:
        print(f"🌐 APPLYING GLOBAL WAKE-UP PROPAGATION...")
        
        # Find the first wake-up point in the entire time series
        first_wakeup_idx = np.where(wakeup_states == 1)[0]
        if len(first_wakeup_idx) > 0:
            global_wakeup_start = first_wakeup_idx[0]
            global_wakeup_time = times[global_wakeup_start]
            
            print(f"🚨 Global wake-up starts at index {global_wakeup_start}, time {global_wakeup_time:.3f}s")
            
            # From the wake-up point forward, change ALL Running_State=3 to 0 and set Wakeup_State=1
            remaining_sleep_mask = (corrected_modes[global_wakeup_start:] == 3)
            remaining_sleep_indices = np.where(remaining_sleep_mask)[0] + global_wakeup_start
            
            if len(remaining_sleep_indices) > 0:
                print(f"🔄 Converting {len(remaining_sleep_indices)} additional Running_State=3 samples to 0")
                corrected_modes[remaining_sleep_indices] = 0  # Change Running_State from 3 to 0
                wakeup_states[remaining_sleep_indices] = 1    # Set Wakeup_State to 1
                
                print(f"🌐 Global propagation: All Running_State=3 after {global_wakeup_time:.3f}s → Running_State=0, Wakeup_State=1")
            else:
                print(f"ℹ️  No additional Running_State=3 samples found after wake-up point")

    return corrected_modes, wakeup_states


def main():
    parser = argparse.ArgumentParser(description="Enhanced plotting with exact guest app canvas reproduction")
    parser.add_argument("--file", default=HARDCODED_FILE, help="Path to saved CSV file (auto-detects multiple parts)")
    parser.add_argument("--mode", choices=['guest_app', 'linear', 'statistical', 'envelope'], 
                        default='guest_app', help="Plotting mode (guest_app = exact canvas reproduction)")
    parser.add_argument("--bins", type=int, default=None, 
                        help="Number of bins (None=adaptive like guest app, or specify fixed number)")
    parser.add_argument("--time-window", type=float, default=None,
                        help="Time window for 100kHz data (e.g., 0.01=10ms, 0.1=100ms, 1.0=1s zoom levels)")
    parser.add_argument("--zoom-detail", action="store_true",
                        help="Use maximum detail binning for zoom-level inspection (like guest app zoomed in)")
    parser.add_argument("--zoom-window", type=float, default=0.01,
                        help="Zoom window duration in seconds for --zoom-detail mode (default: 0.01 = 10ms)")
    parser.add_argument("--filter-outliers", action="store_true", default=True,
                        help="Apply outlier filter to remove extreme current spikes (default: enabled)")
    parser.add_argument("--no-filter", action="store_true", default=False,
                        help="Disable outlier filtering (show all original data)")
    parser.add_argument("--outlier-factor", type=float, default=DEFAULT_OUTLIER_FACTOR,
                        help=f"Outlier detection factor - remove currents X times higher than surrounding (default: {DEFAULT_OUTLIER_FACTOR})")
    parser.add_argument("--outlier-window", type=int, default=DEFAULT_OUTLIER_WINDOW,
                        help=f"Window size for outlier detection surrounding average (default: {DEFAULT_OUTLIER_WINDOW} samples)")
    parser.add_argument("--advanced-detection", action="store_true", default=True,
                        help="Use advanced IR/BLE pattern detection (analyzes current signatures for precise timing)")
    parser.add_argument("--save-refined", action="store_true", default=True,
                        help="Save refined dataset with adjusted IR/BLE timing to *_refined.csv (default: True)")
    parser.add_argument("--no-save-refined", dest="save_refined", action="store_false",
                        help="Disable saving refined dataset")
    parser.add_argument("--wakeup-detection", action="store_true", default=False,
                        help="Apply wake-up detection algorithm to correct running_state from 3 to 0 and set wakeup_state to 1")
    parser.add_argument("--refine-wakeup-transitions", action="store_true", default=True,
                        help="Refine wakeup transition points (3→0) by finding exact current jump locations (default: True)")
    parser.add_argument("--no-refine-wakeup-transitions", dest="refine_wakeup_transitions", action="store_false",
                        help="Disable wakeup transition refinement")
    parser.add_argument("--plot-running-state", action="store_true", default=True,
                        help="Plot running state timeline below main plot (default: True)")
    parser.add_argument("--no-plot-running-state", dest="plot_running_state", action="store_false",
                        help="Disable running state timeline")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument("--save", default="", help="PNG save path")
    parser.add_argument("--save-pdf", default="", help="PDF save path")
    parser.add_argument("--save-svg", default="", help="SVG save path")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (higher for guest app quality, default: 300)")
    
    args = parser.parse_args()

    # Print current detection parameters
    print("🎯 DETECTION PARAMETERS:")
    print(f"   Baseline Window: {BASELINE_START_TIME}s - {BASELINE_END_TIME}s")
    print(
        f"   IR: {IR_THRESHOLD_MULTIPLIER}x baseline, {IR_MIN_SAMPLES} samples ({IR_MIN_SAMPLES / 100:.1f}ms @ 100kHz)")
    print(
        f"   BLE: {BLE_THRESHOLD_MULTIPLIER}x baseline, {BLE_MIN_SAMPLES} samples ({BLE_MIN_SAMPLES / 100:.1f}ms @ 100kHz)")
    print(f"   Search Window: ±{SEARCH_WINDOW_SECONDS}s around parameter periods")
    print()

    # Apply hardcoded overrides if enabled
    if HARDCODED_USE:
        args.file = HARDCODED_FILE or args.file
        args.mode = HARDCODED_MODE
        args.bins = HARDCODED_BINS
        # Always apply hardcoded save paths when HARDCODED_USE is True
        if not args.save:  # Only override if not specified by user
            args.save = HARDCODED_SAVE_PNG
        if not args.save_pdf:
            args.save_pdf = HARDCODED_SAVE_PDF
        if not args.save_svg:
            args.save_svg = HARDCODED_SAVE_SVG

    # Set detection method based on argument
    global USE_ADVANCED_DETECTION
    USE_ADVANCED_DETECTION = args.advanced_detection

    if USE_ADVANCED_DETECTION:
        print("🎯 DETECTION MODE: Advanced IR/BLE pattern detection enabled")
    else:
        print("📊 DETECTION MODE: Simple parameter-based detection enabled")

    # Ensure output directory exists
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
    except Exception:
        pass

    def _into_save_dir(path: str) -> str:
        if not path:
            return path
        if not os.path.isabs(path):
            return os.path.join(SAVE_DIR, os.path.basename(path))
        return path

    args.save = _into_save_dir(args.save)
    args.save_pdf = _into_save_dir(args.save_pdf)
    args.save_svg = _into_save_dir(args.save_svg)

    # Print save locations
    if args.save or args.save_pdf or args.save_svg:
        print("\n📁 OUTPUT FILES WILL BE SAVED TO:")
        if args.save:
            print(f"   PNG: {os.path.abspath(args.save)}")
        if args.save_pdf:
            print(f"   PDF: {os.path.abspath(args.save_pdf)}")
        if args.save_svg:
            print(f"   SVG: {os.path.abspath(args.save_svg)}")
        print()

    # Close any existing plots
    plt.close('all')

    try:
        # DEBUG: Show what files are detected
        print(f"🔍 DEBUG: Requested file = '{args.file}'")
        detected_parts = find_all_csv_parts(args.file)
        print(f"🔍 DEBUG: Detected parts = {detected_parts}")
        
        # Load data including running_state, running_mode, and comm_mode
        times, currents, running_states, running_modes, comm_modes = read_time_current(args.file)
        print(f"📊 Loaded {len(times):,} samples from {args.file}")
        
        # Read node parameters for display
        csv_file_for_params = detected_parts[0] if detected_parts else args.file
        node_params = read_node_parameters(csv_file_for_params)
        print(f"📋 Node Parameters: ID={node_params['guest_id']}, Tx Power={node_params['tx_power']}, Adv Interval={node_params['adv_interval']}, Conn Interval={node_params['conn_interval']}")
        
        # For backward compatibility, modes refers to running_modes for IR/BLE detection
        modes = running_modes
        
        # Store all CSV file paths for refined dataset saving
        if args.file.lower() == 'latest':
            latest_files = find_latest_guest_files()
            actual_csv_files = latest_files if latest_files else [args.file]
        else:
            # Use detected_parts which includes all CSV files (part1, part2, etc.)
            actual_csv_files = detected_parts if detected_parts else [args.file]
        print(f"🔍 DEBUG: CSV files for saving: {actual_csv_files}")

        # Store data for potential refined dataset saving
        original_times = times.copy()
        original_currents = currents.copy()
        original_modes = modes.copy() if modes is not None else None

        # Apply outlier filter to remove extremely high current spikes (if enabled)
        if not args.no_filter:
            times, currents, modes, running_states, running_modes, comm_modes, outlier_info = filter_extreme_outliers(
                times, currents, modes,
                running_states=running_states,
                running_modes=running_modes,
                comm_modes=comm_modes,
                outlier_factor=args.outlier_factor,  # Remove currents X times higher than surrounding values
                window_size=args.outlier_window  # Use specified samples for surrounding average
            )

            # Don't modify title with filter info - keep it clean
        else:
            print(f"🔧 OUTLIER FILTER: Disabled by --no-filter flag")
            outlier_info = {"removed": 0, "total": len(currents), "kept": len(currents), "percentage": 0}

        # Apply wake-up detection if enabled
        if args.wakeup_detection:
            print(f"🔍 Wake-up detection enabled")
            corrected_modes, wakeup_states = apply_wakeup_detection(times, currents, modes)
            modes = corrected_modes  # Update modes with corrected running states

        # Check if we have running_mode data for IR/BLE markers
        if modes is not None and len(modes) > 0:
            unique_modes = np.unique(modes[modes > 0])  # Exclude mode 0
            if len(unique_modes) > 0:
                print(f"🎯 IR/BLE markers available - detected modes: {unique_modes}")
                if 1 in unique_modes:
                    print(f"🟢 BLE periods detected (running_mode = 1)")
                if 2 in unique_modes:
                    print(f"🔴 IR periods detected (running_mode = 2)")
            else:
                print(f"📊 No IR/BLE periods detected (all running_mode = 0)")
        else:
            print(f"⚠️  No running_mode column found - IR/BLE markers not available")

        # Detect data type and provide guidance  
        data_duration = times[-1] - times[0]
        sample_rate = len(times) / data_duration if data_duration > 0 else 1
        
        print(f"🔍 DATA ANALYSIS:")
        print(f"   Sample rate: {sample_rate:.0f} Hz")
        print(f"   Duration: {data_duration:.2f} seconds")
        
        if sample_rate >= 80000:
            print(f"🚀 100kHz RAW DATA: Full dataset with guest app binning")
            print(f"🔍 ZOOM OPTIONS:")
            print(f"   --zoom-detail = Maximum detail (like guest app zoomed in)")
            print(f"   --zoom-detail --zoom-window 0.001 = 1ms extreme zoom (raw points)")
            print(f"   --zoom-detail --zoom-window 0.01 = 10ms zoom (default)")
            print(f"   --time-window 0.1 = 100ms window detail view") 
            print(f"📊 FULL CANVAS REPRODUCTION: Complete data like guest app")
        elif sample_rate >= 8000:
            print(f"📊 {sample_rate:.0f}Hz DATA: Standard guest app binning")
            print(f"🎨 OPTIMIZED: Adaptive binning for smooth visualization")
        print(f"📈 Time range: {times[0]:.6f}s to {times[-1]:.6f}s ({times[-1] - times[0]:.3f}s duration)")
        print(f"⚡ Current range: {currents.min():.2f} to {currents.max():.2f} μA")
        bins_info = "adaptive (like guest app)" if args.bins is None else f"{args.bins} fixed"
        print(f"🎨 Using {args.mode} mode with {bins_info} binning")
        
        # Create figure with optional running state subplot
        if args.plot_running_state and running_states is not None:
            # Create subplot layout: main plot on top, running state timeline below
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1-RUNNING_STATE_HEIGHT_RATIO, RUNNING_STATE_HEIGHT_RATIO], 
                                 hspace=0.5)  # Increased spacing to prevent legend overlay
            ax = fig.add_subplot(gs[0])
            ax_state = fig.add_subplot(gs[1], sharex=ax)
            print(f"📊 RUNNING STATE TIMELINE: Enabled (subplot created)")
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax_state = None
            if args.plot_running_state:
                print(f"⚠️ RUNNING STATE TIMELINE: Disabled (no Running_State data available)")
            else:
                print(f"📊 RUNNING STATE TIMELINE: Disabled by user (--no-plot-running-state)")
        
        # Apply selected plotting mode with IR/BLE markers
        if args.mode == 'linear':
            plot_linear_mode(times, currents, ax, modes=modes)
        elif args.mode == 'statistical':
            plot_statistical_mode(times, currents, ax, modes=modes, bins=args.bins)
        elif args.mode == 'envelope':
            plot_envelope_mode(times, currents, ax, modes=modes, bins=args.bins)
        elif args.mode == 'guest_app':
            plot_guest_app_mode(times, currents, ax, modes=modes, bins=args.bins, time_window=args.time_window,
                                zoom_detail=args.zoom_detail, zoom_window=args.zoom_window)
        
        # Styling based on mode
        if args.mode == 'guest_app':
            # EXACT GUEST APP CANVAS STYLING - WHITE BACKGROUND
            colors = GUEST_APP_COLORS
            
            # White background (exact match: setBackground('w'))
            ax.set_facecolor(colors['background'])
            fig.patch.set_facecolor(colors['background'])
            
            # Black text on white background
            ax.set_xlabel("Time (s)", color=colors['text'], fontsize=16)
            ax.set_ylabel("Current (μA)", color=colors['text'], fontsize=16)
            # Use clean title without additional info
            ax.set_title(args.title, color=colors['text'], fontsize=18, fontweight='bold')
            
            # Add node parameters in upper left corner
            param_text = f"Node ID: {node_params['guest_id']}\n"
            param_text += f"Tx Power: {node_params['tx_power']}\n"
            param_text += f"Adv Interval: {node_params['adv_interval']}\n"
            param_text += f"Conn Interval: {node_params['conn_interval']}"
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
                   color='black', family='monospace')
            
            # Grid styling (exact match: showGrid alpha=0.3)
            ax.grid(True, color='gray', alpha=0.3, linewidth=0.5)
            
            # Legend styling (minimal, like guest app)
            legend = ax.legend(loc='upper right', framealpha=0.9, facecolor='white', fontsize=12)
            legend.get_frame().set_edgecolor('gray')
            for text in legend.get_texts():
                text.set_color('black')
            
            # Axis tick colors and sizes (black on white)
            ax.tick_params(colors='black', labelsize=14)
            
            # Spines (borders) styling
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(0.8)
                
            print("🎨 Applied exact guest app canvas styling: WHITE background")
            
        else:
            # Standard matplotlib styling for other modes
            ax.set_xlabel("Time (s)", fontsize=16)
            ax.set_ylabel("Current (μA)", fontsize=16)
            ax.set_title(f"{args.title} - {args.mode.title()} Mode", fontsize=18, fontweight='bold')
            
            # Add node parameters in upper left corner
            param_text = f"Node ID: {node_params['guest_id']}\n"
            param_text += f"Tx Power: {node_params['tx_power']}\n"
            param_text += f"Adv Interval: {node_params['adv_interval']}\n"
            param_text += f"Conn Interval: {node_params['conn_interval']}"
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
                   color='black', family='monospace')
            
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', framealpha=0.9, fontsize=12)
            ax.tick_params(labelsize=14)
            
            # White background for all modes
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
        
        # Apply refinements to running_states, running_modes, and comm_modes BEFORE plotting
        # This ensures the timeline shows the same refined data that will be saved to *_refined.csv
        refined_running_states = running_states.copy() if running_states is not None else None
        refined_running_modes = running_modes.copy() if running_modes is not None else None
        refined_comm_modes = comm_modes.copy() if comm_modes is not None else None
        
        if refined_running_modes is not None and args.advanced_detection:
            print(f"\n🔧 APPLYING REFINEMENTS FOR RUNNING STATE TIMELINE...")
            
            # Calculate baseline
            baseline_start_time = 9.0
            baseline_end_time = 10.0
            baseline_start_idx = np.searchsorted(times, baseline_start_time)
            baseline_end_idx = np.searchsorted(times, baseline_end_time)
            
            if baseline_end_idx > baseline_start_idx:
                baseline_currents_calc = currents[baseline_start_idx:baseline_end_idx]
                baseline = np.mean(baseline_currents_calc)
            else:
                baseline = np.median(currents[:1000])
            
            # Calculate refined boundaries
            refined_boundaries = calculate_refined_boundaries(times, currents, modes, baseline)
            
            # Apply boundary adjustments to Running_Mode
            for adj in refined_boundaries:
                # Reset original period to idle (mode 0)
                original_mask = (times >= adj['original_start']) & (times <= adj['original_end'])
                refined_running_modes[original_mask] = 0
                
                # Set refined period to correct mode
                refined_mask = (times >= adj['refined_start']) & (times <= adj['refined_end'])
                mode_value = 2 if adj['type'] == 'IR' else 1
                refined_running_modes[refined_mask] = mode_value
            
            print(f"✅ Applied {len(refined_boundaries)} boundary adjustments to Running_Mode")
        
        # Apply wakeup detection refinements to Running_State
        if args.wakeup_detection and refined_running_states is not None:
            print(f"🔍 Applying wake-up detection to Running_State...")
            refined_running_states, wakeup_states = apply_wakeup_detection(times, currents, refined_running_states)
            print(f"✅ Wake-up detection applied to Running_State")
        
        # REFINE WAKEUP TRANSITION POINTS: Find exact current jump locations
        if args.refine_wakeup_transitions and refined_running_states is not None:
            refined_running_states = refine_wakeup_transition_points(times, currents, refined_running_states, search_window_seconds=5.0)
        
        # Plot running state timeline if enabled (using refined data)
        if args.plot_running_state and ax_state is not None and refined_running_states is not None:
            plot_running_state_timeline(times, refined_running_states, refined_running_modes, refined_comm_modes, ax_state)
        
        plt.tight_layout()
        
        # INTERACTIVE MODE: Enable zoom-responsive plotting for high-res data
        if sample_rate >= 80000 and args.mode == 'guest_app':
            print(f"\n🔍 INTERACTIVE MODE ENABLED:")
            print(f"   - Zoom will trigger high-resolution recalculation")
            print(f"   - Like guest app canvas: more detail when zoomed in")
            print(f"   - Use mouse wheel or zoom tool to see effect")
            
            # Store data for interactive updates
            ax._original_times = times
            ax._original_currents = currents
            ax._original_modes = modes  # Store modes for IR/BLE markers in zoom
            ax._colors = GUEST_APP_COLORS
            ax._sample_rate = sample_rate
            
            # Connect zoom event handler
            def on_zoom(ax):
                """Recalculate data when zoom changes (like guest app canvas)"""
                try:
                    # Get current zoom window
                    xlim = ax.get_xlim()
                    t0, t1 = float(xlim[0]), float(xlim[1])
                    
                    # Filter data to visible window
                    mask = (ax._original_times >= t0) & (ax._original_times <= t1)
                    window_times = ax._original_times[mask]
                    window_currents = ax._original_currents[mask]
                    
                    if len(window_times) < 10:  # Too few points
                        return
                    
                    # Calculate adaptive bins (like guest app)
                    fig = ax.get_figure()
                    width_inches = fig.get_size_inches()[0]
                    dpi = fig.get_dpi()
                    plot_width_pixels = int(width_inches * dpi * 0.8)
                    bins = max(300, plot_width_pixels)
                    
                    # Determine plotting strategy
                    samples_per_bin = len(window_times) / bins if bins > 0 else 1
                    
                    # Clear and redraw with new resolution
                    ax.clear()
                    
                    if samples_per_bin <= 3 and len(window_times) <= 10000:
                        # RAW DATA MODE: Individual points
                        ax.plot(window_times, window_currents, 
                                color=ax._colors['dark_blue'], 
                                linewidth=0.5,
                                alpha=0.8, 
                                marker='.',
                                markersize=0.5,
                                label=f'Raw Points ({len(window_times):,} samples)')
                        print(f"🎯 ZOOM UPDATE: Raw mode - {len(window_times):,} individual points")
                    else:
                        # BINNING MODE: Statistical curves
                        centers, mins, maxs, means, _ = DataAccumulator.bin_min_max_mean(
                            window_times, window_currents, t0, t1, bins)
                        
                        if centers.size > 0:
                            light_blue_rgb = tuple(c / 255.0 for c in ax._colors['light_blue'])
                            envelope_alpha = ax._colors['envelope_alpha'] / 255.0
                            
                            # Plot 4-layer system
                            ax.plot(centers, mins, color=light_blue_rgb, linewidth=1, alpha=1.0)
                            ax.plot(centers, maxs, color=light_blue_rgb, linewidth=1, alpha=1.0)
                            ax.fill_between(centers, mins, maxs, color=light_blue_rgb, 
                                           alpha=envelope_alpha, label='Current Range')
                            ax.plot(centers, means, color=ax._colors['dark_blue'], 
                                   linewidth=1, label='Mean Current')
                        
                        print(f"🎯 ZOOM UPDATE: Binning mode - {bins} bins, {samples_per_bin:.1f} samples/bin")
                    
                    # CRITICAL: Add IR/BLE markers for the zoomed window
                    if hasattr(ax, '_original_modes') and ax._original_modes is not None:
                        window_modes = ax._original_modes[mask]
                        add_vlc_ble_markers(window_times, window_currents, window_modes, ax)

                    # Restore styling
                    ax.set_facecolor(ax._colors['background'])
                    ax.set_xlabel("Time (s)", color=ax._colors['text'], fontsize=16)
                    ax.set_ylabel("Current (μA)", color=ax._colors['text'], fontsize=16)
                    ax.grid(True, color='gray', alpha=0.3)
                    ax.legend(fontsize=12)
                    ax.tick_params(colors='black', labelsize=14)
                    
                    # Redraw
                    ax.figure.canvas.draw()
                    
                except Exception as e:
                    print(f"Zoom update error: {e}")
            
            # Connect to zoom events
            ax.callbacks.connect('xlim_changed', lambda ax: on_zoom(ax))
            
        # Save refined dataset if requested
        if args.save_refined and original_modes is not None:
            # Calculate baseline from 9th to 10th seconds
            baseline_start_time = 9.0
            baseline_end_time = 10.0
            baseline_start_idx = np.searchsorted(original_times, baseline_start_time)
            baseline_end_idx = np.searchsorted(original_times, baseline_end_time)

            if baseline_end_idx > baseline_start_idx:
                baseline_currents = original_currents[baseline_start_idx:baseline_end_idx]
                baseline = np.mean(baseline_currents)
            else:
                baseline = np.median(original_currents[:1000])

            print(f"💾 CALCULATING REFINED BOUNDARIES (once for all {len(actual_csv_files)} files)...")

            # Calculate refined boundaries ONCE from the combined dataset
            refined_boundaries = calculate_refined_boundaries(original_times, original_currents, original_modes,
                                                              baseline)

            # Apply refined boundaries to each CSV file
            print(f"📂 APPLYING BOUNDARIES TO {len(actual_csv_files)} FILES...")

            # Global wake-up detection: Combine all files and treat as continuous time series
            global_wakeup_results = {}  # Store wake-up results for each file

            if args.wakeup_detection:
                print(f"🌐 GLOBAL WAKE-UP DETECTION: Integrating all {len(actual_csv_files)} files as continuous time series")

                # Step 1: Sort files chronologically and load all data
                sorted_csv_files = sorted(actual_csv_files, key=lambda x: (
                    int(x.split('_part')[-1].split('.')[0]) if '_part' in x else 0
                ))
                print(f"🔄 Chronological file order: {[os.path.basename(f) for f in sorted_csv_files]}")

                # Load and combine all files into one continuous time series
                all_file_data = []
                combined_times = []
                combined_currents = []
                combined_running_states = []

                for i, csv_file in enumerate(sorted_csv_files):
                    print(f"  📄 Loading: {os.path.basename(csv_file)}")
                    file_times, file_currents, file_running_states = read_running_state_for_wakeup(csv_file)
                    
                    # Store file info for later mapping
                    all_file_data.append({
                        'file': csv_file,
                        'times': file_times,
                        'currents': file_currents,
                        'running_states': file_running_states,
                        'start_idx': len(combined_times),
                        'length': len(file_times)
                    })
                    
                    # Add to combined arrays
                    combined_times.extend(file_times)
                    combined_currents.extend(file_currents)
                    combined_running_states.extend(file_running_states)
                    
                    print(f"    📊 {len(file_times)} samples, time {file_times[0]:.3f}s-{file_times[-1]:.3f}s")

                # Convert to numpy arrays and sort by timestamp
                combined_times = np.array(combined_times)
                combined_currents = np.array(combined_currents)
                combined_running_states = np.array(combined_running_states)
                
                # Sort by timestamp to ensure chronological order
                sort_indices = np.argsort(combined_times)
                combined_times = combined_times[sort_indices]
                combined_currents = combined_currents[sort_indices]
                combined_running_states = combined_running_states[sort_indices]

                print(f"🌐 Combined & sorted dataset: {len(combined_times)} samples, time {combined_times[0]:.3f}s-{combined_times[-1]:.3f}s")

                # Step 2: Apply wake-up detection to the entire combined time series
                print(f"🌐 Applying wake-up detection to entire time series...")
                corrected_combined_states, wakeup_combined_states = apply_wakeup_detection(
                    combined_times, combined_currents, combined_running_states
                )

                # Step 3: Map corrected states back to individual files
                print(f"🌐 Mapping corrected states back to individual files...")
                
                # Create reverse mapping from sorted indices back to original file positions
                reverse_sort_indices = np.argsort(sort_indices)
                original_corrected_states = corrected_combined_states[reverse_sort_indices]
                original_wakeup_states = wakeup_combined_states[reverse_sort_indices]
                
                # Map back to individual files
                for file_data in all_file_data:
                    start_idx = file_data['start_idx']
                    length = file_data['length']
                    end_idx = start_idx + length
                    
                    # Extract corrected states for this file
                    file_corrected_states = original_corrected_states[start_idx:end_idx]
                    file_wakeup_states = original_wakeup_states[start_idx:end_idx]
                    
                    # Calculate changes
                    original_states = file_data['running_states']
                    running_state_changes = np.sum(original_states != file_corrected_states)
                    wakeup_state_changes = np.sum(file_wakeup_states > 0)
                    
                    global_wakeup_results[file_data['file']] = {
                        'corrected_states': file_corrected_states,
                        'wakeup_states': file_wakeup_states,
                        'changes_made': running_state_changes > 0 or wakeup_state_changes > 0,
                        'running_state_changes': running_state_changes,
                        'wakeup_state_changes': wakeup_state_changes
                    }
                    
                    if running_state_changes > 0 or wakeup_state_changes > 0:
                        print(f"    📄 {os.path.basename(file_data['file'])}: {running_state_changes} Running_State changes, {wakeup_state_changes} Wakeup_State changes")
                    else:
                        print(f"    📄 {os.path.basename(file_data['file'])}: No wake-up changes")

            # Process each file with boundary adjustments and global wake-up results
            for csv_file in actual_csv_files:
                print(f"  📄 Processing: {csv_file}")

                # Get global wake-up results for this file (if any)
                wakeup_results = global_wakeup_results.get(csv_file, None) if args.wakeup_detection else None

                file_wakeup_detected = save_refined_dataset_with_boundaries(
                    refined_boundaries, csv_file,
                    enable_wakeup_detection=args.wakeup_detection,
                    global_wakeup_results=wakeup_results,
                    enable_advanced_detection=args.advanced_detection,
                    enable_wakeup_refinement=args.refine_wakeup_transitions
                )

        # Save files with white background (guest app uses white) - BEFORE showing plot
        save_facecolor = 'white'
        
        if args.save:
            print(f"💾 Saving high-resolution PNG to {args.save} (DPI: {args.dpi})...")
            fig.savefig(args.save, dpi=args.dpi, facecolor=save_facecolor, edgecolor='none', bbox_inches='tight')
            print(f"✅ Saved PNG to {args.save}")
        
        if args.save_pdf:
            print(f"📄 Saving high-resolution PDF to {args.save_pdf}...")
            with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
                fig.savefig(args.save_pdf, format='pdf', facecolor=save_facecolor, edgecolor='none', bbox_inches='tight')
            print(f"✅ Saved PDF to {args.save_pdf}")
        
        if args.save_svg:
            print(f"🎨 Saving SVG to {args.save_svg}...")
            with plt.rc_context({'svg.fonttype': 'none'}):
                fig.savefig(args.save_svg, format='svg', facecolor=save_facecolor, edgecolor='none', bbox_inches='tight')
            print(f"✅ Saved SVG to {args.save_svg}")
            
        if args.mode == 'guest_app' and (args.save or args.save_pdf or args.save_svg):
            print("🎨 All files saved with WHITE background (exact guest app style)")

        # Show interactive window AFTER saving
        print("\n📊 Displaying interactive plot window...")
        plt.show()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
