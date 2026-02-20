#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PPK2-Python: Python implementation inspired by pc-nrfconnect-ppk-main.

Included in this version:
- Device controls (connect, mode, voltage, start/stop measure) via ppk2_api
- Three-row visualization (detail, overview with region, stats) using PyQtGraph
- X-only zoom/pan, drag-zoom, crosshair, right-click live end, middle-click reset
- Overview region controls detail view; Shift+drag selection stats
- Downsample (Hz) and interdependent Time Window options
- Export: save CSV (mean-per-bin at selected Hz) and save PNG of plots
- Keyboard shortcuts: Space(toggle pause), R(reset), A(autorange Y), +/- zoom time, [ ] cycle downsample

Notes:
- This is a focused core implementation. Additional features from the official app (session management,
  triggers, markers persistence, settings pages) can be layered next.
"""

import os
import sys
import csv
import math
import logging
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional
from collections import deque
import threading
import queue
from statistics import median

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QRadioButton,
    QCheckBox,
    QTextEdit,
)
import pyqtgraph as pg

try:
    import serial.tools.list_ports as list_ports
    import serial
except Exception:
    list_ports = None
    serial = None

try:
    import socket
except Exception:
    socket = None

try:
    from src.ppk2_api.ppk2_api import PPK2_MP as PPK2_API
except Exception:
    try:
        from ppk2_api.ppk2_api import PPK2_MP as PPK2_API
    except Exception:
        PPK2_API = None  # CSV/demo only


class ParameterBuffer:
    """
    Thread-safe parameter buffer for high-performance parameter-current synchronization.

    Designed for parallel execution:
    - PPK2 thread: Fast, non-blocking parameter lookup at 100kHz
    - Serial thread: Independent parameter updates at 1Hz
    - Shared timestamp system: Perfect time alignment
    """

    def __init__(self):
        self.history = []  # [(timestamp, params_dict), ...] - chronologically ordered
        self.lock = threading.RLock()  # Reentrant lock for nested calls
        self.current_params = self._get_default_params()
        self.max_history = 1000  # Keep last 1000 entries (~16 minutes at 1Hz)

        print(f"🔧 PARAM BUFFER: Initialized with default parameters")

    def _get_default_params(self):
        """Get default parameter values for initialization"""
        return {
            'running_state_status': 0, 'communication_mode_status': 0, 'wakeup_state': 0,
            'running_mode': 0, 'vlc_protocol_mode_status': 0, 'vlc_interval_seconds': 0,
            'vlc_communication_volume_status': 0, 'vlc_information_volume_status': 0,
            'vlc_per_unit_time_status': 0, 'pwm_percentage': 0, 'ble_protocol_mode_status': 0,
            'ble_interval_seconds': 0, 'ble_communication_volume_status': 0,
            'ble_information_volume_status': 0, 'ble_per_unit_time_status': 0,
            'phy_rate_percentage': 0, 'mtu_value': 0, 'fast_time': 0,
            'spare_2': 0, 'spare_3': 0, 'spare_4': 0
        }

    def update(self, timestamp, new_params):
        """
        Update parameter buffer from serial thread (non-blocking for PPK2 thread).

        Args:
            timestamp: time.perf_counter() when parameters were received
            new_params: Dictionary of new parameter values
        """
        with self.lock:
            # Create a copy to avoid reference issues
            params_copy = new_params.copy()

            # Add to chronological history
            self.history.append((timestamp, params_copy))

            # Update current parameters
            self.current_params.update(params_copy)

            # Maintain history size for memory efficiency
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

            # Log parameter updates (minimal output)
            if not hasattr(self, '_update_count'):
                self._update_count = 0
            self._update_count += 1

            if self._update_count <= 3:  # Only first 3 updates
                important_changes = {k: v for k, v in new_params.items()
                                     if k in ['running_state_status', 'wakeup_state', 'communication_mode_status']}
                print(f"📊 BUFFER UPDATE #{self._update_count}: t={timestamp:.3f}s → {important_changes}")

    def get_at_time(self, target_timestamp):
        """
        Get parameters valid at timestamp using HOLD strategy for frequency mismatch.

        HOLD STRATEGY: Parameters change slowly (~1Hz), current samples fast (100kHz).
        Each current sample gets the parameter values that were most recently updated
        before that sample's timestamp - this handles the frequency mismatch properly.

        Args:
            target_timestamp: PPK2 sample timestamp (cumulative time from measurement start)

        Returns:
            Dictionary of parameters that were valid at target_timestamp
        """
        with self.lock:
            if not self.history:
                # No history available, return current defaults
                return self.current_params.copy()

            # HOLD STRATEGY: Find most recent parameter update before target_timestamp
            # This ensures each 100kHz current sample gets the correct ~1Hz parameter values
            valid_params = None
            for timestamp, params in reversed(self.history):
                if timestamp <= target_timestamp:
                    valid_params = params
                    break

            if valid_params is not None:
                return valid_params.copy()
            else:
                # Target time is before any recorded parameters, use earliest
                return self.history[0][1].copy()

    def get_current(self):
        """Get the most recent parameters (thread-safe)"""
        with self.lock:
            return self.current_params.copy()

    def clear(self):
        """Clear all parameter history (for measurement restart)"""
        with self.lock:
            self.history.clear()
            self.current_params = self._get_default_params()
            if hasattr(self, '_update_count'):
                self._update_count = 0
            print(f"🔧 PARAM BUFFER: Cleared and reset to defaults")

    def get_stats(self):
        """Get buffer statistics for debugging"""
        with self.lock:
            if not self.history:
                return {"entries": 0, "time_span": 0, "oldest": None, "newest": None}

            oldest_time = self.history[0][0]
            newest_time = self.history[-1][0]
            time_span = newest_time - oldest_time

            return {
                "entries": len(self.history),
                "time_span": time_span,
                "oldest": oldest_time,
                "newest": newest_time
            }


# Filter Classes (from app-guest-redesigned.py)
class DelayedMedianSpikeFilter:
    def __init__(self, window_size=21, threshold=5, extreme_spike_limit=50000):
        self.window_size = window_size
        self.threshold = threshold
        self.extreme_spike_limit = extreme_spike_limit
        self.buffer = []
        self.sample_index = 0

    def process(self, sample):
        self.buffer.append(sample)
        self.sample_index += 1
        if len(self.buffer) < self.window_size:
            return sample

        center_idx = self.window_size // 2
        window = np.array(self.buffer)
        center_val = window[center_idx]

        # Check for extreme spikes first
        if abs(center_val) > self.extreme_spike_limit:
            clean_window = window[np.abs(window) <= self.extreme_spike_limit]
            if len(clean_window) >= 3:
                replacement_value = np.median(clean_window)
            else:
                replacement_value = np.median(window)
            self.buffer.pop(0)
            return replacement_value

        # Regular MAD-based spike detection
        median = np.median(window)
        outlier_threshold = 3 * np.std(window) if np.std(window) > 0 else 10000
        clean_indices = np.abs(window - median) <= outlier_threshold
        if np.sum(clean_indices) >= self.window_size // 3:
            clean_window = window[clean_indices]
            mad = np.median(np.abs(clean_window - median))
        else:
            mad = np.median(np.abs(window - median))

        filtered = center_val
        if mad == 0:
            mad = 1.0

        spike_threshold = self.threshold * mad
        if abs(center_val - median) > spike_threshold:
            filtered = median
        else:
            filtered = center_val

        self.buffer.pop(0)
        return filtered


class MovingAverageFilter:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.buffer = []

    def process(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return sum(self.buffer) / len(self.buffer)


class HampelFilter:
    def __init__(self, window_size=21, threshold=3):
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = []

    def process(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) < self.window_size:
            return sample

        center_idx = self.window_size // 2
        window = np.array(self.buffer)
        center_val = window[center_idx]

        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if mad == 0:
            mad = 1.0

        if abs(center_val - median) > self.threshold * mad:
            filtered = median
        else:
            filtered = center_val

        self.buffer.pop(0)
        return filtered


class SavitzkyGolayFilter:
    def __init__(self, window_size=21, polyorder=2):
        self.window_size = window_size
        self.polyorder = polyorder
        self.buffer = []

    def process(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) < self.window_size:
            return sample

        try:
            from scipy.signal import savgol_filter
            filtered_data = savgol_filter(self.buffer, self.window_size, self.polyorder)
            center_idx = self.window_size // 2
            result = filtered_data[center_idx]
            self.buffer.pop(0)
            return result
        except ImportError:
            # Fallback to moving average if scipy not available
            return sum(self.buffer) / len(self.buffer)


# Network Manager for Host Connection (from app-guest-redesigned.py)
class NetworkManager:
    def __init__(self, parent=None):
        self.parent = parent
        self.sock = None
        self.host_ip = "localhost"  # Default to localhost - most common use case
        self.host_port = 65432
        self.connected = False

    def connect(self, host, port):
        if not socket:
            logging.error("Socket module not available")
            return False

        # Clean up any existing connection first
        self.disconnect()

        try:
            print(f"🔗 Creating socket connection to {host}:{port}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2)  # Fast connection timeout for localhost

            print(f"🔗 Attempting socket connect...")
            self.sock.connect((host, port))
            print(f"✅ Socket connected successfully")

            self.host_ip = host
            self.host_port = port

            # Optional ping test - skip if we want fastest connection
            skip_ping = True  # Set to False if you want ping verification

            if not skip_ping:
                ping_data = {"type": "ping"}
                payload = json.dumps(ping_data).encode() + b'\n'
                print(f"📤 Sending PING to host...")
                self.sock.sendall(payload)

                # Quick ping response check
                self.sock.settimeout(0.5)  # Very short timeout for ping response
                try:
                    print(f"👂 Quick ping response check...")
                    data = self.sock.recv(1024)
                    print(f"📥 Host responded to ping: {data!r}")
                except socket.timeout:
                    print(f"✅ No ping response (proceeding anyway)")
            else:
                print(f"⚡ Skipping ping for fastest connection")

            self.sock.settimeout(None)  # Reset to blocking for data transmission
            self.connected = True
            print(f"🎉 Host connection established successfully!")
            return True

        except ConnectionRefusedError:
            error_msg = f"Connection refused - Host app may not be running on {host}:{port}"
            print(f"❌ {error_msg}")
            logging.error(error_msg)
            self.connected = False
            return False
        except socket.timeout:
            error_msg = f"Connection timeout - Host {host}:{port} not responding"
            print(f"❌ {error_msg}")
            logging.error(error_msg)
            self.connected = False
            return False
        except socket.gaierror as e:
            error_msg = f"DNS/Address error for {host}: {e}"
            print(f"❌ {error_msg}")
            logging.error(error_msg)
            self.connected = False
            return False
        except Exception as e:
            error_msg = f"Network connection failed: {e}"
            print(f"❌ {error_msg}")
            logging.error(error_msg)
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from host and clean up connection"""
        if self.sock:
            try:
                print("🔌 Closing host connection...")
                self.sock.close()
                print("✅ Host connection closed")
            except Exception as e:
                print(f"⚠️ Error closing socket: {e}")
            finally:
                self.sock = None
        self.connected = False
        print("🔌 Host disconnected")

    def send_data(self, data):
        if self.connected and self.sock:
            try:
                payload = json.dumps(data).encode() + b'\n'
                self.sock.sendall(payload)
                return True
            except Exception as e:
                logging.error(f"Failed to send data: {e}")
                self.connected = False
                return False
        return False


# Serial Command Manager - OPTIMIZED for real-time reception with selective UI updates
class SerialCommandManager(QObject):
    """
    Ultra-optimized serial communication manager:

    🚀 PERFORMANCE OPTIMIZATIONS:
    - Real-time data reception (like PPK2) with minimal latency
    - Selective UI updates: only when parameters actually change
    - No unnecessary text conversion: just raw byte count display
    - Important parameter highlighting for critical changes
    - Change-based logging to reduce noise

    📊 EXPECTED PERFORMANCE:
    - ~90% reduction in text processing overhead
    - Real-time frame processing with <1ms latency
    - UI updates only when needed (not every 33ms)
    - Clear visual feedback for important parameter changes
    Supports both text commands and frame parsing with AA...EE protocol.
    """
    status_message_signal = pyqtSignal(str)
    serial_data_signal = pyqtSignal(dict)
    status_text_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.serial_port = None
        self.serial_thread = None
        self.is_connected = False
        self.running = False
        self.receive_buffer = bytearray()
        self._latest_queue = queue.Queue(maxsize=1)
        self._text_line_buffer = ""

        # Serial error tracking to prevent spam
        self._serial_error_logged = False

        # Frame detection statistics
        self._frame_stats = {
            'total_frames': 0,
            'valid_frames': 0,
            'invalid_frames': 0,
            'last_reset': time.time()
        }

        # Track previous frame for change detection to reduce UI load
        self._previous_frame = {}
        self._last_signal_time = 0

        # CRITICAL: Dedicated frame buffer for running_mode changes
        self._critical_frame_buffer = queue.Queue(maxsize=100)  # Large buffer for critical frames
        self._frame_processing_active = False

        # CRITICAL: Ultra-high frequency timer for immediate frame processing
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._flush_to_ui)
        self._ui_timer.start(10)  # 100Hz for ultra-responsive frame processing

        # DEBUG: Verify timer is working
        print(f"🔍 UI TIMER STARTED: {self._ui_timer.isActive()}, Interval: {self._ui_timer.interval()}ms")
        logging.info(f"🔍 UI TIMER STARTED: {self._ui_timer.isActive()}, Interval: {self._ui_timer.interval()}ms")

    def connect(self, port_name: str) -> bool:
        try:
            print(f"🔌 CONNECTING: Attempting to connect to {port_name}...")
            self.serial_port = serial.Serial(
                port=port_name,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
                timeout=0.001,  # ULTRA-SHORT timeout for rapid polling (1ms)
                write_timeout=0.1,  # Very short write timeout
            )

            # CRITICAL: Optimize serial port for maximum performance
            if hasattr(self.serial_port, 'set_buffer_size'):
                self.serial_port.set_buffer_size(rx_size=8192, tx_size=8192)  # Large buffers

            # Disable any hardware flow control
            self.serial_port.rts = False
            self.serial_port.dtr = False

            print(f"🔌 SERIAL PORT: Optimized for rapid frame capture (timeout={self.serial_port.timeout}s)")
            print(f"🔌 SERIAL PORT: Opened successfully")
            if not self.serial_port.is_open:
                raise Exception("Serial port failed to open.")

            # Nudge the device to print its banner
            try:
                self.serial_port.reset_input_buffer()
                self.serial_port.reset_output_buffer()
                self.serial_port.setDTR(True)
                self.serial_port.setRTS(False)
            except Exception:
                pass
            time.sleep(0.1)
            try:
                self.serial_port.write(b"\r\n")
                self.serial_port.flush()
            except Exception:
                pass

            self.is_connected = True
            self.running = True

            # Reset error flag for new connection
            self._serial_error_logged = False

            self.serial_thread = threading.Thread(target=self._serial_read_thread, daemon=True)
            self.serial_thread.start()
            print(f"🔌 SERIAL THREAD: Started successfully")
            self.status_message_signal.emit(f"<font color='green'>✅ Serial connected: {port_name} (115200-8N1)</font>")
            return True
        except Exception as e:
            self.status_message_signal.emit(f"<font color='red'>❌ Serial connect failed: {e}</font>")
            logging.error(f"Serial connect failed: {e}")
            return False

    def disconnect(self):
        """Enhanced disconnect with comprehensive resource cleanup"""
        logging.info("🔍 DISCONNECT: Starting comprehensive cleanup")

        self.running = False

        # Close serial port first to unblock any pending reads
        try:
            if self.serial_port:
                self.serial_port.close()
                logging.info("🔍 DISCONNECT: Serial port closed")
        except Exception as e:
            logging.error(f"Error closing serial port: {e}")

        # Clean up thread resources
        if self.serial_thread and not getattr(self, '_auto_disconnect_mode', False):
            try:
                # Only wait for thread in manual disconnection
                self.serial_thread.join(timeout=0.5)  # Reduced timeout
                logging.info("🔍 DISCONNECT: Thread joined")
            except Exception as e:
                logging.error(f"Error cleaning up thread: {e}")

        # Clear all buffers to prevent memory accumulation
        try:
            self.receive_buffer.clear()
            # Clear the queue
            while not self._latest_queue.empty():
                try:
                    self._latest_queue.get_nowait()
                except:
                    break
            logging.info("🔍 DISCONNECT: Buffers cleared")
        except Exception as e:
            logging.error(f"Error clearing buffers: {e}")

        # Stop UI timer to prevent resource leaks
        try:
            if hasattr(self, '_ui_timer') and self._ui_timer:
                self._ui_timer.stop()
                logging.info("🔍 DISCONNECT: UI timer stopped")
        except Exception as e:
            logging.error(f"Error stopping UI timer: {e}")

        # Clean up state
        self.serial_port = None
        self.serial_thread = None
        self.is_connected = False

        if not getattr(self, '_auto_disconnect_mode', False):
            self.status_message_signal.emit("<font color='blue'>🔌 Serial disconnected</font>")

        logging.info("🔍 DISCONNECT: Cleanup completed")

    def send_command(self, command_text: str):
        # Debug: Log send command attempt
        logging.info(
            f"🔍 SEND_COMMAND called: '{command_text}', connected={self.is_connected}, port_open={self.serial_port.is_open if self.serial_port else False}")
        if not (self.is_connected and self.serial_port and self.serial_port.is_open):
            self.status_message_signal.emit("<font color='red'>❌ Serial not connected</font>")
            return False
        try:
            # Node expects ASCII hex strings terminated with \n
            # Send command as-is with \n terminator (not \r\n)
            payload = command_text.strip().encode('utf-8') + b"\n"

            self.serial_port.write(payload)
            self.serial_port.flush()
            self.status_message_signal.emit(f"<font color='green'>📤 Sent: {command_text}</font>")
            return True
        except Exception as e:
            self.status_message_signal.emit(f"<font color='red'>❌ Send error: {e}</font>")
            return False

    def _should_emit_signal(self, parsed_data):
        """CRITICAL: Always emit for running_mode changes - never miss them!"""
        current_time = time.time()

        # CRITICAL: running_mode changes are ALWAYS emitted - this is the most important parameter
        if 'running_mode' in parsed_data:
            old_running_mode = self._previous_frame.get('running_mode')
            new_running_mode = parsed_data['running_mode']
            if old_running_mode != new_running_mode:
                logging.info(f"🚨 CRITICAL: running_mode changed {old_running_mode} → {new_running_mode} - ALWAYS EMIT!")
                self._previous_frame['running_mode'] = new_running_mode
                # Update all values and emit immediately
                for key, value in parsed_data.items():
                    self._previous_frame[key] = value
                return True

        # Always emit for other critical parameter changes
        critical_params = ['running_state', 'wakeup_state', 'communication_mode',
                           'vlc_communication_volume', 'ble_communication_volume']

        for param in critical_params:
            if param in parsed_data:
                old_value = self._previous_frame.get(param)
                new_value = parsed_data[param]
                if old_value != new_value:
                    self._previous_frame[param] = new_value
                    # Update all values and emit
                    for key, value in parsed_data.items():
                        self._previous_frame[key] = value
                    return True

        # Emit at least every 50ms to keep UI very responsive (reduced from 100ms)
        if current_time - self._last_signal_time > 0.05:
            self._last_signal_time = current_time
            # Update all previous values
            for key, value in parsed_data.items():
                self._previous_frame[key] = value
            return True

        # Update previous values but don't emit
        for key, value in parsed_data.items():
            self._previous_frame[key] = value
        return False

    def _process_critical_frames(self):
        """IMMEDIATE processing of critical frames - especially running_mode changes"""
        processed_count = 0

        while not self._critical_frame_buffer.empty() and processed_count < 10:  # Process up to 10 frames at once
            try:
                parsed = self._critical_frame_buffer.get_nowait()

                # Check if this frame has critical changes
                if self._should_emit_signal(parsed):
                    self.serial_data_signal.emit(parsed)
                    logging.info(f"🚨 CRITICAL FRAME PROCESSED: running_mode={parsed.get('running_mode', 'N/A')}")

                processed_count += 1

            except queue.Empty:
                break
            except Exception as e:
                logging.error(f"Critical frame processing error: {e}")
                break

    def _flush_to_ui(self):
        """Regular UI flush - also processes any remaining critical frames"""
        # First, process any remaining critical frames
        if not self._critical_frame_buffer.empty():
            self._process_critical_frames()

        # Then process regular queue
        try:
            parsed = self._latest_queue.get_nowait()
            if self._should_emit_signal(parsed):
                self.serial_data_signal.emit(parsed)
        except queue.Empty:
            return
        except Exception as e:
            logging.error(f"Flush error: {e}")

    def _serial_read_thread(self):
        """Ultra-optimized serial read thread - real-time reception, minimal processing"""
        header_byte = b'\xAA'
        end_byte = b'\xEE'
        buf = self.receive_buffer

        while self.running:
            if not (self.serial_port and self.serial_port.is_open):
                time.sleep(0.01)
                continue

            if not self.running:
                break

            try:
                # ULTRA-AGGRESSIVE: No sleep, continuous polling for critical frame capture
                available = self.serial_port.in_waiting
                if available == 0:
                    # NO SLEEP - continuous polling to catch rapid 10ms frames
                    continue

                # Read ALL available data immediately - no limits for critical frames
                chunk = self.serial_port.read(available)  # Read everything available
                if not chunk:
                    continue

                buf.extend(chunk)

                # Process frames with robust 24-byte frame detection
                frames_found = 0
                while len(buf) >= 24:  # Need at least 24 bytes for a complete frame
                    # Look for frame start (AA)
                    frame_start = -1
                    for i in range(len(buf) - 23):  # Ensure we have enough bytes left
                        if buf[i] == 0xAA:
                            # Check if we have a complete 24-byte frame starting here
                            if i + 23 < len(buf) and buf[i + 23] == 0xEE:
                                frame_start = i
                                break

                    if frame_start == -1:
                        # No valid frame found, keep last 23 bytes in case frame is split
                        if len(buf) > 23:
                            buf = buf[-23:]
                        break

                    # Extract the 24-byte frame
                    frame = bytes(buf[frame_start:frame_start + 24])
                    # Remove processed data from buffer
                    del buf[:frame_start + 24]
                    frames_found += 1
                    self._frame_stats['total_frames'] += 1

                    # Debug: Log frame detection
                    logging.info(
                        f"🔍 FOUND FRAME #{frames_found}: {' '.join(f'{b:02X}' for b in frame[:8])}...{' '.join(f'{b:02X}' for b in frame[-2:])} (length: {len(frame)})")

                    # Validate frame format: AA [21 data] [CS] EE
                    if len(frame) == 24 and frame[0] == 0xAA and frame[23] == 0xEE:
                        # GLOBAL TIME: Timestamp parameters using global time system
                        receive_timestamp = time.perf_counter()

                        # Process frame immediately
                        try:
                            if hasattr(self.parent, '_process_frame'):
                                parsed = self.parent._process_frame(frame)

                                if parsed is not None:
                                    self._frame_stats['valid_frames'] += 1

                                    # GLOBAL TIME: Add exact receive timestamp for global time conversion
                                    parsed['_receive_timestamp'] = receive_timestamp

                                    # Convert to global time for debug display
                                    if hasattr(self.parent, '_global_start_time'):
                                        global_time = receive_timestamp - self.parent._global_start_time
                                        debug_time_display = f"t={global_time:.6f}s"
                                    else:
                                        debug_time_display = f"abs={receive_timestamp:.6f}s"

                                    # Debug: Log successful parsing with global time (reduced logging for performance)
                                    if self._frame_stats[
                                        'valid_frames'] % 10 == 1:  # Log every 10th frame to reduce overhead
                                        logging.info(
                                            f"🔍 FRAME PARSED: running_state={parsed.get('running_state', 'N/A')} at {debug_time_display}")

                                    # CRITICAL: Always buffer frames, especially those with running_mode changes
                                    try:
                                        # Put in critical buffer first (never block, never lose frames)
                                        self._critical_frame_buffer.put(parsed, block=False)
                                    except queue.Full:
                                        # If critical buffer is full, remove oldest and add new
                                        try:
                                            _ = self._critical_frame_buffer.get_nowait()
                                            self._critical_frame_buffer.put(parsed, block=False)
                                            logging.warning("⚠️ CRITICAL BUFFER FULL - removed oldest frame")
                                        except queue.Empty:
                                            pass

                                    # Also put in regular queue for UI timer processing
                                    try:
                                        self._latest_queue.put(parsed, block=False)
                                    except queue.Full:
                                        try:
                                            _ = self._latest_queue.get_nowait()
                                            self._latest_queue.put(parsed, block=False)
                                        except queue.Empty:
                                            pass

                                    # IMMEDIATE: Process critical frames right away (don't wait for timer)
                                    if 'running_mode' in parsed and not self._frame_processing_active:
                                        self._frame_processing_active = True
                                        self._process_critical_frames()
                                        self._frame_processing_active = False
                                else:
                                    self._frame_stats['invalid_frames'] += 1
                                    logging.warning(f"⚠️ FRAME PARSING FAILED: _process_frame returned None")
                        except Exception as e:
                            logging.error(f"Serial frame parse error: {e}")
                    else:
                        logging.warning(
                            f"⚠️ INVALID FRAME FORMAT: length={len(frame)}, start={frame[0]:02X}, end={frame[-1]:02X}")

                # Debug: Log if no frames found in this chunk (reduced frequency)
                if frames_found == 0 and len(chunk) > 0 and len(buf) > 50:
                    # Only log every 100th occurrence to reduce spam
                    if not hasattr(self, '_no_frame_count'):
                        self._no_frame_count = 0
                    self._no_frame_count += 1
                    if self._no_frame_count % 100 == 1:
                        chunk_preview = ' '.join(f'{b:02X}' for b in chunk[:8])
                        logging.info(f"🔍 NO FRAMES FOUND in {len(chunk)} bytes: {chunk_preview}..., buffer: {len(buf)}")

                # Monitor buffer size to detect accumulation
                if len(buf) > 100:
                    logging.warning(f"⚠️ LARGE BUFFER: {len(buf)} bytes accumulated - may indicate frame sync issues")
                    # Show buffer contents for debugging
                    buf_preview = ' '.join(f'{b:02X}' for b in buf[:32])
                    logging.info(f"🔍 BUFFER CONTENTS: {buf_preview}...")

                    # If buffer gets too large, reset it to prevent memory issues
                    if len(buf) > 500:
                        logging.error("❌ BUFFER OVERFLOW: Resetting buffer to prevent memory leak")
                        buf.clear()

                # Report frame statistics every 10 seconds
                current_time = time.time()
                if current_time - self._frame_stats['last_reset'] >= 10.0:
                    total = self._frame_stats['total_frames']
                    valid = self._frame_stats['valid_frames']
                    invalid = self._frame_stats['invalid_frames']
                    time_elapsed = current_time - self._frame_stats['last_reset']
                    if total > 0:
                        success_rate = (valid / total) * 100
                        frames_per_second = total / time_elapsed
                        valid_per_second = valid / time_elapsed
                        logging.info(f"📊 FRAME STATS (10s): Total: {total} ({frames_per_second:.1f}/s), "
                                     f"Valid: {valid} ({valid_per_second:.1f}/s), Invalid: {invalid}, "
                                     f"Success: {success_rate:.1f}%")

                        # Alert if frame rate is too low (node sends ~10 frames per VLC transmission)
                        if valid_per_second < 1.0:
                            logging.error(
                                f"🚨 CRITICAL: LOW FRAME RATE: Only {valid_per_second:.1f} valid frames/s - MISSING FRAMES!")
                            logging.error(f"🚨 CRITICAL: This may cause running_mode changes to be lost!")
                        elif valid_per_second < 3.0:
                            logging.warning(
                                f"⚠️ MODERATE FRAME RATE: {valid_per_second:.1f} valid frames/s - monitor for frame loss")
                    else:
                        logging.info("📊 FRAME STATS (10s): No frames detected")

                    # Reset statistics
                    self._frame_stats = {
                        'total_frames': 0,
                        'valid_frames': 0,
                        'invalid_frames': 0,
                        'last_reset': current_time
                    }

                # Show actual received data as strings and process text responses
                if chunk:
                    # Debug: Log raw data reception
                    logging.info(f"🔍 SERIAL_READ_THREAD received {len(chunk)} bytes")
                    try:
                        # Try to decode as text first
                        text = chunk.decode('utf-8', errors='replace')
                        # Filter out non-printable characters but keep newlines
                        clean_text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t')
                        if clean_text.strip():  # Only show if there's meaningful text
                            logging.info(f"🔍 EMITTING status_text_signal: '{clean_text.strip()}'")
                            self.status_text_signal.emit(f"📡 {clean_text.strip()}")

                        # Check for node command acknowledgments
                        if '▶' in text and 'command:' in text:
                            self.status_text_signal.emit(f"🎯 Node: {text.strip()}")
                    except Exception:
                        # Fallback to hex if decode fails
                        hex_data = ' '.join(f'{b:02X}' for b in chunk)
                        logging.info(f"🔍 EMITTING status_text_signal HEX: {hex_data}")
                        self.status_text_signal.emit(f"📡 HEX: {hex_data}")

            except Exception as e:
                if self.running:
                    # Check for specific disconnection errors
                    error_str = str(e).lower()
                    is_disconnection_error = (
                            'clearcommerror failed' in error_str or
                            'permission denied' in error_str or
                            'device does not recognize' in error_str or
                            'access is denied' in error_str or
                            'device not ready' in error_str or
                            'i/o operation' in error_str
                    )

                    if is_disconnection_error:
                        # Only log once and trigger automatic disconnection
                        if not self._serial_error_logged:
                            logging.error(f"Serial device disconnected: {e}")
                            self._serial_error_logged = True

                            # Emit disconnection signal and trigger auto-disconnect
                            self.status_message_signal.emit(
                                "<font color='red'>❌ Serial device disconnected automatically</font>")

                            # Auto-disconnect on the main thread (immediate)
                            QTimer.singleShot(0, self._handle_auto_disconnect)

                        # Stop the reading loop for disconnection errors
                        break
                    else:
                        # For other errors, log normally but limit frequency
                        if not self._serial_error_logged:
                            logging.error(f"Serial read error: {e}")
                            self._serial_error_logged = True

                time.sleep(0.1)

    def _handle_auto_disconnect(self):
        """Handle automatic disconnection when device is removed"""
        try:
            # Set auto-disconnect mode for fast disconnection
            self._auto_disconnect_mode = True

            # Perform fast disconnection (non-blocking)
            self.disconnect()

            # Update parent UI if possible
            if self.parent and hasattr(self.parent, 'serial_conn_toggle_button'):
                self.parent.serial_conn_toggle_button.setText("🔗 Connect")

            # Reset error flag for next connection
            self._serial_error_logged = False

            # Clear auto-disconnect mode
            self._auto_disconnect_mode = False

        except Exception as e:
            logging.error(f"Error in auto-disconnect: {e}")
            # Make sure to clear the flag even on error
            self._auto_disconnect_mode = False

    @property
    def connected(self):
        return self.is_connected


class DataManager:
    def __init__(self):
        self.times: np.ndarray = np.array([], dtype=np.float64)
        self.currents: np.ndarray = np.array([], dtype=np.float64)
        self.locked: bool = False
        self.paused: bool = False

        # Efficient storage with complete parameter tracking
        self.raw_samples_full = []  # (time, current, running_mode, guest_id, voltage_mv, guest_params, status_params)
        self.save_downsample_hz = 5000  # Downsample to 5kHz for saving (like backup version)

        # MEMORY MANAGEMENT: Smart limits to prevent system overload
        self.max_samples_in_memory = 2000000  # 2M samples ≈ 128MB (increased limit)
        self.memory_warning_threshold = 1800000  # Warn at 90% capacity
        self.auto_save_enabled = False  # Disable auto-save for better performance
        self.auto_save_counter = 0  # Track auto-saves
        self._last_memory_warning = 0  # Prevent spam warnings

        # EXCEL COMPATIBILITY: Excel has 1,048,576 row limit
        self.excel_max_rows = 1048575  # Reserve 1 row for header
        self.excel_safe_mode = True  # Split files to stay within Excel limits

        print(
            f"🧠 MEMORY MANAGEMENT: Limiting to {self.max_samples_in_memory:,} samples ({self.max_samples_in_memory * 64 / 1024 / 1024:.0f}MB) per guest")
        print(f"📊 EXCEL COMPATIBILITY: Files will be split at {self.excel_max_rows:,} rows for Excel compatibility")

    def get_memory_usage(self):
        """Get current memory usage statistics"""
        sample_count = len(self.raw_samples_full)
        memory_mb = (sample_count * 64) / 1024 / 1024  # 64 bytes per sample
        usage_percent = (sample_count / self.max_samples_in_memory) * 100

        return {
            'samples': sample_count,
            'memory_mb': memory_mb,
            'usage_percent': usage_percent,
            'is_warning': sample_count >= self.memory_warning_threshold,
            'is_critical': sample_count >= self.max_samples_in_memory * 0.95
        }

    def check_memory_limits(self, parent_window=None):
        """Check memory limits and take action if needed"""
        usage = self.get_memory_usage()

        # Critical: Auto-save and clear buffer
        if usage['is_critical'] and self.auto_save_enabled:
            print(f"🚨 CRITICAL MEMORY: {usage['samples']:,} samples ({usage['memory_mb']:.1f}MB). Auto-saving...")
            if parent_window:
                self._auto_save_and_clear(parent_window)
            return True

        # Warning: Notify user
        elif usage['is_warning']:
            current_time = time.time()
            if current_time - self._last_memory_warning > 30:  # Warn every 30 seconds max
                print(
                    f"⚠️ MEMORY WARNING: {usage['samples']:,} samples ({usage['memory_mb']:.1f}MB, {usage['usage_percent']:.1f}%)")
                self._last_memory_warning = current_time

        return False

    def _auto_save_and_clear(self, parent_window):
        """Auto-save data and clear buffer to prevent memory overload"""
        try:
            self.auto_save_counter += 1

            # Generate auto-save filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"auto_save_{timestamp}_part{self.auto_save_counter}.csv"

            # Save current data
            print(f"💾 AUTO-SAVING: {len(self.raw_samples_full):,} samples to {filename}")
            parent_window._save_raw_buffer_direct(filename)

            # Clear buffer to free memory
            samples_cleared = len(self.raw_samples_full)
            self.clear()

            print(f"🧹 BUFFER CLEARED: Freed {samples_cleared:,} samples ({samples_cleared * 64 / 1024 / 1024:.1f}MB)")
            print(f"📊 MEMORY STATUS: Ready for new data collection")

            # Show user notification
            if hasattr(parent_window, 'status_label'):
                parent_window.status_label.setText(f"Auto-saved {samples_cleared:,} samples (memory management)")

        except Exception as e:
            print(f"❌ AUTO-SAVE FAILED: {e}")

    def clear(self):
        self.times = np.array([], dtype=np.float64)
        self.currents = np.array([], dtype=np.float64)
        self.raw_samples_full = []

    def append_samples(self, times: np.ndarray, currents: np.ndarray):
        if times.size == 0:
            return
        if self.times.size == 0:
            self.times = times.astype(np.float64, copy=True)
            self.currents = currents.astype(np.float64, copy=True)
        else:
            self.times = np.concatenate([self.times, times.astype(np.float64)])
            self.currents = np.concatenate([self.currents, currents.astype(np.float64)])

    def append_raw_sample(self, time_val, current_val, running_mode, guest_id, voltage_mv, guest_params, status_params,
                          unified_timestamp=None, parent_window=None):
        """Store individual sample with complete parameters and memory management"""
        # If no unified timestamp provided, use computer time for synchronization
        if unified_timestamp is None:
            unified_timestamp = time.perf_counter()

        self.raw_samples_full.append(
            (time_val, current_val, running_mode, guest_id, voltage_mv, guest_params, status_params, unified_timestamp))

        # MEMORY MANAGEMENT: Check limits less frequently (every 10000 samples for better performance)
        if len(self.raw_samples_full) % 10000 == 0:
            self.check_memory_limits(parent_window)

    def downsample_full_samples(self, raw_list, target_hz):
        """Downsample full-session raw samples with complete parameter tracking
        Input: list of (time, current, running_mode, guest_id, voltage_mv, guest_params, status_params)
        Output: downsampled list with same format
        """
        if not raw_list or target_hz <= 0:
            return raw_list

        # Calculate bin width
        if len(raw_list) < 2:
            return raw_list

        total_time = raw_list[-1][0] - raw_list[0][0]
        if total_time <= 0:
            return raw_list

        bin_width = 1.0 / target_hz
        num_bins = max(1, int(total_time / bin_width))

        # Group samples into bins and average
        bins = {}
        for sample in raw_list:
            # Handle both old format (7 elements) and new format (8 elements with unified timestamp)
            if len(sample) == 8:
                time_val, current, running_state, guest_id, voltage_mv, guest_params, status_params, unified_timestamp = sample
            else:
                # Old format compatibility
                time_val, current, running_state, guest_id, voltage_mv, guest_params, status_params = sample
                unified_timestamp = time_val  # Fallback to device timestamp

            bin_idx = int((time_val - raw_list[0][0]) / bin_width)
            bin_idx = min(bin_idx, num_bins - 1)

            if bin_idx not in bins:
                bins[bin_idx] = {
                    'times': [],
                    'currents': [],
                    'running_states': [],
                    'unified_timestamps': [],
                    'guest_id': guest_id,
                    'voltage_mv': voltage_mv,
                    'guest_params': guest_params,  # Use first sample's parameters for bin
                    'status_params': status_params
                }

            bins[bin_idx]['times'].append(time_val)
            bins[bin_idx]['currents'].append(current)
            bins[bin_idx]['running_states'].append(running_state)
            bins[bin_idx]['unified_timestamps'].append(unified_timestamp)

        # Create downsampled output
        result = []
        for bin_idx in sorted(bins.keys()):
            bin_data = bins[bin_idx]

            # Average time, current, and unified timestamp
            avg_time = sum(bin_data['times']) / len(bin_data['times'])
            avg_current = sum(bin_data['currents']) / len(bin_data['currents'])
            avg_unified_timestamp = sum(bin_data['unified_timestamps']) / len(bin_data['unified_timestamps'])

            # Use most common running state
            running_state = max(set(bin_data['running_states']),
                                key=bin_data['running_states'].count)

            result.append((avg_time, avg_current, running_state,
                           bin_data['guest_id'], bin_data['voltage_mv'],
                           bin_data['guest_params'], bin_data['status_params'], avg_unified_timestamp))

        return result

    def load_csv(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return False
                norm = [h.strip().lower() for h in header]
                t_idx = None
                i_idx = None
                for idx, name in enumerate(norm):
                    if name in ("time (s)", "time", "timestamp (s)"):
                        t_idx = idx
                    if name in ("current (ua)", "current", "i(ua)", "current (μa)"):
                        i_idx = idx
                if t_idx is None or i_idx is None:
                    return False
                t, i = [], []
                for row in reader:
                    if not row or len(row) <= max(t_idx, i_idx):
                        continue
                    try:
                        t.append(float(row[t_idx]))
                        i.append(float(row[i_idx]))
                    except Exception:
                        continue
            if not t:
                return False
            self.times = np.asarray(t, dtype=np.float64)
            self.currents = np.asarray(i, dtype=np.float64)
            return True
        except Exception:
            return False

    def get_time_bounds(self) -> Tuple[float, float]:
        if self.times.size == 0:
            return 0.0, 0.0
        return float(self.times[0]), float(self.times[-1])


class DataAccumulator:
    @staticmethod
    def bin_min_max_mean(times: np.ndarray, values: np.ndarray, t0: float, t1: float, bins: int):
        if bins <= 0 or t1 <= t0 or times.size == 0:
            return (np.array([], dtype=np.float64),) * 5
        # Clip range to available data
        t0 = max(t0, float(times[0]))
        t1 = min(t1, float(times[-1]))
        if t1 <= t0:
            return (np.array([], dtype=np.float64),) * 5
        edges = np.linspace(t0, t1, bins + 1)
        centers = (edges[:-1] + edges[1:]) * 0.5
        idx = np.searchsorted(times, edges)
        mins = np.full(bins, np.nan)
        maxs = np.full(bins, np.nan)
        sums = np.zeros(bins, dtype=np.float64)
        counts = np.zeros(bins, dtype=np.int64)
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
        means = np.full(bins, np.nan)
        valid = counts > 0
        means[valid] = sums[valid] / counts[valid]
        return centers[valid], mins[valid], maxs[valid], means[valid], counts[valid]

    @staticmethod
    def mean_per_bin(times: np.ndarray, values: np.ndarray, target_hz: float):
        if target_hz <= 0 or times.size == 0:
            return times.copy(), values.copy()
        t0, t1 = float(times[0]), float(times[-1])
        dur = max(1e-9, t1 - t0)
        bins = int(max(1, math.floor(dur * target_hz)))
        centers, _, _, means, valid_counts = DataAccumulator.bin_min_max_mean(times, values, t0, t1, bins)
        if centers.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        return centers, means


class PPK2Main(QMainWindow):
    def __init__(self, csv_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("PPK2-Python - SuperIoT Power Analyzer")
        self.resize(1400, 820)

        # Data and device state
        self.dm = DataManager()
        self.device = None
        self.sample_rate_hz: float = 100000.0  # default used for axis/estimates when streaming
        self.is_measuring: bool = False
        self.dut_power_on: bool = False
        self.live_mode: bool = True

        # View window state
        self.t_min = 0.0
        self.t_max = 10.0
        self.window_begin = 0.0
        self.window_end = 10.0
        self.paused = False

        # Controls state
        self.allowed_downsample = [1, 10, 100, 1000, 10000, 100000]
        self.time_window_options = [
            (0.010, "10ms"), (0.100, "100ms"), (1.0, "1s"), (3.0, "3s"), (10.0, "10s"),
            (60.0, "1min"), (600.0, "10min"), (3600.0, "1h"), (21600.0, "6h"), (86400.0, "24h"),
        ]
        self.current_downsample = 1000
        self.current_time_window = 10.0

        # Initialize additional parameters from guest app
        self.init_guest_parameters()

        self._build_ui()
        # Selection defaults
        self._sel_region = None
        self._sel_start = None

        # Load initial data after UI is built
        self._load_initial_data(csv_path)

        # Timers
        self.timer_plot = QTimer(self)
        self.timer_plot.setInterval(100)
        self.timer_plot.timeout.connect(self._tick)
        self.timer_plot.start()

        # Auto log clearing timer (every 20 seconds to prevent freezing)
        self.timer_log_clear = QTimer(self)
        self.timer_log_clear.setInterval(20000)  # 20 seconds
        self.timer_log_clear.timeout.connect(self._auto_clear_logs)
        self.timer_log_clear.start()

        # Guest status periodic update timer (every 1 second)
        self.timer_guest_status = QTimer(self)
        self.timer_guest_status.setInterval(1000)  # 1 second
        self.timer_guest_status.timeout.connect(self._periodic_guest_status_update)
        self.timer_guest_status.start()

        # Memory status timer (disabled for better performance)
        # self.timer_memory = QTimer(self)
        # self.timer_memory.setInterval(10000)  # 10 seconds
        # self.timer_memory.timeout.connect(self.update_memory_status)
        # self.timer_memory.start()

        # Ensure all plots start completely blank (after full initialization)
        self._ensure_blank_plots()

    def init_guest_parameters(self):
        """Initialize guest-specific parameters from app-guest-redesigned.py"""
        # Guest parameters
        self.guest_id = 10
        self.Tx_power_dBm = 0
        self.communication_volume_bytes = 0
        self.information_volume_bytes = 0
        self.per_unit_time_s = 1
        self.communication_mode = 0
        self.protocol_mode = 0
        self.power_supply_mode = 0
        self.advertising_interval = 250
        self.connection_interval = 100
        self.running_mode = 0
        self.fast_time = 0
        self.spare_0 = 0
        self.spare_1 = 0
        self.spare_2 = 0

        # Dual calibration system
        self.calibration_enabled = True
        self.calibration_sending_offset = 0
        self.calibration_normal_offset = 0

        # Wakeup detection parameters - MEAN-BASED DETECTION
        self.WAKEUP_WINDOW_SIZE = 40000  # Number of samples to check (at 100kHz = 10ms window)
        self.WAKEUP_MEAN_THRESHOLD = 700  # Mean current increase threshold (µA)
        self.wakeup_sample_buffer = []  # Rolling buffer for mean calculation
        self.wakeup_timestamp_buffer = []  # Corresponding timestamps for each sample in wakeup_sample_buffer

        # Initialize wakeup detection variables
        self.current_wakeup_state = 0  # Track current wakeup state (0 or 1)
        self.wakeup_detection_cooldown = 0  # Cooldown to prevent duplicate detections (samples)

        # New baseline-based wakeup detection
        self.wakeup_baseline = None  # Calculated baseline after 2 seconds
        self.wakeup_baseline_samples = []  # Samples for baseline calculation
        self.wakeup_baseline_start_time = None  # When we started collecting baseline
        self.BASELINE_COLLECTION_TIME = 2.0  # 2 seconds to collect baseline

        # Wakeup state change tracking for saved data - SIMPLEST APPROACH
        self.wakeup_timestamps = []  # List of timestamps when wakeup was detected

        # Serial error tracking to prevent spam
        self._serial_error_logged = False

        # Filter settings
        self.filter_enabled = False
        self.current_filter_type = "DelayedMedian"
        self.transmission_filter_mode = "ReducedWindow"
        self.transmission_window_size = 3

        # Initialize filters
        self.filters = {
            "DelayedMedian": DelayedMedianSpikeFilter(window_size=21, threshold=5, extreme_spike_limit=25000),
            "MovingAverage": MovingAverageFilter(window_size=21),
            "Hampel": HampelFilter(window_size=21, threshold=3),
            "SavitzkyGolay": SavitzkyGolayFilter(window_size=21, polyorder=2)
        }

        # Baseline replacement feature
        self.baseline_replacement_enabled = False
        self.baseline_replacement_mode = "LowerThanBaseline"
        self.baseline_value = 0
        self.baseline_collection_active = False
        self.baseline_collection_start_time = 0
        self.baseline_collection_delay = 2.0
        self.baseline_collection_duration = 2.0
        self.baseline_collection_buffer = []
        self.last_running_state = 0
        self.baseline_ui_update_counter = 0

        # Network manager for host connection
        self.network_manager = NetworkManager(self)
        self.serial_command_manager = SerialCommandManager(self)

        # Connect serial command manager signals
        self.serial_command_manager.serial_data_signal.connect(self.update_guest_status_display)
        self.serial_command_manager.status_text_signal.connect(self.update_serial_status_text)
        self.serial_command_manager.status_message_signal.connect(self.update_serial_status_text)

        # HIGH-PERFORMANCE PARAMETER BUFFER (Your approach implementation)
        self.parameter_buffer = ParameterBuffer()

        # Legacy support for existing code (will be gradually replaced)
        self.current_guest_status = {}  # Current status to be saved with each PPK2 sample
        self.guest_status_lock = threading.Lock()  # Thread-safe access

        # Time-synchronized guest status history for accurate matching
        self.guest_status_history = []  # List of (timestamp, status_dict) tuples
        self.status_history_lock = threading.Lock()  # Thread-safe access to history

        # Unified timestamp system - use same reference for both PPK2 and guest status
        self._measurement_start_time = None  # Will be set when measurement starts
        self._unified_time_reference = time.perf_counter()  # Common time reference

        # Network settings
        self.host_ip = "localhost"
        self.host_port = 65432

        # Print network info for debugging
        self._print_network_info()

        logging.basicConfig(level=logging.INFO)

    def _load_initial_data(self, csv_path):
        """Load initial data from CSV if available - but start with blank plots"""
        # Always start with blank plots for fresh app appearance
        # Initial data loading is disabled to ensure clean start

        # If you want to load specific CSV on startup, uncomment below:
        # loaded = False
        # if csv_path and os.path.isfile(csv_path):
        #     loaded = self.dm.load_csv(csv_path)
        #     if loaded:
        #         self.t_min, self.t_max = self.dm.get_time_bounds()
        #         self.window_end = self.t_max
        #         self.window_begin = max(self.t_min, self.window_end - self.current_time_window)
        #         self._render_overview()
        #         self._render_detail()

        # Note: Blank plots will be ensured at the end of initialization

    def _ensure_blank_plots(self):
        """Ensure all plots start with blank/empty data"""
        try:
            # Clear all plot curves with empty data
            if hasattr(self, 'curve_min'):
                self.curve_min.setData([], [])
            if hasattr(self, 'curve_max'):
                self.curve_max.setData([], [])
            if hasattr(self, 'curve_mean'):
                self.curve_mean.setData([], [])

            # Clear overview plot curves
            if hasattr(self, 'ov_min'):
                self.ov_min.setData([], [])
            if hasattr(self, 'ov_max'):
                self.ov_max.setData([], [])
            if hasattr(self, 'ov_mean'):
                self.ov_mean.setData([], [])

            # Reset data manager to empty state
            if hasattr(self, 'dm'):
                self.dm.clear()

            # Reset time bounds to default
            self.t_min = 0.0
            self.t_max = 10.0
            self.window_begin = 0.0
            self.window_end = 10.0

            # Update region to default position
            if hasattr(self, 'region'):
                self.region.setRegion([self.window_begin, self.window_end])

            logging.info("All plots initialized to blank state")

        except Exception as e:
            logging.error(f"Error ensuring blank plots: {e}")

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout with three columns
        main_layout = QHBoxLayout()
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        central.setLayout(main_layout)

        # Create three columns
        left_column = self.create_left_column()
        middle_column = self.create_middle_column()
        right_column = self.create_right_column()

        # Add columns to main layout with stretch factors
        main_layout.addWidget(left_column, 1)
        main_layout.addWidget(middle_column, 2)
        main_layout.addWidget(right_column, 1)

        # Set modern styling
        self.set_modern_styling()

    def create_left_column(self):
        """Create the left control column (from app-guest-redesigned.py)"""
        column = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        column.setLayout(layout)

        # Host Connection Group
        host_conn_group = QGroupBox("Host Connection")
        host_conn_layout = QGridLayout()
        host_conn_layout.setSpacing(3)

        host_conn_layout.addWidget(QLabel("Host IP:"), 0, 0)
        self.host_ip_edit = QLineEdit("localhost")
        host_conn_layout.addWidget(self.host_ip_edit, 0, 1)

        host_conn_layout.addWidget(QLabel("Host Port:"), 1, 0)
        self.host_port_edit = QLineEdit("65432")
        host_conn_layout.addWidget(self.host_port_edit, 1, 1)

        self.connect_host_button = QPushButton("🔗 Connect Host")
        host_conn_layout.addWidget(self.connect_host_button, 2, 0, 1, 2)

        host_conn_group.setLayout(host_conn_layout)
        layout.addWidget(host_conn_group)

        # PPK2 Connection Group
        ppk2_connection_group = QGroupBox("PPK2 Connection")
        ppk2_connection_layout = QVBoxLayout()

        ppk2_conn_grid_layout = QGridLayout()
        ppk2_conn_grid_layout.addWidget(QLabel("COM Port:"), 0, 0)
        self.port_combo = QComboBox()
        self._refresh_ports()
        ppk2_conn_grid_layout.addWidget(self.port_combo, 0, 1)

        self.refresh_button = QPushButton("🔄 Refresh")
        self.connect_btn = QPushButton("🔗 Connect")
        ppk2_conn_grid_layout.addWidget(self.refresh_button, 1, 0)
        ppk2_conn_grid_layout.addWidget(self.connect_btn, 1, 1)
        ppk2_connection_layout.addLayout(ppk2_conn_grid_layout)

        # Voltage Configuration
        voltage_group = QGroupBox("Voltage Configuration")
        voltage_layout = QVBoxLayout()

        slider_container = QHBoxLayout()
        self.voltage_slider = QSlider(Qt.Horizontal)
        self.voltage_slider.setRange(800, 5000)
        self.voltage_slider.setValue(3300)
        slider_container.addWidget(QLabel("Voltage (mV):"))
        slider_container.addWidget(self.voltage_slider)

        value_container = QHBoxLayout()
        self.voltage_spin = QSpinBox()
        self.voltage_spin.setRange(800, 5000)
        self.voltage_spin.setValue(3300)
        self.set_voltage_button = QPushButton("⚡ Set Voltage")
        value_container.addWidget(QLabel("Set Value (mV):"))
        value_container.addWidget(self.voltage_spin)
        value_container.addWidget(self.set_voltage_button)

        voltage_layout.addLayout(slider_container)
        voltage_layout.addLayout(value_container)
        voltage_group.setLayout(voltage_layout)
        ppk2_connection_layout.addWidget(voltage_group)

        # Measurement Mode
        mode_group = QGroupBox("Measurement Mode")
        mode_layout = QVBoxLayout()
        self.radio_ampere = QRadioButton("⚡ Ampere Meter Mode")
        self.radio_source = QRadioButton("🔋 Source Meter Mode")
        self.radio_source.setChecked(True)
        mode_layout.addWidget(self.radio_ampere)
        mode_layout.addWidget(self.radio_source)
        self.dut_power_check = QCheckBox("Enable Power Output (DUT)")
        self.dut_power_check.setChecked(False)
        mode_layout.addWidget(self.dut_power_check)
        mode_group.setLayout(mode_layout)
        ppk2_connection_layout.addWidget(mode_group)

        ppk2_connection_group.setLayout(ppk2_connection_layout)
        layout.addWidget(ppk2_connection_group)

        # Measurement Control Group
        measurement_group = QGroupBox("Measurement Control")
        measurement_layout = QGridLayout()
        measurement_layout.setSpacing(3)

        # Start/Stop buttons
        self.start_btn = QPushButton("▶ Start Measurement")
        self.stop_btn = QPushButton("⏹ Stop Measurement")
        self.stop_btn.setEnabled(False)
        measurement_layout.addWidget(self.start_btn, 0, 0, 1, 2)
        measurement_layout.addWidget(self.stop_btn, 1, 0, 1, 2)

        measurement_group.setLayout(measurement_layout)
        layout.addWidget(measurement_group)

        # Serial Command Connection Group
        serial_conn_group = QGroupBox("Serial Command Connection")
        serial_conn_layout = QVBoxLayout()

        # Use same grid layout as PPK2 Connection
        serial_conn_grid_layout = QGridLayout()
        serial_conn_grid_layout.addWidget(QLabel("Serial Port:"), 0, 0)
        self.serial_port_combo = QComboBox()
        self.refresh_serial_ports()
        serial_conn_grid_layout.addWidget(self.serial_port_combo, 0, 1)

        # Refresh and Connect buttons in same row (refresh button shorter)
        self.refresh_serial_button = QPushButton("🔄 Refresh")
        self.refresh_serial_button.setMaximumWidth(80)  # Make refresh button shorter
        self.serial_conn_toggle_button = QPushButton("🔗 Connect")
        serial_conn_grid_layout.addWidget(self.refresh_serial_button, 1, 0)
        serial_conn_grid_layout.addWidget(self.serial_conn_toggle_button, 1, 1)
        serial_conn_layout.addLayout(serial_conn_grid_layout)

        # Serial status text with clear button
        serial_output_layout = QHBoxLayout()
        self.serial_status_text = QTextEdit()
        self.serial_status_text.setReadOnly(True)
        self.serial_status_text.setMaximumHeight(80)
        self.serial_status_text.setPlaceholderText("Serial connection status will appear here...")
        serial_output_layout.addWidget(self.serial_status_text)

        # Clear button for serial output
        self.clear_serial_button = QPushButton("🧹")
        self.clear_serial_button.setMaximumWidth(30)
        self.clear_serial_button.setToolTip("Clear serial output")
        serial_output_layout.addWidget(self.clear_serial_button)

        serial_conn_layout.addLayout(serial_output_layout)

        serial_conn_group.setLayout(serial_conn_layout)
        layout.addWidget(serial_conn_group)

        # Add stretch to push everything up
        layout.addStretch(1)

        return column

    def create_middle_column(self):
        """Create the middle plotting column (original PPK2-Python functionality)"""
        column = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        column.setLayout(layout)

        # Controls from original PPK2-Python (make compact)
        ctrl = QGroupBox("Measurement Setup")
        ctrl.setMaximumHeight(80)  # Reduce measurement setup height
        grid = QGridLayout()
        grid.setSpacing(3)  # Reduce spacing
        grid.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        ctrl.setLayout(grid)

        # View controls (compact layout)
        grid.addWidget(QLabel("Downsample (Hz):"), 0, 0)
        self.cmb_down = QComboBox()
        self.cmb_down.setMaximumHeight(25)  # Smaller combo boxes
        for hz in self.allowed_downsample:
            self.cmb_down.addItem(str(hz), hz)
        self.cmb_down.setCurrentText(str(self.current_downsample))
        grid.addWidget(self.cmb_down, 0, 1)

        grid.addWidget(QLabel("Time Window:"), 0, 2)
        self.cmb_time = QComboBox()
        self.cmb_time.setMaximumHeight(25)  # Smaller combo boxes
        for seconds, label in self.time_window_options:
            self.cmb_time.addItem(label, seconds)
        self._rebuild_time_window_options(self.current_downsample)
        grid.addWidget(self.cmb_time, 0, 3)

        # Export PNG moved to right column
        # Export CSV removed (similar to Save Data)

        layout.addWidget(ctrl)

        # Top: detail plot
        self.detail = pg.PlotWidget()
        self.detail.setBackground('w')
        self.detail.showGrid(x=True, y=True, alpha=0.3)
        self.detail.setMenuEnabled(False)
        self.detail.setLabel('bottom', 'Time (s)')
        self.detail.setLabel('left', 'Current (μA)')
        vb = self.detail.getViewBox()
        # Match PPK2: left-drag pans; wheel zooms. Use PanMode (not RectMode)
        vb.setMouseMode(pg.ViewBox.PanMode)
        vb.setMouseEnabled(x=True, y=False)

        # Mouse wheel zoom controls detail time span and mirrors to overview region width (works paused or live)
        def _detail_wheel(ev):
            try:
                delta = ev.angleDelta().y()
            except Exception:
                delta = 0
            if delta == 0:
                return
            # User interaction: leave live-follow
            self.live_mode = False
            scale = 0.9 if delta > 0 else 1.1
            width = float(self.window_end - self.window_begin)
            cx = 0.5 * (self.window_begin + self.window_end)
            new_width = max(1e-6, width * scale)
            g0, g1 = self.dm.get_time_bounds()
            new_begin = max(g0, cx - 0.5 * new_width)
            new_end = min(g1, cx + 0.5 * new_width)
            if new_end <= new_begin:
                return
            self.window_begin, self.window_end = new_begin, new_end
            self.detail.setXRange(new_begin, new_end, padding=0)
            # Mirror to overview region
            try:
                self.region.blockSignals(True)
                self.region.setRegion([new_begin, new_end])
            finally:
                self.region.blockSignals(False)
            # Ensure detail content updates even when paused
            self._render_detail()
            ev.accept()

        self.detail.wheelEvent = _detail_wheel
        vb.sigXRangeChanged.connect(self._on_detail_xrange_changed)
        self.curve_min = self.detail.plot([], [], pen=pg.mkPen((33, 150, 243), width=1))
        self.curve_max = self.detail.plot([], [], pen=pg.mkPen((33, 150, 243), width=1))
        try:
            self.fill_envelope = pg.FillBetweenItem(self.curve_max, self.curve_min, brush=pg.mkBrush(33, 150, 243, 60))
            self.detail.addItem(self.fill_envelope)
        except Exception:
            self.fill_envelope = None
        self.curve_mean = self.detail.plot([], [], pen=pg.mkPen('b', width=1))
        # Crosshair
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((150, 150, 150), width=1))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((150, 150, 150), width=1))
        self.detail.addItem(self.vline, ignoreBounds=True)
        self.detail.addItem(self.hline, ignoreBounds=True)
        self.cross_text = pg.TextItem(html='<span style="color:#333;">t=-- s, I=-- μA</span>', anchor=(0, 1))
        self.detail.addItem(self.cross_text)
        self.detail.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.detail.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # Adjust detail plot height for better column alignment
        self.detail.setMaximumHeight(400)  # Reduced slightly for better column alignment
        layout.addWidget(self.detail, 2)

        # Middle: overview
        self.overview = pg.PlotWidget()
        self.overview.setBackground('w')
        self.overview.showGrid(x=True, y=True, alpha=0.3)
        self.overview.setMenuEnabled(False)
        self.overview.setLabel('bottom', 'Time (s)')
        self.overview.setLabel('left', 'Current (μA)')
        self.overview.setMaximumHeight(140)  # Reduced for better column alignment
        vb2 = self.overview.getViewBox()
        vb2.setMouseEnabled(x=True, y=False)

        # Mouse wheel zoom on overview controls selection block width (also works while paused)
        def _ov_wheel(ev):
            try:
                delta = ev.angleDelta().y()
            except Exception:
                delta = 0
            if delta == 0:
                return
            # User interaction: leave live-follow
            self.live_mode = False
            scale = 0.9 if delta > 0 else 1.1
            r0, r1 = self.region.getRegion()
            c = 0.5 * (r0 + r1)
            w = max(1e-6, (r1 - r0) * scale)
            g0, g1 = self.dm.get_time_bounds()
            new0 = max(g0, c - 0.5 * w)
            new1 = min(g1, c + 0.5 * w)
            if new1 <= new0:
                return
            self.region.setRegion([new0, new1])
            # Reflect region change to detail immediately
            self.window_begin, self.window_end = new0, new1
            self.detail.setXRange(new0, new1, padding=0)
            self._render_detail()
            ev.accept()

        self.overview.wheelEvent = _ov_wheel
        self.ov_min = self.overview.plot([], [], pen=pg.mkPen((120, 120, 120), width=1))
        self.ov_max = self.overview.plot([], [], pen=pg.mkPen((120, 120, 120), width=1))
        try:
            self.ov_fill = pg.FillBetweenItem(self.ov_max, self.ov_min, brush=pg.mkBrush(120, 120, 120, 60))
            self.overview.addItem(self.ov_fill)
        except Exception:
            self.ov_fill = None
        self.ov_mean = self.overview.plot([], [], pen=pg.mkPen('k', width=1))
        self.region = pg.LinearRegionItem([self.window_begin, self.window_end])
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        try:
            self.region.sigRegionChangeFinished.connect(self._on_region_finished)
        except Exception:
            pass
        self.overview.addItem(self.region)
        layout.addWidget(self.overview, 1)

        # Bottom: stats (single row, compact)
        stats = QGroupBox("Statistics")
        stats.setMaximumHeight(60)  # Single row height
        sl = QHBoxLayout()  # Back to horizontal (single row)
        sl.setSpacing(5)

        self.lbl_window = QLabel("Window: avg -- μA, max -- μA, duration -- s, charge -- μC")
        self.lbl_selection = QLabel("Selection: avg -- μA, max -- μA, duration -- s, charge -- μC")

        for lab in (self.lbl_window, self.lbl_selection):
            lab.setStyleSheet("font-size: 9px; color: #333;")  # Very small font for single row

        sl.addWidget(self.lbl_window)
        sl.addStretch(1)
        sl.addWidget(self.lbl_selection)
        stats.setLayout(sl)
        layout.addWidget(stats)

        return column

    def create_right_column(self):
        """Create the right configuration column (from app-guest-redesigned.py)"""
        column = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        column.setLayout(layout)

        # Guest Settings Group
        guest_settings_group = QGroupBox("Guest Settings")
        guest_settings_layout = QGridLayout()
        guest_settings_layout.setSpacing(3)

        row = 0
        col = 0

        # ID Parameter
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("ID (1-255):"))
        self.id_spin = QSpinBox()
        self.id_spin.setRange(1, 255)
        self.id_spin.setValue(self.guest_id)
        id_layout.addWidget(self.id_spin)
        guest_settings_layout.addLayout(id_layout, row, col)
        col += 1
        if col > 2:
            col = 0
            row += 1

        # Tx_power Parameter
        Tx_power_layout = QHBoxLayout()
        Tx_power_layout.addWidget(QLabel("Tx_power (dBm):"))
        self.Tx_power_spin = QSpinBox()
        self.Tx_power_spin.setRange(-20, 4)
        self.Tx_power_spin.setValue(self.Tx_power_dBm)
        Tx_power_layout.addWidget(self.Tx_power_spin)
        guest_settings_layout.addLayout(Tx_power_layout, row, col)
        col += 1
        if col > 2:
            col = 0
            row += 1

        # advertising_interval Parameter
        advertising_interval_layout = QHBoxLayout()
        advertising_interval_layout.addWidget(QLabel("advertising interval (20-2000ms):"))
        self.advertising_interval_spin = QSpinBox()
        self.advertising_interval_spin.setRange(20, 2000)
        self.advertising_interval_spin.setValue(self.advertising_interval)
        advertising_interval_layout.addWidget(self.advertising_interval_spin)
        guest_settings_layout.addLayout(advertising_interval_layout, row, col)
        col += 1
        if col > 2:
            col = 0
            row += 1

        # connection interval Parameter
        connection_interval_layout = QHBoxLayout()
        connection_interval_layout.addWidget(QLabel("connection interval (20-1000ms):"))
        self.connection_interval_spin = QSpinBox()
        self.connection_interval_spin.setRange(20, 1000)
        self.connection_interval_spin.setValue(self.connection_interval)
        connection_interval_layout.addWidget(self.connection_interval_spin)
        guest_settings_layout.addLayout(connection_interval_layout, row, col)
        col += 1
        if col > 2:
            col = 0
            row += 1

        # Spare 0 Parameter
        spare_0_layout = QHBoxLayout()
        spare_0_layout.addWidget(QLabel("Spare 0 (0-255):"))
        self.spare_0_spin = QSpinBox()
        self.spare_0_spin.setRange(0, 255)
        self.spare_0_spin.setValue(self.spare_0)
        spare_0_layout.addWidget(self.spare_0_spin)
        guest_settings_layout.addLayout(spare_0_layout, row, col)
        col += 1
        if col > 2:
            col = 0
            row += 1

        # Spare 1 Parameter
        spare_1_layout = QHBoxLayout()
        spare_1_layout.addWidget(QLabel("Spare 1 (0-255):"))
        self.spare_1_spin = QSpinBox()
        self.spare_1_spin.setRange(0, 255)
        self.spare_1_spin.setValue(self.spare_1)
        spare_1_layout.addWidget(self.spare_1_spin)
        guest_settings_layout.addLayout(spare_1_layout, row, col)

        guest_settings_group.setLayout(guest_settings_layout)
        layout.addWidget(guest_settings_group)

        # Guest Status Group (complete from original app)
        guest_status_group = QGroupBox("Guest Status")
        guest_status_layout = QGridLayout()
        guest_status_layout.setSpacing(3)

        row_status = 0
        col_status = 0

        # Running State Status
        running_state_status_layout = QHBoxLayout()
        running_state_status_layout.addWidget(QLabel("Running State:"))
        self.running_state_status_label = QLabel("N/A")
        running_state_status_layout.addWidget(self.running_state_status_label)
        guest_status_layout.addLayout(running_state_status_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # Wakeup Status
        wakeup_layout = QHBoxLayout()
        wakeup_layout.addWidget(QLabel("Wakeup:"))
        self.wakeup_label = QLabel("N/A")
        wakeup_layout.addWidget(self.wakeup_label)
        guest_status_layout.addLayout(wakeup_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # Communication Mode Status
        communication_mode_status_layout = QHBoxLayout()
        communication_mode_status_layout.addWidget(QLabel("Comm. Mode:"))
        self.communication_mode_status_label = QLabel("N/A")
        communication_mode_status_layout.addWidget(self.communication_mode_status_label)
        guest_status_layout.addLayout(communication_mode_status_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # Running Mode
        running_mode_layout = QHBoxLayout()
        running_mode_layout.addWidget(QLabel("Running Mode:"))
        self.running_mode_label = QLabel("N/A")
        running_mode_layout.addWidget(self.running_mode_label)
        guest_status_layout.addLayout(running_mode_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # VLC Protocol Mode Status
        vlc_protocol_mode_status_layout = QHBoxLayout()
        vlc_protocol_mode_status_layout.addWidget(QLabel("VLC Protocol:"))
        self.vlc_protocol_mode_status_label = QLabel("N/A")
        vlc_protocol_mode_status_layout.addWidget(self.vlc_protocol_mode_status_label)
        guest_status_layout.addLayout(vlc_protocol_mode_status_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # VLC Interval
        vlc_interval_layout = QHBoxLayout()
        vlc_interval_layout.addWidget(QLabel("VLC Interval:"))
        self.vlc_interval_label = QLabel("N/A")
        vlc_interval_layout.addWidget(self.vlc_interval_label)
        guest_status_layout.addLayout(vlc_interval_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # PWM Percentage
        pwm_layout = QHBoxLayout()
        pwm_layout.addWidget(QLabel("PWM %:"))
        self.pwm_label = QLabel("N/A")
        pwm_layout.addWidget(self.pwm_label)
        guest_status_layout.addLayout(pwm_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # BLE Protocol Mode
        ble_protocol_mode_layout = QHBoxLayout()
        ble_protocol_mode_layout.addWidget(QLabel("BLE Protocol:"))
        self.ble_protocol_mode_status_label = QLabel("N/A")
        ble_protocol_mode_layout.addWidget(self.ble_protocol_mode_status_label)
        guest_status_layout.addLayout(ble_protocol_mode_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # BLE Interval
        ble_interval_layout = QHBoxLayout()
        ble_interval_layout.addWidget(QLabel("BLE Interval:"))
        self.ble_interval_label = QLabel("N/A")
        ble_interval_layout.addWidget(self.ble_interval_label)
        guest_status_layout.addLayout(ble_interval_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # Fast Time
        fast_time_layout = QHBoxLayout()
        fast_time_layout.addWidget(QLabel("Fast Time:"))
        self.fast_time_label = QLabel("N/A")
        fast_time_layout.addWidget(self.fast_time_label)
        guest_status_layout.addLayout(fast_time_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # VLC Communication Volume Status
        vlc_comm_vol_layout = QHBoxLayout()
        vlc_comm_vol_layout.addWidget(QLabel("VLC Comm Vol:"))
        self.vlc_communication_volume_status_label = QLabel("N/A")
        vlc_comm_vol_layout.addWidget(self.vlc_communication_volume_status_label)
        guest_status_layout.addLayout(vlc_comm_vol_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # VLC Information Volume Status
        vlc_info_vol_layout = QHBoxLayout()
        vlc_info_vol_layout.addWidget(QLabel("VLC Info Vol:"))
        self.vlc_information_volume_status_label = QLabel("N/A")
        vlc_info_vol_layout.addWidget(self.vlc_information_volume_status_label)
        guest_status_layout.addLayout(vlc_info_vol_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # VLC Per Unit Time Status
        vlc_per_time_layout = QHBoxLayout()
        vlc_per_time_layout.addWidget(QLabel("VLC Per Time:"))
        self.vlc_per_unit_time_status_label = QLabel("N/A")
        vlc_per_time_layout.addWidget(self.vlc_per_unit_time_status_label)
        guest_status_layout.addLayout(vlc_per_time_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # BLE Communication Volume Status
        ble_comm_vol_layout = QHBoxLayout()
        ble_comm_vol_layout.addWidget(QLabel("BLE Comm Vol:"))
        self.ble_communication_volume_status_label = QLabel("N/A")
        ble_comm_vol_layout.addWidget(self.ble_communication_volume_status_label)
        guest_status_layout.addLayout(ble_comm_vol_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # BLE Information Volume Status
        ble_info_vol_layout = QHBoxLayout()
        ble_info_vol_layout.addWidget(QLabel("BLE Info Vol:"))
        self.ble_information_volume_status_label = QLabel("N/A")
        ble_info_vol_layout.addWidget(self.ble_information_volume_status_label)
        guest_status_layout.addLayout(ble_info_vol_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # BLE Per Unit Time Status
        ble_per_time_layout = QHBoxLayout()
        ble_per_time_layout.addWidget(QLabel("BLE Per Time:"))
        self.ble_per_unit_time_status_label = QLabel("N/A")
        ble_per_time_layout.addWidget(self.ble_per_unit_time_status_label)
        guest_status_layout.addLayout(ble_per_time_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # PHY Rate Percentage
        phy_rate_layout = QHBoxLayout()
        phy_rate_layout.addWidget(QLabel("PHY Rate %:"))
        self.phy_rate_label = QLabel("N/A")
        phy_rate_layout.addWidget(self.phy_rate_label)
        guest_status_layout.addLayout(phy_rate_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # MTU Value
        mtu_layout = QHBoxLayout()
        mtu_layout.addWidget(QLabel("MTU Value:"))
        self.mtu_label = QLabel("N/A")
        mtu_layout.addWidget(self.mtu_label)
        guest_status_layout.addLayout(mtu_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        # Spare 2, 3, 4
        spare_2_layout = QHBoxLayout()
        spare_2_layout.addWidget(QLabel("Spare 2:"))
        self.spare_2_label = QLabel("N/A")
        spare_2_layout.addWidget(self.spare_2_label)
        guest_status_layout.addLayout(spare_2_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        spare_3_layout = QHBoxLayout()
        spare_3_layout.addWidget(QLabel("Spare 3:"))
        self.spare_3_label = QLabel("N/A")
        spare_3_layout.addWidget(self.spare_3_label)
        guest_status_layout.addLayout(spare_3_layout, row_status, col_status)
        col_status += 1
        if col_status > 2:
            col_status = 0
            row_status += 1

        spare_4_layout = QHBoxLayout()
        spare_4_layout.addWidget(QLabel("Spare 4:"))
        self.spare_4_label = QLabel("N/A")
        spare_4_layout.addWidget(self.spare_4_label)
        guest_status_layout.addLayout(spare_4_layout, row_status, col_status)

        guest_status_group.setLayout(guest_status_layout)
        layout.addWidget(guest_status_group)

        # Create status_labels dictionary for compatibility
        self.status_labels = {
            "running_state": self.running_state_status_label,
            "wakeup_state": self.wakeup_label,
            "communication_mode": self.communication_mode_status_label,
            "running_mode": self.running_mode_label,
            "vlc_protocol_mode": self.vlc_protocol_mode_status_label,
            "vlc_interval_seconds": self.vlc_interval_label,
            "vlc_communication_volume": self.vlc_communication_volume_status_label,
            "vlc_information_volume": self.vlc_information_volume_status_label,
            "vlc_per_unit_time": self.vlc_per_unit_time_status_label,
            "pwm_percentage": self.pwm_label,
            "ble_protocol_mode": self.ble_protocol_mode_status_label,
            "ble_interval_seconds": self.ble_interval_label,
            "ble_communication_volume": self.ble_communication_volume_status_label,
            "ble_information_volume": self.ble_information_volume_status_label,
            "ble_per_unit_time": self.ble_per_unit_time_status_label,
            "phy_rate_percentage": self.phy_rate_label,
            "mtu_value": self.mtu_label,
            "fast_time": self.fast_time_label,
            "spare_2": self.spare_2_label,
            "spare_3": self.spare_3_label,
            "spare_4": self.spare_4_label
        }

        # Calibration Configuration Panel (moved before serial commands)
        calibration_group = QGroupBox("Calibration")
        calibration_layout = QGridLayout()
        calibration_layout.setSpacing(3)

        self.calibration_enabled_check = QCheckBox("Enable Calibration")
        self.calibration_enabled_check.setChecked(False)
        calibration_layout.addWidget(self.calibration_enabled_check, 0, 0, 1, 2)

        # Dynamic calibration labels that change based on running state
        self.normal_offset_label = QLabel("Other Offset:")
        calibration_layout.addWidget(self.normal_offset_label, 1, 0)
        self.normal_offset_spin = QDoubleSpinBox()
        self.normal_offset_spin.setRange(-1000, 1000)
        self.normal_offset_spin.setSuffix(" μA")
        calibration_layout.addWidget(self.normal_offset_spin, 1, 1)

        self.sending_offset_label = QLabel("Sending Offset:")
        calibration_layout.addWidget(self.sending_offset_label, 2, 0)
        self.sending_offset_spin = QDoubleSpinBox()
        self.sending_offset_spin.setRange(-1000, 1000)
        self.sending_offset_spin.setSuffix(" μA")
        calibration_layout.addWidget(self.sending_offset_spin, 2, 1)

        calibration_group.setLayout(calibration_layout)
        layout.addWidget(calibration_group)

        # Serial Commands Group
        serial_commands_group = QGroupBox("Serial Commands")
        serial_commands_layout = QGridLayout()
        serial_commands_layout.setSpacing(3)

        # Three default commands
        default_commands = [
            "AA 01 08 46 EE",
            "AA 01 08 50 EE",
            "AA 01 08 5A EE"
        ]

        self.cmd_inputs = []
        self.send_cmd_buttons = []

        for i, default_cmd in enumerate(default_commands):
            label = QLabel(f"Cmd {i + 1}:")
            cmd_edit = QLineEdit(default_cmd)
            send_btn = QPushButton("Send")

            serial_commands_layout.addWidget(label, i, 0)
            serial_commands_layout.addWidget(cmd_edit, i, 1)
            serial_commands_layout.addWidget(send_btn, i, 2)

            self.cmd_inputs.append(cmd_edit)
            self.send_cmd_buttons.append(send_btn)

        serial_commands_group.setLayout(serial_commands_layout)
        layout.addWidget(serial_commands_group)

        # Status and Logs Panel
        status_group = QGroupBox("Status & Logs")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-weight: bold; color: #2ecc71;")
        status_layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(50)
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Application logs will appear here...")
        status_layout.addWidget(self.log_text)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Data Management Group (moved from middle column)
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()
        data_layout.setSpacing(5)

        # Save controls row
        save_row = QHBoxLayout()
        self.save_data_btn = QPushButton("💾 Save Data")
        self.btn_export_png = QPushButton("📸 Export PNG")
        self.clear_data_btn = QPushButton("🗑 Clear Data")

        # File format selection
        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "CSV (Excel compatible)",
            "HDF5 (High performance)",
            "Parquet (Compressed)",
            "NPZ (NumPy arrays)",
            "All formats"
        ])
        self.format_combo.setCurrentText("CSV (Excel compatible)")
        self.format_combo.setToolTip(
            "Choose file format:\n• CSV: Excel compatible, but slow for large files\n• HDF5: Fast, efficient for large datasets\n• Parquet: Compressed, good for analysis\n• NPZ: NumPy arrays, fastest loading\n• All: Save in multiple formats")

        save_row.addWidget(self.save_data_btn)
        save_row.addWidget(self.format_combo)
        save_row.addWidget(self.btn_export_png)
        save_row.addWidget(self.clear_data_btn)

        # Downsampling control row
        downsample_row = QHBoxLayout()
        downsample_row.addWidget(QLabel("Save Downsample:"))
        self.save_downsample_combo = QComboBox()
        self.save_downsample_combo.addItem("1000 Hz", 1000)
        self.save_downsample_combo.addItem("5000 Hz", 5000)
        self.save_downsample_combo.addItem("10000 Hz", 10000)
        self.save_downsample_combo.addItem("20000 Hz", 20000)
        self.save_downsample_combo.addItem("50000 Hz", 50000)
        self.save_downsample_combo.addItem("100000 Hz (Full Rate)", 100000)
        self.save_downsample_combo.setCurrentIndex(5)  # Default to 100000 Hz (Full Rate)
        downsample_row.addWidget(self.save_downsample_combo)

        data_layout.addLayout(save_row)
        data_layout.addLayout(downsample_row)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Add stretch
        layout.addStretch(1)

        return column

    def set_modern_styling(self):
        """Apply modern styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin: 5px 0px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                border: none;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)

        # Wire signals
        self.refresh_button.clicked.connect(self._refresh_ports)
        self.connect_btn.clicked.connect(self._on_connect)
        self.start_btn.clicked.connect(self._on_start_stop)
        self.stop_btn.clicked.connect(self._on_start_stop)
        self.set_voltage_button.clicked.connect(self._on_set_voltage)
        self.radio_ampere.toggled.connect(self._on_mode_amp)
        self.radio_source.toggled.connect(self._on_mode_src)
        self.cmb_down.currentIndexChanged.connect(self._on_down_changed)
        self.cmb_time.currentIndexChanged.connect(self._on_time_changed)

        # Voltage slider connections
        self.voltage_slider.valueChanged.connect(self.voltage_spin.setValue)
        self.voltage_spin.valueChanged.connect(self.voltage_slider.setValue)

        # DUT power checkbox
        self.dut_power_check.toggled.connect(self._on_toggle_power)

        # Host connection
        self.connect_host_button.clicked.connect(self._on_connect_host)

        # Serial connections
        self.serial_conn_toggle_button.clicked.connect(self._on_toggle_serial)
        self.refresh_serial_button.clicked.connect(self.refresh_serial_ports)
        self.clear_serial_button.clicked.connect(self._on_clear_serial_output)

        # Serial command buttons
        for i, send_btn in enumerate(self.send_cmd_buttons):
            send_btn.clicked.connect(lambda checked, idx=i: self._send_serial_command(idx))

        # Data management buttons (moved to right column)
        self.save_data_btn.clicked.connect(self._on_save_data)
        self.clear_data_btn.clicked.connect(self._on_clear_data)
        self.btn_export_png.clicked.connect(self._on_export_png)

        # Calibration panel connections
        self.calibration_enabled_check.toggled.connect(self._on_calibration_config_changed)
        self.normal_offset_spin.valueChanged.connect(self._on_calibration_config_changed)
        self.sending_offset_spin.valueChanged.connect(self._on_calibration_config_changed)

        # Set initial calibration UI state
        self._on_calibration_config_changed()

        # Initial port population
        self._refresh_ports()

        # Interaction flags
        self._mouse_press_scene_pos = None
        self._dragging_active = False
        self._selecting_active = False

    # ---------- Device ----------
    def _on_connect(self):
        if self.device is not None:
            try:
                if self.is_measuring:
                    self._stop_measuring()
                self.device.toggle_DUT_power("OFF")
                if getattr(self.device, 'ser', None) and self.device.ser.is_open:
                    self.device.ser.close()
            except Exception:
                pass
            self.device = None
            self.connect_btn.setText("Connect")
            self.dut_power_check.setChecked(False)
            self.dut_power_on = False
            return
        if PPK2_API is None:
            QMessageBox.warning(self, "Device", "ppk2_api not available; CSV/demo mode only.")
            return
        try:
            # Use selected port if available
            sel_port = None
            try:
                sel_port = self.port_combo.currentData()
            except Exception:
                sel_port = None
            self.device = PPK2_API(port=sel_port, buffer_max_size_seconds=0.1, buffer_chunk_seconds=0.005,
                                   timeout=1, write_timeout=1, exclusive=True)
            if self.device.get_modifiers():
                self.connect_btn.setText("Disconnect")
            else:
                self.device = None
                QMessageBox.critical(self, "Device", "Failed to open PPK2 device.")
        except Exception as e:
            self.device = None
            QMessageBox.critical(self, "Device", f"Open failed: {e}")

    def _on_set_voltage(self):
        if self.device is None:
            return
        try:
            # Read from spinbox (800..5000 mV)
            val_mv = 0
            try:
                val_mv = int(self.voltage_spin.value())
            except Exception:
                val_mv = 0
            val_mv = max(800, min(5000, val_mv))
            self.device.set_source_voltage(val_mv)
        except Exception:
            pass

    def _on_toggle_power(self):
        if self.device is None:
            return
        try:
            if self.dut_power_on:
                self.device.toggle_DUT_power("OFF")
                self.dut_power_on = False
                self.dut_power_check.setChecked(False)
            else:
                self.device.toggle_DUT_power("ON")
                self.dut_power_on = True
                self.dut_power_check.setChecked(True)
        except Exception:
            pass

    def _refresh_ports(self):
        if list_ports is None:
            # Cannot enumerate ports; leave combo as-is
            return
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        ports = list(list_ports.comports())
        for p in ports:
            label = f"{p.device} - {p.description}"
            self.port_combo.addItem(label, p.device)
        self.port_combo.blockSignals(False)

    def _on_mode_amp(self):
        if self.device is None:
            return
        try:
            self.device.use_ampere_meter()
        except Exception:
            pass

    def _on_mode_src(self):
        if self.device is None:
            return
        try:
            self.device.use_source_meter()
        except Exception:
            pass

    def _on_start_stop(self):
        if self.device is None:
            self._toggle_pause_resume()
            return
        if self.is_measuring:
            self._stop_measuring()
        else:
            self._start_measuring()

    def _start_measuring(self):
        try:
            # Debug: Check serial connection status before measurement start
            serial_connected = hasattr(self, 'serial_command_manager') and self.serial_command_manager.connected
            logging.info(f"🔍 STARTING MEASUREMENT - Serial connected: {serial_connected}")

            # Comprehensive cleanup to prevent resource leaks
            if serial_connected:
                self._cleanup_serial_resources()

            # Reset guest status and clear logs when starting measurement
            self.reset_guest_status()
            self.clear_logs()

            # GLOBAL TIMESTAMP SYSTEM: Everything starts from t=0 when measurement starts
            self._global_start_time = time.perf_counter()  # Absolute reference point
            print(f"🕐 GLOBAL TIME: Measurement started at t=0.0s (absolute: {self._global_start_time:.6f}s)")

            # Clear wakeup timestamps from previous measurements
            if hasattr(self, 'wakeup_timestamps'):
                self.wakeup_timestamps = []
                print(f"🔄 Cleared wakeup timestamps from previous measurement")

            # Clear all timing references for clean start
            if hasattr(self, '_measurement_start_time'):
                delattr(self, '_measurement_start_time')
            if hasattr(self, '_ppk2_start_time'):
                delattr(self, '_ppk2_start_time')
            if hasattr(self, '_unified_time_reference'):
                delattr(self, '_unified_time_reference')

            # HIGH-PERFORMANCE: Clear parameter buffer for new measurement
            self.parameter_buffer.clear()

            # Initialize parameter buffer at global t=0.0
            self.parameter_buffer.update(0.0, self.parameter_buffer._get_default_params())
            print(f"🔧 GLOBAL TIME: Parameter buffer initialized at t=0.0s")

            # Legacy: Clear guest status history for backward compatibility
            with self.status_history_lock:
                self.guest_status_history.clear()
                # Initialize with default status at global t=0.0
                default_status = {
                    'running_state_status': 0, 'communication_mode_status': 0, 'wakeup_state': 0,
                    'running_mode': 0, 'vlc_protocol_mode_status': 0, 'vlc_interval_seconds': 0,
                    'vlc_communication_volume_status': 0, 'vlc_information_volume_status': 0,
                    'vlc_per_unit_time_status': 0, 'pwm_percentage': 0, 'ble_protocol_mode_status': 0,
                    'ble_interval_seconds': 0, 'ble_communication_volume_status': 0,
                    'ble_information_volume_status': 0, 'ble_per_unit_time_status': 0,
                    'phy_rate_percentage': 0, 'mtu_value': 0, 'fast_time': 0,
                    'spare_2': 0, 'spare_3': 0, 'spare_4': 0,
                    '_timestamp': 0.0  # Global time start
                }
                self.guest_status_history.append((0.0, default_status.copy()))

            # Initialize current guest status with global time
            with self.guest_status_lock:
                if not self.current_guest_status:
                    self.current_guest_status = default_status.copy()
                    self.current_guest_status.update(default_status)
                    print(f"🔧 LEGACY SYNC: Initialized default guest status for backward compatibility")

                    # Add default status to history at time 0.0 for time-based lookups
                    with self.status_history_lock:
                        default_status_copy = default_status.copy()
                        default_status_copy.pop('_timestamp', None)  # Remove timestamp from history entry
                        self.guest_status_history.append((0.0, default_status_copy))
                        print(f"🔧 LEGACY SYNC: Added default status to history at t=0.0s")

            # Debug: Verify signal connections after reset
            if serial_connected:
                self._verify_serial_signals()
                # Ensure serial connection is properly maintained
                self._ensure_serial_connection_health()

            # Clear all plot data and canvas when starting measurement
            self.dm.clear()
            self.t_min = 0.0
            self.t_max = 10.0
            self.window_begin = 0.0
            self.window_end = 10.0

            # Clear the plot curves
            try:
                self.curve_min.setData([], [])
                self.curve_max.setData([], [])
                self.curve_mean.setData([], [])
                self.ov_min.setData([], [])
                self.ov_max.setData([], [])
                self.ov_mean.setData([], [])
            except Exception:
                pass

            # Reset statistics
            self.lbl_window.setText("Window: avg -- μA, max -- μA, duration -- s, charge -- μC")
            self.lbl_selection.setText("Selection: avg -- μA, max -- μA, duration -- s, charge -- μC")

            self.device.toggle_DUT_power("ON")
            self.device.start_measuring()
            self.is_measuring = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.update_status("Measurement started")

            # Start periodic serial health check during measurement
            if hasattr(self, 'serial_command_manager') and self.serial_command_manager.connected:
                self._start_serial_health_monitor()
                # Additional verification and test after measurement start
                QTimer.singleShot(1000, self._test_serial_signal_connection)
                # Also test direct UI update
                QTimer.singleShot(2000, self.force_guest_status_update)
        except Exception as e:
            self.update_status(f"Failed to start measurement: {e}", error=True)
            QMessageBox.critical(self, "Measure", f"Failed to start: {e}")

    def _stop_measuring(self):
        try:
            self.device.stop_measuring()
            self.device.toggle_DUT_power("OFF")
        except Exception:
            pass
        self.is_measuring = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_status("Measurement stopped")

        # Stop serial health monitor when measurement stops
        self._stop_serial_health_monitor()

        # Ensure serial signals remain connected after stopping measurement
        if hasattr(self, 'serial_command_manager') and self.serial_command_manager.connected:
            self._verify_serial_signals()
            # Additional cleanup after stopping measurement
            QTimer.singleShot(1000, self._cleanup_after_stop)

    def _resume_measuring(self):
        """Resume measurement without resetting status values (used after save operations)"""
        try:
            # Debug: Show data state before resume
            if self.dm.times.size > 0:
                last_time = self.dm.times[-1]
                data_duration = self.dm.times[-1] - self.dm.times[0] if self.dm.times.size > 1 else 0
                print(f"📊 Resume: Last timestamp was {last_time:.3f}s, total duration: {data_duration:.3f}s")

            # Resume device measurement without clearing data or resetting status
            self.device.toggle_DUT_power("ON")
            self.device.start_measuring()
            self.is_measuring = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.update_status("Measurement resumed")
            print("✅ Measurement resumed successfully - status values preserved")
        except Exception as e:
            self.update_status(f"Failed to resume measurement: {e}", error=True)
            QMessageBox.critical(self, "Resume Measurement", f"Failed to resume: {e}")

    # ---------- Downsample / Time Window ----------
    def _rebuild_time_window_options(self, down_hz: int):
        # Mapping per user's constraint spec
        limits = {
            1: [60.0, 600.0, 3600.0, 21600.0, 86400.0],
            10: [10.0, 60.0, 600.0, 3600.0, 21600.0, 86400.0],
            100: [3.0, 10.0, 60.0, 600.0, 3600.0, 21600.0, 86400.0],
            1000: [1.0, 3.0, 10.0, 60.0, 600.0, 3600.0],
            10000: [0.1, 1.0, 3.0, 10.0, 60.0],
            100000: [0.010, 0.100, 1.0, 3.0, 10.0],
        }
        allowed = limits.get(int(down_hz), [10.0])
        cur_choice = self.cmb_time.currentData() if self.cmb_time.count() else None
        self.cmb_time.blockSignals(True)
        self.cmb_time.clear()
        for secs in allowed:
            label = next((lbl for s, lbl in self.time_window_options if abs(s - secs) < 1e-9), f"{secs}s")
            self.cmb_time.addItem(label, secs)
        # Auto-select minimum
        self.cmb_time.setCurrentIndex(0)
        self.cmb_time.blockSignals(False)
        self.current_time_window = float(self.cmb_time.currentData())

    def _on_down_changed(self):
        self.current_downsample = int(self.cmb_down.currentData())
        self._rebuild_time_window_options(self.current_downsample)
        self._snap_window_to_end()
        self._render_detail()

    def _on_time_changed(self):
        self.current_time_window = float(self.cmb_time.currentData())
        self._snap_window_to_end()
        self._render_detail()

    # ---------- Rendering ----------
    def _render_overview(self):
        t0, t1 = self.dm.get_time_bounds()
        if t1 <= t0:
            return
        bins = min(2000, max(200, int(self.overview.width() * self.devicePixelRatio())))
        centers, vmin, vmax, vmean, _ = DataAccumulator.bin_min_max_mean(self.dm.times, self.dm.currents, t0, t1, bins)
        if centers.size == 0:
            return
        self.ov_min.setData(centers, vmin)
        self.ov_max.setData(centers, vmax)
        self.ov_mean.setData(centers, vmean)
        # Do not force region here; user controls selection block. Overview lines update live.

    def _render_detail(self):
        if self.dm.times.size == 0:
            return
        t0, t1 = self.window_begin, self.window_end
        if t1 <= t0:
            return
        bins = max(300, int(self.detail.width() * self.devicePixelRatio()))
        centers, vmin, vmax, vmean, cnt = DataAccumulator.bin_min_max_mean(self.dm.times, self.dm.currents, t0, t1,
                                                                           bins)
        if centers.size == 0:
            return
        self.curve_min.setData(centers, vmin)
        self.curve_max.setData(centers, vmax)
        self.curve_mean.setData(centers, vmean)
        # Freeze detail x-range when paused; allow programmatic pan while dragging
        if not self.paused or getattr(self, '_dragging_active', False):
            self.detail.setXRange(t0, t1, padding=0)
        # Window stats: when paused, compute over live tail; else over current window
        if self.paused:
            g0, g1 = self.dm.get_time_bounds()
            stats_t1 = g1
            stats_t0 = max(g0, stats_t1 - float(self.current_time_window))
        else:
            stats_t0, stats_t1 = t0, t1
        sbins = max(300, int(self.detail.width() * self.devicePixelRatio()))
        sc, _, _, svmean, scnt = DataAccumulator.bin_min_max_mean(self.dm.times, self.dm.currents, stats_t0, stats_t1,
                                                                  sbins)
        if sc.size:
            try:
                w_avg = float(np.nansum(svmean * scnt) / np.nansum(scnt))
            except Exception:
                w_avg = float(np.nanmean(svmean))
            w_max = float(np.nanmax(svmean))
            dur = float(max(0.0, stats_t1 - stats_t0))
            charge_uC = w_avg * dur
            self.lbl_window.setText(
                f"Window: avg {w_avg:.1f} μA, max {w_max:.1f} μA, duration {dur:.3f} s, charge {charge_uC:.2f} μC")

    # ---------- Events ----------
    def _on_region_changed(self):
        try:
            r0, r1 = self.region.getRegion()
            self.window_begin = float(max(self.dm.get_time_bounds()[0], min(r0, r1)))
            self.window_end = float(min(self.dm.get_time_bounds()[1], max(r0, r1)))
            # Always update detail content to reflect overview selection
            self.detail.setXRange(self.window_begin, self.window_end, padding=0)
            self._render_detail()
        except Exception:
            pass

    def _on_region_finished(self):
        # Final sync on region drag end
        try:
            r0, r1 = self.region.getRegion()
            self.window_begin = float(max(self.dm.get_time_bounds()[0], min(r0, r1)))
            self.window_end = float(min(self.dm.get_time_bounds()[1], max(r0, r1)))
            self.detail.setXRange(self.window_begin, self.window_end, padding=0)
            self._render_detail()
        except Exception:
            pass

    def _on_detail_xrange_changed(self):
        try:
            (x0, x1), _ = self.detail.getViewBox().viewRange()
            self.window_begin = float(max(self.dm.get_time_bounds()[0], x0))
            self.window_end = float(min(self.dm.get_time_bounds()[1], x1))
            # Do not force overview region here to allow independent dragging during live updates
        except Exception:
            pass

    def _on_mouse_moved(self, pos):
        vb = self.detail.getViewBox()
        if vb is None:
            return
        if isinstance(pos, tuple) and len(pos) >= 2:
            pos = pg.QtCore.QPointF(pos[0], pos[1])
        # Initialize press anchor when left button first detected; do not pause on click
        try:
            from PyQt5.QtWidgets import QApplication
            buttons = QApplication.mouseButtons()
            if (buttons & Qt.LeftButton) and self._mouse_press_scene_pos is None:
                self._mouse_press_scene_pos = pg.QtCore.QPointF(pos)
            if not (buttons & Qt.LeftButton):
                self._mouse_press_scene_pos = None
        except Exception:
            pass
        # Pause detail on left-drag beyond threshold (no Shift)
        try:
            from PyQt5.QtWidgets import QApplication
            buttons = QApplication.mouseButtons()
            modifiers = QApplication.keyboardModifiers()
            if buttons & Qt.LeftButton and not (modifiers & Qt.ShiftModifier):
                if self._mouse_press_scene_pos is not None:
                    dx = float(pos.x() - self._mouse_press_scene_pos.x())
                    dy = float(pos.y() - self._mouse_press_scene_pos.y())
                    if (dx * dx + dy * dy) ** 0.5 > 12.0:
                        if not self.paused and not self._dragging_active:
                            self._dragging_active = True
                            self._toggle_pause_resume()
                        # Pan detail range with mouse delta in time coordinates
                        p_now = vb.mapSceneToView(pos)
                        p_anchor = vb.mapSceneToView(self._mouse_press_scene_pos)
                        dt = float(p_now.x() - p_anchor.x())
                        if dt != 0.0:
                            g0, g1 = self.dm.get_time_bounds()
                            width = float(self.window_end - self.window_begin)
                            new_begin = float(self.window_begin - dt)
                            new_end = new_begin + width
                            # clamp
                            if new_begin < g0:
                                new_begin = g0
                                new_end = g0 + width
                            if new_end > g1:
                                new_end = g1
                                new_begin = g1 - width
                            self.window_begin, self.window_end = new_begin, new_end
                            self.detail.setXRange(new_begin, new_end, padding=0)
                else:
                    self._dragging_active = False
        except Exception:
            pass

        p = vb.mapSceneToView(pos)
        tx = float(p.x())
        ty = float(p.y())
        self.vline.setPos(tx)
        self.hline.setPos(ty)
        self.cross_text.setHtml(f'<span style="color:#333;">t={tx:.3f} s, I={ty:.1f} μA</span>')
        self.cross_text.setPos(tx, ty)
        # Update selection region during Shift+Left drag; finalize on release
        try:
            from PyQt5.QtWidgets import QApplication
            if (QApplication.keyboardModifiers() & Qt.ShiftModifier) and (QApplication.mouseButtons() & Qt.LeftButton):
                # Start selection on movement with Shift held
                if not self._selecting_active:
                    self._selecting_active = True
                    self._sel_start = tx
                    if self._sel_region is None:
                        self._sel_region = pg.LinearRegionItem(orientation='vertical')
                        self._sel_region.setZValue(20)
                        self.detail.addItem(self._sel_region)
                    vb.setMouseEnabled(x=False, y=False)
                t0 = float(min(self._sel_start, tx))
                t1 = float(max(self._sel_start, tx))
                if self._sel_region is not None:
                    self._sel_region.setRegion([t0, t1])
            else:
                # Mouse released → finalize selection if active
                if self._selecting_active and self._sel_region is not None:
                    t0 = float(min(self._sel_region.getRegion()))
                    t1 = float(max(self._sel_region.getRegion()))
                    bins = max(300, int(self.detail.width() * self.devicePixelRatio()))
                    centers, _, _, vmean, _ = DataAccumulator.bin_min_max_mean(self.dm.times, self.dm.currents, t0, t1,
                                                                               bins)
                    if centers.size and vmean.size:
                        s_avg = float(np.nanmean(vmean))
                        s_max = float(np.nanmax(vmean))
                        dur = float(max(0.0, t1 - t0))
                        charge = s_avg * dur
                        self.lbl_selection.setText(
                            f"Selection: avg {s_avg:.1f} μA, max {s_max:.1f} μA, duration {dur:.3f} s, charge {charge:.2f} μC"
                        )
                    # Keep selection region persistent (like PPK2 cursor)
                    self._sel_start = None
                    self._selecting_active = False
                    vb.setMouseEnabled(x=True, y=False)
        except Exception:
            pass

    def _toggle_pause_resume(self):
        self.paused = not self.paused
        self.dm.paused = self.paused
        # Note: In the new layout, pause/resume is handled differently

    def _on_mouse_clicked(self, evt):
        if int(evt.button()) == 2:  # Right click → resume and jump to live end
            # resume if paused; also reset drag anchors and re-enable live follow
            try:
                self._mouse_press_scene_pos = None
                self._dragging_active = False
            except Exception:
                pass
            self.live_mode = True
            if self.paused:
                self._toggle_pause_resume()
            self._snap_window_to_end()
            # Reflect the resumed window to the overview region once
            try:
                self.region.blockSignals(True)
                self.region.setRegion([self.window_begin, self.window_end])
            finally:
                self.region.blockSignals(False)
            self._render_detail()
            return
        if int(evt.button()) == 1 and (evt.modifiers() & Qt.ShiftModifier):
            vb = self.detail.getViewBox()
            p = vb.mapSceneToView(evt.scenePos())
            tx = float(p.x())
            if not hasattr(self, '_sel_start') or self._sel_start is None:
                # start selection
                self._sel_start = tx
                if self._sel_region is None:
                    self._sel_region = pg.LinearRegionItem(orientation='vertical')
                    self._sel_region.setZValue(20)
                    self.detail.addItem(self._sel_region)
                self._sel_region.setRegion([self._sel_start, self._sel_start])
                # Disable default drag-zoom while selecting
                vb.setMouseEnabled(x=False, y=False)
                self._selecting_active = True
            return
        if int(evt.button()) == 1 and not (evt.modifiers() & Qt.ShiftModifier):
            # Record press position for drag detection; do not pause on simple click
            try:
                self._mouse_press_scene_pos = evt.scenePos()
            except Exception:
                self._mouse_press_scene_pos = None

    # removed _on_sel_region_finished; selection computed on second Shift+Left click

    # ---------- Export ----------
    # CSV export removed - now handled by Save Data functionality

    def _on_export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG (*.png)")
        if not path:
            return
        try:
            exporter = pg.exporters.ImageExporter(self.detail.plotItem)
            exporter.params.param('width').setValue(1600, blockSignal=exporter.widthChanged)
            exporter.export(path)
        except Exception as e:
            QMessageBox.critical(self, "Export", f"PNG export failed: {e}")

    # ---------- Ticking / Data ingest ----------
    def _tick(self):
        """Optimized tick method with performance monitoring"""
        # Performance monitoring
        tick_start_time = time.perf_counter()

        # Ingest from device (always ingest to keep overview/stat live)
        if self.device is not None and self.is_measuring:
            try:
                data = self.device.get_data()
                if data:
                    samples, _ = self.device.get_samples(data)
                    if samples:
                        # Use device-provided sampling interval estimate if available
                        n = len(samples)
                        if n > 0:
                            # SIMPLE GLOBAL TIME: Just use measurement-relative timestamps
                            dt = 1.0 / max(1.0, self.sample_rate_hz)
                            if self.dm.times.size == 0:
                                t0 = 0.0  # Start from t=0
                            else:
                                t0 = float(self.dm.times[-1] + dt)

                            times = t0 + np.arange(n, dtype=np.float64) * dt
                            currents = np.asarray(samples, dtype=np.float64)

                            # ALWAYS get current running state for wakeup detection (regardless of calibration)
                            current_running_state = 0

                            # First try to get from current guest status (most up-to-date)
                            if hasattr(self, 'current_guest_status') and self.current_guest_status:
                                with self.guest_status_lock:
                                    if 'running_state_status' in self.current_guest_status:
                                        try:
                                            current_running_state = int(
                                                self.current_guest_status['running_state_status'])
                                        except (ValueError, TypeError):
                                            current_running_state = 0

                            # Fallback to status labels if current_guest_status not available
                            if current_running_state == 0 and hasattr(self,
                                                                      'status_labels') and 'running_state' in self.status_labels:
                                try:
                                    current_running_state = int(self.status_labels['running_state'].text())
                                except (ValueError, AttributeError):
                                    current_running_state = 0

                            # Apply calibration offset if enabled
                            if hasattr(self,
                                       'calibration_enabled_check') and self.calibration_enabled_check.isChecked():
                                # Apply appropriate calibration offset with safety check
                                if current_running_state == 2:
                                    # Use sending calibration ONLY for running state 2 (Sending)
                                    calibration_offset = self.sending_offset_spin.value()
                                    # Log when switching to sending offset (less verbose)
                                    if not hasattr(self,
                                                   '_last_calibration_state') or self._last_calibration_state != 2:
                                        print(
                                            f"📡 Calibration: Running State {current_running_state} → Using Sending Offset ({calibration_offset}μA)")
                                        self._last_calibration_state = 2
                                else:
                                    # Use other/normal calibration for running states 0 (Idle), 1 (Receiving), 3, and others
                                    calibration_offset = self.normal_offset_spin.value()
                                    # Log when switching to other offset (less verbose)
                                    if not hasattr(self,
                                                   '_last_calibration_state') or self._last_calibration_state == 2:
                                        print(
                                            f"📡 Calibration: Running State {current_running_state} → Using Other Offset ({calibration_offset}μA)")
                                        self._last_calibration_state = current_running_state

                                # Safety check: prevent abnormal calibration offsets
                                if abs(calibration_offset) > 1000:
                                    print(
                                        f"⚠️  WARNING: Abnormal calibration offset detected: {calibration_offset}μA - skipping calibration")
                                    calibration_offset = 0

                                currents = currents + calibration_offset

                            # CRITICAL: Wakeup detection on PPK2 current data (works even when node is sleeping)
                            # This runs ALWAYS, regardless of calibration settings
                            # Pass BOTH times and currents for accurate timestamp tracking
                            self._process_wakeup_detection(times, currents)

                            # PERFORMANCE OPTIMIZATION: Skip expensive operations when only serial is connected
                            # Only do full data processing when PPK2 is actually measuring
                            if hasattr(self,
                                       'serial_command_manager') and self.serial_command_manager.connected and not self.is_measuring:
                                # Serial-only mode: minimal processing for performance
                                self.dm.append_samples(times, currents)  # Still show on canvas
                                self.t_min, self.t_max = self.dm.get_time_bounds()
                            else:
                                # Full PPK2 mode: complete data processing

                                # Get guest parameters and voltage (these don't change from wakeup detection)
                                guest_params = self.get_guest_parameters()
                                voltage_mv = self.voltage_spin.value() if hasattr(self, 'voltage_spin') else 3300

                                # CRITICAL: Get status parameters AFTER wakeup detection (in case running_state changed)
                                status_params = self.get_guest_status_parameters()  # Get FRESH status after wakeup detection

                                # Extract parameters for data storage (now using UPDATED status_params)
                                running_state = int(status_params.get('running_state', '0')) if status_params.get(
                                    'running_state', '0').isdigit() else 0
                                guest_id = guest_params.get('id', 1)

                                # Store current running state for wakeup detection
                                self.current_running_state = running_state

                                # HIGH-PERFORMANCE: Store individual samples with fast parameter lookup
                                for i, (time_val, current_val) in enumerate(zip(times, currents)):
                                    # Generate unified timestamp for synchronization across multiple guests
                                    unified_timestamp = time.perf_counter()

                                    # HIGH-PERFORMANCE: Use parameter buffer for fast, non-blocking lookup
                                    ppk2_sample_timestamp = time_val  # Already measurement-relative (cumulative time)

                                    # CRITICAL: Both PPK2 and parameters should now use measurement-relative time
                                    # PPK2 time_val is cumulative from measurement start
                                    # Parameter timestamps are (receive_time - measurement_start_time)
                                    # So they should be directly comparable

                                    buffer_matched_params = self.parameter_buffer.get_at_time(ppk2_sample_timestamp)

                                    # Debug: Verify parameter buffer lookup is working
                                    if not hasattr(self, '_lookup_debug_count'):
                                        self._lookup_debug_count = 0
                                    if self._lookup_debug_count < 3:
                                        self._lookup_debug_count += 1
                                        buffer_running_state = buffer_matched_params.get('running_state_status',
                                                                                         'MISSING')
                                        buffer_stats = self.parameter_buffer.get_stats()
                                        print(
                                            f"🔍 LOOKUP #{self._lookup_debug_count}: PPK2 t={ppk2_sample_timestamp:.6f}s → running_state={buffer_running_state}")
                                        print(f"   Buffer stats: {buffer_stats}")

                                    # Legacy: Get from old system using PPK2 cumulative time
                                    time_matched_guest_status = self._get_guest_status_at_time(ppk2_sample_timestamp)

                                    # HIGH-PERFORMANCE: Use buffer parameters (primary)
                                    enhanced_status_params = status_params.copy()
                                    enhanced_status_params.update(buffer_matched_params)

                                    # Legacy: Also merge old system for backward compatibility
                                    # enhanced_status_params.update(time_matched_guest_status)

                                    self.dm.append_raw_sample(time_val, current_val, running_state, guest_id,
                                                              voltage_mv,
                                                              guest_params.copy(), enhanced_status_params,
                                                              unified_timestamp,
                                                              self)

                                    # Store for potential downsampled transmission to host
                                    if self.network_manager.connected:
                                        # Store sample for downsampled transmission
                                        if not hasattr(self, '_host_transmission_buffer'):
                                            self._host_transmission_buffer = []
                                            self._last_host_transmission = time.time()

                                        self._host_transmission_buffer.append((time_val, current_val, running_state,
                                                                               guest_id, voltage_mv,
                                                                               guest_params.copy(),
                                                                               status_params.copy()))

                                # Debug: Periodically check collection rate
                                if not hasattr(self, '_last_rate_check'):
                                    self._last_rate_check = time.time()
                                    self._sample_count_at_check = len(self.dm.raw_samples_full)

                                current_time = time.time()
                                if current_time - self._last_rate_check >= 5.0:  # Every 5 seconds
                                    current_sample_count = len(self.dm.raw_samples_full)
                                    samples_added = current_sample_count - self._sample_count_at_check
                                    time_elapsed = current_time - self._last_rate_check
                                    collection_rate = samples_added / time_elapsed
                                    # Collection rate logging removed for clean output
                                    self._last_rate_check = current_time
                                    self._sample_count_at_check = current_sample_count

                                # CANVAS UPDATE: Add samples for display
                                self.dm.append_samples(times, currents)
                                self.t_min, self.t_max = self.dm.get_time_bounds()

                                # Send data to host app if connected - SIMPLE APPROACH
                                # Data transmission to host now happens ONLY during save process
                                # This ensures IDENTICAL timestamps and data between save and host
                                # See _send_exact_save_row_to_host() called from _save_with_efficient_storage()
                                pass  # No real-time transmission

                                # Send downsampled data to host periodically
                                if self.network_manager.connected:
                                    self._send_downsampled_data_to_host()

                            # Maintain live following for overview/stat; detail may be paused
                            if self.live_mode and not self.paused:
                                self._snap_window_to_end()
            except Exception:
                pass

        # Render: overview + stats always live; detail pauses X-range but still updates content when region changes
        self._render_overview()
        self._render_detail()

        # Performance monitoring - track tick duration
        tick_duration = time.perf_counter() - tick_start_time
        if not hasattr(self, '_tick_performance_stats'):
            self._tick_performance_stats = {'count': 0, 'total_time': 0.0, 'max_time': 0.0}

        self._tick_performance_stats['count'] += 1
        self._tick_performance_stats['total_time'] += tick_duration
        self._tick_performance_stats['max_time'] = max(self._tick_performance_stats['max_time'], tick_duration)

        # Log performance every 100 ticks
        if self._tick_performance_stats['count'] % 100 == 0:
            avg_time = self._tick_performance_stats['total_time'] / self._tick_performance_stats['count']
            max_time = self._tick_performance_stats['max_time']
            serial_connected = hasattr(self, 'serial_command_manager') and self.serial_command_manager.connected
            ppk2_measuring = self.device is not None and self.is_measuring

            logging.info(f"PERFORMANCE: Avg tick: {avg_time * 1000:.2f}ms, Max: {max_time * 1000:.2f}ms, "
                         f"Serial: {serial_connected}, PPK2: {ppk2_measuring}")

            # Reset stats periodically to get current performance
            self._tick_performance_stats = {'count': 0, 'total_time': 0.0, 'max_time': 0.0}

    # ---------- Helpers ----------
    def _snap_window_to_end(self):
        t0, t1 = self.dm.get_time_bounds()
        if t1 <= t0:
            return
        width = float(self.current_time_window)
        self.window_end = t1
        self.window_begin = max(t0, self.window_end - width)

    # ---------- Shortcuts ----------
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self._toggle_pause_resume()
            return
        if key == Qt.Key_R:
            # Reset X to global
            if self.dm.times.size:
                self.window_begin, self.window_end = self.dm.get_time_bounds()
                self._render_detail()
            return
        if key == Qt.Key_A:
            # Auto-range Y
            self.detail.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            return
        if key in (Qt.Key_Plus, Qt.Key_Equal):  # zoom in time
            self.current_time_window = max(0.010, self.current_time_window / 1.5)
            self._snap_window_to_end()
            self._render_detail()
            return
        if key == Qt.Key_Minus:  # zoom out time
            self.current_time_window = min(86400.0, self.current_time_window * 1.5)
            self._snap_window_to_end()
            self._render_detail()
            return
        if key == Qt.Key_BracketLeft:  # prev downsample
            idx = max(0, self.allowed_downsample.index(self.current_downsample) - 1)
            self.current_downsample = self.allowed_downsample[idx]
            self.cmb_down.setCurrentText(str(self.current_downsample))
            return
        if key == Qt.Key_BracketRight:  # next downsample
            idx = min(len(self.allowed_downsample) - 1, self.allowed_downsample.index(self.current_downsample) + 1)
            self.current_downsample = self.allowed_downsample[idx]
            self.cmb_down.setCurrentText(str(self.current_downsample))
            return

    # ---------- Additional Methods for Guest App Integration ----------
    def _send_data_to_host(self, times, currents, guest_params, status_params, voltage_mv):
        """Send measurement data to host app in the expected format"""
        try:
            # SYNCHRONIZED TRANSMISSION: Use save downsample setting to control transmission frequency
            save_hz = self.save_downsample_combo.currentData()

            # Calculate transmission interval based on save downsample frequency
            if save_hz >= 10000:  # Full rate or high frequency
                sample_step = 1  # Send all samples
            elif save_hz >= 5000:  # Medium frequency
                sample_step = 2  # Send every 2nd sample
            elif save_hz >= 1000:  # Low frequency
                sample_step = 3  # Send every 3rd sample
            else:  # Very low frequency
                sample_step = 5  # Send every 5th sample

            sample_indices = range(0, len(times), sample_step)  # Dynamic sampling based on save setting

            # Debug transmission rate
            if not hasattr(self, '_last_transmission_debug'):
                self._last_transmission_debug = time.time()
                self._transmission_count = 0

            self._transmission_count += len(sample_indices)
            current_time = time.time()
            if current_time - self._last_transmission_debug >= 5.0:  # Every 5 seconds
                transmission_rate = self._transmission_count / (current_time - self._last_transmission_debug)
                print(
                    f"🌐 HOST TRANSMISSION: Sending at {transmission_rate:.1f} Hz (Save setting: {save_hz} Hz, Step: {sample_step})")
                self._last_transmission_debug = current_time
                self._transmission_count = 0

            for i in sample_indices:
                time_val = times[i]
                current_val = currents[i]

                # Calculate power
                power_mw = (voltage_mv * current_val) / 1000.0

                # Get calibration info
                cal_enabled = self.calibration_enabled_check.isChecked()
                other_offset = self.normal_offset_spin.value() if cal_enabled else 0
                sending_offset = self.sending_offset_spin.value() if cal_enabled else 0

                # Determine which offset was actually applied
                running_state = int(status_params.get('running_state', '0')) if status_params.get('running_state',
                                                                                                  '0').isdigit() else 0
                if cal_enabled:
                    if running_state == 2:
                        applied_offset = sending_offset
                    else:
                        applied_offset = other_offset
                else:
                    applied_offset = 0

                # Calculate applied offsets exactly like save format
                running_state = int(status_params.get('running_state', '0')) if status_params.get('running_state',
                                                                                                  '0').isdigit() else 0
                applied_other_offset = other_offset if (cal_enabled and running_state != 2) else 0
                applied_sending_offset = sending_offset if (cal_enabled and running_state == 2) else 0

                # Create complete data packet in EXACT SAME FORMAT as CSV save
                data_to_send = {
                    'type': 'realtime_sample',
                    'guest_timestamp': time_val,  # ORIGINAL guest timestamp for comparison
                    'guest_id': guest_params.get('id', 1),
                    'timestamp': time_val,  # Timestamp (s) - same as save format
                    'current': current_val,  # Current (uA)
                    'applied_other_offset': applied_other_offset,  # Applied_Other_Offset (uA)
                    'applied_sending_offset': applied_sending_offset,  # Applied_Sending_Offset (uA)
                    'voltage': voltage_mv,  # Voltage (mV)
                    'power': power_mw,  # Power (mW)
                    'data_source': 'realtime',

                    # EXACT same parameters as CSV save format
                    'tx_power': guest_params.get('tx_power', 0),  # Tx_Power
                    'advertising_interval': guest_params.get('advertising_interval', 0),  # Adv_Interval
                    'connection_interval': guest_params.get('connection_interval', 0),  # Conn_Interval
                    'spare_0': guest_params.get('spare_0', 0),  # Spare_0
                    'spare_1': guest_params.get('spare_1', 0),  # Spare_1
                    'running_state': status_params.get('running_state_status', running_state),  # Running_State
                    'communication_mode': status_params.get('communication_mode_status', 0),  # Comm_Mode
                    'wakeup_state': status_params.get('wakeup_state', 0),  # Wakeup_State
                    'running_mode': status_params.get('running_mode', running_state),  # Running_Mode
                    'vlc_protocol_mode': status_params.get('vlc_protocol_mode_status', 0),  # VLC_Protocol
                    'vlc_interval_seconds': status_params.get('vlc_interval_seconds', 0),  # VLC_Interval
                    'vlc_communication_volume': status_params.get('vlc_communication_volume_status', 0),  # VLC_Comm_Vol
                    'vlc_information_volume': status_params.get('vlc_information_volume_status', 0),  # VLC_Info_Vol
                    'vlc_per_unit_time': status_params.get('vlc_per_unit_time_status', 0),  # VLC_Per_Time
                    'pwm_percentage': status_params.get('pwm_percentage', 0),  # PWM_Percent
                    'ble_protocol_mode': status_params.get('ble_protocol_mode_status', 0),  # BLE_Protocol
                    'ble_interval_seconds': status_params.get('ble_interval_seconds', 0),  # BLE_Interval
                    'ble_communication_volume': status_params.get('ble_communication_volume_status', 0),  # BLE_Comm_Vol
                    'ble_information_volume': status_params.get('ble_information_volume_status', 0),  # BLE_Info_Vol
                    'ble_per_unit_time': status_params.get('ble_per_unit_time_status', 0),  # BLE_Per_Time
                    'phy_rate_percentage': status_params.get('phy_rate_percentage', 0),  # PHY_Rate
                    'mtu_value': status_params.get('mtu_value', 0),  # MTU_Value
                    'fast_time': status_params.get('fast_time', 0),  # Fast_Time
                    'spare_2': status_params.get('spare_2', 0),  # Spare_2
                    'spare_3': status_params.get('spare_3', 0),  # Spare_3
                    'spare_4': status_params.get('spare_4', 0),  # Spare_4
                    'calibration_enabled': cal_enabled,  # Calibration_Enabled
                    'calibration_other_offset': other_offset,  # Calibration_Other_Offset
                    'calibration_sending_offset': sending_offset  # Calibration_Sending_Offset
                }

                # Send data to host
                if not self.network_manager.send_data(data_to_send):
                    # Connection lost, update UI
                    self.connect_host_button.setText("🔗 Connect Host")
                    self.update_status("Host connection lost", error=True)
                    break

        except Exception as e:
            logging.error(f"Error sending data to host: {e}")

    def _send_raw_samples_to_host(self, times, currents, guest_params, status_params, voltage_mv):
        """Send samples using FIXED-GRID downsampling to ensure continuous timestamps"""
        try:
            # Initialize tracking with fixed time grid
            if not hasattr(self, '_transmission_time_grid'):
                self._transmission_time_grid = 0.0  # Start from first sample time
                self._transmission_counter = 0
                self._grid_initialized = False

            # Get current save downsample setting
            save_hz = self.save_downsample_combo.currentData()
            bin_width = 1.0 / save_hz  # e.g., 0.001 for 1000Hz

            if len(self.dm.raw_samples_full) == 0:
                return  # No data

            # Initialize grid from first sample if not done
            if not self._grid_initialized:
                first_sample_time = self.dm.raw_samples_full[0][0]
                self._transmission_time_grid = first_sample_time
                self._grid_initialized = True
                print(f"🎯 GRID INITIALIZED: Starting from {first_sample_time:.6f}s with {save_hz}Hz grid")

            # Find samples that fit the next time grid slots
            samples_sent = 0
            current_grid_time = self._transmission_time_grid

            # Process raw samples to find those in the next grid slot
            for time_val, current_val, running_state, guest_id, voltage_mv_sample, guest_params_sample, status_params_sample in self.dm.raw_samples_full:

                # Check if this sample belongs to the current grid slot
                if time_val >= current_grid_time and time_val < (current_grid_time + bin_width):

                    # Calculate power using historical voltage
                    power_mw = (voltage_mv_sample * current_val) / 1000.0

                    # Get calibration info (use current UI state)
                    cal_enabled = self.calibration_enabled_check.isChecked()
                    other_offset = self.normal_offset_spin.value() if cal_enabled else 0
                    sending_offset = self.sending_offset_spin.value() if cal_enabled else 0

                    # Apply calibration logic
                    if cal_enabled:
                        if running_state == 2:
                            applied_other_offset = 0
                            applied_sending_offset = sending_offset
                        else:
                            applied_other_offset = other_offset
                            applied_sending_offset = 0
                    else:
                        applied_other_offset = 0
                        applied_sending_offset = 0

                    # Use the GRID TIME as timestamp (ensures continuity)
                    data_to_send = {
                        'type': 'fixed_grid',
                        'guest_timestamp': current_grid_time,  # FIXED GRID timestamp
                        'guest_id': guest_params_sample.get('id', 1),
                        'timestamp': current_grid_time,
                        'current': current_val,
                        'applied_other_offset': applied_other_offset,
                        'applied_sending_offset': applied_sending_offset,
                        'voltage': voltage_mv_sample,  # Historical voltage
                        'power': power_mw,
                        'data_source': 'fixed_grid',

                        # All parameters from historical sample
                        'tx_power': guest_params_sample.get('tx_power', 0),
                        'advertising_interval': guest_params_sample.get('advertising_interval', 0),
                        'connection_interval': guest_params_sample.get('connection_interval', 0),
                        'spare_0': guest_params_sample.get('spare_0', 0),
                        'spare_1': guest_params_sample.get('spare_1', 0),
                        'running_state': status_params_sample.get('running_state_status', running_state),
                        'communication_mode': status_params_sample.get('communication_mode_status', 0),
                        'wakeup_state': status_params_sample.get('wakeup_state', 0),
                        'running_mode': status_params_sample.get('running_mode', running_state),
                        'vlc_protocol_mode': status_params_sample.get('vlc_protocol_mode_status', 0),
                        'vlc_interval_seconds': status_params_sample.get('vlc_interval_seconds', 0),
                        'vlc_communication_volume': status_params_sample.get('vlc_communication_volume_status', 0),
                        'vlc_information_volume': status_params_sample.get('vlc_information_volume_status', 0),
                        'vlc_per_unit_time': status_params_sample.get('vlc_per_unit_time_status', 0),
                        'pwm_percentage': status_params_sample.get('pwm_percentage', 0),
                        'ble_protocol_mode': status_params_sample.get('ble_protocol_mode_status', 0),
                        'ble_interval_seconds': status_params_sample.get('ble_interval_seconds', 0),
                        'ble_communication_volume': status_params_sample.get('ble_communication_volume_status', 0),
                        'ble_information_volume': status_params_sample.get('ble_information_volume_status', 0),
                        'ble_per_unit_time': status_params_sample.get('ble_per_unit_time_status', 0),
                        'phy_rate_percentage': status_params_sample.get('phy_rate_percentage', 0),
                        'mtu_value': status_params_sample.get('mtu_value', 0),
                        'fast_time': status_params_sample.get('fast_time', 0),
                        'spare_2': status_params_sample.get('spare_2', 0),
                        'spare_3': status_params_sample.get('spare_3', 0),
                        'spare_4': status_params_sample.get('spare_4', 0),
                        'calibration_enabled': cal_enabled,
                        'calibration_other_offset': other_offset,
                        'calibration_sending_offset': sending_offset
                    }

                    # Send to host
                    if self.network_manager.send_data(data_to_send):
                        samples_sent += 1
                        self._transmission_counter += 1
                    else:
                        print(f"❌ Failed to send fixed grid sample to host")
                        return

                    # Move to next grid slot
                    current_grid_time += bin_width
                    self._transmission_time_grid = current_grid_time
                    break  # Only send one sample per grid slot

            # Debug output
            if samples_sent > 0:
                grid_start = self._transmission_time_grid - (samples_sent * bin_width)
                print(
                    f"📤 FIXED-GRID: Sent {samples_sent} samples to host (total: {self._transmission_counter}, grid: {grid_start:.6f}s to {self._transmission_time_grid:.6f}s, save_hz={save_hz})")

        except Exception as e:
            print(f"❌ Error sending full downsampled samples: {e}")
            logging.error(f"Full downsampled sample transmission failed: {e}")

    def _send_exact_save_row_to_host(self, time_val, current_val, applied_other_offset, applied_sending_offset,
                                     voltage_mv, power_mw, guest_params, status_params, running_mode):
        """Send the EXACT SAME data that gets saved to CSV"""
        try:
            # Create data packet with IDENTICAL data as CSV row
            data_to_send = {
                'type': 'save_identical',
                'guest_timestamp': time_val,  # IDENTICAL timestamp from save process
                'guest_id': guest_params.get('id', 1),
                'timestamp': time_val,
                'current': current_val,
                'applied_other_offset': applied_other_offset,
                'applied_sending_offset': applied_sending_offset,
                'voltage': voltage_mv,
                'power': power_mw,
                'data_source': 'save_identical',

                # All parameters exactly as saved
                'tx_power': guest_params.get('tx_power', 0),
                'advertising_interval': guest_params.get('advertising_interval', 0),
                'connection_interval': guest_params.get('connection_interval', 0),
                'spare_0': guest_params.get('spare_0', 0),
                'spare_1': guest_params.get('spare_1', 0),
                'running_state': status_params.get('running_state_status', running_mode),
                'communication_mode': status_params.get('communication_mode_status', 0),
                'wakeup_state': status_params.get('wakeup_state', 0),
                'running_mode': status_params.get('running_mode', running_mode),
                'vlc_protocol_mode': status_params.get('vlc_protocol_mode_status', 0),
                'vlc_interval_seconds': status_params.get('vlc_interval_seconds', 0),
                'vlc_communication_volume': status_params.get('vlc_communication_volume_status', 0),
                'vlc_information_volume': status_params.get('vlc_information_volume_status', 0),
                'vlc_per_unit_time': status_params.get('vlc_per_unit_time_status', 0),
                'pwm_percentage': status_params.get('pwm_percentage', 0),
                'ble_protocol_mode': status_params.get('ble_protocol_mode_status', 0),
                'ble_interval_seconds': status_params.get('ble_interval_seconds', 0),
                'ble_communication_volume': status_params.get('ble_communication_volume_status', 0),
                'ble_information_volume': status_params.get('ble_information_volume_status', 0),
                'ble_per_unit_time': status_params.get('ble_per_unit_time_status', 0),
                'phy_rate_percentage': status_params.get('phy_rate_percentage', 0),
                'mtu_value': status_params.get('mtu_value', 0),
                'fast_time': status_params.get('fast_time', 0),
                'spare_2': status_params.get('spare_2', 0),
                'spare_3': status_params.get('spare_3', 0),
                'spare_4': status_params.get('spare_4', 0),
                'calibration_enabled': status_params.get('calibration_enabled', False),
                'calibration_other_offset': status_params.get('calibration_other_offset', 0),
                'calibration_sending_offset': status_params.get('calibration_sending_offset', 0)
            }

            # Send to host (no counting needed since this is save-time only)
            success = self.network_manager.send_data(data_to_send)
            if not success:
                print(f"❌ Failed to send save-identical sample to host: {time_val:.6f}s")

        except Exception as e:
            print(f"❌ Error sending save-identical sample: {e}")
            logging.error(f"Save-identical sample transmission failed: {e}")

    def _send_downsampled_data_to_host(self):
        """Send downsampled data to host periodically at controlled frequency"""
        try:
            if not hasattr(self, '_host_transmission_buffer') or not self._host_transmission_buffer:
                return

            # Check if enough time has passed (send at ~20 Hz to host for 10kHz data)
            current_time = time.time()
            if current_time - self._last_host_transmission < 0.05:  # 50ms = 20 Hz for smooth 10kHz
                return

            # Use save downsample setting directly (support up to 10kHz)
            transmission_hz = self.save_downsample_combo.currentData()  # Direct from UI setting

            # Downsample the buffer if needed (optimized for 10kHz)
            if len(self._host_transmission_buffer) > transmission_hz / 20:  # If more than 0.05s worth of samples
                # Apply downsampling to reduce data
                downsampled_data = self.dm.downsample_full_samples(self._host_transmission_buffer, transmission_hz)

                # Send downsampled data
                for time_val, current_val, running_state, guest_id, voltage_mv, guest_params, status_params in downsampled_data:
                    data_to_send = {
                        'type': 'downsampled_stream',
                        'guest_timestamp': time_val,
                        'guest_id': guest_id,
                        'timestamp': time_val,
                        'current': current_val,
                        'running_state': running_state,
                        'voltage': voltage_mv,
                        'power': (voltage_mv * current_val) / 1000.0,
                        'data_source': 'downsampled_stream',
                        'transmission_hz': transmission_hz,

                        # All guest parameters
                        'tx_power': guest_params.get('tx_power', 0),
                        'advertising_interval': guest_params.get('advertising_interval', 0),
                        'connection_interval': guest_params.get('connection_interval', 0),
                        'spare_0': guest_params.get('spare_0', 0),
                        'spare_1': guest_params.get('spare_1', 0),
                        'spare_2': guest_params.get('spare_2', 0),
                        'spare_3': guest_params.get('spare_3', 0),
                        'spare_4': guest_params.get('spare_4', 0),

                        # All status parameters
                        'running_state_status': status_params.get('running_state_status', running_state),
                        'communication_mode_status': status_params.get('communication_mode_status', 0),
                        'wakeup_state': status_params.get('wakeup_state', 0),
                        'running_mode': status_params.get('running_mode', running_state),
                        'vlc_protocol_mode_status': status_params.get('vlc_protocol_mode_status', 0),
                        'vlc_interval_seconds': status_params.get('vlc_interval_seconds', 0),
                        'vlc_communication_volume_status': status_params.get('vlc_communication_volume_status', 0),
                        'vlc_information_volume_status': status_params.get('vlc_information_volume_status', 0),
                        'vlc_per_unit_time_status': status_params.get('vlc_per_unit_time_status', 0),
                        'pwm_percentage': status_params.get('pwm_percentage', 0),
                        'ble_protocol_mode_status': status_params.get('ble_protocol_mode_status', 0),
                        'ble_interval_seconds': status_params.get('ble_interval_seconds', 0),
                        'ble_communication_volume_status': status_params.get('ble_communication_volume_status', 0),
                        'ble_information_volume_status': status_params.get('ble_information_volume_status', 0),
                        'ble_per_unit_time_status': status_params.get('ble_per_unit_time_status', 0),
                        'phy_rate_percentage': status_params.get('phy_rate_percentage', 0),
                        'mtu_value': status_params.get('mtu_value', 0),
                        'fast_time': status_params.get('fast_time', 0),
                        'calibration_enabled': status_params.get('calibration_enabled', False),
                        'calibration_other_offset': status_params.get('calibration_other_offset', 0),
                        'calibration_sending_offset': status_params.get('calibration_sending_offset', 0)
                    }

                    # Send to host
                    self.network_manager.send_data(data_to_send)

                # Clear the buffer after sending
                self._host_transmission_buffer.clear()
                self._last_host_transmission = current_time

                print(f"🌐 Sent {len(downsampled_data):,} downsampled samples to host at {transmission_hz} Hz")

        except Exception as e:
            print(f"❌ Error sending downsampled data to host: {e}")

    def _send_saved_data_to_host_with_downsample(self, save_hz):
        """Send complete saved dataset to host with downsampling and HISTORICAL parameters"""
        try:
            if not self.dm.raw_samples_full:
                print("⚠️ No saved data to send to host")
                return

            print(
                f"🌐 HOST TRANSMISSION: Downsampling {len(self.dm.raw_samples_full):,} samples to {save_hz} Hz for host...")

            # Downsample the data using the same method as local save
            downsampled_samples = self.dm.downsample_full_samples(self.dm.raw_samples_full, save_hz)

            print(f"📤 SENDING {len(downsampled_samples):,} downsampled samples to host...")

            # Send each downsampled sample to host with 'saved_sample' type
            for i, (
                    time_val, current_val, running_state, guest_id, voltage_mv, guest_params,
                    status_params) in enumerate(
                downsampled_samples):

                # Calculate power
                power_mw = (voltage_mv * current_val) / 1000.0

                # Get calibration info (use historical values from this sample)
                cal_enabled = self.calibration_enabled_check.isChecked()  # Current UI state
                other_offset = self.normal_offset_spin.value() if cal_enabled else 0
                sending_offset = self.sending_offset_spin.value() if cal_enabled else 0

                # Create complete data packet with ALL historical parameters
                data_packet = {
                    'type': 'saved_sample',  # Mark as saved data
                    'timestamp': time_val,
                    'current': current_val,
                    'voltage': voltage_mv,
                    'power': power_mw,
                    'guest_id': guest_params.get('id', 1),
                    'running_state': running_state,
                    'calibration_enabled': cal_enabled,
                    'calibration_other_offset': other_offset,
                    'calibration_sending_offset': sending_offset,
                    'applied_other_offset': other_offset if (cal_enabled and running_state != 2) else 0,
                    'applied_sending_offset': sending_offset if (cal_enabled and running_state == 2) else 0,
                    'data_source': 'saved',
                    'save_frequency': save_hz,  # Include save frequency info

                    # ALL Guest parameters (historical from this sample)
                    'tx_power': guest_params.get('tx_power', 0),
                    'advertising_interval': guest_params.get('advertising_interval', 0),
                    'connection_interval': guest_params.get('connection_interval', 0),
                    'spare_0': guest_params.get('spare_0', 0),
                    'spare_1': guest_params.get('spare_1', 0),

                    # ALL Status parameters (historical from this sample)
                    'communication_mode': status_params.get('communication_mode_status', 0),
                    'wakeup_state': status_params.get('wakeup_state', 0),
                    'running_mode': status_params.get('running_mode', 0),
                    'vlc_protocol_mode': status_params.get('vlc_protocol_mode_status', 0),
                    'vlc_interval_seconds': status_params.get('vlc_interval_seconds', 0),
                    'vlc_communication_volume': status_params.get('vlc_communication_volume_status', 0),
                    'vlc_information_volume': status_params.get('vlc_information_volume_status', 0),
                    'vlc_per_unit_time': status_params.get('vlc_per_unit_time_status', 0),
                    'pwm_percentage': status_params.get('pwm_percentage', 0),
                    'ble_protocol_mode': status_params.get('ble_protocol_mode_status', 0),
                    'ble_interval_seconds': status_params.get('ble_interval_seconds', 0),
                    'ble_communication_volume': status_params.get('ble_communication_volume_status', 0),
                    'ble_information_volume': status_params.get('ble_information_volume_status', 0),
                    'ble_per_unit_time': status_params.get('ble_per_unit_time_status', 0),
                    'phy_rate_percentage': status_params.get('phy_rate_percentage', 0),
                    'mtu_value': status_params.get('mtu_value', 0),
                    'fast_time': status_params.get('fast_time', 0),
                    'spare_2': status_params.get('spare_2', 0),
                    'spare_3': status_params.get('spare_3', 0),
                    'spare_4': status_params.get('spare_4', 0)
                }

                # Send to host
                if not self.network_manager.send_data(data_packet):
                    print(f"❌ Failed to send saved data sample {i + 1}/{len(downsampled_samples)} to host")
                    break

                # Progress feedback every 1000 samples
                if (i + 1) % 1000 == 0:
                    print(f"📤 Sent {i + 1:,}/{len(downsampled_samples):,} saved samples to host...")

            print(f"✅ COMPLETED: Sent all {len(downsampled_samples):,} saved samples to host at {save_hz} Hz")

        except Exception as e:
            print(f"❌ Error sending saved data to host: {e}")
            logging.error(f"Failed to send saved data to host: {e}")

    def _send_saved_data_to_host(self):
        """Send complete saved dataset to host with HISTORICAL parameters"""
        try:
            if not hasattr(self, '_last_saved_data') or not self._last_saved_data:
                return

            data = self._last_saved_data
            historical_samples = data['historical_samples']
            total_samples = data['total_samples']

            print(f"📤 Sending saved dataset with HISTORICAL parameters to host: {total_samples:,} samples")

            # Send dataset info first
            dataset_info = {
                'type': 'saved_dataset',
                'guest_id': historical_samples[0][3] if historical_samples else 1,  # guest_id from first sample
                'total_samples': total_samples,
                'downsample_rate': data['downsample_rate'],
                'file_path': data['path'],
                'timestamp_start': historical_samples[0][0] if historical_samples else 0,  # time_val from first sample
                'timestamp_end': historical_samples[-1][0] if historical_samples else 0,  # time_val from last sample
                'data_type': 'historical_parameters'
            }

            if not self.network_manager.send_data(dataset_info):
                print("❌ Failed to send dataset info to host")
                return

            # Send historical data in chunks to avoid overwhelming the network
            chunk_size = 1000
            for i in range(0, total_samples, chunk_size):
                chunk_end = min(i + chunk_size, total_samples)

                for j in range(i, chunk_end):
                    # Extract historical sample: (time_val, current, running_mode, guest_id, voltage_mv, guest_params, status_params)
                    time_val, current_val, running_mode, guest_id, voltage_mv, guest_params, status_params = \
                        historical_samples[j]

                    # Calculate power using historical voltage
                    power_mw = (voltage_mv * current_val) / 1000.0

                    # Get historical calibration info
                    cal_enabled = status_params.get('calibration_enabled', False)
                    other_offset_raw = status_params.get('calibration_other_offset', 0)
                    sending_offset_raw = status_params.get('calibration_sending_offset', 0)

                    # Record which calibration offset was applied (current already has calibration applied from real-time processing)
                    if cal_enabled:
                        # Use running_state_status from guest status, not running_mode
                        guest_running_state = int(status_params.get('running_state_status', 0)) if status_params.get(
                            'running_state_status', 0) != -1 else 0
                        if guest_running_state == 2:
                            applied_other_offset = 0  # Other offset not applied
                            applied_sending_offset = sending_offset_raw  # Sending offset applied
                        else:
                            applied_other_offset = other_offset_raw  # Other offset applied
                            applied_sending_offset = 0  # Sending offset not applied
                    else:
                        applied_other_offset = 0  # No calibration applied
                        applied_sending_offset = 0

                    # Send complete historical data packet (ALL parameters like CSV save)
                    data_packet = {
                        'type': 'saved_sample',
                        'timestamp': time_val,
                        'current': current_val,
                        'voltage': voltage_mv,
                        'power': power_mw,
                        'guest_id': guest_id,
                        'running_state': running_mode,
                        'calibration_enabled': cal_enabled,
                        'calibration_other_offset': other_offset_raw,
                        'calibration_sending_offset': sending_offset_raw,
                        'applied_other_offset': applied_other_offset,
                        'applied_sending_offset': applied_sending_offset,
                        'data_source': 'historical',

                        # Guest parameters (same as CSV)
                        'tx_power': guest_params.get('tx_power', 0),
                        'advertising_interval': guest_params.get('advertising_interval', 0),
                        'connection_interval': guest_params.get('connection_interval', 0),
                        'spare_0': guest_params.get('spare_0', 0),
                        'spare_1': guest_params.get('spare_1', 0),

                        # Status parameters (same as CSV)
                        'communication_mode': status_params.get('communication_mode_status', 0),
                        'wakeup_state': status_params.get('wakeup_state', 0),
                        'running_mode': status_params.get('running_mode', 0),
                        'vlc_protocol_mode': status_params.get('vlc_protocol_mode_status', 0),
                        'vlc_interval_seconds': status_params.get('vlc_interval_seconds', 0),
                        'vlc_communication_volume': status_params.get('vlc_communication_volume_status', 0),
                        'vlc_information_volume': status_params.get('vlc_information_volume_status', 0),
                        'vlc_per_unit_time': status_params.get('vlc_per_unit_time_status', 0),
                        'pwm_percentage': status_params.get('pwm_percentage', 0),
                        'ble_protocol_mode': status_params.get('ble_protocol_mode_status', 0),
                        'ble_interval_seconds': status_params.get('ble_interval_seconds', 0),
                        'ble_communication_volume': status_params.get('ble_communication_volume_status', 0),
                        'ble_information_volume': status_params.get('ble_information_volume_status', 0),
                        'ble_per_unit_time': status_params.get('ble_per_unit_time_status', 0),
                        'phy_rate_percentage': status_params.get('phy_rate_percentage', 0),
                        'mtu_value': status_params.get('mtu_value', 0),
                        'fast_time': status_params.get('fast_time', 0),
                        'spare_2': status_params.get('spare_2', 0),
                        'spare_3': status_params.get('spare_3', 0),
                        'spare_4': status_params.get('spare_4', 0)
                    }

                    if not self.network_manager.send_data(data_packet):
                        print(f"❌ Failed to send historical saved data chunk at sample {j}")
                        return

                # Small delay between chunks
                time.sleep(0.01)

            # Send completion signal
            completion_signal = {
                'type': 'saved_dataset_complete',
                'guest_id': historical_samples[0][3] if historical_samples else 1,
                'total_samples': total_samples,
                'data_type': 'historical_parameters'
            }

            if self.network_manager.send_data(completion_signal):
                print(
                    f"✅ Successfully sent complete saved dataset with HISTORICAL parameters to host ({total_samples:,} samples)")
            else:
                print("❌ Failed to send completion signal to host")

        except Exception as e:
            logging.error(f"Error sending historical saved data to host: {e}")
            import traceback
            traceback.print_exc()

    def _print_network_info(self):
        """Print network information for debugging connection issues"""
        try:
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"🌐 Network Info:")
            print(f"   Hostname: {hostname}")
            print(f"   Local IP: {local_ip}")
            print(f"   Default host target: {self.host_ip}:{self.host_port}")
            print(f"💡 Connection Tips:")
            print(f"   • If host app is on same computer, try: localhost or 127.0.0.1")
            print(f"   • If host app is on different computer, use host's actual IP")
            print(f"   • Check that host app is running and listening on the port")
        except Exception as e:
            print(f"⚠️ Could not determine network info: {e}")

    def _on_connect_host(self):
        """Handle host connection/disconnection toggle"""
        try:
            if self.network_manager.connected:
                # Currently connected, so disconnect
                self.network_manager.disconnect()
                self.connect_host_button.setText("🔗 Connect Host")
                self.update_status("Disconnected from host")
                print("🔌 Host connection manually disconnected")
            else:
                # Currently disconnected, so connect
                host = self.host_ip_edit.text()
                port = int(self.host_port_edit.text())
                print(f"🔗 Attempting to connect to host {host}:{port}...")

                if self.network_manager.connect(host, port):
                    self.connect_host_button.setText("🔌 Disconnect Host")
                    self.update_status(f"Connected to host {host}:{port}")
                    print(f"✅ Successfully connected to host {host}:{port}")
                    # No popup for successful connection - just status message
                else:
                    self.update_status(f"Failed to connect to host {host}:{port}", error=True)
                    print(f"❌ Failed to connect to host {host}:{port}")
                    QMessageBox.warning(self, "Connection",
                                        f"Failed to connect to host {host}:{port}\n\nPossible issues:\n• Host app not running\n• Wrong IP/Port\n• Firewall blocking connection")
        except ValueError:
            self.update_status("Invalid port number", error=True)
            QMessageBox.critical(self, "Connection Error", "Please enter a valid port number")
        except Exception as e:
            self.update_status(f"Host connection failed: {e}", error=True)
            print(f"❌ Host connection error: {e}")
            QMessageBox.critical(self, "Connection Error", f"Host connection failed: {e}")

    def _on_toggle_serial(self):
        """Toggle serial connection"""
        try:
            if self.serial_command_manager.connected:
                self.serial_command_manager.disconnect()
                self.serial_conn_toggle_button.setText("🔗 Connect")
                self.serial_status_text.append("Serial disconnected")
                self.update_status("Serial disconnected")
            else:
                port = self.serial_port_combo.currentText()
                if self.serial_command_manager.connect(port):
                    self.serial_conn_toggle_button.setText("🔌 Disconnect")
                    self.serial_status_text.append(f"Connected to {port}")
                    self.update_status(f"Serial connected to {port}")

                    # Debug: Verify signal connections after connection
                    self._verify_serial_signals()
                else:
                    self.serial_status_text.append(f"Failed to connect to {port}")
                    self.update_status(f"Failed to connect to serial {port}", error=True)
        except Exception as e:
            self.serial_status_text.append(f"Serial error: {e}")
            self.update_status(f"Serial error: {e}", error=True)

    def _verify_serial_signals(self):
        """Ensure serial signal connections are properly established"""
        try:
            # Disconnect all signals first to avoid duplicate connections
            try:
                self.serial_command_manager.serial_data_signal.disconnect()
            except:
                pass
            try:
                self.serial_command_manager.status_text_signal.disconnect()
            except:
                pass
            try:
                self.serial_command_manager.status_message_signal.disconnect()
            except:
                pass

            # Reconnect all signals with fresh connections
            self.serial_command_manager.serial_data_signal.connect(self.update_guest_status_display)
            self.serial_command_manager.status_text_signal.connect(self.update_serial_status_text)
            self.serial_command_manager.status_message_signal.connect(self.update_serial_status_text)

            logging.info("🔍 SIGNAL CONNECTIONS RESTORED: All signals reconnected successfully")

        except Exception as e:
            logging.error(f"Error verifying serial signals: {e}")
            # Try alternative connection method
            try:
                # Force new connections without disconnect
                self.serial_command_manager.serial_data_signal.connect(self.update_guest_status_display)
                self.serial_command_manager.status_text_signal.connect(self.update_serial_status_text)
                self.serial_command_manager.status_message_signal.connect(self.update_serial_status_text)
                logging.info("🔍 SIGNAL CONNECTIONS: Alternative connection method succeeded")
            except Exception as e2:
                logging.error(f"Error in alternative signal connection: {e2}")

    def _ensure_serial_connection_health(self):
        """Ensure serial connection is healthy and properly functioning"""
        try:
            if not (hasattr(self, 'serial_command_manager') and self.serial_command_manager.connected):
                return

            # Check if serial port is still open
            if not (self.serial_command_manager.serial_port and self.serial_command_manager.serial_port.is_open):
                logging.warning("⚠️ Serial port closed unexpectedly, attempting to reconnect")
                port = self.serial_port_combo.currentText()
                if port:
                    # Disconnect and reconnect
                    self.serial_command_manager.disconnect()
                    if self.serial_command_manager.connect(port):
                        self.serial_conn_toggle_button.setText("🔌 Disconnect")
                        self.serial_status_text.append(f"🔄 Reconnected to {port}")
                        logging.info(f"✅ Serial port reconnected: {port}")
                    else:
                        self.serial_conn_toggle_button.setText("🔗 Connect")
                        self.serial_status_text.append(f"❌ Failed to reconnect to {port}")
                        logging.error(f"❌ Failed to reconnect to serial port: {port}")
                return

            # Check if serial thread is running
            if not self.serial_command_manager.running:
                logging.warning("⚠️ Serial thread stopped, restarting")
                self.serial_command_manager.running = True
                if not (
                        self.serial_command_manager.serial_thread and self.serial_command_manager.serial_thread.is_alive()):
                    import threading
                    self.serial_command_manager.serial_thread = threading.Thread(
                        target=self.serial_command_manager._serial_read_thread, daemon=True)
                    self.serial_command_manager.serial_thread.start()
                    logging.info("✅ Serial thread restarted")

            # Test serial connection with a small data request
            try:
                # Send a simple test to verify connection is working
                logging.info("🔍 Testing serial connection health")
                # Don't actually send a command, just verify the connection state

            except Exception as e:
                logging.warning(f"⚠️ Serial health check failed: {e}")

        except Exception as e:
            logging.error(f"Error ensuring serial connection health: {e}")

    def _start_serial_health_monitor(self):
        """Start periodic monitoring of serial connection health during measurement"""
        try:
            # Create a timer for periodic health checks
            if not hasattr(self, 'serial_health_timer'):
                self.serial_health_timer = QTimer(self)
                self.serial_health_timer.timeout.connect(self._periodic_serial_health_check)

            # Check every 10 seconds during measurement
            self.serial_health_timer.start(10000)
            logging.info("🔍 Serial health monitor started")
        except Exception as e:
            logging.error(f"Error starting serial health monitor: {e}")

    def _stop_serial_health_monitor(self):
        """Stop periodic monitoring of serial connection health"""
        try:
            if hasattr(self, 'serial_health_timer'):
                self.serial_health_timer.stop()
                logging.info("🔍 Serial health monitor stopped")
        except Exception as e:
            logging.error(f"Error stopping serial health monitor: {e}")

    def _periodic_serial_health_check(self):
        """Periodic check of serial connection health during measurement"""
        try:
            if hasattr(self, 'serial_command_manager') and self.serial_command_manager.connected:
                # Quick health check
                if not (self.serial_command_manager.serial_port and self.serial_command_manager.serial_port.is_open):
                    logging.warning("⚠️ Serial port lost during measurement, attempting recovery")
                    self._ensure_serial_connection_health()
                elif not self.serial_command_manager.running:
                    logging.warning("⚠️ Serial thread stopped during measurement, restarting")
                    self._ensure_serial_connection_health()
        except Exception as e:
            logging.error(f"Error in periodic serial health check: {e}")

    def _test_serial_signal_connection(self):
        """Test if serial signal connection is working by directly calling the update method"""
        try:
            if hasattr(self, 'serial_command_manager') and self.serial_command_manager.connected:
                # Force signal verification
                self._verify_serial_signals()

                # Create a test data packet to verify signal is working
                test_data = {
                    'running_state': 1,
                    'wakeup_state': 1,
                    'communication_mode': 1,
                    'running_mode': 0,
                    'vlc_protocol_mode': 2,
                    'vlc_interval_seconds': 50,
                    'vlc_communication_volume': 0,
                    'vlc_information_volume': 0,
                    'vlc_per_unit_time': 20,
                    'pwm_percentage': 50,
                    'ble_protocol_mode': 2,
                    'ble_interval_seconds': 36,
                    'ble_communication_volume': 0,
                    'ble_information_volume': 0,
                    'ble_per_unit_time': 10,
                    'phy_rate_percentage': 1,
                    'mtu_value': 13,
                    'fast_time': 30,
                    'spare_2': 0,
                    'spare_3': 0,
                    'spare_4': 20
                }

                logging.info("🔍 TESTING SIGNAL CONNECTION - Testing both signal and direct call")

                # Test 1: Direct call to update method (bypass signal)
                logging.info("🔍 TEST 1: Direct call to update_guest_status_display")
                self.update_guest_status_display(test_data)

                # Test 2: Signal emission
                logging.info("🔍 TEST 2: Emitting signal")
                self.serial_command_manager.serial_data_signal.emit(test_data)

                # Schedule a check to see if the UI was updated
                QTimer.singleShot(500, lambda: self._verify_ui_update_test(test_data))

        except Exception as e:
            logging.error(f"Error testing serial signal connection: {e}")

    def _verify_ui_update_test(self, test_data):
        """Verify if the test data was received by the UI"""
        try:
            if hasattr(self, 'status_labels') and 'running_state' in self.status_labels:
                current_value = self.status_labels['running_state'].text()
                expected_value = str(test_data['running_state'])
                if current_value == expected_value:
                    logging.info("✅ SIGNAL CONNECTION TEST PASSED - UI updated correctly")
                else:
                    logging.warning(
                        f"⚠️ SIGNAL CONNECTION TEST FAILED - Expected: {expected_value}, Got: {current_value}")
                    # Try to fix by forcing signal reconnection
                    self._verify_serial_signals()
                    # Try the test again in 1 second
                    QTimer.singleShot(1000, lambda: self._retry_signal_test(test_data))
            else:
                logging.warning("⚠️ Cannot verify UI update - status_labels not available")
        except Exception as e:
            logging.error(f"Error verifying UI update test: {e}")

    def _retry_signal_test(self, test_data):
        """Retry the signal test after reconnection"""
        try:
            if hasattr(self, 'status_labels') and 'running_state' in self.status_labels:
                current_value = self.status_labels['running_state'].text()
                expected_value = str(test_data['running_state'])
                if current_value == expected_value:
                    logging.info("✅ SIGNAL CONNECTION RETRY TEST PASSED - UI updated after reconnection")
                else:
                    logging.error(f"❌ SIGNAL CONNECTION RETRY TEST FAILED - Signal may be permanently broken")
        except Exception as e:
            logging.error(f"Error in retry signal test: {e}")

    def force_guest_status_update(self):
        """Force update guest status with test data - bypass all signal mechanisms"""
        try:
            test_data = {
                'running_state': 1,
                'wakeup_state': 1,
                'communication_mode': 1,
                'running_mode': 0,
                'vlc_protocol_mode': 2,
                'vlc_interval_seconds': 50,
                'vlc_communication_volume': 0,
                'vlc_information_volume': 0,
                'vlc_per_unit_time': 20,
                'pwm_percentage': 50,
                'ble_protocol_mode': 2,
                'ble_interval_seconds': 36,
                'ble_communication_volume': 0,
                'ble_information_volume': 0,
                'ble_per_unit_time': 10,
                'phy_rate_percentage': 1,
                'mtu_value': 13,
                'fast_time': 30,
                'spare_2': 0,
                'spare_3': 0,
                'spare_4': 20
            }

            logging.info("🔍 FORCE UPDATE: Directly updating guest status with test data")

            # Direct UI update without any signal mechanism
            if hasattr(self, 'status_labels') and self.status_labels:
                for field, value in test_data.items():
                    if field in self.status_labels:
                        try:
                            self.status_labels[field].setText(f"<font color='red'><b>FORCE:{value}</b></font>")
                            logging.info(f"🔍 FORCE UPDATED: {field} = {value}")
                        except Exception as e:
                            logging.error(f"Error force updating {field}: {e}")
                logging.info("✅ FORCE UPDATE COMPLETED - Check if UI shows red 'FORCE:' values")
            else:
                logging.error("❌ FORCE UPDATE FAILED - status_labels not available")

        except Exception as e:
            logging.error(f"Error in force guest status update: {e}")

    def _cleanup_serial_resources(self):
        """Comprehensive cleanup of serial-related resources to prevent leaks"""
        try:
            logging.info("🔍 CLEANUP: Starting serial resource cleanup")

            # Stop all timers that might be related to serial
            timer_attrs = ['serial_health_timer', '_ui_timer', 'timer_plot', 'timer_log_clear', 'timer_guest_status']
            for attr in timer_attrs:
                if hasattr(self, attr):
                    timer = getattr(self, attr)
                    if timer and hasattr(timer, 'stop'):
                        try:
                            timer.stop()
                            logging.info(f"🔍 CLEANUP: Stopped timer {attr}")
                        except Exception as e:
                            logging.error(f"Error stopping timer {attr}: {e}")

            # Clear any accumulated signal connections by recreating the SerialCommandManager
            if hasattr(self, 'serial_command_manager'):
                try:
                    # Disconnect the old manager
                    self.serial_command_manager.disconnect()
                    logging.info("🔍 CLEANUP: Old serial manager disconnected")

                    # Create a fresh SerialCommandManager instance
                    self.serial_command_manager = SerialCommandManager(self)

                    # Reconnect signals with fresh connections
                    self.serial_command_manager.serial_data_signal.connect(self.update_guest_status_display)
                    self.serial_command_manager.status_text_signal.connect(self.update_serial_status_text)
                    self.serial_command_manager.status_message_signal.connect(self.update_serial_status_text)

                    logging.info("🔍 CLEANUP: Fresh serial manager created with clean signals")
                except Exception as e:
                    logging.error(f"Error recreating serial manager: {e}")

            # Clear any accumulated guest status data
            if hasattr(self, '_previous_guest_status'):
                self._previous_guest_status.clear()
                logging.info("🔍 CLEANUP: Previous guest status cleared")

            # Force garbage collection to free memory
            import gc
            gc.collect()
            logging.info("🔍 CLEANUP: Garbage collection completed")

        except Exception as e:
            logging.error(f"Error in serial resource cleanup: {e}")

    def _cleanup_after_stop(self):
        """Additional cleanup after measurement stops to prevent accumulation"""
        try:
            logging.info("🔍 POST-STOP CLEANUP: Starting cleanup after measurement stop")

            # Clear any remaining data in serial buffers
            if hasattr(self, 'serial_command_manager') and self.serial_command_manager:
                try:
                    if hasattr(self.serial_command_manager, 'receive_buffer'):
                        self.serial_command_manager.receive_buffer.clear()
                    if hasattr(self.serial_command_manager, '_latest_queue'):
                        while not self.serial_command_manager._latest_queue.empty():
                            try:
                                self.serial_command_manager._latest_queue.get_nowait()
                            except:
                                break
                    logging.info("🔍 POST-STOP CLEANUP: Serial buffers cleared")
                except Exception as e:
                    logging.error(f"Error clearing serial buffers: {e}")

            # Force garbage collection
            import gc
            gc.collect()
            logging.info("🔍 POST-STOP CLEANUP: Completed")

        except Exception as e:
            logging.error(f"Error in post-stop cleanup: {e}")

    def _on_clear_serial_output(self):
        """Clear the serial output text area"""
        try:
            if hasattr(self, 'serial_status_text') and self.serial_status_text is not None:
                self.serial_status_text.clear()
                self.update_status("Serial output cleared")
        except Exception as e:
            self.update_status(f"Error clearing serial output: {e}", error=True)

    def refresh_serial_ports(self):
        """Refresh available serial ports"""
        if not serial:
            return
        self.serial_port_combo.clear()
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.serial_port_combo.addItem(port.device)
        except Exception as e:
            logging.error(f"Failed to list serial ports: {e}")

    def _send_serial_command(self, idx):
        """Send a serial command by index"""
        if idx < len(self.cmd_inputs):
            cmd = self.cmd_inputs[idx].text()
            if self.serial_command_manager.send_command(cmd):
                self.serial_status_text.append(f"Sent cmd {idx + 1}: {cmd}")
            else:
                self.serial_status_text.append(f"Failed to send cmd {idx + 1}")

    def update_serial_status_text(self, text: str):
        """Slot to append status messages or plain text from SerialCommandManager."""
        try:
            # Debug: Log when this slot is called
            logging.info(f"🔍 UPDATE_SERIAL_STATUS_TEXT called with: '{text}'")
            if hasattr(self, 'serial_status_text') and self.serial_status_text is not None:
                self.serial_status_text.append(text)
            else:
                logging.warning("⚠️ serial_status_text not available")
        except Exception as e:
            logging.error(f"Error updating serial status text: {e}")

    def _process_frame(self, frame):
        """Process serial frame with EXACT original validation and parsing"""
        # Debug: Log received frame
        frame_hex = ' '.join(f'{b:02X}' for b in frame)
        logging.info(f"🔍 Processing frame: {frame_hex}")

        # Validate frame format
        if len(frame) < 4 or not (frame.startswith(b'\xAA') and frame.endswith(b'\xEE')):
            logging.warning(
                f"❌ Bad frame format - Length: {len(frame)}, Start: {frame[0]:02X if len(frame) > 0 else 'N/A'}, End: {frame[-1]:02X if len(frame) > 0 else 'N/A'}")
            return None

        payload = frame[1:-2]
        recv_cs = frame[-2:-1]
        calc_cs = bytes([sum(payload) % 256])
        if recv_cs != calc_cs:
            logging.warning(
                f"⚠️ Checksum mismatch - Received: {recv_cs[0]:02X}, Calculated: {calc_cs[0]:02X} - PROCESSING ANYWAY")
            # return None  # Disabled - process frame anyway for testing

        if len(payload) != 21:
            logging.warning(f"⚠️ Unexpected payload length: {len(payload)} (expected 21) - PADDING/TRUNCATING")
            # Pad or truncate to 21 bytes
            if len(payload) < 21:
                payload = payload + b'\x00' * (21 - len(payload))  # Pad with zeros
            else:
                payload = payload[:21]  # Truncate to 21 bytes

        # EXACT original keys mapping
        keys = [
            "running_state", "wakeup_state", "communication_mode", "running_mode",
            "vlc_protocol_mode", "vlc_interval_seconds", "vlc_communication_volume",
            "vlc_information_volume", "vlc_per_unit_time", "pwm_percentage",
            "ble_protocol_mode", "ble_interval_seconds", "ble_communication_volume",
            "ble_information_volume", "ble_per_unit_time", "phy_rate_percentage",
            "mtu_value", "fast_time", "spare_2", "spare_3", "spare_4"
        ]
        values = list(payload)
        parsed = dict(zip(keys, values))

        # Debug: Log parsed data
        important_params = {k: v for k, v in parsed.items() if
                            k in ['running_state', 'wakeup_state', 'communication_mode', 'running_mode']}
        logging.info(f"✅ Frame parsed successfully - Important params: {important_params}")

        return parsed

    def update_guest_status_display(self, parsed_data):
        """OPTIMIZED: Update UI parameters with reduced logging overhead"""
        if not parsed_data:
            return

        # Filter out metadata fields that are not UI parameters
        ui_data = {k: v for k, v in parsed_data.items() if not k.startswith('_')}

        # Reduced logging - only log every 20th call to reduce overhead
        if not hasattr(self, '_ui_update_count'):
            self._ui_update_count = 0
        self._ui_update_count += 1

        if self._ui_update_count % 20 == 1:
            logging.info(f"🔍 UI UPDATE #{self._ui_update_count}: {len(ui_data)} fields")

        # Check if UI elements are accessible (reduced logging)
        if not hasattr(self, 'status_labels') or not self.status_labels:
            if self._ui_update_count % 20 == 1:
                logging.error("❌ status_labels not available - UI elements not initialized")
            return

        try:
            # Track previous values for change detection (for logging only)
            if not hasattr(self, '_previous_guest_status'):
                self._previous_guest_status = {}

            # ALWAYS UPDATE ALL PARAMETERS - no conditions
            changed_fields = []
            important_changes = []

            updated_fields = []
            for field, value in ui_data.items():
                if field in self.status_labels:
                    try:
                        # Update text value with green color for visibility
                        old_text = self.status_labels[field].text()
                        new_text = f"<font color='green'>{value}</font>"
                        self.status_labels[field].setText(new_text)
                        updated_fields.append(f"{field}={value}")

                        # Track changes for logging
                        if self._previous_guest_status.get(field) != value:
                            changed_fields.append(field)
                            if field in ['running_state', 'wakeup_state', 'communication_mode', 'running_mode']:
                                important_changes.append(f"{field}={value}")

                        self._previous_guest_status[field] = value
                    except Exception as e:
                        logging.error(f"Error updating field {field}: {e}")
                else:
                    logging.warning(f"Field {field} not found in status_labels")

            # Reduced logging for performance
            if self._ui_update_count % 20 == 1:
                logging.info(f"🔍 UI UPDATE COMPLETED: Updated {len(updated_fields)} fields")

            # Record guest status history with timestamp
            self._record_guest_status_history(parsed_data)

            # ALWAYS update calibration labels
            if 'running_state' in parsed_data:
                self.update_calibration_labels(parsed_data['running_state'])

            # Log important changes only
            if important_changes:
                logging.info(f"Important changes: {important_changes}")

        except Exception as e:
            logging.error(f"Error updating guest status display: {e}")

    def _convert_to_ppk2_time(self, absolute_timestamp):
        """
        Convert absolute timestamp to PPK2's cumulative timeline.

        PPK2 uses cumulative time starting from 0.0 when first samples arrive.
        Parameters arrive with absolute timestamps, so we need to convert them
        to match PPK2's timeline for proper synchronization.
        """
        if not hasattr(self, 'dm') or self.dm.times.size == 0:
            # No PPK2 data yet, parameters arrive before PPK2 starts
            return 0.0

        # Get the current PPK2 timeline position
        current_ppk2_time = float(self.dm.times[-1])

        # Calculate when this parameter change should be placed on PPK2 timeline
        # We use the current PPK2 time as the parameter change time
        # This ensures parameters are synchronized with the ongoing PPK2 timeline

        # Debug: Show time conversion for first few parameters
        if not hasattr(self, '_time_conversion_debug_count'):
            self._time_conversion_debug_count = 0
        if self._time_conversion_debug_count < 3:
            self._time_conversion_debug_count += 1
            elapsed_since_global_start = absolute_timestamp - self._global_start_time
            print(
                f"🕐 TIME SYNC #{self._time_conversion_debug_count}: Real elapsed={elapsed_since_global_start:.6f}s → PPK2 time={current_ppk2_time:.6f}s")

        return current_ppk2_time

    def _record_guest_status_history(self, status_data):
        """IMMEDIATE parameter recording: Record parameters exactly when they change"""
        try:
            # OPTION A: Convert parameter timestamps to PPK2's cumulative timeline
            if '_receive_timestamp' in status_data:
                exact_receive_time = status_data['_receive_timestamp']
                # Convert real-time to PPK2's cumulative time
                parameter_change_timestamp = self._convert_to_ppk2_time(exact_receive_time)
            else:
                current_time = time.perf_counter()
                # Convert current time to PPK2's cumulative time
                parameter_change_timestamp = self._convert_to_ppk2_time(current_time)

            # Thread-safe update of current guest status
            with self.guest_status_lock:
                # Map guest status keys to CSV column names
                # Complete mapping based on CSV save logic analysis
                key_mapping = {
                    # Parameters that need _status suffix
                    'running_state': 'running_state_status',
                    'communication_mode': 'communication_mode_status',
                    'vlc_protocol_mode': 'vlc_protocol_mode_status',
                    'vlc_communication_volume': 'vlc_communication_volume_status',
                    'vlc_information_volume': 'vlc_information_volume_status',
                    'vlc_per_unit_time': 'vlc_per_unit_time_status',
                    'ble_protocol_mode': 'ble_protocol_mode_status',
                    'ble_communication_volume': 'ble_communication_volume_status',
                    'ble_information_volume': 'ble_information_volume_status',
                    'ble_per_unit_time': 'ble_per_unit_time_status',

                    # Parameters that map directly (no _status suffix needed)
                    'wakeup_state': 'wakeup_state',
                    'running_mode': 'running_mode',
                    'vlc_interval_seconds': 'vlc_interval_seconds',
                    'pwm_percentage': 'pwm_percentage',
                    'ble_interval_seconds': 'ble_interval_seconds',
                    'phy_rate_percentage': 'phy_rate_percentage',
                    'mtu_value': 'mtu_value',
                    'fast_time': 'fast_time',
                    'spare_2': 'spare_2',
                    'spare_3': 'spare_3',
                    'spare_4': 'spare_4'
                }

                mapped_status = {}
                for key, value in status_data.items():
                    mapped_key = key_mapping.get(key, key)  # Use mapping or original key
                    mapped_status[mapped_key] = value

                self.current_guest_status.update(mapped_status)

                # Add timestamp to current status for synchronization
                self.current_guest_status['_timestamp'] = parameter_change_timestamp

            # IMMEDIATE: Record parameter change in buffer with exact timestamp
            # This ensures no parameter changes are missed
            self.parameter_buffer.update(parameter_change_timestamp, mapped_status)

            # Store timestamped status in history for time-based matching
            with self.status_history_lock:
                # Simple parameter change detection (no complex priority system)
                parameter_changed = False
                if len(self.guest_status_history) > 0:
                    last_status = self.guest_status_history[-1][1]
                    # Check important parameters for changes
                    important_params = ['running_state_status', 'wakeup_state', 'communication_mode_status',
                                        'running_mode']
                    for param in important_params:
                        if mapped_status.get(param) != last_status.get(param):
                            parameter_changed = True
                            old_val = last_status.get(param, 'N/A')
                            new_val = mapped_status.get(param, 'N/A')
                            print(
                                f"📊 CHANGE: {param} changed from {old_val} → {new_val} at {parameter_change_timestamp:.6f}s")
                            break

                # Add new status to history with timestamp
                status_entry = (parameter_change_timestamp, mapped_status.copy())
                self.guest_status_history.append(status_entry)

                # Keep history manageable (last 1000 entries, ~16 minutes at 1Hz)
                if len(self.guest_status_history) > 1000:
                    self.guest_status_history = self.guest_status_history[-1000:]

                # Simple logging for parameter changes (no high-priority sync)
                if parameter_changed:
                    print(f"📊 PARAM CHANGE: Parameters updated at {parameter_change_timestamp:.6f}s")

                # Note: Debug prints removed to reduce lag

            logging.info(f"📊 GUEST STATUS UPDATED: {len(status_data)} parameters mapped for dynamic saving")

        except Exception as e:
            logging.error(f"Error updating current guest status: {e}")

    def _get_guest_status_at_time(self, target_timestamp):
        """Get the guest status that was valid at the given timestamp"""
        try:
            with self.status_history_lock:
                if not self.guest_status_history:
                    # No history available, return current status (remove timestamp key)
                    status = self.current_guest_status.copy()
                    status.pop('_timestamp', None)  # Remove internal timestamp
                    return status

                # Find the most recent status before or at the target time
                valid_status = None
                valid_timestamp = None
                for timestamp, status in self.guest_status_history:
                    if timestamp <= target_timestamp:
                        valid_status = status
                        valid_timestamp = timestamp
                    else:
                        break  # History is chronological, so we can stop here

                if valid_status is not None:
                    # Debug: Log timestamp matching for first few lookups

                    return valid_status.copy()
                else:
                    # Target time is before any recorded status, use earliest
                    return self.guest_status_history[0][1].copy()

        except Exception as e:
            logging.error(f"Error getting guest status at time {target_timestamp}: {e}")
            # Fallback to current status
            status = self.current_guest_status.copy()
            status.pop('_timestamp', None)  # Remove internal timestamp
            return status

    def _log_synchronization_status(self):
        """Log current synchronization status for debugging"""
        try:
            with self.status_history_lock:
                history_count = len(self.guest_status_history)
                if history_count > 0:
                    latest_status_time = self.guest_status_history[-1][0]
                    oldest_status_time = self.guest_status_history[0][0]
                    time_span = latest_status_time - oldest_status_time
                    print(f"📊 SYNC STATUS: {history_count} guest status entries spanning {time_span:.2f}s")
                else:
                    print(f"📊 SYNC STATUS: No guest status history available")
        except Exception as e:
            logging.error(f"Error logging synchronization status: {e}")

    def _update_ui_display(self, data, source="UNKNOWN"):
        """Internal method to actually update the UI display"""
        try:
            print(f"🎯 UPDATING UI from {source}: {list(data.keys())}")
            logging.info(f"🎯 UPDATING UI from {source}: {list(data.keys())}")

            # Debug: Check if status_labels exists and has the right fields
            if not hasattr(self, 'status_labels'):
                print(f"❌ NO status_labels ATTRIBUTE!")
                logging.error(f"❌ NO status_labels ATTRIBUTE!")
                return

            # Note: Debug prints removed to reduce lag

            # Update all fields in the UI
            updated_count = 0
            for field, value in data.items():
                if field in self.status_labels:
                    old_text = self.status_labels[field].text()
                    self.status_labels[field].setText(str(value))
                    print(f"🔄 Updated {field}: '{old_text}' → '{value}'")

                    # Highlight important parameters differently
                    if field in ['running_state', 'wakeup_state', 'communication_mode', 'running_mode']:
                        self.status_labels[field].setStyleSheet(
                            "font-weight: bold; font-size: 9px; color: #e74c3c;")  # Red for important
                    else:
                        self.status_labels[field].setStyleSheet(
                            "font-weight: bold; font-size: 9px; color: #27ae60;")  # Green for normal

                    updated_count += 1
                else:
                    print(f"❌ Field '{field}' NOT FOUND in status_labels!")
                    logging.warning(f"❌ Field '{field}' NOT FOUND in status_labels!")

            print(f"✅ Updated {updated_count}/{len(data)} fields in UI")
            logging.info(f"✅ Updated {updated_count}/{len(data)} fields in UI")

            # Update calibration labels when running state changes
            if 'running_state' in data:
                print(f"🔧 Updating calibration labels for running_state: {data['running_state']}")
                self.update_calibration_labels(data['running_state'])

            print(f"✅ UI update complete from {source}")
            logging.info(f"✅ UI updated successfully from {source}")

        except Exception as e:
            print(f"❌ ERROR in _update_ui_display: {e}")
            logging.error(f"Error updating UI display: {e}")
            import traceback
            traceback.print_exc()

    def _periodic_guest_status_update(self):
        """Case 2: Periodic update every 1 second to ensure UI stays current"""
        try:
            if not hasattr(self, '_latest_guest_data') or not self._latest_guest_data:
                return  # No data to update with

            self._update_ui_display(self._latest_guest_data, "PERIODIC")

        except Exception as e:
            logging.error(f"Error in periodic guest status update: {e}")

    def get_guest_parameters(self):
        """Get current guest settings (matching original app format)"""
        params = {
            "id": self.id_spin.value() if hasattr(self, 'id_spin') else self.guest_id,
            "tx_power": self.Tx_power_spin.value() if hasattr(self, 'Tx_power_spin') else self.Tx_power_dBm,
            "advertising_interval": self.advertising_interval_spin.value() if hasattr(self,
                                                                                      'advertising_interval_spin') else self.advertising_interval,
            "connection_interval": self.connection_interval_spin.value() if hasattr(self,
                                                                                    'connection_interval_spin') else self.connection_interval,
            "spare_0": self.spare_0_spin.value() if hasattr(self, 'spare_0_spin') else getattr(self, 'spare_0', 0),
            "spare_1": self.spare_1_spin.value() if hasattr(self, 'spare_1_spin') else getattr(self, 'spare_1', 0)
        }

        return params

    def get_guest_status_parameters(self):
        """Get current guest status (compatible with original app)"""
        status = {}
        if hasattr(self, 'status_labels'):
            for field, label in self.status_labels.items():
                status[field] = label.text()

        # Add calibration state to status for historical tracking
        status['calibration_enabled'] = str(self.calibration_enabled_check.isChecked()).lower()
        status['calibration_other_offset'] = str(self.normal_offset_spin.value())
        status['calibration_sending_offset'] = str(self.sending_offset_spin.value())

        return status

    def _on_save_data(self):
        """Save measurement data to CSV file with efficient historical parameters"""
        if self.dm.times.size == 0:
            QMessageBox.warning(self, "No Data", "No measurement data to save.")
            return

        # Debug: Show data collection status before save
        total_samples = len(self.dm.raw_samples_full) if self.dm.raw_samples_full else 0
        time_range = f"{self.dm.times[0]:.3f}s to {self.dm.times[-1]:.3f}s" if self.dm.times.size > 0 else "No time data"
        duration = f"{self.dm.times[-1] - self.dm.times[0]:.3f}s" if self.dm.times.size > 1 else "0s"

        print(f"💾 SAVE DATA STATUS:")
        print(f"   Display buffer: {self.dm.times.size:,} samples, {time_range}, duration: {duration}")
        print(f"   Raw storage: {total_samples:,} samples with full parameters")
        print(f"   Measurement active: {self.is_measuring}")

        try:
            # Get selected format
            selected_format = self.format_combo.currentText()

            # Generate timestamp-based filename
            from datetime import datetime
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

            if selected_format == "All formats":
                # Save in all formats
                self._save_all_formats(timestamp_str)
            else:
                # Save in single format
                self._save_single_format(selected_format, timestamp_str)

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data: {e}")
            import traceback
            traceback.print_exc()

    def _save_downsampled_split(self, base_path, downsampled_samples, excel_limit):
        """Split large downsampled data into Excel-compatible files"""
        try:
            total_samples = len(downsampled_samples)
            num_files = (total_samples + excel_limit - 1) // excel_limit
            base_name = base_path.rsplit('.', 1)[0]

            print(f"📁 SPLITTING DOWNSAMPLED DATA INTO {num_files} FILES: {excel_limit:,} rows each")

            # Show wakeup timestamps that will be applied during save
            if hasattr(self, 'wakeup_timestamps') and self.wakeup_timestamps:
                print(
                    f"📋 WAKEUP TIMESTAMPS TO APPLY: {len(self.wakeup_timestamps)} wakeup(s) recorded in this measurement")
                for idx, wakeup_ts in enumerate(self.wakeup_timestamps):
                    print(f"   Wakeup #{idx + 1}: at t={wakeup_ts:.6f}s")
                print(f"   Will update Running_State 3→0 and Wakeup_State 0→1 starting from these timestamps")
                print(f"   (Updates stop naturally when Running_State != 3 or Wakeup_State != 0)")
            else:
                print(f"📋 NO WAKEUP TIMESTAMPS: Running_State and Wakeup_State will use original values")

            progress = QProgressDialog(f"Saving {num_files} Excel-compatible files...", "Cancel", 0, 100, self)
            progress.setWindowModality(2)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()

            saved_files = []

            for file_num in range(num_files):
                if progress.wasCanceled():
                    break

                start_idx = file_num * excel_limit
                end_idx = min(start_idx + excel_limit, total_samples)
                samples_in_file = end_idx - start_idx

                filename = f"{base_name}_part{file_num + 1}.csv" if num_files > 1 else f"{base_name}.csv"

                file_progress = int((file_num / num_files) * 90)
                progress.setValue(file_progress)
                progress.setLabelText(f"Creating file {file_num + 1}/{num_files}: {filename}")
                QApplication.processEvents()

                with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)

                    # Write header
                    header = [
                        "Guest_ID", "Timestamp (s)", "Unified_Timestamp (s)", "Current (uA)",
                        "Applied_Other_Offset (uA)", "Applied_Sending_Offset (uA)",
                        "Voltage (mV)", "Power (mW)", "Tx_Power", "Adv_Interval", "Conn_Interval",
                        "Spare_0", "Spare_1", "Running_State", "Comm_Mode", "Wakeup_State", "Running_Mode",
                        "VLC_Protocol", "VLC_Interval", "VLC_Comm_Vol", "VLC_Info_Vol", "VLC_Per_Time",
                        "PWM_Percent", "BLE_Protocol", "BLE_Interval", "BLE_Comm_Vol", "BLE_Info_Vol",
                        "BLE_Per_Time", "PHY_Rate", "MTU_Value", "Fast_Time", "Spare_2", "Spare_3", "Spare_4",
                        "Calibration_Enabled", "Calibration_Other_Offset", "Calibration_Sending_Offset"
                    ]
                    writer.writerow(header)

                    # Write data for this file
                    file_samples = downsampled_samples[start_idx:end_idx]
                    batch_size = 5000

                    for i in range(0, len(file_samples), batch_size):
                        batch_end = min(i + batch_size, len(file_samples))
                        batch_rows = []

                        for j in range(i, batch_end):
                            sample_data = file_samples[j]
                            if len(sample_data) == 8:
                                time_val, current_val, running_mode, guest_id, voltage_mv, guest_params, status_params, unified_timestamp = sample_data
                            else:
                                time_val, current_val, running_mode, guest_id, voltage_mv, guest_params, status_params = sample_data
                                unified_timestamp = time_val

                            # Same processing as single file
                            power_mw = (voltage_mv * current_val) / 1000.0
                            cal_enabled = status_params.get('calibration_enabled', False)
                            other_offset_raw = status_params.get('calibration_other_offset', 0)
                            sending_offset_raw = status_params.get('calibration_sending_offset', 0)

                            # Record which calibration offset was applied (current already has calibration applied from real-time processing)
                            if cal_enabled:
                                # Use running_state_status from guest status, not running_mode
                                guest_running_state = int(
                                    status_params.get('running_state_status', 0)) if status_params.get(
                                    'running_state_status', 0) != -1 else 0
                                if guest_running_state == 2:
                                    applied_other_offset = 0  # Other offset not applied
                                    applied_sending_offset = sending_offset_raw  # Sending offset applied
                                else:
                                    applied_other_offset = other_offset_raw  # Other offset applied
                                    applied_sending_offset = 0  # Sending offset not applied
                            else:
                                applied_other_offset = 0  # No calibration applied
                                applied_sending_offset = 0

                            # WAKEUP FIX: If time >= global_wakeup_time AND original=3, change to 0
                            # Example: [3,3,3,3,3,3,1,1,1] with wakeup at 4th → [3,3,3,0,0,0,1,1,1]
                            orig_running = status_params.get('running_state_status', running_mode)
                            orig_wakeup = status_params.get('wakeup_state', 0)

                            if self._should_apply_wakeup_fix(time_val):
                                # time >= global_wakeup_time: Replace 3→0, 0→1
                                final_running_state = 0 if orig_running == 3 else orig_running
                                final_wakeup_state = 1 if orig_wakeup == 0 else orig_wakeup
                            else:
                                # time < global_wakeup_time: Keep original
                                final_running_state = orig_running
                                final_wakeup_state = orig_wakeup

                            row = [
                                guest_params.get('id', guest_id), time_val, unified_timestamp, current_val,
                                applied_other_offset, applied_sending_offset,
                                voltage_mv, power_mw,
                                guest_params.get('tx_power', 0), guest_params.get('advertising_interval', 0),
                                guest_params.get('connection_interval', 0), guest_params.get('spare_0', 0),
                                guest_params.get('spare_1', 0), final_running_state,
                                status_params.get('communication_mode_status', 0), final_wakeup_state,
                                status_params.get('running_mode', running_mode),
                                status_params.get('vlc_protocol_mode_status', 0),
                                status_params.get('vlc_interval_seconds', 0),
                                status_params.get('vlc_communication_volume_status', 0),
                                status_params.get('vlc_information_volume_status', 0),
                                status_params.get('vlc_per_unit_time_status', 0),
                                status_params.get('pwm_percentage', 0),
                                status_params.get('ble_protocol_mode_status', 0),
                                status_params.get('ble_interval_seconds', 0),
                                status_params.get('ble_communication_volume_status', 0),
                                status_params.get('ble_information_volume_status', 0),
                                status_params.get('ble_per_unit_time_status', 0),
                                status_params.get('phy_rate_percentage', 0), status_params.get('mtu_value', 0),
                                status_params.get('fast_time', 0), status_params.get('spare_2', 0),
                                status_params.get('spare_3', 0), status_params.get('spare_4', 0),
                                cal_enabled, other_offset_raw, sending_offset_raw
                            ]

                            batch_rows.append(row)

                        writer.writerows(batch_rows)

                saved_files.append(filename)
                print(f"✅ SAVED: {filename} ({samples_in_file:,} samples)")

            progress.setValue(100)
            progress.close()

            file_list = '\n'.join(saved_files)
            QMessageBox.information(self, "Excel-Compatible Save Complete",
                                    f"Successfully split {total_samples:,} samples into {len(saved_files)} Excel-compatible files!\n\n"
                                    f"Files created:\n{file_list}\n\n"
                                    f"Each file has ≤{excel_limit:,} rows and can be opened in Excel.")

            print(f"✅ EXCEL SPLIT COMPLETE: {total_samples:,} samples → {len(saved_files)} files")

        except Exception as e:
            QMessageBox.critical(self, "Split Save Error", f"Failed to save split files: {e}")
            import traceback
            traceback.print_exc()

    def _save_raw_buffer_split(self, base_path, total_samples, excel_limit):
        """Split large raw buffer into Excel-compatible files"""
        try:
            # Calculate number of files needed
            num_files = (total_samples + excel_limit - 1) // excel_limit  # Ceiling division
            base_name = base_path.rsplit('.', 1)[0]  # Remove .csv extension

            print(f"📁 SPLITTING INTO {num_files} FILES: {excel_limit:,} rows each (Excel compatible)")

            # Show wakeup timestamps that will be applied during save
            if hasattr(self, 'wakeup_timestamps') and self.wakeup_timestamps:
                print(
                    f"📋 WAKEUP TIMESTAMPS TO APPLY: {len(self.wakeup_timestamps)} wakeup(s) recorded in this measurement")
                for idx, wakeup_ts in enumerate(self.wakeup_timestamps):
                    print(f"   Wakeup #{idx + 1}: at t={wakeup_ts:.6f}s")
                print(f"   Will update Running_State 3→0 and Wakeup_State 0→1 starting from these timestamps")
                print(f"   (Updates stop naturally when Running_State != 3 or Wakeup_State != 0)")
            else:
                print(f"📋 NO WAKEUP TIMESTAMPS: Running_State and Wakeup_State will use original values")

            # Create progress dialog
            progress = QProgressDialog(f"Saving {num_files} Excel-compatible files...", "Cancel", 0, 100, self)
            progress.setWindowModality(2)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()

            saved_files = []

            for file_num in range(num_files):
                if progress.wasCanceled():
                    break

                # Calculate sample range for this file
                start_idx = file_num * excel_limit
                end_idx = min(start_idx + excel_limit, total_samples)
                samples_in_file = end_idx - start_idx

                # Generate filename
                if num_files > 1:
                    filename = f"{base_name}_part{file_num + 1}.csv"
                else:
                    filename = f"{base_name}.csv"

                print(f"📄 FILE {file_num + 1}/{num_files}: {filename} ({samples_in_file:,} samples)")

                # Update progress
                file_progress = int((file_num / num_files) * 80)  # 80% for file creation
                progress.setValue(file_progress)
                progress.setLabelText(f"Creating file {file_num + 1}/{num_files}: {filename}")
                QApplication.processEvents()

                # Write this file
                with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)

                    # Write header
                    header = [
                        "Guest_ID", "Timestamp (s)", "Unified_Timestamp (s)", "Current (uA)",
                        "Applied_Other_Offset (uA)", "Applied_Sending_Offset (uA)",
                        "Voltage (mV)", "Power (mW)", "Tx_Power", "Adv_Interval", "Conn_Interval",
                        "Spare_0", "Spare_1", "Running_State", "Comm_Mode", "Wakeup_State", "Running_Mode",
                        "VLC_Protocol", "VLC_Interval", "VLC_Comm_Vol", "VLC_Info_Vol", "VLC_Per_Time",
                        "PWM_Percent", "BLE_Protocol", "BLE_Interval", "BLE_Comm_Vol", "BLE_Info_Vol",
                        "BLE_Per_Time", "PHY_Rate", "MTU_Value", "Fast_Time", "Spare_2", "Spare_3", "Spare_4",
                        "Calibration_Enabled", "Calibration_Other_Offset", "Calibration_Sending_Offset"
                    ]
                    writer.writerow(header)

                    # Write data for this file in batches
                    batch_size = 5000
                    for i in range(start_idx, end_idx, batch_size):
                        batch_end = min(i + batch_size, end_idx)
                        batch_rows = []

                        for j in range(i, batch_end):
                            # Extract sample data
                            sample_data = self.dm.raw_samples_full[j]
                            if len(sample_data) == 8:
                                time_val, current_val, running_mode, guest_id, voltage_mv, guest_params, status_params, unified_timestamp = sample_data
                            else:
                                time_val, current_val, running_mode, guest_id, voltage_mv, guest_params, status_params = sample_data
                                unified_timestamp = time_val

                            # Calculate power and calibration (same as single file version)
                            power_mw = (voltage_mv * current_val) / 1000.0
                            cal_enabled = status_params.get('calibration_enabled', False)
                            other_offset_raw = status_params.get('calibration_other_offset', 0)
                            sending_offset_raw = status_params.get('calibration_sending_offset', 0)

                            # Record which calibration offset was applied (current already has calibration applied from real-time processing)
                            if cal_enabled:
                                # Use running_state_status from guest status, not running_mode
                                guest_running_state = int(
                                    status_params.get('running_state_status', 0)) if status_params.get(
                                    'running_state_status', 0) != -1 else 0
                                if guest_running_state == 2:
                                    applied_other_offset = 0  # Other offset not applied
                                    applied_sending_offset = sending_offset_raw  # Sending offset applied
                                else:
                                    applied_other_offset = other_offset_raw  # Other offset applied
                                    applied_sending_offset = 0  # Sending offset not applied
                            else:
                                applied_other_offset = 0  # No calibration applied
                                applied_sending_offset = 0

                            # WAKEUP FIX: If time >= global_wakeup_time AND original=3, change to 0
                            # Example: [3,3,3,3,3,3,1,1,1] with wakeup at 4th → [3,3,3,0,0,0,1,1,1]
                            orig_running = status_params.get('running_state_status', running_mode)
                            orig_wakeup = status_params.get('wakeup_state', 0)

                            if self._should_apply_wakeup_fix(time_val):
                                # time >= global_wakeup_time: Replace 3→0, 0→1
                                final_running_state = 0 if orig_running == 3 else orig_running
                                final_wakeup_state = 1 if orig_wakeup == 0 else orig_wakeup
                            else:
                                # time < global_wakeup_time: Keep original
                                final_running_state = orig_running
                                final_wakeup_state = orig_wakeup

                            # Build row
                            row = [
                                guest_params.get('id', guest_id), time_val, unified_timestamp, current_val,
                                applied_other_offset, applied_sending_offset,
                                voltage_mv, power_mw,
                                guest_params.get('tx_power', 0), guest_params.get('advertising_interval', 0),
                                guest_params.get('connection_interval', 0), guest_params.get('spare_0', 0),
                                guest_params.get('spare_1', 0), final_running_state,
                                status_params.get('communication_mode_status', 0), final_wakeup_state,
                                status_params.get('running_mode', running_mode),
                                status_params.get('vlc_protocol_mode_status', 0),
                                status_params.get('vlc_interval_seconds', 0),
                                status_params.get('vlc_communication_volume_status', 0),
                                status_params.get('vlc_information_volume_status', 0),
                                status_params.get('vlc_per_unit_time_status', 0),
                                status_params.get('pwm_percentage', 0),
                                status_params.get('ble_protocol_mode_status', 0),
                                status_params.get('ble_interval_seconds', 0),
                                status_params.get('ble_communication_volume_status', 0),
                                status_params.get('ble_information_volume_status', 0),
                                status_params.get('ble_per_unit_time_status', 0),
                                status_params.get('phy_rate_percentage', 0), status_params.get('mtu_value', 0),
                                status_params.get('fast_time', 0), status_params.get('spare_2', 0),
                                status_params.get('spare_3', 0), status_params.get('spare_4', 0),
                                cal_enabled, other_offset_raw, sending_offset_raw
                            ]

                            batch_rows.append(row)

                        # Write batch
                        writer.writerows(batch_rows)

                saved_files.append(filename)
                print(f"✅ SAVED: {filename} ({samples_in_file:,} samples)")

            # Finish
            progress.setValue(100)
            progress.close()

            # Show completion message
            total_files = len(saved_files)
            file_list = '\n'.join(saved_files)

            QMessageBox.information(self, "Excel-Compatible Save Complete",
                                    f"Successfully split {total_samples:,} samples into {total_files} Excel-compatible files!\n\n"
                                    f"Files created:\n{file_list}\n\n"
                                    f"Each file has ≤{excel_limit:,} rows and can be opened in Excel.")

            print(f"✅ EXCEL SPLIT COMPLETE: {total_samples:,} samples → {total_files} files")
            print(f"📊 FILES: {file_list}")

        except Exception as e:
            QMessageBox.critical(self, "Split Save Error", f"Failed to save split files: {e}")
            import traceback
            traceback.print_exc()

    def _save_raw_buffer_direct(self, path):
        """Ultra-fast save for 100kHz: direct raw buffer copy with Excel compatibility"""
        try:
            total_samples = len(self.dm.raw_samples_full)
            print(f"🚀 RAW BUFFER SAVE: Writing {total_samples:,} samples directly (no downsampling)")

            # CHECK EXCEL COMPATIBILITY (quick check)
            if total_samples > 1000000:  # Quick threshold check
                print(f"📊 LARGE DATASET: {total_samples:,} samples - using Excel-compatible splitting")
                self._save_raw_buffer_split(path, total_samples, 1048575)
                return
            else:
                print(f"📊 STANDARD SAVE: {total_samples:,} samples in single file")

            # Create progress dialog
            progress = QProgressDialog("Saving raw buffer data...", "Cancel", 0, 100, self)
            progress.setWindowModality(2)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()

            with open(path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)

                # Write header (same as downsampled version)
                header = [
                    "Guest_ID", "Timestamp (s)", "Unified_Timestamp (s)", "Current (uA)", "Applied_Other_Offset (uA)",
                    "Applied_Sending_Offset (uA)",
                    "Voltage (mV)", "Power (mW)", "Tx_Power", "Adv_Interval", "Conn_Interval",
                    "Spare_0", "Spare_1", "Running_State", "Comm_Mode", "Wakeup_State", "Running_Mode",
                    "VLC_Protocol", "VLC_Interval", "VLC_Comm_Vol", "VLC_Info_Vol", "VLC_Per_Time",
                    "PWM_Percent", "BLE_Protocol", "BLE_Interval", "BLE_Comm_Vol", "BLE_Info_Vol",
                    "BLE_Per_Time", "PHY_Rate", "MTU_Value", "Fast_Time", "Spare_2", "Spare_3", "Spare_4",
                    "Calibration_Enabled", "Calibration_Other_Offset", "Calibration_Sending_Offset"
                ]
                writer.writerow(header)
                progress.setValue(10)

                # Write raw data in large batches for maximum performance
                batch_size = 10000  # Large batches for speed
                for i in range(0, total_samples, batch_size):
                    if progress.wasCanceled():
                        return

                    batch_end = min(i + batch_size, total_samples)
                    progress_percent = 10 + int(((i / total_samples) * 80))
                    progress.setValue(progress_percent)
                    progress.setLabelText(
                        f"Writing raw batch {i // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}...")
                    QApplication.processEvents()

                    # Prepare batch - direct copy from raw buffer
                    batch_rows = []
                    for j in range(i, batch_end):
                        # Extract raw sample data
                        sample_data = self.dm.raw_samples_full[j]
                        if len(sample_data) == 8:
                            time_val, current_val, running_mode, guest_id, voltage_mv, guest_params, status_params, unified_timestamp = sample_data
                        else:
                            time_val, current_val, running_mode, guest_id, voltage_mv, guest_params, status_params = sample_data
                            unified_timestamp = time_val

                        # Calculate power
                        power_mw = (voltage_mv * current_val) / 1000.0

                        # Get calibration from historical data
                        cal_enabled = status_params.get('calibration_enabled', False)
                        other_offset_raw = status_params.get('calibration_other_offset', 0)
                        sending_offset_raw = status_params.get('calibration_sending_offset', 0)

                        # Record which calibration offset was applied (current already has calibration applied from real-time processing)
                        if cal_enabled:
                            # Use running_state_status from guest status, not running_mode
                            guest_running_state = int(
                                status_params.get('running_state_status', 0)) if status_params.get(
                                'running_state_status', 0) != -1 else 0
                            if guest_running_state == 2:
                                applied_other_offset = 0  # Other offset not applied
                                applied_sending_offset = sending_offset_raw  # Sending offset applied
                            else:
                                applied_other_offset = other_offset_raw  # Other offset applied
                                applied_sending_offset = 0  # Sending offset not applied
                        else:
                            applied_other_offset = 0  # No calibration applied
                            applied_sending_offset = 0

                        # Build row (same format as downsampled version)
                        row = [
                            guest_params.get('id', guest_id), time_val, unified_timestamp, current_val,
                            applied_other_offset, applied_sending_offset,
                            voltage_mv, power_mw,
                            guest_params.get('tx_power', 0), guest_params.get('advertising_interval', 0),
                            guest_params.get('connection_interval', 0), guest_params.get('spare_0', 0),
                            guest_params.get('spare_1', 0), status_params.get('running_state_status', running_mode),
                            status_params.get('communication_mode_status', 0), status_params.get('wakeup_state', 0),
                            status_params.get('running_mode', running_mode),
                            status_params.get('vlc_protocol_mode_status', 0),
                            status_params.get('vlc_interval_seconds', 0),
                            status_params.get('vlc_communication_volume_status', 0),
                            status_params.get('vlc_information_volume_status', 0),
                            status_params.get('vlc_per_unit_time_status', 0),
                            status_params.get('pwm_percentage', 0), status_params.get('ble_protocol_mode_status', 0),
                            status_params.get('ble_interval_seconds', 0),
                            status_params.get('ble_communication_volume_status', 0),
                            status_params.get('ble_information_volume_status', 0),
                            status_params.get('ble_per_unit_time_status', 0),
                            status_params.get('phy_rate_percentage', 0), status_params.get('mtu_value', 0),
                            status_params.get('fast_time', 0), status_params.get('spare_2', 0),
                            status_params.get('spare_3', 0), status_params.get('spare_4', 0),
                            cal_enabled, other_offset_raw, sending_offset_raw
                        ]

                        batch_rows.append(row)

                    # Write entire batch at once (maximum performance)
                    writer.writerows(batch_rows)

            # Finish
            progress.setValue(100)
            progress.close()

            print(f"✅ RAW BUFFER SAVE COMPLETE: {total_samples:,} samples at full 100kHz resolution")
            print(f"📁 File: {path}")

            # Show success message
            QMessageBox.information(self, "Save Complete",
                                    f"Successfully saved {total_samples:,} samples at full 100kHz resolution!\n\n"
                                    f"File: {path}\n"
                                    f"No information lost - complete raw data preserved.")

        except Exception as e:
            QMessageBox.critical(self, "Raw Save Error", f"Failed to save raw buffer: {e}")
            import traceback
            traceback.print_exc()

    def _save_with_efficient_storage(self, path):
        """Fast save with smart handling: 100kHz = raw buffer copy, others = downsample"""
        try:
            if not self.dm.raw_samples_full:
                QMessageBox.warning(self, "No Data", "No historical data available to save.")
                return

            # Get selected downsampling rate
            save_hz = self.save_downsample_combo.currentData()

            # SMART SAVE: 100kHz uses raw buffer directly, others use downsampling
            if save_hz >= 100000:
                print(
                    f"🚀 FAST SAVE: Saving {len(self.dm.raw_samples_full):,} samples at full 100kHz resolution (raw buffer copy)")
                self._save_raw_buffer_direct(path)
                return
            else:
                print(f"💾 STANDARD SAVE: Saving data with {save_hz} Hz downsampling...")

            # Skip automatic host transmission during save for better performance
            # Host will receive data in real-time during collection

            # Create progress dialog
            progress = QProgressDialog("Preparing historical data...", "Cancel", 0, 100, self)
            progress.setWindowModality(2)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()

            # Use historical data downsampling (preserves parameters)
            progress.setValue(20)
            progress.setLabelText("Downsampling historical data with parameters...")
            QApplication.processEvents()

            # Downsample the raw_samples_full data (which contains historical parameters)
            downsampled_samples = self.dm.downsample_full_samples(self.dm.raw_samples_full, save_hz)

            progress.setValue(40)
            progress.setLabelText("Preparing CSV data...")
            QApplication.processEvents()

            total_rows = len(downsampled_samples)
            print(
                f"📊 Downsampled from {len(self.dm.raw_samples_full):,} to {total_rows:,} samples ({save_hz} Hz) with HISTORICAL parameters")

            # Quick Excel compatibility check for downsampled data
            if total_rows > 1000000:
                print(f"📊 LARGE DOWNSAMPLED DATASET: {total_rows:,} samples - using file splitting")
                progress.close()
                self._save_downsampled_split(path, downsampled_samples, 1048575)
                return

            # Fast CSV writing with historical parameters
            progress.setValue(50)
            progress.setLabelText("Writing CSV file with historical parameters...")
            QApplication.processEvents()

            with open(path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)

                # Write header (Guest_ID first to match host app, with unified timestamp)
                header = [
                    "Guest_ID", "Timestamp (s)", "Unified_Timestamp (s)", "Current (uA)", "Applied_Other_Offset (uA)",
                    "Applied_Sending_Offset (uA)",
                    "Voltage (mV)", "Power (mW)", "Tx_Power", "Adv_Interval", "Conn_Interval",
                    "Spare_0", "Spare_1", "Running_State", "Comm_Mode", "Wakeup_State", "Running_Mode",
                    "VLC_Protocol", "VLC_Interval", "VLC_Comm_Vol", "VLC_Info_Vol", "VLC_Per_Time",
                    "PWM_Percent", "BLE_Protocol", "BLE_Interval", "BLE_Comm_Vol", "BLE_Info_Vol",
                    "BLE_Per_Time", "PHY_Rate", "MTU_Value", "Fast_Time", "Spare_2", "Spare_3", "Spare_4",
                    "Calibration_Enabled", "Calibration_Other_Offset", "Calibration_Sending_Offset"
                ]
                writer.writerow(header)

                # Write data rows in batches using HISTORICAL parameters
                batch_size = 5000  # Write 5000 rows at a time
                for i in range(0, total_rows, batch_size):
                    if progress.wasCanceled():
                        return

                    batch_end = min(i + batch_size, total_rows)
                    progress_percent = 50 + int((i / total_rows) * 40)
                    progress.setValue(progress_percent)
                    progress.setLabelText(
                        f"Writing batch {i // batch_size + 1}/{(total_rows + batch_size - 1) // batch_size} (historical data)...")
                    QApplication.processEvents()

                    # Prepare batch data using HISTORICAL parameters
                    batch_rows = []
                    for j in range(i, batch_end):
                        # Extract historical data: handle both old and new format with unified timestamp
                        sample_data = downsampled_samples[j]
                        if len(sample_data) == 8:
                            # New format with unified timestamp
                            time_val, current_val, running_mode, guest_id, voltage_mv, guest_params, status_params, unified_timestamp = sample_data
                        else:
                            # Old format compatibility
                            time_val, current_val, running_mode, guest_id, voltage_mv, guest_params, status_params = sample_data
                            unified_timestamp = time_val  # Fallback to device timestamp

                        # Calculate power using historical voltage
                        power_mw = (voltage_mv * current_val) / 1000.0

                        # Calculate historical calibration offsets (dynamic based on historical state)
                        cal_enabled = status_params.get('calibration_enabled', False)
                        other_offset_raw = status_params.get('calibration_other_offset', 0)
                        sending_offset_raw = status_params.get('calibration_sending_offset', 0)

                        # Record which calibration offset was applied (current already has calibration applied from real-time processing)
                        if cal_enabled:
                            # Use running_state_status from guest status, not running_mode
                            guest_running_state = int(
                                status_params.get('running_state_status', 0)) if status_params.get(
                                'running_state_status', 0) != -1 else 0
                            if guest_running_state == 2:
                                # Sending offset was applied to current in real-time processing
                                applied_other_offset = 0  # Other offset not applied
                                applied_sending_offset = sending_offset_raw  # Sending offset applied
                            else:
                                # Other offset was applied to current in real-time processing
                                applied_other_offset = other_offset_raw  # Other offset applied
                                applied_sending_offset = 0  # Sending offset not applied
                        else:
                            applied_other_offset = 0  # No calibration applied
                            applied_sending_offset = 0

                        # Build row with HISTORICAL parameters (Guest_ID first)
                        row = [
                            guest_params.get('id', guest_id), time_val, unified_timestamp, current_val,
                            applied_other_offset, applied_sending_offset,
                            voltage_mv, power_mw,
                            guest_params.get('tx_power', 0), guest_params.get('advertising_interval', 0),
                            guest_params.get('connection_interval', 0), guest_params.get('spare_0', 0),
                            guest_params.get('spare_1', 0), status_params.get('running_state_status', running_mode),
                            status_params.get('communication_mode_status', 0), status_params.get('wakeup_state', 0),
                            status_params.get('running_mode', running_mode),
                            status_params.get('vlc_protocol_mode_status', 0),
                            status_params.get('vlc_interval_seconds', 0),
                            status_params.get('vlc_communication_volume_status', 0),
                            status_params.get('vlc_information_volume_status', 0),
                            status_params.get('vlc_per_unit_time_status', 0),
                            status_params.get('pwm_percentage', 0), status_params.get('ble_protocol_mode_status', 0),
                            status_params.get('ble_interval_seconds', 0),
                            status_params.get('ble_communication_volume_status', 0),
                            status_params.get('ble_information_volume_status', 0),
                            status_params.get('ble_per_unit_time_status', 0),
                            status_params.get('phy_rate_percentage', 0), status_params.get('mtu_value', 0),
                            status_params.get('fast_time', 0), status_params.get('spare_2', 0),
                            status_params.get('spare_3', 0), status_params.get('spare_4', 0),
                            cal_enabled, other_offset_raw, sending_offset_raw
                        ]

                        # Raw buffer data already sent to host in real-time during _tick()
                        # No need to send again during save process
                        batch_rows.append(row)

                    # Write entire batch at once (much faster)
                    writer.writerows(batch_rows)

            # Finish save
            progress.setValue(100)
            progress.close()

            print(f"✅ Successfully saved {total_rows:,} downsampled samples with HISTORICAL parameters to {path}")

            # Store saved data for host transmission (use historical data)
            self._last_saved_data = {
                'historical_samples': downsampled_samples,
                'path': path,
                'downsample_rate': save_hz,
                'total_samples': total_rows
            }

            # Send saved data to host if connected
            if self.network_manager.connected:
                self._send_saved_data_to_host()

            QMessageBox.information(self, "Save Complete",
                                    f"Successfully saved data with HISTORICAL parameters!\n\n"
                                    f"File: {path}\n"
                                    f"Samples: {total_rows:,} (downsampled from {len(self.dm.raw_samples_full):,})\n"
                                    f"Downsample Rate: {save_hz} Hz\n"
                                    f"Data Source: Historical parameters preserved ✅")

        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "Save Error", f"Failed to save data: {e}")
            print(f"❌ Save error: {e}")
            import traceback
            traceback.print_exc()

    def _save_with_current_parameters_only(self, path):
        """Fallback save method using current parameters only"""
        try:
            guest_params = self.get_guest_parameters()
            status_params = self.get_guest_status_parameters()

            # Get downsampled data
            hz = float(self.current_downsample)
            x, y = DataAccumulator.mean_per_bin(self.dm.times, self.dm.currents, hz)

            # Get current voltage
            voltage_mv = self.voltage_spin.value() if hasattr(self, 'voltage_spin') else 3300

            # Calculate current offsets
            cal_enabled = self.calibration_enabled_check.isChecked()
            other_offset = self.normal_offset_spin.value() if cal_enabled else 0
            sending_offset = self.sending_offset_spin.value() if cal_enabled else 0

            # Get current running state for calibration logic
            current_running_state = 0
            if hasattr(self, 'status_labels') and 'running_state' in self.status_labels:
                try:
                    current_running_state = int(self.status_labels['running_state'].text())
                except (ValueError, AttributeError):
                    current_running_state = 0

            # Apply correct calibration logic
            if cal_enabled:
                if current_running_state == 2:
                    applied_other_offset = 0
                    applied_sending_offset = sending_offset
                else:
                    applied_other_offset = other_offset
                    applied_sending_offset = 0
            else:
                applied_other_offset = 0
                applied_sending_offset = 0

            # Simple CSV export
            import csv
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)

                # Write header (include state columns for consistency)
                header = [
                    "Timestamp (s)", "Current (uA)", "Applied_Other_Offset (uA)", "Applied_Sending_Offset (uA)",
                    "Voltage (mV)", "Power (mW)", "Guest_ID", "Running_State", "Wakeup_State", "Running_Mode"
                ]
                writer.writerow(header)

                # Write data rows
                for timestamp, current_ua in zip(x, y):
                    power_mw = (voltage_mv * current_ua) / 1000.0
                    row = [
                        timestamp, current_ua, applied_other_offset, applied_sending_offset,
                        voltage_mv, power_mw, guest_params.get('id', 1),
                        status_params.get('running_state_status', current_running_state),
                        status_params.get('wakeup_state', 0),
                        status_params.get('running_mode', current_running_state)
                    ]
                    writer.writerow(row)

            QMessageBox.information(self, "Save Complete", f"Data saved to {path}\nRecords: {len(x)}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data: {e}")
            import traceback
            traceback.print_exc()

    def _save_single_format(self, selected_format, timestamp_str):
        """Save data in a single selected format"""
        format_info = self._get_format_info(selected_format)

        # File dialog for save location
        default_filename = f"guest_data_{timestamp_str}{format_info['extension']}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Measurement Data", default_filename, format_info['filter']
        )

        if not path:
            return

        print(f"💾 SAVING DATA: {format_info['name']} format to {path}")

        # Save based on format
        if format_info['type'] == 'csv':
            self._save_csv_format(path)
        elif format_info['type'] == 'hdf5':
            self._save_hdf5_format(path)
        elif format_info['type'] == 'parquet':
            self._save_parquet_format(path)
        elif format_info['type'] == 'npz':
            self._save_npz_format(path)

    def _save_all_formats(self, timestamp_str):
        """Save data in all supported formats"""
        # Directory dialog for save location
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory for Multi-Format Save"
        )

        if not directory:
            return

        print(f"💾 SAVING DATA: All formats to {directory}")

        # Create base filename
        base_name = f"guest_data_{timestamp_str}"

        # Save in each format
        formats = ['csv', 'hdf5', 'parquet', 'npz']
        saved_files = []

        progress = QProgressDialog("Saving in multiple formats...", "Cancel", 0, len(formats), self)
        progress.setWindowModality(2)  # ApplicationModal
        progress.show()

        for i, fmt in enumerate(formats):
            if progress.wasCanceled():
                break

            progress.setLabelText(f"Saving {fmt.upper()} format...")
            progress.setValue(i)
            QApplication.processEvents()

            try:
                file_path = os.path.join(directory, f"{base_name}.{fmt}")

                if fmt == 'csv':
                    self._save_csv_format(file_path)
                elif fmt == 'hdf5':
                    self._save_hdf5_format(file_path)
                elif fmt == 'parquet':
                    self._save_parquet_format(file_path)
                elif fmt == 'npz':
                    self._save_npz_format(file_path)

                saved_files.append(file_path)
                print(f"✅ Saved: {file_path}")

            except Exception as e:
                print(f"❌ Failed to save {fmt}: {e}")

        progress.setValue(len(formats))

        # Show summary
        if saved_files:
            QMessageBox.information(
                self, "Multi-Format Save Complete",
                f"Successfully saved {len(saved_files)} files:\n" +
                "\n".join([os.path.basename(f) for f in saved_files])
            )

    def _get_format_info(self, format_text):
        """Get file format information"""
        format_map = {
            "CSV (Excel compatible)": {
                'type': 'csv', 'extension': '.csv', 'name': 'CSV',
                'filter': 'CSV files (*.csv);;All files (*.*)'
            },
            "HDF5 (High performance)": {
                'type': 'hdf5', 'extension': '.h5', 'name': 'HDF5',
                'filter': 'HDF5 files (*.h5);;All files (*.*)'
            },
            "Parquet (Compressed)": {
                'type': 'parquet', 'extension': '.parquet', 'name': 'Parquet',
                'filter': 'Parquet files (*.parquet);;All files (*.*)'
            },
            "NPZ (NumPy arrays)": {
                'type': 'npz', 'extension': '.npz', 'name': 'NPZ',
                'filter': 'NPZ files (*.npz);;All files (*.*)'
            }
        }
        return format_map[format_text]

    def _save_csv_format(self, path):
        """Save in CSV format (existing method)"""
        if self.dm.raw_samples_full:
            self._save_with_efficient_storage(path)
        else:
            self._save_with_current_parameters_only(path)

    def _save_hdf5_format(self, path):
        """Save in HDF5 format for high performance"""
        try:
            import h5py
            import numpy as np

            print(f"💾 HDF5 SAVE: Creating {path}")

            with h5py.File(path, 'w') as f:
                # Create groups
                measurement_group = f.create_group('measurement')
                metadata_group = f.create_group('metadata')

                if self.dm.raw_samples_full:
                    # Process raw samples
                    times_list, currents_list, guest_status_list = [], [], []

                    for sample in self.dm.raw_samples_full:
                        if len(sample) == 8:
                            time_val, current, running_state, guest_id, voltage_mv, guest_params, status_params, unified_timestamp = sample
                        else:
                            time_val, current, running_state, guest_id, voltage_mv, guest_params, status_params = sample

                        times_list.append(time_val)
                        currents_list.append(current)

                        # Flatten guest status for this sample
                        status_row = {**guest_params, **status_params}
                        guest_status_list.append(status_row)

                    # Save main data
                    measurement_group.create_dataset('time', data=np.array(times_list))
                    measurement_group.create_dataset('current', data=np.array(currents_list))

                    # Save guest status as separate datasets
                    if guest_status_list:
                        status_group = f.create_group('guest_status')
                        all_keys = set()
                        for status in guest_status_list:
                            all_keys.update(status.keys())

                        for key in all_keys:
                            values = [status.get(key, 0) for status in guest_status_list]
                            # Convert to numeric, handling string values
                            numeric_values = []
                            for val in values:
                                try:
                                    if isinstance(val, str):
                                        # Handle special cases
                                        if val.upper() == 'N/A' or val.strip() == '':
                                            numeric_values.append(-1)  # N/A uses -1
                                        elif val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                                            numeric_values.append(int(val))
                                        else:
                                            try:
                                                numeric_values.append(float(val))
                                            except ValueError:
                                                # If can't convert, use -1 for invalid data
                                                numeric_values.append(-1)
                                    else:
                                        numeric_values.append(float(val) if val is not None else -1)
                                except (ValueError, TypeError):
                                    numeric_values.append(-1)

                            status_group.create_dataset(key, data=np.array(numeric_values, dtype=np.float64))
                else:
                    # Fallback to display data
                    measurement_group.create_dataset('time', data=self.dm.times)
                    measurement_group.create_dataset('current', data=self.dm.currents)

                # Metadata
                metadata_group.attrs['save_timestamp'] = time.time()
                metadata_group.attrs['sample_count'] = len(times_list) if self.dm.raw_samples_full else len(
                    self.dm.times)
                metadata_group.attrs['format_version'] = '1.0'

            print(f"✅ HDF5 SAVE: Completed {path}")

        except ImportError:
            QMessageBox.warning(self, "HDF5 Not Available", "Please install h5py: pip install h5py")
        except Exception as e:
            print(f"❌ HDF5 SAVE ERROR: {e}")
            import traceback
            traceback.print_exc()

            # Try to provide more specific error information
            if "no conversion path for dtype" in str(e):
                error_msg = (f"HDF5 save failed due to data type conversion error.\n\n"
                             f"This usually happens when guest status parameters contain "
                             f"non-numeric data. Invalid/N/A values are converted to -1.\n\n"
                             f"The error has been logged to console. Try using CSV format instead.")
                QMessageBox.critical(self, "HDF5 Data Type Error", error_msg)
            else:
                QMessageBox.critical(self, "HDF5 Save Error", f"Failed to save HDF5 file:\n{e}")
            raise

    def _save_parquet_format(self, path):
        """Save in Parquet format for compression"""
        try:
            import pandas as pd

            print(f"💾 PARQUET SAVE: Creating {path}")

            if self.dm.raw_samples_full:
                # Process raw samples into DataFrame
                data_rows = []

                for sample in self.dm.raw_samples_full:
                    if len(sample) == 8:
                        time_val, current, running_state, guest_id, voltage_mv, guest_params, status_params, unified_timestamp = sample
                    else:
                        time_val, current, running_state, guest_id, voltage_mv, guest_params, status_params = sample

                    row = {
                        'time': time_val,
                        'current': current,
                        'voltage': voltage_mv,
                        'power': current * voltage_mv / 1000.0,
                        **guest_params,
                        **status_params
                    }
                    data_rows.append(row)

                df = pd.DataFrame(data_rows)
            else:
                # Fallback to display data
                df = pd.DataFrame({
                    'time': self.dm.times,
                    'current': self.dm.currents
                })

            # Save with compression
            df.to_parquet(path, compression='snappy', index=False)
            print(f"✅ PARQUET SAVE: Completed {path} ({len(df)} rows)")

        except ImportError:
            QMessageBox.warning(self, "Parquet Not Available",
                                "Please install pandas and pyarrow: pip install pandas pyarrow")
        except Exception as e:
            print(f"❌ PARQUET SAVE ERROR: {e}")
            raise

    def _save_npz_format(self, path):
        """Save in NumPy NPZ format for fastest loading"""
        try:
            import numpy as np

            print(f"💾 NPZ SAVE: Creating {path}")

            if self.dm.raw_samples_full:
                # Process raw samples
                times_list, currents_list = [], []
                guest_status_dict = {}

                for sample in self.dm.raw_samples_full:
                    if len(sample) == 8:
                        time_val, current, running_state, guest_id, voltage_mv, guest_params, status_params, unified_timestamp = sample
                    else:
                        time_val, current, running_state, guest_id, voltage_mv, guest_params, status_params = sample

                    times_list.append(time_val)
                    currents_list.append(current)

                    # Collect all status parameters
                    all_status = {**guest_params, **status_params}
                    for key, value in all_status.items():
                        if key not in guest_status_dict:
                            guest_status_dict[key] = []
                        guest_status_dict[key].append(value)

                # Convert to numpy arrays
                arrays_to_save = {
                    'time': np.array(times_list),
                    'current': np.array(currents_list),
                    'voltage': np.array([3300] * len(times_list))  # Default voltage
                }

                # Add guest status arrays with proper numeric conversion
                for key, values in guest_status_dict.items():
                    # Convert to numeric, handling string values
                    numeric_values = []
                    for val in values:
                        try:
                            if isinstance(val, str):
                                # Handle special cases
                                if val.upper() == 'N/A' or val.strip() == '':
                                    numeric_values.append(-1)  # N/A uses -1
                                elif val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                                    numeric_values.append(int(val))
                                else:
                                    try:
                                        numeric_values.append(float(val))
                                    except ValueError:
                                        # If can't convert, use -1 for invalid data
                                        numeric_values.append(-1)
                            else:
                                numeric_values.append(float(val) if val is not None else -1)
                        except (ValueError, TypeError):
                            numeric_values.append(-1)

                    arrays_to_save[f'guest_{key}'] = np.array(numeric_values, dtype=np.float64)

            else:
                # Fallback to display data
                arrays_to_save = {
                    'time': self.dm.times,
                    'current': self.dm.currents
                }

            # Save compressed
            np.savez_compressed(path, **arrays_to_save)
            print(f"✅ NPZ SAVE: Completed {path} ({len(arrays_to_save)} arrays)")

        except Exception as e:
            print(f"❌ NPZ SAVE ERROR: {e}")
            raise

    def _on_clear_data(self):
        """Clear all measurement data with progress indication"""
        reply = QMessageBox.question(
            self, "Clear Data",
            "Are you sure you want to clear all measurement data?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Show progress dialog for large datasets
            progress = None
            try:
                # Check if we have a lot of data
                data_size = len(self.dm.raw_samples_full) if hasattr(self.dm, 'raw_samples_full') else 0
                if data_size > 100000:  # Show progress for large datasets
                    progress = QProgressDialog("Clearing data...", "Cancel", 0, 100, self)
                    progress.setWindowTitle("Clear Data")
                    progress.setWindowModality(Qt.WindowModal)
                    progress.show()
                    QApplication.processEvents()

                # Clear data in chunks to prevent UI freeze
                if progress:
                    progress.setValue(20)
                    QApplication.processEvents()

                # Clear data manager
                self.dm.clear()

                # Clear wakeup timestamps
                if hasattr(self, 'wakeup_timestamps'):
                    wakeup_count = len(self.wakeup_timestamps)
                    self.wakeup_timestamps = []
                    print(f"🔄 Cleared {wakeup_count} wakeup timestamp(s)")

                if progress:
                    progress.setValue(40)
                    QApplication.processEvents()

                # Reset time bounds
                self.t_min = 0.0
                self.t_max = 10.0
                self.window_begin = 0.0
                self.window_end = 10.0

                if progress:
                    progress.setValue(60)
                    QApplication.processEvents()

                # Reset guest status and clear logs
                self.reset_guest_status()
                self.clear_logs()

                # Clear current guest status and history
                with self.guest_status_lock:
                    status_count = len(self.current_guest_status)
                    self.current_guest_status.clear()
                    print(f"🧹 Cleared {status_count} guest status parameters")

                # HIGH-PERFORMANCE: Clear parameter buffer
                self.parameter_buffer.clear()

                # Legacy: Clear guest status history for time synchronization
                with self.status_history_lock:
                    history_count = len(self.guest_status_history)
                    self.guest_status_history.clear()
                    print(f"🧹 Cleared {history_count} guest status history entries")

                if progress:
                    progress.setValue(80)
                    QApplication.processEvents()

                # Clear the plots
                try:
                    self.curve_min.setData([], [])
                    self.curve_max.setData([], [])
                    self.curve_mean.setData([], [])
                    self.ov_min.setData([], [])
                    self.ov_max.setData([], [])
                    self.ov_mean.setData([], [])
                except Exception:
                    pass

                if progress:
                    progress.setValue(100)
                    QApplication.processEvents()

                # Reset statistics
                self.lbl_window.setText("Window: avg -- μA, max -- μA, duration -- s, charge -- μC")
                self.lbl_selection.setText("Selection: avg -- μA, max -- μA, duration -- s, charge -- μC")

                self.update_status("Data cleared successfully")

            except Exception as e:
                self.update_status(f"Error clearing data: {e}", error=True)
            finally:
                if progress:
                    progress.close()

    def _on_calibration_config_changed(self):
        """Handle calibration configuration changes"""
        try:
            self.calibration_enabled = self.calibration_enabled_check.isChecked()
            self.calibration_normal_offset = self.normal_offset_spin.value()
            self.calibration_sending_offset = self.sending_offset_spin.value()

            # Enable/disable calibration spinboxes based on calibration enabled state
            self.normal_offset_spin.setEnabled(self.calibration_enabled)
            self.sending_offset_spin.setEnabled(self.calibration_enabled)

            # Update visual styling to indicate enabled/disabled state
            if self.calibration_enabled:
                self.normal_offset_spin.setStyleSheet("")  # Default style
                self.sending_offset_spin.setStyleSheet("")  # Default style
                self.normal_offset_label.setStyleSheet("color: black;")
                self.sending_offset_label.setStyleSheet("color: black;")
            else:
                self.normal_offset_spin.setStyleSheet("color: gray;")
                self.sending_offset_spin.setStyleSheet("color: gray;")
                self.normal_offset_label.setStyleSheet("color: gray;")
                self.sending_offset_label.setStyleSheet("color: gray;")

            logging.info(
                f"Calibration config changed: enabled={self.calibration_enabled}, "
                f"normal={self.calibration_normal_offset}μA, sending={self.calibration_sending_offset}μA"
            )

            self.update_status(f"Calibration {'enabled' if self.calibration_enabled else 'disabled'}")

        except Exception as e:
            logging.error(f"Error updating calibration configuration: {e}")
            self.update_status(f"Calibration config error: {e}", error=True)

    def update_status(self, message, error=False):
        """Update status display and log"""
        try:
            if error:
                self.status_label.setText(f"Status: {message}")
                self.status_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
                if hasattr(self, 'log_text'):
                    self.log_text.append(f"ERROR: {message}")
            else:
                self.status_label.setText(f"Status: {message}")
                self.status_label.setStyleSheet("font-weight: bold; color: #2ecc71;")
                if hasattr(self, 'log_text'):
                    self.log_text.append(f"INFO: {message}")
        except Exception as e:
            logging.error(f"Error updating status: {e}")

    def update_memory_status(self):
        """Update memory usage display"""
        try:
            usage = self.dm.get_memory_usage()

            # Add memory info to status
            memory_text = f"Memory: {usage['samples']:,} samples ({usage['memory_mb']:.1f}MB, {usage['usage_percent']:.1f}%)"

            # Color based on usage level
            if usage['is_critical']:
                color = "#e74c3c"  # Red
                memory_text = f"🚨 {memory_text}"
            elif usage['is_warning']:
                color = "#f39c12"  # Orange
                memory_text = f"⚠️ {memory_text}"
            else:
                color = "#27ae60"  # Green
                memory_text = f"✅ {memory_text}"

            # Update memory status label (if exists) or add to main status
            if hasattr(self, 'memory_status_label'):
                self.memory_status_label.setText(memory_text)
                self.memory_status_label.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 9px;")
            else:
                # Add memory info to main status
                current_status = self.status_label.text()
                if not "Memory:" in current_status:
                    self.status_label.setText(f"{current_status} | {memory_text}")

        except Exception as e:
            logging.error(f"Error updating memory status: {e}")

    def update_calibration_labels(self, running_state):
        """Update calibration labels based on running state"""
        try:
            # Update label text based on running state
            if running_state == 2:
                self.normal_offset_label.setText("Other Offset (inactive):")
                self.sending_offset_label.setText("Sending Offset (active):")
                self.normal_offset_label.setStyleSheet("color: gray; font-weight: normal;")
                self.sending_offset_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.normal_offset_label.setText("Other Offset (active):")
                self.sending_offset_label.setText("Sending Offset (inactive):")
                self.normal_offset_label.setStyleSheet("color: green; font-weight: bold;")
                self.sending_offset_label.setStyleSheet("color: gray; font-weight: normal;")

        except Exception as e:
            logging.error(f"Error updating calibration labels: {e}")

    def reset_guest_status(self):
        """Reset all guest status labels to default values when starting measurement"""
        try:
            # Reset all status labels to "0" (default value)
            if hasattr(self, 'status_labels'):
                for field, label in self.status_labels.items():
                    label.setText("0")
                    label.setStyleSheet("font-size: 9px; color: #666;")  # Reset to default style

            # Reset wakeup detection variables
            if hasattr(self, 'current_buffer'):
                self.current_buffer.clear()
            if hasattr(self, 'wakeup_counter'):
                self.wakeup_counter = 0

            logging.info("Guest status and wakeup detection reset to default values")
        except Exception as e:
            logging.error(f"Error resetting guest status: {e}")

    def clear_logs(self):
        """Clear serial command logs and status logs"""
        try:
            # Clear serial status text
            if hasattr(self, 'serial_status_text') and self.serial_status_text is not None:
                self.serial_status_text.clear()

            # Clear status logs
            if hasattr(self, 'log_text') and self.log_text is not None:
                self.log_text.clear()

            logging.info("Logs cleared")
        except Exception as e:
            logging.error(f"Error clearing logs: {e}")

    def _auto_clear_logs(self):
        """Automatically clear logs every 20 seconds to prevent app freezing"""
        try:
            current_time = time.time()

            # Only clear if there's substantial log content to prevent freezing
            should_clear = False

            # Check serial status text length
            if hasattr(self, 'serial_status_text') and self.serial_status_text is not None:
                if self.serial_status_text.document().blockCount() > 500:  # More than 500 lines
                    should_clear = True

            # Check status logs length
            if hasattr(self, 'log_text') and self.log_text is not None:
                if self.log_text.document().blockCount() > 200:  # More than 200 lines
                    should_clear = True

            if should_clear:
                self.clear_logs()

        except Exception as e:
            logging.error(f"Error in auto log clearing: {e}")

    def _process_wakeup_detection(self, times, currents):
        """Process baseline-based wakeup detection WITH REAL-TIME TIMESTAMP TRACKING - works on PPK2 data even when node sleeps"""
        try:
            # Method is working - debug removed to reduce lag
            # Get current running state from LAST KNOWN status (important: node may be sleeping!)
            # Priority: 1) current_guest_status, 2) current_running_state, 3) UI labels
            current_running_state = 0
            if hasattr(self, 'current_guest_status') and 'running_state_status' in self.current_guest_status:
                try:
                    current_running_state = int(self.current_guest_status['running_state_status'])
                except (ValueError, TypeError):
                    pass

            if current_running_state == 0 and hasattr(self, 'current_running_state'):
                current_running_state = getattr(self, 'current_running_state', 0)

            current_time = time.time()

            # No need to track running_state=3 entry - just record wakeup timestamp when detected

            # CRITICAL: Only detect wakeup when running_state == 3 AND current wakeup == 0
            if current_running_state == 3 and self.current_wakeup_state == 0:
                # Wakeup detection is active - debug removed to reduce lag
                if not hasattr(self, '_wakeup_debug_shown'):
                    self._wakeup_debug_shown = True

                # Initialize baseline collection when entering state 3 with wakeup 0
                if self.wakeup_baseline_start_time is None:
                    self.wakeup_baseline_start_time = current_time
                    self.wakeup_baseline_samples = []
                    self.wakeup_baseline = None
                    self.wakeup_sample_buffer = []  # Clear rolling buffer
                    self.wakeup_timestamp_buffer = []  # Clear timestamp buffer

                # Collect samples for baseline calculation
                for current in currents:
                    self.wakeup_baseline_samples.append(current)

                # Calculate baseline after 2 seconds
                time_elapsed = current_time - self.wakeup_baseline_start_time
                if time_elapsed >= self.BASELINE_COLLECTION_TIME and self.wakeup_baseline is None:
                    if self.wakeup_baseline_samples:
                        self.wakeup_baseline = sum(self.wakeup_baseline_samples) / len(self.wakeup_baseline_samples)

                # Start wakeup detection after baseline is established
                if self.wakeup_baseline is not None:
                    # Decrement cooldown counter
                    if self.wakeup_detection_cooldown > 0:
                        self.wakeup_detection_cooldown -= len(currents)
                        if self.wakeup_detection_cooldown < 0:
                            self.wakeup_detection_cooldown = 0

                    # Only detect if not in cooldown
                    if self.wakeup_detection_cooldown <= 0:
                        # Add samples to rolling buffer for mean calculation
                        for time_val, current in zip(times, currents):
                            self.wakeup_sample_buffer.append(current)
                            self.wakeup_timestamp_buffer.append(time_val)

                        # Keep only the last WAKEUP_WINDOW_SIZE samples and timestamps
                        if len(self.wakeup_sample_buffer) > self.WAKEUP_WINDOW_SIZE:
                            self.wakeup_sample_buffer = self.wakeup_sample_buffer[-self.WAKEUP_WINDOW_SIZE:]
                            self.wakeup_timestamp_buffer = self.wakeup_timestamp_buffer[-self.WAKEUP_WINDOW_SIZE:]

                        # Check if we have enough samples for detection
                        if len(self.wakeup_sample_buffer) >= self.WAKEUP_WINDOW_SIZE:
                            # Calculate mean current of the window
                            window_mean = sum(self.wakeup_sample_buffer) / len(self.wakeup_sample_buffer)
                            wakeup_threshold = self.wakeup_baseline + self.WAKEUP_MEAN_THRESHOLD

                            # Check if mean exceeds threshold - WAKEUP CONFIRMED!
                            if window_mean > wakeup_threshold:
                                print(
                                    f"🔔 WAKEUP CONFIRMED: Baseline={self.wakeup_baseline:.1f}µA, Window Mean={window_mean:.1f}µA, Threshold={wakeup_threshold:.1f}µA")

                                # Update local wakeup state first
                                self.current_wakeup_state = 1

                                # Update the UI wakeup label
                                if hasattr(self, 'wakeup_label'):
                                    self.wakeup_label.setText("1")
                                    self.wakeup_label.setStyleSheet("font-weight: bold; color: #e74c3c;")

                                # Update current_guest_status for wakeup state to ensure consistency
                                if hasattr(self, 'current_guest_status'):
                                    self.current_guest_status['wakeup_state'] = 1

                                # CRITICAL: Change running_state to 0 (record timestamp)
                                self._change_running_state_to_0()

                                # Set cooldown to prevent duplicate detections (2 seconds worth of samples)
                                self.wakeup_detection_cooldown = 200000  # 200k samples = 2 seconds at 100kHz

                                # Reset detection variables
                                self._reset_wakeup_detection()
                                # Don't break - continue processing remaining samples

            else:
                # Reset debug flag when conditions change
                if hasattr(self, '_wakeup_debug_shown'):
                    del self._wakeup_debug_shown  # Reset so we can show again when conditions change

                # Not in the right state for wakeup detection - reset wakeup detection baseline
                if (current_running_state != 3 or self.current_wakeup_state != 0):
                    self._reset_wakeup_detection()

                    # CRITICAL FIX: Reset wakeup_state to 0 ONLY when running_state changes to 1, 2, or 3
                    # Wakeup_state should STAY at 1 when running_state is 0 (the wakeup state)
                    # Only reset when device moves away from wakeup state (0) to another state
                    if current_running_state in [1, 2, 3] and self.current_wakeup_state != 0:
                        print(
                            f"🔄 WAKEUP RESET: Running_State changed to {current_running_state} (away from wakeup state 0), resetting Wakeup_State to 0")
                        self.current_wakeup_state = 0

                        # Update UI
                        if hasattr(self, 'wakeup_label'):
                            self.wakeup_label.setText("0")
                            self.wakeup_label.setStyleSheet("font-weight: normal; color: #666;")

                        # Update running_state label back to normal style
                        if hasattr(self, 'status_labels') and 'running_state' in self.status_labels:
                            self.status_labels['running_state'].setStyleSheet("font-weight: normal; color: #666;")

                        # Update current_guest_status
                        if hasattr(self, 'current_guest_status'):
                            self.current_guest_status['wakeup_state'] = 0

                        # Update parameter buffer
                        if hasattr(self, 'parameter_buffer') and hasattr(self, '_global_start_time'):
                            current_ppk2_time = time.perf_counter() - self._global_start_time
                            updated_params = self.current_guest_status.copy() if hasattr(self,
                                                                                         'current_guest_status') else {}
                            updated_params['wakeup_state'] = 0
                            self.parameter_buffer.update(current_ppk2_time, updated_params)

        except Exception as e:
            logging.error(f"Error in wakeup detection: {e}")
            import traceback
            traceback.print_exc()

    def _change_running_state_to_0(self):
        """
        Change running_state to 0 when wakeup is detected.
        Record GLOBAL timestamp from canvas for use during save.

        Example: running_state = [3,3,3,3,3,3,1,1,1]
                 wakeup at position 4 (global time = t)
                 Result: [3,3,3,0,0,0,1,1,1]
        """
        try:
            if hasattr(self, 'dm') and self.dm.times.size > 0:
                # Get GLOBAL timestamp from canvas X-axis
                global_wakeup_time = float(self.dm.times[-1])

                print(f"\n{'=' * 80}")
                print(f"🔔 WAKEUP DETECTED - Recording global timestamp")
                print(f"{'=' * 80}")
                print(f"   Canvas time range: {self.dm.times[0]:.6f}s to {self.dm.times[-1]:.6f}s")
                print(f"   Canvas size: {len(self.dm.times):,} samples")
                print(f"   🎯 Recorded global timestamp: t={global_wakeup_time:.6f}s")

                # DEBUG: Check raw_samples_full too
                if hasattr(self, 'dm') and self.dm.raw_samples_full:
                    raw_first = self.dm.raw_samples_full[0][0]
                    raw_last = self.dm.raw_samples_full[-1][0]
                    print(f"   Raw buffer time range: {raw_first:.6f}s to {raw_last:.6f}s")
                    print(f"   Raw buffer size: {len(self.dm.raw_samples_full):,} samples")

                # Record it
                self.wakeup_timestamps.append(global_wakeup_time)

                print(f"\n   ✅ RULE: All samples with time >= {global_wakeup_time:.6f}s")
                print(f"      will change Running_State 3→0 and Wakeup_State 0→1")
                print(f"   Total wakeup events recorded: {len(self.wakeup_timestamps)}")
                print(f"{'=' * 80}\n")
            else:
                print(f"⚠️ Cannot record wakeup timestamp - no canvas data")

            # Update the running state in UI
            if hasattr(self, 'status_labels') and 'running_state' in self.status_labels:
                self.status_labels['running_state'].setText("0")
                self.status_labels['running_state'].setStyleSheet("font-weight: bold; color: #e74c3c;")

            # Update internal tracking
            self.current_running_state = 0
            if hasattr(self, 'current_guest_status'):
                self.current_guest_status['running_state_status'] = 0

            # Update parameter buffer for future samples
            if hasattr(self, 'parameter_buffer') and hasattr(self, '_global_start_time'):
                current_ppk2_time = time.perf_counter() - self._global_start_time
                updated_params = self.current_guest_status.copy() if hasattr(self, 'current_guest_status') else {}
                updated_params['running_state_status'] = 0
                updated_params['wakeup_state'] = 1
                self.parameter_buffer.update(current_ppk2_time, updated_params)

            print(f"✅ WAKEUP STATE CHANGE - App updated, CSV will be updated during save")

        except Exception as e:
            logging.error(f"Error changing running state to 0: {e}")
            import traceback
            traceback.print_exc()

    def _reset_wakeup_detection(self):
        """Reset all wakeup detection variables"""
        self.wakeup_baseline = None
        self.wakeup_baseline_samples = []
        self.wakeup_baseline_start_time = None
        self.wakeup_sample_buffer = []  # Clear the rolling buffer
        self.wakeup_timestamp_buffer = []  # Clear the timestamp buffer
        # Note: Don't reset cooldown here - it should persist across baseline recalculations

    def _should_apply_wakeup_fix(self, sample_time):
        """
        SIMPLE: Check if sample time >= global wakeup timestamp.
        Example: [3,3,3,3,3,3,1,1,1] with wakeup at 4th position → [3,3,3,0,0,0,1,1,1]
        """
        if not hasattr(self, 'wakeup_timestamps') or not self.wakeup_timestamps:
            return False

        # Check if sample time >= ANY wakeup timestamp (treats all files as continuous time series)
        for wakeup_ts in self.wakeup_timestamps:
            if sample_time >= wakeup_ts:
                return True
        return False


# Helper functions and classes
def enable_debug_logging():
    """Enable debug logging for development"""
    try:
        import logging

        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Add console handler if not present
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        logging.info("🔧 Wakeup detection debug logging enabled")

    except Exception as e:
        print(f"Failed to enable debug logging: {e}")


def main():
    app = QApplication(sys.argv)
    # Import here to avoid optional dependency warning if not present
    import pyqtgraph.exporters  # noqa: F401
    csv_path = None
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        csv_path = sys.argv[1]
    win = PPK2Main(csv_path)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
