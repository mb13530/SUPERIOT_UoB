import sys
import joblib
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
                             QPushButton, QGroupBox, QMessageBox, QProgressBar,
                             QComboBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (QPainter, QColor, QPen, QPainterPath, QLinearGradient,
                         QFont, QBrush, QConicalGradient, QPixmap, QIcon)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec  # Import GridSpec
import math
import pandas as pd
import json


class MeterWidget(QWidget):
    def __init__(self, parent=None, max_value=100, unit=""):
        super().__init__(parent)
        self.value = 0
        self.maximum = max_value
        self.unit = unit
        self.setFixedSize(60, 60)
        self.setMinimumSize(60, 60)
        self.setMaximumSize(60, 60)

        # Define scale points and their angles for each meter type
        # First number at 0° (3 o'clock position)
        # Second number at 180° (9 o'clock position)
        if unit == "mWh":  # Energy meter (0-10 mWh)
            self.scale_points = [(2.5, 0), (7.5, 180)]  # Values and their angles
        elif unit == "mW":  # Power meter (0-100 mW)
            self.scale_points = [(25, 0), (75, 180)]
        elif unit == "mA":  # Current meter (0-20 mA)
            self.scale_points = [(5, 0), (15, 180)]
        else:
            self.scale_points = [(25, 0), (75, 180)]

        self.setValue(0)  # Initialize with 0

    def setValue(self, value):
        """Convert the input value to a percentage based on the meter type"""
        try:
            if self.unit == "mWh":
                # Scale for 0-10 mWh range
                self.value = min((value / 10.0) * 100, 100)
            elif self.unit == "mW":
                # Scale for 0-100 mW range
                self.value = min((value / 100.0) * 100, 100)
            elif self.unit == "mA":
                # Scale for 0-20 mA range
                self.value = min((value / 20.0) * 100, 100)
            else:
                self.value = min(value, 100)

            self.update()  # Request a repaint
        except Exception as e:
            print(f"Error in setValue: {e}")

    def setMaximum(self, maximum):
        try:
            self.maximum = maximum
            self.update()
        except Exception as e:
            print(f"Error setting meter maximum: {e}")

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Draw background circle with gradient
            gradient = QLinearGradient(0, 0, 60, 60)
            gradient.setColorAt(0, QColor("#34495e"))
            gradient.setColorAt(1, QColor("#2c3e50"))
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(QColor("#ecf0f1"), 2))
            painter.drawEllipse(2, 2, 56, 56)

            # Draw conical gradient background for the meter (Dark Red to Light Green)
            conical_gradient = QConicalGradient(30, 30, 90)  # Keep at 90 degrees for color gradient
            # Left side (max) - Dark Red to Red
            conical_gradient.setColorAt(0.0, QColor("#8B0000"))  # Dark Red
            conical_gradient.setColorAt(0.25, QColor("#FF0000"))  # Red
            # Right side (min) - Light Green to Very Light Green
            conical_gradient.setColorAt(0.75, QColor("#90EE90"))  # Light Green
            conical_gradient.setColorAt(1.0, QColor("#98FB98"))  # Very Light Green

            painter.setBrush(QBrush(conical_gradient))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(6, 6, 48, 48)  # Draw gradient circle

            # Draw the arc for the current value
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(conical_gradient))

            start_angle = -90 * 16  # Start from -90 degrees (right)
            span_angle = int(self.value * 3.6 * 16)  # Convert value to angle

            path_arc = QPainterPath()
            path_arc.moveTo(30, 30)
            path_arc.arcTo(6, 6, 48, 48, start_angle / 16.0,
                           span_angle / 16.0)
            path_arc.closeSubpath()

            painter.drawPath(path_arc)

            # Draw inner white circle
            painter.setBrush(QBrush(QColor("#ecf0f1")))
            painter.drawEllipse(10, 10, 40, 40)

            # Draw scale marks with two points
            painter.setPen(QPen(QColor("#2c3e50"), 1))

            for value, angle in self.scale_points:
                angle_rad = angle * 3.14159 / 180
                # Draw major tick
                painter.drawLine(
                    int(30 + 22 * math.cos(angle_rad)),
                    int(30 + 22 * math.sin(angle_rad)),
                    int(30 + 26 * math.cos(angle_rad)),
                    int(30 + 26 * math.sin(angle_rad))
                )
                # Draw scale number
                if self.unit == "mWh":
                    text = str(value)  # Show decimal for energy meter
                else:
                    text = str(int(value))  # Show as integer for other meters

                painter.setPen(QPen(QColor("#2c3e50"), 1))
                painter.setFont(QFont("Arial", 8, QFont.Bold))  # Larger and bold font
                # Keep text straight (no rotation)
                painter.save()
                painter.translate(
                    int(30 + 15 * math.cos(angle_rad)),  # Closer to center (was 18)
                    int(30 + 15 * math.sin(angle_rad))  # Closer to center (was 18)
                )
                painter.drawText(-12, -6, 24, 12, Qt.AlignCenter, text)  # Larger text area
                painter.restore()

            # Draw minor ticks
            for i in range(0, 101, 5):  # Minor ticks every 5%
                angle = -90 + (i * 3.6)
                angle_rad = angle * 3.14159 / 180
                painter.drawLine(
                    int(30 + 23 * math.cos(angle_rad)),
                    int(30 + 23 * math.sin(angle_rad)),
                    int(30 + 25 * math.cos(angle_rad)),
                    int(30 + 25 * math.sin(angle_rad))
                )

            # Draw center circle for pointer pivot
            painter.setBrush(QBrush(QColor("#2c3e50")))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(29, 29, 2, 2)

            # Draw pointer - start from -90 degrees
            angle = -90 + (self.value * 3.6)  # Convert percentage to angle
            angle_rad = angle * 3.14159 / 180

            # Draw pointer shadow
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 50)))
            shadow_offset = 1
            path = QPainterPath()
            path.moveTo(30, 30)
            path.lineTo(
                int(30 + 25 * math.cos(angle_rad) + shadow_offset),
                int(30 + 25 * math.sin(angle_rad) + shadow_offset)
            )
            painter.drawPath(path)

            # Draw pointer
            painter.setPen(QPen(QColor("#2c3e50"), 2))
            painter.setBrush(QBrush(QColor("#2c3e50")))
            path = QPainterPath()
            path.moveTo(30, 30)
            path.lineTo(
                int(30 + 25 * math.cos(angle_rad)),
                int(30 + 25 * math.sin(angle_rad))
            )
            painter.drawPath(path)

        except Exception as e:
            print(f"Error in paintEvent: {e}")
            # Draw a simple fallback
            painter.setBrush(QBrush(QColor("#ecf0f1")))
            painter.setPen(QPen(QColor("#2c3e50"), 2))
            painter.drawEllipse(2, 2, 56, 56)

    def updateAnimation(self):
        """DEPRECATED: This method is no longer used. Animation updates happen in update_animation()"""
        # This method is kept for compatibility but is no longer called
        try:
            if not hasattr(self, 'current_time'):
                self.current_time = 0

            time_points = self.animation_data['time_points']
            current_time_index = np.clip(np.searchsorted(time_points, self.current_time), 0, len(time_points) - 1)

            # Update Nodes Energy Indicator (0-4 mWh)
            if self.animation_data and len(self.animation_data['energy_consumption']) > 0:
                energy_val = self.animation_data['energy_consumption'][0][current_time_index]
                self.nodes_energy_indicator_value.setText(f"{energy_val:.2f} mWh")
                self.nodes_energy_indicator_bar.setValue(energy_val)  # MeterWidget will handle scaling
            else:
                self.nodes_energy_indicator_value.setText("-- mWh")
                self.nodes_energy_indicator_bar.setValue(0)

            # Update Nodes Power Indicator (0-100 mW)
            if self.animation_data and len(self.animation_data['power_consumption']) > 0:
                power_val = self.animation_data['power_consumption'][0][current_time_index]
                self.nodes_power_indicator_value.setText(f"{power_val:.2f} mW")
                self.nodes_power_indicator_bar.setValue(power_val)  # MeterWidget will handle scaling
            else:
                self.nodes_power_indicator_value.setText("-- mW")
                self.nodes_power_indicator_bar.setValue(0)

            # Update Current Indicator (0-20 mA)
            if self.animation_data and len(self.animation_data['current_consumption']) > 0:
                current_val = self.animation_data['current_consumption'][0][current_time_index]
                self.current_indicator_value.setText(f"{current_val:.2f} mA")
                self.current_indicator_bar.setValue(current_val)  # MeterWidget will handle scaling
            else:
                self.current_indicator_value.setText("-- mA")
                self.current_indicator_bar.setValue(0)

            # Update plots with animation line
            for i in range(len(self.animation_data['device_ids'])):
                # Plot energy consumption
                if len(self.animation_data['energy_consumption']) > i:
                    ax = self.axes[i * 3]
                    ax.clear()
                    ax.plot(time_points, self.animation_data['energy_consumption'][i], 'b-')
                    ax.axvline(x=self.current_time, color='r', linestyle='--', alpha=0.5)
                    ax.set_ylabel('Energy (mWh)')
                    ax.set_title('Energy')
                    ax.grid(True)
                    ax.tick_params(labelbottom=False)
                    ax.set_ylim([0, 4])

                # Plot power consumption
                if len(self.animation_data['power_consumption']) > i:
                    ax = self.axes[i * 3 + 1]
                    ax.clear()
                    ax.plot(time_points, self.animation_data['power_consumption'][i], 'g-')
                    ax.axvline(x=self.current_time, color='r', linestyle='--', alpha=0.5)
                    ax.set_ylabel('Power (mW)')
                    ax.set_title('Power')
                    ax.grid(True)
                    ax.tick_params(labelbottom=False)
                    ax.set_ylim([0, 100])

                # Plot current consumption
                if len(self.animation_data['current_consumption']) > i:
                    ax = self.axes[i * 3 + 2]
                    current_data = self.animation_data['current_consumption'][i]
                    ax.plot(time_points, current_data, 'r-')
                    ax.axvline(x=self.current_time, color='r', linestyle='--', alpha=0.5)
                    ax.set_ylabel('Current (mA)')
                    ax.set_title('Current')
                    ax.grid(True)
                    # Set y-axis limits for current plot
                    y_max = max(20, math.ceil(max(current_data)))
                    ax.set_ylim([0, y_max])
                    # Format y-axis to show proper mA values (divide by 1000 since data is already in mA)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
                    if i == len(self.animation_data['device_ids']) - 1:
                        ax.set_xlabel('Time (s)')
                    else:
                        ax.tick_params(labelbottom=False)

            self.canvas.draw()

            # Update time
            self.current_time += self.animation_speed
            if self.current_time > max(time_points):
                self.current_time = 0

        except Exception as e:
            print(f"Error updating animation: {e}")
            self.animation_timer.stop()


class PowerPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Keep standard window with custom title bar styling
        self.setWindowTitle("Energy Consumption Predictor")
        
        self.setMinimumWidth(1323)  # Increased by 5% (1260 → 1323)
        self.setMinimumHeight(882)  # Increased by 5% (840 → 882)

        # Initialize model storage
        self.models = {}
        self.current_model = None
        self.current_model_data = None
        self.current_model_name = None
        self.feature_names = []

        # Load available models
        self.load_available_models()

        # Set default values based on model training
        self.voltage_mV = 5000  # Default voltage in mV

        # Communication energy overhead (mJ per byte)
        self.avg_vlc_energy_per_byte = 0.1
        self.avg_ble_energy_per_byte = 0.05
        self.avg_vlc_current_per_byte = 0.1
        self.avg_ble_current_per_byte = 0.05

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create custom title bar
        self.create_custom_title_bar(main_layout)
        
        # Create content area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        main_layout.addWidget(content_widget)

        # Set global stylesheet for larger fonts
        self.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                font-size: 16px;
                font-weight: bold;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QLabel {
                font-size: 14px;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                font-size: 14px;
                min-height: 25px;
            }
            QPushButton {
                font-size: 14px;
                min-height: 30px;
            }
        """)

        # Create left panel for inputs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.create_input_groups(left_layout)
        content_layout.addWidget(left_panel, stretch=1)

        # Create right panel for plots
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.create_plot_area(right_layout)
        content_layout.addWidget(right_panel, stretch=2)

        # Animation variables
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.current_time = 0
        self.animation_data = None
        self.animation_timer.setInterval(10)

        # Initialize node-specific configurations
        self.device_configs = {}  # Will store config for each node
        self.current_config_device = 1  # Currently selected node for configuration

        # Initialize default configuration for node 1
        self.device_configs[1] = self.get_default_device_config()

        # Load node 1 configuration to UI (to sync UI with stored config)
        self.load_device_config_to_ui(1)

        # Now that inputs are created, initialize plots
        self._initialize_plots(self.total_devices.value())
        
        # Mark initialization as complete (for validation messages)
        self._initialization_complete = True


    def create_custom_title_bar(self, layout):
        """Create custom title bar with logo, centered title, matching OS style"""
        title_bar = QWidget()
        title_bar.setFixedHeight(50)  # Wider than standard title bar
        title_bar.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border-bottom: 1px solid #d0d0d0;
            }
        """)
        
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 8, 15, 8)
        
        # Logo on the left
        logo_label = QLabel()
        try:
            pixmap = QPixmap("icon/icon.png")
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_label.setPixmap(scaled_pixmap)
            else:
                logo_label.setText("⚡")
                logo_label.setStyleSheet("font-size: 24px; color: black;")
        except Exception as e:
            logo_label.setText("⚡")
            logo_label.setStyleSheet("font-size: 24px; color: black;")
        
        logo_label.setFixedSize(34, 34)
        title_layout.addWidget(logo_label)
        
        # Spacer to center title
        title_layout.addStretch(1)
        
        # Centered title
        title_label = QLabel("Energy Consumption Predictor")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            color: black;
            font-size: 16px;
            font-weight: bold;
            margin: 0px;
            padding: 0px;
        """)
        title_layout.addWidget(title_label)
        
        # Spacer to balance
        title_layout.addStretch(1)
        
        # Empty spacer on right to balance the logo (no window controls needed)
        spacer_label = QLabel()
        spacer_label.setFixedSize(34, 34)
        title_layout.addWidget(spacer_label)
        
        layout.addWidget(title_bar)



    def load_available_models(self):
        """Load all available current prediction models"""
        import os
        import glob

        # Load all current prediction models (both simple and enhanced)
        model_patterns = [
            '*_enhanced_model.joblib',  # Enhanced models (7 features) - PRIORITY
            '*_model.joblib',  # Simple models (3 features) - FALLBACK
            'current_prediction_enhanced_model.joblib',  # Best enhanced model
            'current_prediction_model.joblib'  # Best simple model (fallback)
        ]
        
        print("Loading current prediction models...")
        models_loaded = 0

        for pattern in model_patterns:
            model_files = glob.glob(pattern)
            for model_file in model_files:
                try:
                    model_data = joblib.load(model_file)
                    
                    # Extract model name from filename
                    base_name = os.path.splitext(model_file)[0]
                    if base_name in ['current_prediction_model', 'current_prediction_enhanced_model']:
                        # Check if it's enhanced model
                        if 'enhanced' in base_name:
                            model_name = 'Best Model (Enhanced 7-Features)'
                        else:
                            model_name = 'Best Model (Simple 3-Features)'
                    else:
                        # Convert filename to display name
                        model_name = base_name.replace('_model', '').replace('_', ' ').title()
                    
                    self.models[model_name] = model_data
                    models_loaded += 1
                    
                    r2_score = model_data.get('r2_score', 'N/A')
                    print(f"✓ Loaded: {model_name} (R² = {r2_score:.4f})")
                    
                    # Set the first loaded model as current
                    if self.current_model is None:
                        self.current_model = model_data
                        self.current_model_data = model_data
                        self.current_model_name = model_name
                        print(f"  -> Set as current model")

                except Exception as e:
                    print(f"✗ Failed to load {model_file}: {str(e)}")

        if models_loaded == 0:
            QMessageBox.critical(self, "Error", 
                               "No current prediction models found!\n"
                               "Please run train_simple_current_prediction.py first.")
            sys.exit(1)

        print(f"Successfully loaded {models_loaded} models!")



    def create_state_button(self):
        """Create a simple state button"""
        button = QPushButton()
        button.setCheckable(True)
        button.setFixedSize(45, 25)  # Smaller, compact size

        # Connect the button event to update the appearance
        button.toggled.connect(lambda: self.update_button_appearance(button))

        # Set initial appearance
        self.update_button_appearance(button)

        return button

    def update_button_appearance(self, button):
        """Update the button appearance based on state"""
        if button.isChecked():
            # ON state - green background
            button.setStyleSheet("""
                QPushButton {
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    border: 2px solid #45a049;
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #388e3c;
                }
            """)
            button.setText("ON")
        else:
            # OFF state - red background
            button.setStyleSheet("""
                QPushButton {
                    border: 2px solid #f44336;
                    border-radius: 8px;
                    background-color: #f44336;
                    color: white;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    border: 2px solid #e53935;
                    background-color: #e53935;
                }
                QPushButton:pressed {
                    background-color: #d32f2f;
                }
            """)
            button.setText("OFF")

    def create_input_groups(self, layout):
        # Create input groups
        self.create_device_group(layout)
        self.create_state_duration_group(layout)
        self.create_ble_parameters_group(layout)
        self.create_communication_group(layout)
        self.create_prediction_group(layout)

    def create_plot_area(self, layout):
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # self.power_ax and self.current_axes will be created dynamically
        # in _initialize_plots.
        self.power_ax = None
        self.current_axes = []

        self.figure.tight_layout()  # Initial tight_layout, will be redrawn later

    def create_device_group(self, layout):
        group = QGroupBox("IoT Network Configuration")
        group_layout = QVBoxLayout()

        # Total devices input
        devices_layout = QHBoxLayout()
        devices_layout.addWidget(QLabel("Total Nodes:"))
        self.total_devices = QSpinBox()
        self.total_devices.setRange(1, 100)
        self.total_devices.setValue(1)
        self.total_devices.valueChanged.connect(self.on_total_devices_changed)
        devices_layout.addWidget(self.total_devices)
        group_layout.addLayout(devices_layout)

        # Device selection for configuration
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("Configure Node:"))
        self.config_device_combo = QComboBox()
        self.config_device_combo.addItem("Node 1")
        self.config_device_combo.currentTextChanged.connect(self.on_config_device_changed)
        config_layout.addWidget(self.config_device_combo)
        group_layout.addLayout(config_layout)

        # Total minilamps and power in the same row
        minilamps_layout = QHBoxLayout()
        minilamps_layout.addWidget(QLabel("Total Minilamps:"))
        self.total_minilamps = QSpinBox()
        self.total_minilamps.setRange(0, 1000)
        self.total_minilamps.setValue(0)
        minilamps_layout.addWidget(self.total_minilamps)
        
        # Add minilamp power in the same row
        minilamps_layout.addWidget(QLabel("   Minilamp Power (mW):"))
        self.minilamp_power_mw = QSpinBox()
        self.minilamp_power_mw.setRange(0, 10000)
        self.minilamp_power_mw.setValue(100)  # Default 100 mW
        minilamps_layout.addWidget(self.minilamp_power_mw)
        group_layout.addLayout(minilamps_layout)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def on_total_devices_changed(self, num_devices):
        """Handle total nodes count change"""
        # Update node selection combo
        self.config_device_combo.clear()
        for i in range(1, num_devices + 1):
            self.config_device_combo.addItem(f"Node {i}")

        # Initialize configurations for new nodes if needed
        for device_id in range(1, num_devices + 1):
            if device_id not in self.device_configs:
                self.device_configs[device_id] = self.get_default_device_config()

        # Update plots
        self._initialize_plots(num_devices)

    def on_config_device_changed(self, device_text):
        """Handle configuration node selection change"""
        if device_text:
            # Save current UI state to the previously selected node
            if self.current_config_device in self.device_configs:
                self.save_current_ui_to_device_config(self.current_config_device)

            # Update current node
            device_id = int(device_text.split()[-1])
            self.current_config_device = device_id

            # Load the new node's configuration to UI
            if device_id not in self.device_configs:
                self.device_configs[device_id] = self.get_default_device_config()

            self.load_device_config_to_ui(device_id)

    def get_default_device_config(self):
        """Get default configuration for a node"""
        return {
            'state_durations': [5.0, 30.0, 40.0, 10.0],  # Default durations
            'state_switches': [True, True, True, True],  # All states ON
            'vlc_payload': 0,
            'ble_payload': 0,
            'time_per_vlc_byte': 0.017,  # Time per VLC byte transmission (ms/byte) - Calculated from dataset
            'time_per_ble_byte': 0.041,  # Time per BLE byte transmission (ms/byte) - Calculated from dataset
            'tx_power': 0,  # Default TxPower (dBm)
            'adv_interval': 40,  # Default Advertising Interval (ms)
            'conn_interval': 45,  # Default Connection Interval (ms)
            'ble_connected_duration': 0.0,  # BLE Connected duration within State 3
            'comm_mode': 0  # Default: BLE Disconnected (0/1 = Disconnected, 2 = Connected)
        }

    def save_current_ui_to_device_config(self, device_id):
        """Save current UI state to node configuration"""
        if device_id not in self.device_configs:
            self.device_configs[device_id] = {}

        config = self.device_configs[device_id]
        config['state_durations'] = [spin.value() for spin in self.state_durations]
        config['state_switches'] = [switch.isChecked() for switch in self.state_switches]
        config['vlc_payload'] = self.vlc_volume.value()
        config['ble_payload'] = self.ble_volume.value()
        config['time_per_vlc_byte'] = self.time_per_vlc_byte_spin.value()
        config['time_per_ble_byte'] = self.time_per_ble_byte_spin.value()
        config['tx_power'] = self.tx_power_spin.value()
        config['adv_interval'] = self.adv_interval_spin.value()
        config['conn_interval'] = self.conn_interval_spin.value()
        config['ble_connected_duration'] = self.ble_connected_duration_spin.value()
        config['comm_mode'] = 2 if self.ble_connected_duration_spin.value() > 0 else 0

    def load_device_config_to_ui(self, device_id):
        """Load node configuration to UI"""
        if device_id not in self.device_configs:
            return

        config = self.device_configs[device_id]

        # Update group titles
        self.state_duration_group.setTitle(f"State Durations (Node {device_id})")
        self.communication_group.setTitle(f"Communication Parameters (Node {device_id})")
        self.ble_params_group.setTitle(f"BLE Parameters (Node {device_id})")

        # Load state durations and switches
        for i, (duration, switch) in enumerate(zip(config['state_durations'], config['state_switches'])):
            if i < len(self.state_durations):
                self.state_durations[i].setValue(duration)
                self.state_switches[i].setChecked(switch)
                self.update_button_appearance(self.state_switches[i])

        # Load communication settings
        self.vlc_volume.setValue(config['vlc_payload'])
        self.ble_volume.setValue(config['ble_payload'])
        self.time_per_vlc_byte_spin.setValue(config.get('time_per_vlc_byte', 0.0))
        self.time_per_ble_byte_spin.setValue(config.get('time_per_ble_byte', 0.0))
        
        # Load BLE parameters
        self.tx_power_spin.setValue(config.get('tx_power', 0))
        self.adv_interval_spin.setValue(config.get('adv_interval', 40))
        self.conn_interval_spin.setValue(config.get('conn_interval', 45))
        self.ble_connected_duration_spin.setValue(config.get('ble_connected_duration', 0.0))

    def create_state_duration_group(self, layout):
        self.state_duration_group = QGroupBox("State Durations (Node 1)")
        group_layout = QVBoxLayout()

        # Duration for each state
        self.state_durations = []
        self.state_switches = []
        default_durations = [10, 30, 40, 10]  # Default durations for states 1-4
        state_names = ["Idle", "Advertising", "Operation", "Deepsleep"]

        for state in range(4):
            duration_layout = QHBoxLayout()
            state_num = state + 1  # Convert to 1-based state numbering
            duration_layout.addWidget(QLabel(f"State {state_num} ({state_names[state]}) Duration (s):"))
            duration_spin = QDoubleSpinBox()
            duration_spin.setRange(0.1, 3600)
            duration_spin.setValue(default_durations[state])
            duration_spin.setDecimals(1)
            self.state_durations.append(duration_spin)
            duration_layout.addWidget(duration_spin)

            # Simple ON/OFF button
            state_button = self.create_state_button()
            state_button.setChecked(True)  # Default to ON
            self.state_switches.append(state_button)
            duration_layout.addWidget(state_button)

            group_layout.addLayout(duration_layout)
            
            # Add BLE Connected duration input after State 3 (Operation)
            if state == 2:  # State 3 (index 2)
                ble_connected_layout = QHBoxLayout()
                ble_connected_label = QLabel("  └─ State 3 (BLE Connected) Duration (s):")
                ble_connected_label.setStyleSheet("color: #1976d2; font-style: italic;")
                ble_connected_layout.addWidget(ble_connected_label)
                
                self.ble_connected_duration_spin = QDoubleSpinBox()
                self.ble_connected_duration_spin.setRange(0.0, 3600)
                self.ble_connected_duration_spin.setValue(0.0)
                self.ble_connected_duration_spin.setDecimals(1)
                self.ble_connected_duration_spin.valueChanged.connect(self.validate_ble_connected_duration)
                ble_connected_layout.addWidget(self.ble_connected_duration_spin)
                
                # Add info label
                info_label = QLabel("(≤ State 3 Duration)")
                info_label.setStyleSheet("font-size: 11px; color: #666;")
                ble_connected_layout.addWidget(info_label)
                
                group_layout.addLayout(ble_connected_layout)

        self.state_duration_group.setLayout(group_layout)
        layout.addWidget(self.state_duration_group)
    
    def validate_ble_connected_duration(self):
        """Validate that BLE Connected duration doesn't exceed State 3 duration"""
        state3_duration = self.state_durations[2].value()  # State 3 is index 2
        ble_connected_duration = self.ble_connected_duration_spin.value()
        
        if ble_connected_duration > state3_duration:
            self.ble_connected_duration_spin.setValue(state3_duration)
            QMessageBox.warning(self, "Invalid Duration", 
                              f"BLE Connected duration cannot exceed State 3 (Operation) duration ({state3_duration}s)")
    
    def update_transmission_durations(self):
        """
        Update and display VLC/BLE transmission durations
        Also validate payload and time_per_byte constraints
        """
        try:
            # Get values
            vlc_payload = self.vlc_volume.value()
            ble_payload = self.ble_volume.value()
            time_per_vlc_byte = self.time_per_vlc_byte_spin.value()
            time_per_ble_byte = self.time_per_ble_byte_spin.value()
            
            # Calculate transmission durations
            vlc_tx_duration = vlc_payload * time_per_vlc_byte  # ms
            ble_tx_duration = ble_payload * time_per_ble_byte  # ms
            
            # Update display labels
            self.vlc_tx_duration_label.setText(f"(→ VLC TX: {vlc_tx_duration:.3f} ms)")
            self.ble_tx_duration_label.setText(f"(→ BLE TX: {ble_tx_duration:.3f} ms)")
            
            # Validation: Check if payload > 0 but time_per_byte = 0
            error_messages = []
            
            if vlc_payload > 0 and time_per_vlc_byte == 0:
                error_messages.append("⚠️ VLC Payload > 0 but Time per VLC Byte = 0!\nPlease set VLC transmission time per byte.")
            
            if ble_payload > 0 and time_per_ble_byte == 0:
                error_messages.append("⚠️ BLE Payload > 0 but Time per BLE Byte = 0!\nPlease set BLE transmission time per byte.")
            
            # Check if transmission times fit within BLE connected duration
            if hasattr(self, 'ble_connected_duration_spin'):
                ble_connected_duration = self.ble_connected_duration_spin.value() * 1000  # Convert s to ms
                if (vlc_tx_duration + ble_tx_duration) > ble_connected_duration:
                    error_messages.append(
                        f"⚠️ Total transmission time ({vlc_tx_duration + ble_tx_duration:.3f} ms) "
                        f"exceeds BLE Connected Duration ({ble_connected_duration:.0f} ms)!\n"
                        f"VLC TX: {vlc_tx_duration:.3f} ms + BLE TX: {ble_tx_duration:.3f} ms"
                    )
            
            # Show errors if any
            if error_messages:
                # Set label color to red to indicate error
                self.vlc_tx_duration_label.setStyleSheet("font-size: 11px; color: red; font-weight: bold; font-style: italic;")
                self.ble_tx_duration_label.setStyleSheet("font-size: 11px; color: red; font-weight: bold; font-style: italic;")
                
                # Only show dialog if user is actively changing values (not during initialization)
                if hasattr(self, '_initialization_complete'):
                    QMessageBox.warning(self, "Validation Error", "\n\n".join(error_messages))
            else:
                # Reset to normal color
                self.vlc_tx_duration_label.setStyleSheet("font-size: 11px; color: #1976d2; font-style: italic;")
                self.ble_tx_duration_label.setStyleSheet("font-size: 11px; color: #1976d2; font-style: italic;")
            
            # Update dynamic prediction if model is loaded
            self.update_dynamic_prediction()
            
        except Exception as e:
            print(f"Error updating transmission durations: {e}")

    def create_ble_parameters_group(self, layout):
        """Create BLE parameters input group"""
        self.ble_params_group = QGroupBox("BLE Parameters (Node 1)")
        group_layout = QVBoxLayout()

        # TxPower input
        tx_power_layout = QHBoxLayout()
        tx_power_layout.addWidget(QLabel("Tx Power (dBm):"))
        self.tx_power_spin = QSpinBox()
        self.tx_power_spin.setRange(-40, 8)
        self.tx_power_spin.setValue(0)
        tx_power_layout.addWidget(self.tx_power_spin)
        group_layout.addLayout(tx_power_layout)

        # Advertising Interval input
        adv_interval_layout = QHBoxLayout()
        adv_interval_layout.addWidget(QLabel("Advertising Interval (ms):"))
        self.adv_interval_spin = QSpinBox()
        self.adv_interval_spin.setRange(20, 10240)
        self.adv_interval_spin.setValue(40)
        adv_interval_layout.addWidget(self.adv_interval_spin)
        group_layout.addLayout(adv_interval_layout)

        # Connection Interval input
        conn_interval_layout = QHBoxLayout()
        conn_interval_layout.addWidget(QLabel("Connection Interval (ms):"))
        self.conn_interval_spin = QSpinBox()
        self.conn_interval_spin.setRange(7, 4000)
        self.conn_interval_spin.setValue(45)
        conn_interval_layout.addWidget(self.conn_interval_spin)
        group_layout.addLayout(conn_interval_layout)

        self.ble_params_group.setLayout(group_layout)
        layout.addWidget(self.ble_params_group)

    def create_communication_group(self, layout):
        self.communication_group = QGroupBox("Payload (Node 1)")
        group_layout = QVBoxLayout()

        # VLC configuration
        vlc_layout = QHBoxLayout()
        vlc_layout.addWidget(QLabel("VLC Payload (Byte):"))
        self.vlc_volume = QSpinBox()
        self.vlc_volume.setRange(0, 1000000)
        self.vlc_volume.setValue(0)
        self.vlc_volume.valueChanged.connect(self.update_transmission_durations)
        vlc_layout.addWidget(self.vlc_volume)
        group_layout.addLayout(vlc_layout)
        
        # VLC time per byte
        vlc_time_layout = QHBoxLayout()
        vlc_time_layout.addWidget(QLabel("  └─ Time per VLC Byte (ms/byte):"))
        self.time_per_vlc_byte_spin = QDoubleSpinBox()
        self.time_per_vlc_byte_spin.setRange(0.0, 1000.0)
        self.time_per_vlc_byte_spin.setValue(0.017)  # Default: 0.017 ms/byte (from dataset)
        self.time_per_vlc_byte_spin.setDecimals(3)
        self.time_per_vlc_byte_spin.setSingleStep(0.001)
        self.time_per_vlc_byte_spin.valueChanged.connect(self.update_transmission_durations)
        vlc_time_layout.addWidget(self.time_per_vlc_byte_spin)
        
        # VLC transmission duration tip (on same row)
        self.vlc_tx_duration_label = QLabel("(→ VLC TX: 0.000 ms)")
        self.vlc_tx_duration_label.setStyleSheet("font-size: 11px; color: #1976d2; font-style: italic;")
        vlc_time_layout.addWidget(self.vlc_tx_duration_label)
        
        group_layout.addLayout(vlc_time_layout)

        # BLE configuration
        ble_layout = QHBoxLayout()
        ble_layout.addWidget(QLabel("BLE Payload (Byte):"))
        self.ble_volume = QSpinBox()
        self.ble_volume.setRange(0, 1000000)
        self.ble_volume.setValue(0)
        self.ble_volume.valueChanged.connect(self.update_transmission_durations)
        ble_layout.addWidget(self.ble_volume)
        group_layout.addLayout(ble_layout)
        
        # BLE time per byte
        ble_time_layout = QHBoxLayout()
        ble_time_layout.addWidget(QLabel("  └─ Time per BLE Byte (ms/byte):"))
        self.time_per_ble_byte_spin = QDoubleSpinBox()
        self.time_per_ble_byte_spin.setRange(0.0, 1000.0)
        self.time_per_ble_byte_spin.setValue(0.041)  # Default: 0.041 ms/byte (from dataset)
        self.time_per_ble_byte_spin.setDecimals(3)
        self.time_per_ble_byte_spin.setSingleStep(0.001)
        self.time_per_ble_byte_spin.valueChanged.connect(self.update_transmission_durations)
        ble_time_layout.addWidget(self.time_per_ble_byte_spin)
        
        # BLE transmission duration tip (on same row)
        self.ble_tx_duration_label = QLabel("(→ BLE TX: 0.000 ms)")
        self.ble_tx_duration_label.setStyleSheet("font-size: 11px; color: #1976d2; font-style: italic;")
        ble_time_layout.addWidget(self.ble_tx_duration_label)
        
        group_layout.addLayout(ble_time_layout)

        self.communication_group.setLayout(group_layout)
        layout.addWidget(self.communication_group)

    def create_prediction_group(self, layout):
        group = QGroupBox("Prediction")
        group_layout = QVBoxLayout()

        # Model selection and prediction button in the same row
        model_button_layout = QHBoxLayout()
        model_button_layout.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.models.keys()))
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_button_layout.addWidget(self.model_combo)
        
        # Add prediction button in the same row with beautiful styling
        self.predict_button = QPushButton("▶ Start Prediction")
        self.predict_button.setFixedSize(160, 35)
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.predict_button.clicked.connect(self.toggle_simulation)
        self.is_running = False  # Track simulation state
        model_button_layout.addWidget(self.predict_button)
        group_layout.addLayout(model_button_layout)

        # Model info display
        self.model_info_label = QLabel("Model: Loading...")
        self.model_info_label.setStyleSheet("font-size: 12px; color: #666;")
        self.model_info_label.setWordWrap(True)
        group_layout.addWidget(self.model_info_label)

        # Update model info for the first model
        if self.models:
            self.update_model_info(list(self.models.keys())[0])

        # Dynamic Energy Indicators (2 columns layout)
        indicator_group = QGroupBox("Indicators")
        indicator_main_layout = QHBoxLayout()  # Main horizontal layout for 2 columns
        
        # Column 1: Original Indicators (Nodes)
        column1_layout = QVBoxLayout()
        
        # Nodes Energy Indicator
        nodes_energy_layout = QHBoxLayout()
        nodes_energy_layout.addWidget(QLabel("Nodes Energy:"))
        self.nodes_energy_indicator_value = QLabel("-- mWh")
        self.nodes_energy_indicator_value.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.nodes_energy_indicator_bar = MeterWidget(max_value=10.0, unit="mWh")  # 0-10 mWh
        nodes_energy_layout.addWidget(self.nodes_energy_indicator_value)
        nodes_energy_layout.addWidget(self.nodes_energy_indicator_bar)
        nodes_energy_layout.addStretch(1)
        column1_layout.addLayout(nodes_energy_layout)

        # Nodes Power Indicator
        nodes_power_layout = QHBoxLayout()
        nodes_power_layout.addWidget(QLabel("Nodes Power:"))
        self.nodes_power_indicator_value = QLabel("-- mW")
        self.nodes_power_indicator_value.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.nodes_power_indicator_bar = MeterWidget(max_value=100.0, unit="mW")  # 0-100 mW
        nodes_power_layout.addWidget(self.nodes_power_indicator_value)
        nodes_power_layout.addWidget(self.nodes_power_indicator_bar)
        nodes_power_layout.addStretch(1)
        column1_layout.addLayout(nodes_power_layout)

        # Current Indicator
        current_layout = QHBoxLayout()
        current_layout.addWidget(QLabel("Current (N1):"))
        self.current_indicator_value = QLabel("-- mA")
        self.current_indicator_value.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.current_indicator_bar = MeterWidget(max_value=20.0, unit="mA")  # 0-20 mA
        current_layout.addWidget(self.current_indicator_value)
        current_layout.addWidget(self.current_indicator_bar)
        current_layout.addStretch(1)
        column1_layout.addLayout(current_layout)
        
        # Column 2: Total Indicators (Nodes + Minilamps)
        column2_layout = QVBoxLayout()
        
        # Total Energy Indicator (Nodes + Minilamps)
        total_energy_layout = QHBoxLayout()
        total_energy_layout.addWidget(QLabel("Total Energy:"))
        self.total_energy_indicator_value = QLabel("-- mWh")
        self.total_energy_indicator_value.setStyleSheet("font-weight: bold; font-size: 16px; color: #1976d2;")
        self.total_energy_indicator_bar = MeterWidget(max_value=10.0, unit="mWh")  # 0-10 mWh
        total_energy_layout.addWidget(self.total_energy_indicator_value)
        total_energy_layout.addWidget(self.total_energy_indicator_bar)
        total_energy_layout.addStretch(1)
        column2_layout.addLayout(total_energy_layout)

        # Total Power Indicator (Nodes + Minilamps)
        total_power_layout = QHBoxLayout()
        total_power_layout.addWidget(QLabel("Total Power:"))
        self.total_power_indicator_value = QLabel("-- mW")
        self.total_power_indicator_value.setStyleSheet("font-weight: bold; font-size: 16px; color: #1976d2;")
        self.total_power_indicator_bar = MeterWidget(max_value=100.0, unit="mW")  # 0-100 mW
        total_power_layout.addWidget(self.total_power_indicator_value)
        total_power_layout.addWidget(self.total_power_indicator_bar)
        total_power_layout.addStretch(1)
        column2_layout.addLayout(total_power_layout)
        
        # Add both columns to main layout
        indicator_main_layout.addLayout(column1_layout)
        indicator_main_layout.addLayout(column2_layout)
        
        indicator_group.setLayout(indicator_main_layout)
        group_layout.addWidget(indicator_group)  # Add the indicator group to the main prediction group layout

        group.setLayout(group_layout)
        layout.addWidget(group)

    def on_model_changed(self, model_name):
        """Handle model selection change"""
        if model_name in self.models:
            self.current_model_data = self.models[model_name]
            self.current_model_name = model_name
            self.update_model_info(model_name)
            print(f"Switched to model: {model_name}")
        else:
            print(f"Model {model_name} not found in available models")

    def get_baseline_currents_from_model(self):
        """
        Get baseline currents from the selected model by predicting with no payloads
        Returns: dict {state: current_mA} where state is 0-3 (app format)
        """
        try:
            if not hasattr(self, 'current_model_data') or self.current_model_data is None:
                raise ValueError("No model selected. Please select a model first.")
            
            print("\n" + "="*60)
            print("🤖 GETTING BASELINE CURRENTS FROM MODEL")
            print("="*60)
            print(f"Using model: {getattr(self, 'current_model_name', 'Unknown')}")
            
            baseline_currents = {}
            
            # Generate baseline predictions for each state (no payloads)
            for state in range(4):  # App uses 0-3, model uses 1-4
                model_state = state + 1  # Convert to model format (1-4)
                
                # Predict with baseline inputs: state + no payloads
                predicted_current = self.predict_with_payload_logic(
                    self.current_model_data,
                    state=model_state,
                    duration=100,  # Duration doesn't affect current prediction
                    voltage=self.voltage_mV,
                    vlc_payload=0,  # Baseline: no VLC payload
                    ble_payload=0,  # Baseline: no BLE payload
                    guest_id=1,
                    tx_power=0,
                    adv_interval=40,
                    conn_interval=45,
                    comm_mode=0
                )
                
                baseline_currents[state] = round(predicted_current, 1)
                print(f"  State {state} (model state {model_state}): {predicted_current:.1f} mA")
            
            print(f"\n✅ Baseline currents: {baseline_currents}")
            print("="*60)
            
            return baseline_currents
            
        except Exception as e:
            print(f"❌ Error getting baseline currents from model: {e}")
            raise ValueError(f"Cannot get baseline currents from model: {str(e)}")

    def update_model_info(self, model_name):
        """Update model info display and show baseline currents for reference"""
        try:
            if model_name not in self.models:
                self.model_info_label.setText("Model not found")
                return
            
            model_data = self.models[model_name]
            r2_score = model_data.get('r2_score', 0)
            model_type = type(model_data['model']).__name__
            
            # Get baseline currents from the selected model for reference
            self.current_model_data = model_data
            self.current_model_name = model_name
            
            try:
                baseline_currents_dict = self.get_baseline_currents_from_model()
                # baseline_currents_dict is already in mA format: {0: current_mA, 1: current_mA, ...}
                print(f"📊 Baseline Currents (mA): {baseline_currents_dict}")
                baseline_text = f"Baseline(VLC=0,BLE=0): S1:{baseline_currents_dict[0]:.1f}mA, S2:{baseline_currents_dict[1]:.1f}mA, S3:{baseline_currents_dict[2]:.1f}mA, S4:{baseline_currents_dict[3]:.1f}mA"
            except Exception as e:
                print(f"Error getting baseline currents: {e}")
                baseline_text = f"Error getting baseline currents: {str(e)}"
            
            info_text = f"R²: {r2_score:.4f}, Type: {model_type}, {baseline_text}"
            self.model_info_label.setText(info_text)
            
            # Also trigger dynamic update to show initial values
            self.update_dynamic_prediction()
            
        except Exception as e:
            self.model_info_label.setText(f"Error loading model info: {str(e)}")
            print(f"Error in update_model_info: {e}")

    def predict_with_selected_model(self, state, duration, voltage, vlc_payload, ble_payload, guest_id=1):
        """Predict current using the currently selected model"""
        try:
            if not hasattr(self, 'current_model_data') or self.current_model_data is None:
                raise ValueError("No model selected. Please select a model first.")
            
            return self.predict_with_current_model(
                self.current_model_data, state, duration, voltage, vlc_payload, ble_payload, guest_id
            )
        except Exception as e:
            print(f"Error in prediction: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def predict_with_current_model(self, model_data, state, duration, voltage, vlc_payload, ble_payload, guest_id):
        """Predict using current model with proper model-based prediction"""
        try:
            print(f"🔍 Prediction request: State {state}, VLC={vlc_payload}, BLE={ble_payload}")
            
            # Use the trained model for all predictions (baseline or with payloads)
            input_data = pd.DataFrame({
                'State': [state],  # State should be 1-4 for model input
                'VLC_Payload': [vlc_payload],
                'BLE_Payload': [ble_payload]
            })
            
            print(f"🔍 Model prediction input: State {state}, VLC={vlc_payload}, BLE={ble_payload}")
            
            model = model_data['model']
            scaler = model_data.get('scaler', None)
            
            if scaler is not None:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
            else:
                prediction = model.predict(input_data)[0]
            
            print(f"🔍 Model prediction result: {prediction:.1f} µA")
            # Convert from µA to mA for consistency with app units
            prediction_mA = prediction / 1000.0
            print(f"🔍 Converted to: {prediction_mA:.1f} mA")
            return prediction_mA
            
        except Exception as e:
            print(f"Error in current model prediction: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Model prediction failed: {str(e)}")

    def calculate_total_energy_baseline(self, state_durations, voltage=5000):
        """
        Calculate total energy using baseline currents from the selected model
        Args:
            state_durations: dict {state: duration_ms} where state is 0-3 (app format)
            voltage: voltage in mV
        Returns:
            dict with energy breakdown and total
        """
        try:
            # Get baseline currents from the selected model
            baseline_currents = self.get_baseline_currents_from_model()
        except Exception as e:
            print(f"❌ Error getting baseline currents: {e}")
            raise ValueError(f"Cannot calculate energy: {str(e)}")
        
        results = {
            'state_energies': {},
            'total_energy_mj': 0,
            'breakdown': []
        }
        
        total_energy = 0
        
        print("\n" + "="*60)
        print("⚡ BASELINE ENERGY CALCULATION")
        print("="*60)
        print(f"Voltage: {voltage} mV")
        print(f"State Durations: {state_durations}")
        print(f"Baseline Currents: {baseline_currents}")
        
        for state, duration in state_durations.items():
            if duration > 0:
                current = baseline_currents.get(state, 0)
                # Energy = Current(mA) × Duration(ms) × Voltage(mV) / 1,000,000 = mJ
                energy_mj = current * duration * voltage / 1_000_000
                
                results['state_energies'][state] = energy_mj
                total_energy += energy_mj
                
                results['breakdown'].append({
                    'state': state,
                    'current_ma': current,
                    'duration_ms': duration,
                    'voltage_mv': voltage,
                    'energy_mj': energy_mj
                })
                
                print(f"  State {state}: {current:.1f} mA × {duration} ms × {voltage} mV = {energy_mj:.2f} mJ")
        
        results['total_energy_mj'] = total_energy
        print(f"\n🔋 TOTAL ENERGY: {total_energy:.2f} mJ")
        print("="*60)
        
        return results

    def run_baseline_energy_simulation(self, num_devices):
        """Run energy simulation using baseline currents"""
        try:
            print("\n" + "="*80)
            print("🚀 BASELINE ENERGY SIMULATION")
            print("="*80)
            
            total_energy_all_devices = 0
            device_energy_results = []
            
            # Process each device
            for device_idx in range(num_devices):
                device_id = device_idx + 1
                device_config = self.device_configs.get(device_id, self.get_default_device_config())
                
                # Prepare state durations for this device
                state_durations = {}
                for state in range(4):
                    duration = float(device_config['state_durations'][state])
                    enabled = device_config['state_switches'][state]
                    state_durations[state] = duration if enabled else 0
                
                # Calculate energy for this device
                device_energy = self.calculate_total_energy_baseline(state_durations, self.voltage_mV)
                device_energy_results.append(device_energy)
                total_energy_all_devices += device_energy['total_energy_mj']
                
                print(f"\n📱 Device {device_id} Energy: {device_energy['total_energy_mj']:.2f} mJ")
            
            # Display summary
            print("\n" + "="*80)
            print("📊 ENERGY SIMULATION SUMMARY")
            print("="*80)
            print(f"Total Devices: {num_devices}")
            print(f"Total Energy (All Devices): {total_energy_all_devices:.2f} mJ")
            print(f"Average Energy per Device: {total_energy_all_devices/num_devices:.2f} mJ")
            
            # Update UI with results
            if hasattr(self, 'total_energy_label'):
                self.total_energy_label.setText(f"Total Energy: {total_energy_all_devices:.2f} mJ")
            
            # Create simplified visualization data
            self.create_baseline_visualization(device_energy_results, num_devices)
            
        except Exception as e:
            print(f"Error in baseline energy simulation: {e}")
            QMessageBox.warning(self, "Simulation Error", f"Error in baseline simulation: {str(e)}")

    def create_baseline_visualization(self, device_energy_results, num_devices):
        """Create simplified visualization for baseline energy calculation"""
        try:
            # Get baseline currents from the selected model
            try:
                baseline_currents_ua = self.get_baseline_currents_from_model()
                # Convert from µA to mA since model outputs µA but we need mA for calculations
                baseline_currents = [current / 1000.0 for current in baseline_currents_ua]
                print(f"📊 Baseline Currents (converted to mA): {[f'{c:.1f}' for c in baseline_currents]}")
            except Exception as e:
                print(f"❌ Error getting baseline currents: {e}")
                QMessageBox.warning(self, "Model Error", f"Cannot get baseline currents from model: {str(e)}")
                return
            
            # Create time series based on actual state durations
            # Calculate total duration from device configurations
            total_duration = 0
            for device_idx in range(num_devices):
                device_config = self.device_configs.get(device_idx + 1, self.get_default_device_config())
                device_duration = sum(
                    float(device_config['state_durations'][i]) 
                    for i in range(4) 
                    if device_config['state_switches'][i]
                )
                total_duration = max(total_duration, device_duration)
            
            # Use actual duration or minimum of 10 seconds
            max_duration = max(total_duration, 10.0)
            time_points = np.arange(0, max_duration, 0.1)
            print(f"📊 Visualization duration: {max_duration:.1f} seconds")
            
            # Calculate dynamic power across all devices based on actual state changes
            total_power_profile = np.zeros_like(time_points)
            
            for device_idx in range(num_devices):
                device_config = self.device_configs.get(device_idx + 1, self.get_default_device_config())
                
                # Create device power profile based on enabled states and their durations
                device_power_profile = np.zeros_like(time_points)
                current_time = 0.0
                
                for state in range(4):
                    duration = float(device_config['state_durations'][state])
                    enabled = device_config['state_switches'][state]
                    
                    if enabled and duration > 0:
                        state_end_time = current_time + duration
                        # Find time points in this state period
                        state_mask = (time_points >= current_time) & (time_points < state_end_time)
                        
                        # Calculate power for this state: Power = Current(mA) × Voltage(V) = mW
                        state_current = baseline_currents[state]
                        state_power = state_current * (self.voltage_mV / 1000.0)  # mW = mA × V
                        device_power_profile[state_mask] = state_power
                        
                        current_time = state_end_time
                
                # Add this device's power to the total
                total_power_profile += device_power_profile
            
            # Use dynamic power consumption for visualization
            power_consumption = total_power_profile
            
            # Create current consumption for each device with proper state-based currents
            current_consumption = []
            for device_idx in range(num_devices):
                device_config = self.device_configs.get(device_idx + 1, self.get_default_device_config())
                
                # Create time-based current profile based on enabled states and their durations
                device_current_profile = np.zeros_like(time_points)
                current_time = 0.0
                
                print(f"📊 Device {device_idx + 1} current profile:")
                
                for state in range(4):
                    duration = float(device_config['state_durations'][state])
                    enabled = device_config['state_switches'][state]
                    
                    if enabled and duration > 0:
                        state_end_time = current_time + duration
                        # Find time points in this state period
                        state_mask = (time_points >= current_time) & (time_points < state_end_time)
                        
                        # Use the appropriate baseline current for this state (in mA)
                        state_current = baseline_currents[state]
                        device_current_profile[state_mask] = state_current
                        
                        print(f"  State {state}: {current_time:.1f}s-{state_end_time:.1f}s -> {state_current:.1f} mA")
                        current_time = state_end_time
                
                current_consumption.append(device_current_profile)
            
            # Store simplified animation data
            avg_total_power = np.mean(total_power_profile) if len(total_power_profile) > 0 else 0
            self.animation_data = {
                'time_points': time_points,
                'power_consumption': power_consumption,
                'current_consumption': current_consumption,
                'state_boundaries': self.calculate_reference_boundaries(max_duration, num_devices),
                'device_boundaries': self.calculate_device_boundaries(max_duration, num_devices),
                'state_colors': ['#e6b3ff', '#cc66ff', '#9900ff', '#6600cc'],  # More distinct bright gradient purples (light to dark)
                'device_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'][:num_devices],
                'state_powers': [avg_total_power/4] * 4,  # Equal distribution for visualization
                'state_actual_durations': [max_duration/4] * 4,
                'state_speed_factors': [1.0] * 4,
                'num_devices': num_devices,
                'baseline_mode': True  # Flag to indicate baseline mode
            }
            
            # Initialize plots and start animation
            self.current_time = 0
            self._initialize_plots(num_devices)
            self.animation_timer.start()
            
            print("✅ Baseline visualization created successfully!")
            
        except Exception as e:
            print(f"Error creating baseline visualization: {e}")

    def calculate_reference_boundaries(self, max_duration, num_devices):
        """Calculate reference state boundaries based on first device configuration"""
        if num_devices > 0:
            device_config = self.device_configs.get(1, self.get_default_device_config())
            boundaries = []
            current_time = 0.0
            
            for state in range(4):
                duration = float(device_config['state_durations'][state])
                enabled = device_config['state_switches'][state]
                
                if enabled and duration > 0:
                    end_time = current_time + duration
                    boundaries.append((current_time, end_time))
                    current_time = end_time
                else:
                    boundaries.append((current_time, current_time))
            
            return boundaries
        else:
            # Default boundaries
            return [(0, max_duration/4), (max_duration/4, max_duration/2), 
                   (max_duration/2, 3*max_duration/4), (3*max_duration/4, max_duration)]

    def calculate_device_boundaries(self, max_duration, num_devices):
        """Calculate state boundaries for each device"""
        device_boundaries = []
        
        for device_idx in range(num_devices):
            device_config = self.device_configs.get(device_idx + 1, self.get_default_device_config())
            boundaries = []
            current_time = 0.0
            
            for state in range(4):
                duration = float(device_config['state_durations'][state])
                enabled = device_config['state_switches'][state]
                
                if enabled and duration > 0:
                    end_time = current_time + duration
                    boundaries.append((current_time, end_time))
                    current_time = end_time
                else:
                    boundaries.append((current_time, current_time))
            
            device_boundaries.append(boundaries)
        
        return device_boundaries

    def toggle_simulation(self):
        """Toggle between start and stop simulation"""
        if not self.is_running:
            # Start simulation
            self.start_simulation()
        else:
            # Stop simulation and reset
            self.stop_simulation()
    
    def stop_simulation(self):
        """Stop the running simulation"""
        try:
            # Stop the animation timer
            if hasattr(self, 'animation_timer'):
                self.animation_timer.stop()
            
            # Reset simulation state
            self.is_running = False
            self.current_time = 0
            
            # Update button to "Start" state
            self.predict_button.setText("▶ Start Prediction")
            self.predict_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: bold;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
            
            print("🛑 Simulation stopped")
            
        except Exception as e:
            print(f"Error stopping simulation: {e}")
    
    def start_simulation(self):
        try:
            # Save current UI state to the currently selected node
            self.save_current_ui_to_device_config(self.current_config_device)

            num_devices = self.total_devices.value()
            
            # Update simulation state
            self.is_running = True
            self.current_time = 0
            
            # Update button to "Stop" state
            self.predict_button.setText("⏹ Stop Prediction")
            self.predict_button.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: bold;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
                QPushButton:pressed {
                    background-color: #c41700;
                }
            """)
            
            # Use dynamic energy calculation with payload-based current prediction
            self.run_dynamic_energy_simulation(num_devices)
            return

        except Exception as e:
            # Reset button state on error
            self.is_running = False
            self.predict_button.setText("▶ Start Prediction")
            self.predict_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: bold;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
            QMessageBox.warning(self, "Simulation Error", f"Error starting simulation: {str(e)}")

    def update_animation(self):
        if self.animation_data is None:
            return

        # Clear previous plots
        self.power_ax.clear()
        for ax in self.current_axes:
            ax.clear()

        # Get data for plotting
        time_points = self.animation_data['time_points']
        power_consumption = self.animation_data['power_consumption']
        current_consumption = self.animation_data['current_consumption']

        # Plot total power in bright red (sum of all nodes) - data is already in mW
        current_time_mask = time_points <= self.current_time
        self.power_ax.plot(time_points[current_time_mask], power_consumption[current_time_mask],
                           color='#ff0000', linewidth=2, label='Total Power')

        # Add state boundary lines and plot current for each device using device-specific boundaries
        epsilon_boundary = 1e-6
        device_boundaries = self.animation_data.get('device_boundaries', [])

        # Plot current for each device with its own state boundaries and colors
        for device_idx, ax in enumerate(self.current_axes):
            if device_idx < len(device_boundaries):
                device_state_boundaries = device_boundaries[device_idx]

                # Add device-specific state boundary lines
                for i, (start_time, end_time) in enumerate(device_state_boundaries):
                    duration = end_time - start_time
                    if duration <= 0:
                        continue

                    if i > 0 and start_time <= self.current_time:
                        ax.axvline(x=start_time, color='gray', linestyle='--', alpha=0.5)
                        # Also add to power plot (use reference boundaries)
                        if device_idx == 0:  # Only add once from first device
                            self.power_ax.axvline(x=start_time, color='gray', linestyle='--', alpha=0.5)

                    # Convert current from µA to mA for plotting
                    current_consumption_mA = current_consumption[device_idx] / 1000.0

                    if end_time - epsilon_boundary <= self.current_time:  # Check if state is fully covered by current_time
                        # Plot complete state - data is already in mA
                        state_mask_plot = (time_points >= start_time - epsilon_boundary) & (
                                time_points <= end_time + epsilon_boundary)

                        ax.plot(time_points[state_mask_plot], current_consumption[device_idx][state_mask_plot],
                                color=self.animation_data['state_colors'][i],
                                label=f'S{i + 1}')

                    elif start_time - epsilon_boundary <= self.current_time:  # Check if state has started
                        # Plot partial state - data is already in mA  
                        state_mask_plot = (time_points >= start_time - epsilon_boundary) & (
                                time_points <= self.current_time + epsilon_boundary)

                        ax.plot(time_points[state_mask_plot], current_consumption[device_idx][state_mask_plot],
                                color=self.animation_data['state_colors'][i],
                                label=f'S{i + 1}')

        # Update plot properties for power
        self.power_ax.set_xlabel('Time (s)')  # Re-adding this as clear() removes it
        self.power_ax.set_ylabel('Power (mW)')
        self.power_ax.set_title('Total Nodes Power')
        # self.power_ax.grid(True)  # Removed background grid lines
        # Only show legend if there are plotted lines
        if len(self.power_ax.get_lines()) > 0:
            self.power_ax.legend(loc='upper right')

        # Set same time scale as current plots
        max_time = self.animation_data['time_points'][-1] if len(self.animation_data['time_points']) > 0 else 10
        self.power_ax.set_xlim(0, max_time)

        # Update plot properties for current (each device has its own subplot)
        for device_idx, ax in enumerate(self.current_axes):
            if device_idx == len(self.current_axes) - 1:  # Only set xlabel for the very bottom subplot
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)

            ax.set_ylabel('Current (mA)')
            # Only show "Current" title for the first device, others have no title
            if device_idx == 0:
                ax.set_title('Current')  # Only first current plot has title
            else:
                ax.set_title('')  # No title for additional current plots

            # Add device ID number in the top-left corner
            device_id = device_idx + 1  # Device IDs start from 1
            ax.text(0.02, 0.95, f'N{device_id}', transform=ax.transAxes,
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # ax.grid(True)  # Removed background grid lines
            ax.legend(loc='upper right')

            # Set same time scale as power plot
            max_time = self.animation_data['time_points'][-1] if len(self.animation_data['time_points']) > 0 else 10
            ax.set_xlim(0, max_time)

        # Update current time with simple increment
        # Timer runs every 10ms (0.01 real seconds)
        # Want 50 simulation seconds per 1 real second for much faster speed
        # So: 50 simulation seconds ÷ 100 ticks per real second = 0.5 simulation seconds per tick
        time_increment_per_tick = 0.5  # 50 simulation seconds per real second (very fast)
        self.current_time += time_increment_per_tick

        # Get the index for the current time to fetch current power/current values
        # Using np.clip to ensure the index stays within bounds
        current_time_index = np.clip(np.searchsorted(time_points, self.current_time), 0, len(time_points) - 1)

        # Calculate current power and current
        current_power_value = power_consumption[current_time_index]

        # Calculate total energy consumption (mW-seconds, then convert to mW-hours)
        # Sum power up to the current time, multiply by time step, convert seconds to hours
        epsilon_boundary = 1e-6
        power_up_to_current_time = power_consumption[time_points <= self.current_time + epsilon_boundary]
        time_step = time_points[1] - time_points[0] if len(time_points) > 1 else 0.1  # Fallback if only one time point
        total_energy_consumption_mWh = (np.sum(power_up_to_current_time) * time_step) / 3600

        # Determine if simulation has ended for the label
        dynamic_suffix = " (Dynamic)"
        if self.current_time >= time_points[-1] - epsilon_boundary:  # Use epsilon for final time check
            self.animation_timer.stop()
            dynamic_suffix = ""  # Remove " (Dynamic)" suffix when simulation ends
            
            # Reset button state when animation completes
            self.is_running = False
            self.predict_button.setText("▶ Start Prediction")
            self.predict_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: bold;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)

        # --- Update Dynamic Indicators ---
        # Get max values from animation data for scaling bars
        max_power_overall = max(power_consumption) if power_consumption.size > 0 else 0.1
        max_current_overall = 0.1
        for dev_current_array in current_consumption:
            if dev_current_array.size > 0:
                max_current_overall = max(max_current_overall, np.max(dev_current_array))
        max_energy_overall = (np.sum(power_consumption) * time_step) / 3600 if power_consumption.size > 0 else 0.1

        # Ensure max values are not zero for division
        max_power_overall = max(0.1, max_power_overall)
        max_current_overall = max(0.1, max_current_overall)
        max_energy_overall = max(0.1, max_energy_overall)

        # Calculate minilamp contributions
        num_minilamps = self.total_minilamps.value() if hasattr(self, 'total_minilamps') else 0
        minilamp_power_per_lamp_mW = self.minilamp_power_mw.value() if hasattr(self, 'minilamp_power_mw') else 100.0
        
        # Calculate total minilamp power
        total_minilamp_power_mW = num_minilamps * minilamp_power_per_lamp_mW
        
        # Calculate minilamp energy: duration = current_time (they run for the same duration as simulation)
        # Energy(mWh) = Power(mW) × Time(hours)
        minilamp_energy_mWh = (total_minilamp_power_mW * self.current_time) / 3600.0
        
        # Calculate totals (nodes + minilamps)
        total_energy_with_minilamps_mWh = total_energy_consumption_mWh + minilamp_energy_mWh
        total_power_with_minilamps_mW = current_power_value + total_minilamp_power_mW

        # Update Nodes Energy Indicator - data is already in mWh
        self.nodes_energy_indicator_value.setText(f"{total_energy_consumption_mWh:.2f} mWh")
        self.nodes_energy_indicator_bar.setValue(total_energy_consumption_mWh)

        # Update Nodes Power Indicator - data is already in mW
        self.nodes_power_indicator_value.setText(f"{current_power_value:.2f} mW")
        self.nodes_power_indicator_bar.setValue(current_power_value)

        # Update Total Energy Indicator (Nodes + Minilamps)
        self.total_energy_indicator_value.setText(f"{total_energy_with_minilamps_mWh:.2f} mWh")
        self.total_energy_indicator_bar.setValue(total_energy_with_minilamps_mWh)

        # Update Total Power Indicator (Nodes + Minilamps)
        self.total_power_indicator_value.setText(f"{total_power_with_minilamps_mW:.2f} mW")
        self.total_power_indicator_bar.setValue(total_power_with_minilamps_mW)

        # Update Current Indicator - data is already in mA
        if current_consumption and len(current_consumption) > 0:
            current_time_index = np.clip(np.searchsorted(time_points, self.current_time), 0, len(time_points) - 1)
            current_val_id1 = current_consumption[0][current_time_index]
            # Data is already in mA, no conversion needed
            self.current_indicator_value.setText(f"{current_val_id1:.2f} mA")
            self.current_indicator_bar.setValue(current_val_id1)
        else:
            self.current_indicator_value.setText("-- mA")
            self.current_indicator_bar.setValue(0)

        # Force update of all indicators
        self.nodes_energy_indicator_bar.update()
        self.nodes_power_indicator_bar.update()
        self.total_energy_indicator_bar.update()
        self.total_power_indicator_bar.update()
        self.current_indicator_bar.update()

        self.figure.tight_layout()
        self.canvas.draw()

    def _initialize_plots(self, num_devices):
        """Initializes or re-initializes the matplotlib plots and axes."""
        self.figure.clear()

        # Define GridSpec: power plot (top), current plots (bottom)
        gs_main = GridSpec(2, 1, figure=self.figure, height_ratios=[1, 1])  # Power and all current plots each take half

        self.power_ax = self.figure.add_subplot(gs_main[0, 0])
        self.power_ax.set_xlabel('Time (s)')
        self.power_ax.set_ylabel('Power (mW)')
        self.power_ax.set_title('Total Nodes Power')  # Power from all nodes
        # self.power_ax.grid(True)  # Removed background grid lines
        # Don't show legend initially when there's no data
        self.power_ax.set_xlim(0, 10)  # Default initial x-limit
        self.power_ax.set_ylim(0, 100)  # Default initial y-limit for mW

        # Create a sub-GridSpec for the current plots within the bottom half
        gs_current = gs_main[1, 0].subgridspec(num_devices, 1,
                                               hspace=0.1)  # num_devices rows for current plots, minimal hspace

        self.current_axes = []
        for i in range(num_devices):
            ax = self.figure.add_subplot(gs_current[i, 0])
            if i < num_devices - 1:  # Only set xlabel for the very bottom subplot
                ax.set_xlabel('')  # Remove xlabel for intermediate current plots
                ax.tick_params(labelbottom=False)  # Remove x-axis tick labels for intermediate plots

            ax.set_ylabel('Current (mA)')
            # Only show "Current" title for the first device, others have no title
            if i == 0:
                ax.set_title('Current')  # Only first current plot has title
            else:
                ax.set_title('')  # No title for additional current plots

            # Add device ID number in the top-left corner
            device_id = i + 1  # Device IDs start from 1
            ax.text(0.02, 0.95, f'N{device_id}', transform=ax.transAxes,
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # ax.grid(True)  # Removed background grid lines
            # Don't show legend initially when there's no data
            ax.set_xlim(0, 10)  # Default initial x-limit
            ax.set_ylim(0, 20)  # Default initial y-limit for mA
            self.current_axes.append(ax)

        self.figure.tight_layout()
        self.canvas.draw()  # Draw immediately

    def _get_color_for_percentage(self, percentage):
        """Helper method to get color based on percentage."""
        if percentage < 33:
            return "#2ecc71"  # Green
        elif percentage < 66:
            return "#f1c40f"  # Yellow
        else:
            return "#e74c3c"  # Red

    def _get_darker_color_for_percentage(self, percentage):
        """Helper method to get darker shade of color based on percentage."""
        if percentage < 33:
            return "#27ae60"  # Dark Green
        elif percentage < 66:
            return "#f39c12"  # Dark Yellow
        else:
            return "#c0392b"  # Dark Red

    def loadData(self, filename):
        try:
            with open(filename, 'r') as f:
                self.animation_data = json.load(f)

            # Data is loaded as-is (µA, mW, mWh) - conversion happens during display
            # No conversion needed at load time

            # Store the maximum values for scaling
            self.max_current_ma = 0
            self.max_energy_mwh = 0
            for current_data in self.animation_data['current_consumption']:
                # Convert µA to mA for max calculation
                current_data_ma = [x / 1000.0 for x in current_data]
                self.max_current_ma = max(self.max_current_ma, max(current_data_ma))
            for energy_data in self.animation_data['energy_consumption']:
                self.max_energy_mwh = max(self.max_energy_mwh, max(energy_data))

            self.plotData()
            self.animation_timer.start(20)  # Update more frequently (50Hz instead of 20Hz)

        except Exception as e:
            print(f"Error loading data: {e}")

    def plotData(self):
        try:
            if not self.animation_data:
                return

            time_points = self.animation_data['time_points']
            num_devices = len(self.animation_data['device_ids'])

            # Create subplots
            self.figure.clear()

            # Calculate the number of rows needed (3 plots per device)
            num_rows = num_devices * 3

            # Create a grid of subplots
            self.axes = self.figure.subplots(num_rows, 1, sharex=True)
            if num_rows == 3:  # If only one device, axes will be 1D
                self.axes = [self.axes]

            # Adjust the spacing between subplots
            self.figure.subplots_adjust(hspace=0.3)

            for i in range(num_devices):
                # Plot energy consumption
                if len(self.animation_data['energy_consumption']) > i:
                    ax = self.axes[i * 3]
                    ax.plot(time_points, self.animation_data['energy_consumption'][i], 'b-')
                    ax.set_ylabel('Energy (mWh)')
                    ax.set_title('Energy')
                    ax.grid(True)
                    ax.tick_params(labelbottom=False)
                    ax.set_ylim([0, 4])

                # Plot power consumption
                if len(self.animation_data['power_consumption']) > i:
                    ax = self.axes[i * 3 + 1]
                    ax.plot(time_points, self.animation_data['power_consumption'][i], 'g-')
                    ax.set_ylabel('Power (mW)')
                    ax.set_title('Power')
                    ax.grid(True)
                    ax.tick_params(labelbottom=False)
                    ax.set_ylim([0, 100])

                # Plot current consumption
                if len(self.animation_data['current_consumption']) > i:
                    ax = self.axes[i * 3 + 2]
                    current_data = self.animation_data['current_consumption'][i]
                    ax.plot(time_points, current_data, 'r-')
                    ax.set_ylabel('Current (mA)')
                    # Only show "Current" title for the first device, others have no title
                    if i == 0:
                        ax.set_title('Current')  # Only first current plot has title
                    else:
                        ax.set_title('')  # No title for additional current plots

                    # Add device ID number in the top-left corner
                    device_id = i + 1  # Device IDs start from 1
                    ax.text(0.02, 0.95, f'N{device_id}', transform=ax.transAxes,
                            fontsize=10, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                    # ax.grid(True)  # Removed background grid lines
                    ax.legend(loc='upper right')
                    # Set y-axis limits for current plot
                    y_max = max(20, math.ceil(max(current_data)))
                    ax.set_ylim([0, y_max])
                    # Format y-axis to show proper mA values (divide by 1000 since data is already in mA)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
                    if i == num_devices - 1:
                        ax.set_xlabel('Time (s)')
                    else:
                        ax.tick_params(labelbottom=False)

            self.canvas.draw()

        except Exception as e:
            print(f"Error plotting data: {e}")

    def startPrediction(self):
        try:
            # Get the selected model and device
            selected_model = self.model_combo.currentText()
            selected_device = self.device_combo.currentText()

            if not selected_model or not selected_device:
                QMessageBox.warning(self, "Warning", "Please select both a model and a device.")
                return

            # Load and preprocess the data
            device_id = int(selected_device.split()[-1])  # Extract device number

            # Get the data for prediction
            if self.animation_data and len(self.animation_data['current_consumption']) >= device_id:
                current_data = self.animation_data['current_consumption'][device_id - 1]
                power_data = self.animation_data['power_consumption'][device_id - 1]
                energy_data = self.animation_data['energy_consumption'][device_id - 1]
                time_points = self.animation_data['time_points']

                # Create prediction plots
                self.figure.clear()
                self.axes = self.figure.subplots(3, 1, sharex=True)

                # Plot energy data and prediction
                self.axes[0].plot(time_points, energy_data, 'b-', label='Actual')
                self.axes[0].set_ylabel('Energy (mWh)')
                self.axes[0].set_title(f'Energy Consumption - Device {device_id}')
                # self.axes[0].grid(True)  # Removed background grid lines
                self.axes[0].legend()
                self.axes[0].set_ylim([0, 4])

                # Plot power data and prediction
                self.axes[1].plot(time_points, power_data, 'g-', label='Actual')
                self.axes[1].set_ylabel('Power (mW)')
                self.axes[1].set_title(f'Power Consumption - Device {device_id}')
                # self.axes[1].grid(True)  # Removed background grid lines
                self.axes[1].legend()
                self.axes[1].set_ylim([0, 100])

                # Plot current data and prediction
                self.axes[2].plot(time_points, current_data, 'r-', label='Actual')
                self.axes[2].set_ylabel('Current (mA)')
                self.axes[2].set_title(f'Current Consumption - Device {device_id}')
                # self.axes[2].grid(True)  # Removed background grid lines
                self.axes[2].legend()
                # Set y-axis limits for current plot
                y_max = max(20, math.ceil(max(current_data)))
                self.axes[2].set_ylim([0, y_max])
                # Format y-axis to show proper mA values (divide by 1000 since data is already in mA)
                self.axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
                self.axes[2].set_xlabel('Time (s)')

                # Add predictions if a model is selected
                if selected_model != "None":
                    # Simulate predictions (replace with actual model predictions)
                    pred_energy = [x * 1.1 for x in energy_data]  # Example: 10% higher
                    pred_power = [x * 1.1 for x in power_data]  # Example: 10% higher
                    pred_current = [x * 1.1 for x in current_data]  # Example: 10% higher

                    self.axes[0].plot(time_points, pred_energy, 'b--', label='Predicted')
                    self.axes[1].plot(time_points, pred_power, 'g--', label='Predicted')
                    self.axes[2].plot(time_points, pred_current, 'r--', label='Predicted')

                    for ax in self.axes:
                        ax.legend()

                self.canvas.draw()

            else:
                QMessageBox.warning(self, "Warning", "No data available for the selected device.")

        except Exception as e:
            print(f"Error in startPrediction: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred during prediction: {str(e)}")

    def create_dynamic_visualization(self, device_energy_results, num_devices):
        """Create visualization with dynamic current prediction based on actual payload values"""
        try:
            # Create time series based on actual state durations
            total_duration = 0
            for device_idx in range(num_devices):
                device_config = self.device_configs.get(device_idx + 1, self.get_default_device_config())
                device_duration = sum(
                    float(device_config['state_durations'][i]) 
                    for i in range(4) 
                    if device_config['state_switches'][i]
                )
                total_duration = max(total_duration, device_duration)
            
            # Use actual duration or minimum of 10 seconds
            max_duration = max(total_duration, 10.0)
            time_points = np.arange(0, max_duration, 0.1)
            print(f"📊 Visualization duration: {max_duration:.1f} seconds")
            
            # Calculate dynamic power across all devices based on actual state changes and payloads
            total_power_profile = np.zeros_like(time_points)
            
            for device_idx in range(num_devices):
                device_config = self.device_configs.get(device_idx + 1, self.get_default_device_config())
                
                # Get actual payload values from device configuration
                vlc_payload = device_config['vlc_payload']
                ble_payload = device_config['ble_payload']
                
                print(f"📊 Device {device_idx + 1} config: VLC={vlc_payload}, BLE={ble_payload}")
                
                # Create device power profile based on enabled states and their durations
                device_power_profile = np.zeros_like(time_points)
                current_time = 0.0
                
                for state in range(4):
                    duration = float(device_config['state_durations'][state])
                    enabled = device_config['state_switches'][state]
                    
                    if enabled and duration > 0:
                        # Special handling for State 3 (Operation) - 4 situations for power
                        if state == 2:  # State 3 in app format
                            # Get transmission parameters
                            time_per_vlc_byte = device_config.get('time_per_vlc_byte', 0.017)
                            time_per_ble_byte = device_config.get('time_per_ble_byte', 0.041)
                            ble_connected_duration_s = device_config.get('ble_connected_duration', 0.0)
                            
                            # Calculate transmission durations (convert to seconds)
                            vlc_tx_duration_s = (vlc_payload * time_per_vlc_byte) / 1000.0
                            ble_tx_duration_s = (ble_payload * time_per_ble_byte) / 1000.0
                            
                            state3_start_time = current_time
                            
                            # Situation 1: BLE Disconnected
                            duration_1 = duration - ble_connected_duration_s
                            if duration_1 > 0:
                                try:
                                    current_1 = self.predict_with_payload_logic(
                                        self.current_model_data, state=3, duration=duration_1*1000,
                                        voltage=self.voltage_mV, vlc_payload=0, ble_payload=0, guest_id=1,
                                        tx_power=device_config.get('tx_power', 0),
                                        adv_interval=device_config.get('adv_interval', 40),
                                        conn_interval=device_config.get('conn_interval', 45),
                                        comm_mode=1
                                    )
                                    power_1 = current_1 * (self.voltage_mV / 1000.0)
                                    situation1_end = current_time + duration_1
                                    situation1_mask = (time_points >= current_time) & (time_points < situation1_end)
                                    device_power_profile[situation1_mask] = power_1
                                    current_time = situation1_end
                                except: pass
                            
                            # Situation 2: BLE Connected Idle
                            duration_2 = ble_connected_duration_s - vlc_tx_duration_s - ble_tx_duration_s
                            if duration_2 > 0:
                                try:
                                    current_2 = self.predict_with_payload_logic(
                                        self.current_model_data, state=3, duration=duration_2*1000,
                                        voltage=self.voltage_mV, vlc_payload=0, ble_payload=0, guest_id=1,
                                        tx_power=device_config.get('tx_power', 0),
                                        adv_interval=device_config.get('adv_interval', 40),
                                        conn_interval=device_config.get('conn_interval', 45),
                                        comm_mode=2
                                    )
                                    power_2 = current_2 * (self.voltage_mV / 1000.0)
                                    situation2_end = current_time + duration_2
                                    situation2_mask = (time_points >= current_time) & (time_points < situation2_end)
                                    device_power_profile[situation2_mask] = power_2
                                    current_time = situation2_end
                                except: pass
                            
                            # Situation 3: BLE Transmission
                            if ble_tx_duration_s > 0 and ble_payload > 0:
                                try:
                                    current_3 = self.predict_with_payload_logic(
                                        self.current_model_data, state=3, duration=ble_tx_duration_s*1000,
                                        voltage=self.voltage_mV, vlc_payload=0, ble_payload=ble_payload, guest_id=1,
                                        tx_power=device_config.get('tx_power', 0),
                                        adv_interval=device_config.get('adv_interval', 40),
                                        conn_interval=device_config.get('conn_interval', 45),
                                        comm_mode=2
                                    )
                                    power_3 = current_3 * (self.voltage_mV / 1000.0)
                                    situation3_end = current_time + ble_tx_duration_s
                                    situation3_mask = (time_points >= current_time) & (time_points < situation3_end)
                                    device_power_profile[situation3_mask] = power_3
                                    current_time = situation3_end
                                except: pass
                            
                            # Situation 4: VLC Transmission
                            if vlc_tx_duration_s > 0 and vlc_payload > 0:
                                try:
                                    current_4 = self.predict_with_payload_logic(
                                        self.current_model_data, state=3, duration=vlc_tx_duration_s*1000,
                                        voltage=self.voltage_mV, vlc_payload=vlc_payload, ble_payload=0, guest_id=1,
                                        tx_power=device_config.get('tx_power', 0),
                                        adv_interval=device_config.get('adv_interval', 40),
                                        conn_interval=device_config.get('conn_interval', 45),
                                        comm_mode=2
                                    )
                                    power_4 = current_4 * (self.voltage_mV / 1000.0)
                                    situation4_end = current_time + vlc_tx_duration_s
                                    situation4_mask = (time_points >= current_time) & (time_points < situation4_end)
                                    device_power_profile[situation4_mask] = power_4
                                    current_time = situation4_end
                                except: pass
                            
                            # Update current_time to end of State 3
                            state_end_time = state3_start_time + duration
                            current_time = state_end_time
                            
                        else:
                            # States 0, 1, 3 (Idle, Advertisement, Sleep) - Standard single power
                            state_end_time = current_time + duration
                            # Find time points in this state period
                            state_mask = (time_points >= current_time) & (time_points < state_end_time)
                            
                            # Dynamically predict current for this state with smart payload logic
                            try:
                                predicted_current = self.predict_with_payload_logic(
                                    self.current_model_data,
                                    state=state + 1,  # Convert to model format (1-4)
                                    duration=duration,
                                    voltage=self.voltage_mV,
                                    vlc_payload=0,
                                    ble_payload=0,
                                    guest_id=1,
                                    tx_power=device_config.get('tx_power', 0),
                                    adv_interval=device_config.get('adv_interval', 40),
                                    conn_interval=device_config.get('conn_interval', 45),
                                    comm_mode=0
                                )
                                
                                # Calculate power for this state: Power = Current(mA) × Voltage(V) = mW
                                state_power = predicted_current * (self.voltage_mV / 1000.0)  # mW = mA × V
                                device_power_profile[state_mask] = state_power
                                
                            except Exception as e:
                                print(f"❌ Error predicting current for Device {device_idx + 1}, State {state}: {e}")
                                device_power_profile[state_mask] = 0  # Fallback to zero
                            
                            current_time = state_end_time
                
                # Add this device's power to the total
                total_power_profile += device_power_profile
            
            # Use dynamic power consumption for visualization
            power_consumption = total_power_profile
            
            # Create current consumption for each device with dynamic predictions
            current_consumption = []
            for device_idx in range(num_devices):
                device_config = self.device_configs.get(device_idx + 1, self.get_default_device_config())
                
                # Get actual payload values from device configuration
                vlc_payload = device_config['vlc_payload']
                ble_payload = device_config['ble_payload']
                
                # Create time-based current profile based on enabled states and their durations
                device_current_profile = np.zeros_like(time_points)
                current_time = 0.0
                
                print(f"📊 Device {device_idx + 1} dynamic current profile:")
                
                for state in range(4):
                    duration = float(device_config['state_durations'][state])
                    enabled = device_config['state_switches'][state]
                    
                    if enabled and duration > 0:
                        # Special handling for State 3 (Operation) - 4 situations
                        if state == 2:  # State 3 in app format
                            print(f"  📊 State 3 (Operation) - 4 Situations at {current_time:.3f}s:")
                            
                            # Get transmission parameters
                            time_per_vlc_byte = device_config.get('time_per_vlc_byte', 0.017)
                            time_per_ble_byte = device_config.get('time_per_ble_byte', 0.041)
                            ble_connected_duration_s = device_config.get('ble_connected_duration', 0.0)
                            
                            # Calculate transmission durations (convert to seconds)
                            vlc_tx_duration_s = (vlc_payload * time_per_vlc_byte) / 1000.0
                            ble_tx_duration_s = (ble_payload * time_per_ble_byte) / 1000.0
                            
                            state3_start_time = current_time
                            
                            # Situation 1: BLE Disconnected
                            duration_1 = duration - ble_connected_duration_s
                            if duration_1 > 0:
                                situation1_end = current_time + duration_1
                                situation1_mask = (time_points >= current_time) & (time_points < situation1_end)
                                
                                try:
                                    current_1 = self.predict_with_payload_logic(
                                        self.current_model_data, state=3, duration=duration_1*1000,
                                        voltage=self.voltage_mV, vlc_payload=0, ble_payload=0, guest_id=1,
                                        tx_power=device_config.get('tx_power', 0),
                                        adv_interval=device_config.get('adv_interval', 40),
                                        conn_interval=device_config.get('conn_interval', 45),
                                        comm_mode=1  # BLE Disconnected
                                    )
                                    device_current_profile[situation1_mask] = current_1
                                    print(f"    S1 (Disconnected): {current_time:.3f}s-{situation1_end:.3f}s -> {current_1:.3f} mA")
                                except Exception as e:
                                    print(f"    ❌ Error in Situation 1: {e}")
                                    device_current_profile[situation1_mask] = 0
                                
                                current_time = situation1_end
                            
                            # Situation 2: BLE Connected Idle
                            duration_2 = ble_connected_duration_s - vlc_tx_duration_s - ble_tx_duration_s
                            if duration_2 > 0:
                                situation2_end = current_time + duration_2
                                situation2_mask = (time_points >= current_time) & (time_points < situation2_end)
                                
                                try:
                                    current_2 = self.predict_with_payload_logic(
                                        self.current_model_data, state=3, duration=duration_2*1000,
                                        voltage=self.voltage_mV, vlc_payload=0, ble_payload=0, guest_id=1,
                                        tx_power=device_config.get('tx_power', 0),
                                        adv_interval=device_config.get('adv_interval', 40),
                                        conn_interval=device_config.get('conn_interval', 45),
                                        comm_mode=2  # BLE Connected
                                    )
                                    device_current_profile[situation2_mask] = current_2
                                    print(f"    S2 (BLE Idle): {current_time:.3f}s-{situation2_end:.3f}s -> {current_2:.3f} mA")
                                except Exception as e:
                                    print(f"    ❌ Error in Situation 2: {e}")
                                    device_current_profile[situation2_mask] = 0
                                
                                current_time = situation2_end
                            
                            # Situation 3: BLE Transmission
                            if ble_tx_duration_s > 0 and ble_payload > 0:
                                situation3_end = current_time + ble_tx_duration_s
                                situation3_mask = (time_points >= current_time) & (time_points < situation3_end)
                                
                                try:
                                    current_3 = self.predict_with_payload_logic(
                                        self.current_model_data, state=3, duration=ble_tx_duration_s*1000,
                                        voltage=self.voltage_mV, vlc_payload=0, ble_payload=ble_payload, guest_id=1,
                                        tx_power=device_config.get('tx_power', 0),
                                        adv_interval=device_config.get('adv_interval', 40),
                                        conn_interval=device_config.get('conn_interval', 45),
                                        comm_mode=2  # BLE Connected
                                    )
                                    device_current_profile[situation3_mask] = current_3
                                    print(f"    S3 (BLE TX): {current_time:.3f}s-{situation3_end:.3f}s -> {current_3:.3f} mA")
                                except Exception as e:
                                    print(f"    ❌ Error in Situation 3: {e}")
                                    device_current_profile[situation3_mask] = 0
                                
                                current_time = situation3_end
                            
                            # Situation 4: VLC Transmission
                            if vlc_tx_duration_s > 0 and vlc_payload > 0:
                                situation4_end = current_time + vlc_tx_duration_s
                                situation4_mask = (time_points >= current_time) & (time_points < situation4_end)
                                
                                try:
                                    current_4 = self.predict_with_payload_logic(
                                        self.current_model_data, state=3, duration=vlc_tx_duration_s*1000,
                                        voltage=self.voltage_mV, vlc_payload=vlc_payload, ble_payload=0, guest_id=1,
                                        tx_power=device_config.get('tx_power', 0),
                                        adv_interval=device_config.get('adv_interval', 40),
                                        conn_interval=device_config.get('conn_interval', 45),
                                        comm_mode=2  # BLE Connected
                                    )
                                    device_current_profile[situation4_mask] = current_4
                                    print(f"    S4 (VLC TX): {current_time:.3f}s-{situation4_end:.3f}s -> {current_4:.3f} mA")
                                except Exception as e:
                                    print(f"    ❌ Error in Situation 4: {e}")
                                    device_current_profile[situation4_mask] = 0
                                
                                current_time = situation4_end
                            
                            # Update current_time to end of State 3
                            state_end_time = state3_start_time + duration
                            current_time = state_end_time
                            
                        else:
                            # States 0, 1, 3 (Idle, Advertisement, Sleep) - Standard single current
                            state_end_time = current_time + duration
                            # Find time points in this state period
                            state_mask = (time_points >= current_time) & (time_points < state_end_time)
                            
                            # Dynamically predict current for this state with smart payload logic
                            try:
                                predicted_current = self.predict_with_payload_logic(
                                    self.current_model_data,
                                    state=state + 1,  # Convert to model format (1-4)
                                    duration=duration,
                                    voltage=self.voltage_mV,
                                    vlc_payload=0,  # Always 0 for non-operation states
                                    ble_payload=0,  # Always 0 for non-operation states
                                    guest_id=1,
                                    tx_power=device_config.get('tx_power', 0),
                                    adv_interval=device_config.get('adv_interval', 40),
                                    conn_interval=device_config.get('conn_interval', 45),
                                    comm_mode=0
                                )
                                
                                device_current_profile[state_mask] = predicted_current
                                
                                print(f"  State {state}: {current_time:.1f}s-{state_end_time:.1f}s -> {predicted_current:.1f} mA")
                                
                            except Exception as e:
                                print(f"❌ Error predicting current for Device {device_idx + 1}, State {state}: {e}")
                                device_current_profile[state_mask] = 0  # Fallback to zero
                            
                            current_time = state_end_time
                
                current_consumption.append(device_current_profile)
            
            # Store dynamic animation data
            avg_total_power = np.mean(total_power_profile) if len(total_power_profile) > 0 else 0
            self.animation_data = {
                'time_points': time_points,
                'power_consumption': power_consumption,
                'current_consumption': current_consumption,
                'state_boundaries': self.calculate_reference_boundaries(max_duration, num_devices),
                'device_boundaries': self.calculate_device_boundaries(max_duration, num_devices),
                'state_colors': ['#e6b3ff', '#cc66ff', '#9900ff', '#6600cc'],  # More distinct bright gradient purples (light to dark)
                'device_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'][:num_devices],
                'state_powers': [avg_total_power/4] * 4,  # Equal distribution for visualization
                'state_actual_durations': [max_duration/4] * 4,
                'state_speed_factors': [1.0] * 4,
                'num_devices': num_devices,
                'dynamic_mode': True  # Flag to indicate dynamic mode
            }
            
            # Initialize plots and start animation
            self.current_time = 0
            self._initialize_plots(num_devices)
            self.animation_timer.start()
            
            print("✅ Dynamic visualization created successfully!")
            
        except Exception as e:
            print(f"Error creating dynamic visualization: {e}")
            QMessageBox.warning(self, "Visualization Error", f"Error creating dynamic visualization: {str(e)}")

    def calculate_total_energy_dynamic(self, state_durations, voltage, vlc_payload, ble_payload, 
                                       time_per_vlc_byte=0, time_per_ble_byte=0, ble_connected_duration_s=0,
                                       tx_power=0, adv_interval=40, conn_interval=45):
        """
        Calculate total energy using dynamic current prediction with 4-situation State 3 breakdown
        Args:
            state_durations: dict {state: duration_ms} where state is 0-3 (app format)
            voltage: voltage in mV
            vlc_payload: VLC payload in bytes
            ble_payload: BLE payload in bytes
            time_per_vlc_byte: Time per VLC byte (ms/byte)
            time_per_ble_byte: Time per BLE byte (ms/byte)
            ble_connected_duration_s: BLE connected duration in seconds
            tx_power, adv_interval, conn_interval: BLE parameters
        Returns:
            dict with energy breakdown and total
        """
        try:
            results = {
                'state_energies': {},
                'total_energy_mj': 0,
                'breakdown': []
            }
            
            total_energy = 0
            
            print("\n" + "="*60)
            print("⚡ DYNAMIC ENERGY CALCULATION (4-SITUATION STATE 3)")
            print("="*60)
            print(f"Voltage: {voltage} mV")
            print(f"VLC Payload: {vlc_payload} bytes, Time/byte: {time_per_vlc_byte} ms/byte")
            print(f"BLE Payload: {ble_payload} bytes, Time/byte: {time_per_ble_byte} ms/byte")
            print(f"BLE Connected Duration: {ble_connected_duration_s} s")
            print(f"State Durations: {state_durations}")
            
            voltage_v = voltage / 1000.0  # Convert mV to V
            
            for state, duration_ms in state_durations.items():
                if duration_ms > 0:
                    # Special handling for State 3 (state == 2 in app format)
                    if state == 2:  # State 3 (Operation)
                        print(f"\n  📊 State 3 (Operation) - 4 Situations:")
                        
                        # Calculate transmission durations
                        vlc_tx_duration_ms = vlc_payload * time_per_vlc_byte
                        ble_tx_duration_ms = ble_payload * time_per_ble_byte
                        ble_connected_duration_ms = ble_connected_duration_s * 1000
                        
                        print(f"    VLC TX Duration: {vlc_tx_duration_ms:.3f} ms")
                        print(f"    BLE TX Duration: {ble_tx_duration_ms:.3f} ms")
                        print(f"    BLE Connected Duration: {ble_connected_duration_ms:.0f} ms")
                        
                        state3_energy = 0
                        
                        # Situation 1: Baseline (BLE Disconnected)
                        duration_1 = duration_ms - ble_connected_duration_ms
                        if duration_1 > 0:
                            current_1 = self.predict_with_payload_logic(
                                self.current_model_data, state=3, duration=duration_1, voltage=voltage,
                                vlc_payload=0, ble_payload=0, guest_id=1,
                                tx_power=tx_power, adv_interval=adv_interval, conn_interval=conn_interval,
                                comm_mode=1  # BLE Disconnected
                            )
                            energy_1 = current_1 * duration_1 * voltage_v / 1000.0
                            state3_energy += energy_1
                            print(f"    Situation 1 (BLE Disconnected): {current_1:.3f} mA × {duration_1:.3f} ms = {energy_1:.6f} mJ")
                            
                            results['breakdown'].append({
                                'state': state,
                                'situation': 1,
                                'description': 'BLE Disconnected',
                                'current_ma': current_1,
                                'duration_ms': duration_1,
                                'energy_mj': energy_1
                            })
                        
                        # Situation 2: BLE Connected Idle (no transmission)
                        duration_2 = ble_connected_duration_ms - vlc_tx_duration_ms - ble_tx_duration_ms
                        if duration_2 > 0:
                            current_2 = self.predict_with_payload_logic(
                                self.current_model_data, state=3, duration=duration_2, voltage=voltage,
                                vlc_payload=0, ble_payload=0, guest_id=1,
                                tx_power=tx_power, adv_interval=adv_interval, conn_interval=conn_interval,
                                comm_mode=2  # BLE Connected
                            )
                            energy_2 = current_2 * duration_2 * voltage_v / 1000.0
                            state3_energy += energy_2
                            print(f"    Situation 2 (BLE Connected Idle): {current_2:.3f} mA × {duration_2:.3f} ms = {energy_2:.6f} mJ")
                            
                            results['breakdown'].append({
                                'state': state,
                                'situation': 2,
                                'description': 'BLE Connected Idle',
                                'current_ma': current_2,
                                'duration_ms': duration_2,
                                'energy_mj': energy_2
                            })
                        
                        # Situation 3: BLE Data Transmission
                        duration_3 = ble_tx_duration_ms
                        if duration_3 > 0 and ble_payload > 0:
                            current_3 = self.predict_with_payload_logic(
                                self.current_model_data, state=3, duration=duration_3, voltage=voltage,
                                vlc_payload=0, ble_payload=ble_payload, guest_id=1,
                                tx_power=tx_power, adv_interval=adv_interval, conn_interval=conn_interval,
                                comm_mode=2  # BLE Connected
                            )
                            energy_3 = current_3 * duration_3 * voltage_v / 1000.0
                            state3_energy += energy_3
                            print(f"    Situation 3 (BLE TX): {current_3:.3f} mA × {duration_3:.3f} ms = {energy_3:.6f} mJ")
                            
                            results['breakdown'].append({
                                'state': state,
                                'situation': 3,
                                'description': f'BLE TX ({ble_payload} bytes)',
                                'current_ma': current_3,
                                'duration_ms': duration_3,
                                'energy_mj': energy_3
                            })
                        
                        # Situation 4: VLC Data Transmission
                        duration_4 = vlc_tx_duration_ms
                        if duration_4 > 0 and vlc_payload > 0:
                            current_4 = self.predict_with_payload_logic(
                                self.current_model_data, state=3, duration=duration_4, voltage=voltage,
                                vlc_payload=vlc_payload, ble_payload=0, guest_id=1,
                                tx_power=tx_power, adv_interval=adv_interval, conn_interval=conn_interval,
                                comm_mode=2  # BLE Connected
                            )
                            energy_4 = current_4 * duration_4 * voltage_v / 1000.0
                            state3_energy += energy_4
                            print(f"    Situation 4 (VLC TX): {current_4:.3f} mA × {duration_4:.3f} ms = {energy_4:.6f} mJ")
                            
                            results['breakdown'].append({
                                'state': state,
                                'situation': 4,
                                'description': f'VLC TX ({vlc_payload} bytes)',
                                'current_ma': current_4,
                                'duration_ms': duration_4,
                                'energy_mj': energy_4
                            })
                        
                        results['state_energies'][state] = state3_energy
                        total_energy += state3_energy
                        print(f"    💡 Total State 3 Energy: {state3_energy:.6f} mJ")
                    
                    else:
                        # States 0, 1, 3 (Idle, Advertisement, Sleep) - Standard calculation
                        predicted_current = self.predict_with_payload_logic(
                            self.current_model_data,
                            state=state + 1,  # Convert to model format (1-4)
                            duration=duration_ms,
                            voltage=voltage,
                            vlc_payload=0,  # Always 0 for non-operation states
                            ble_payload=0,  # Always 0 for non-operation states
                            guest_id=1,
                            tx_power=tx_power,
                            adv_interval=adv_interval,
                            conn_interval=conn_interval,
                            comm_mode=0
                        )
                        
                        energy_mj = predicted_current * duration_ms * voltage_v / 1000.0
                        results['state_energies'][state] = energy_mj
                        total_energy += energy_mj
                        
                        results['breakdown'].append({
                            'state': state,
                            'current_ma': predicted_current,
                            'duration_ms': duration_ms,
                            'voltage_mv': voltage,
                            'energy_mj': energy_mj
                        })
                        
                        print(f"  State {state}: {predicted_current:.3f} mA × {duration_ms:.0f} ms = {energy_mj:.6f} mJ")
            
            results['total_energy_mj'] = total_energy
            print(f"\n🔋 TOTAL ENERGY: {total_energy:.6f} mJ")
            print("="*60)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in dynamic energy calculation: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Cannot calculate energy: {str(e)}")

    def run_dynamic_energy_simulation(self, num_devices):
        """Run energy simulation using dynamic current prediction based on actual payload values"""
        try:
            print("\n" + "="*80)
            print("🚀 DYNAMIC ENERGY SIMULATION")
            print("="*80)
            
            total_energy_all_devices = 0
            device_energy_results = []
            
            # Process each device
            for device_idx in range(num_devices):
                device_id = device_idx + 1
                device_config = self.device_configs.get(device_id, self.get_default_device_config())
                
                # Get actual payload values from device configuration
                vlc_payload = device_config['vlc_payload']
                ble_payload = device_config['ble_payload']
                time_per_vlc_byte = device_config.get('time_per_vlc_byte', 0.0)
                time_per_ble_byte = device_config.get('time_per_ble_byte', 0.0)
                ble_connected_duration = device_config.get('ble_connected_duration', 0.0)
                tx_power = device_config.get('tx_power', 0)
                adv_interval = device_config.get('adv_interval', 40)
                conn_interval = device_config.get('conn_interval', 45)
                
                # Prepare state durations for this device (convert to milliseconds)
                state_durations = {}
                for state in range(4):
                    duration_s = float(device_config['state_durations'][state])
                    enabled = device_config['state_switches'][state]
                    state_durations[state] = (duration_s * 1000) if enabled else 0  # Convert s to ms
                
                # Calculate energy for this device using dynamic prediction with 4-situation State 3
                device_energy = self.calculate_total_energy_dynamic(
                    state_durations, self.voltage_mV, vlc_payload, ble_payload,
                    time_per_vlc_byte, time_per_ble_byte, ble_connected_duration,
                    tx_power, adv_interval, conn_interval
                )
                device_energy_results.append(device_energy)
                total_energy_all_devices += device_energy['total_energy_mj']
                
                print(f"\n📱 Device {device_id} Energy: {device_energy['total_energy_mj']:.2f} mJ (VLC={vlc_payload}, BLE={ble_payload})")
            
            # Display summary
            print("\n" + "="*80)
            print("📊 DYNAMIC ENERGY SIMULATION SUMMARY")
            print("="*80)
            print(f"Total Devices: {num_devices}")
            print(f"Total Energy (All Devices): {total_energy_all_devices:.2f} mJ")
            print(f"Average Energy per Device: {total_energy_all_devices/num_devices:.2f} mJ")
            
            # Update UI with results
            if hasattr(self, 'total_energy_label'):
                self.total_energy_label.setText(f"Total Energy: {total_energy_all_devices:.2f} mJ")
            
            # Create dynamic visualization data
            self.create_dynamic_visualization(device_energy_results, num_devices)
            
        except Exception as e:
            print(f"Error in dynamic energy simulation: {e}")
            QMessageBox.warning(self, "Simulation Error", f"Error in dynamic simulation: {str(e)}")

    def predict_with_payload_logic(self, model_data, state, duration, voltage, vlc_payload, ble_payload, guest_id, tx_power=0, adv_interval=40, conn_interval=45, comm_mode=0):
        """
        Predict current with enhanced 7-feature model
        Features: State, VLC Payload, BLE Payload, TxPower, AdvInterval, ConnInterval, Comm_Mode
        """
        try:
            print(f"🔍 Enhanced prediction: State {state}, VLC={vlc_payload}, BLE={ble_payload}, TxPower={tx_power}, AdvInterval={adv_interval}, ConnInterval={conn_interval}, Comm_Mode={comm_mode}")
            
            # State mapping: app uses 0-3, model uses 1-4
            # Only State 3 (model) / State 2 (app) should be affected by payloads
            if state == 3:  # This is State 3 in model format, State 2 in app format
                # For State 3, use actual payload values (this state had payloads in training)
                input_data = pd.DataFrame({
                    'State': [state],
                    'VLC_Payload': [vlc_payload],
                    'BLE_Payload': [ble_payload],
                    'TxPower': [tx_power],
                    'AdvInterval': [adv_interval],
                    'ConnInterval': [conn_interval],
                    'Comm_Mode': [comm_mode]
                })
                print(f"🔍 State 3 prediction with all features")
            else:
                # For States 1, 2, 4, always use zero payloads (these states never had payloads in training)
                input_data = pd.DataFrame({
                    'State': [state],
                    'VLC_Payload': [0],  # Force to 0 for non-payload states
                    'BLE_Payload': [0],   # Force to 0 for non-payload states
                    'TxPower': [tx_power],
                    'AdvInterval': [adv_interval],
                    'ConnInterval': [conn_interval],
                    'Comm_Mode': [comm_mode]
                })
                print(f"🔍 State {state} prediction (forced VLC=0, BLE=0)")
            
            model = model_data['model']
            scaler = model_data.get('scaler', None)
            
            # Validate model compatibility (check if it's enhanced or simple)
            use_enhanced = model_data.get('use_enhanced', False)
            expected_features = 7 if use_enhanced else 3
            actual_features = len(input_data.columns)
            
            if expected_features != actual_features:
                raise ValueError(f"Feature mismatch! Model expects {expected_features} features "
                               f"but {actual_features} were provided. "
                               f"Model type: {'Enhanced (7-feature)' if use_enhanced else 'Simple (3-feature)'}")
            
            print(f"✓ Model validation passed: {expected_features} features")
            
            if scaler is not None:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
            else:
                prediction = model.predict(input_data)[0]
            
            print(f"🔍 Model prediction result: {prediction:.1f} µA")
            # Convert from µA to mA for consistency with app units
            prediction_mA = prediction / 1000.0
            print(f"🔍 Converted to: {prediction_mA:.1f} mA")
            return prediction_mA
            
        except Exception as e:
            print(f"Error in enhanced prediction: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Enhanced prediction failed: {str(e)}")

    def update_dynamic_prediction(self):
        """Update model info display with dynamic current predictions based on current VLC/BLE payload values"""
        try:
            if not hasattr(self, 'current_model_data') or self.current_model_data is None:
                print("⚠️ No model loaded yet, skipping dynamic prediction update")
                return
            
            # Get current VLC/BLE payload values, with fallback to 0 if widgets not created yet
            try:
                vlc_payload = self.vlc_volume.value() if hasattr(self, 'vlc_volume') else 0
                ble_payload = self.ble_volume.value() if hasattr(self, 'ble_volume') else 0
                tx_power = self.tx_power_spin.value() if hasattr(self, 'tx_power_spin') else 0
                adv_interval = self.adv_interval_spin.value() if hasattr(self, 'adv_interval_spin') else 40
                conn_interval = self.conn_interval_spin.value() if hasattr(self, 'conn_interval_spin') else 45
                ble_connected_duration = self.ble_connected_duration_spin.value() if hasattr(self, 'ble_connected_duration_spin') else 0
                comm_mode = 2 if ble_connected_duration > 0 else 0
            except Exception as e:
                print(f"⚠️ Error accessing widgets, using defaults: {e}")
                vlc_payload = 0
                ble_payload = 0
                tx_power = 0
                adv_interval = 40
                conn_interval = 45
                comm_mode = 0
            
            print(f"\n📊 UPDATING DYNAMIC PREDICTION")
            print(f"📊 Current payloads: VLC={vlc_payload}, BLE={ble_payload}")
            print(f"📊 BLE params: TxPower={tx_power}, AdvInterval={adv_interval}, ConnInterval={conn_interval}, CommMode={comm_mode}")
            
            # Calculate dynamic currents for each state with current payload values
            dynamic_currents = []
            for state in range(4):  # App uses 0-3, model uses 1-4
                model_state = state + 1  # Convert to model format (1-4)
                
                try:
                    predicted_current = self.predict_with_payload_logic(
                        self.current_model_data,
                        state=model_state,
                        duration=100,  # Duration doesn't affect current prediction
                        voltage=self.voltage_mV,
                        vlc_payload=vlc_payload,
                        ble_payload=ble_payload,
                        guest_id=1,
                        tx_power=tx_power,
                        adv_interval=adv_interval,
                        conn_interval=conn_interval,
                        comm_mode=comm_mode
                    )
                    dynamic_currents.append(predicted_current)
                    print(f"📊 State {state}: {predicted_current:.1f} mA")
                except Exception as e:
                    print(f"❌ Error predicting for state {state}: {e}")
                    dynamic_currents.append(0)
            
            # Update model info display with dynamic predictions
            model_data = self.current_model_data
            r2_score = model_data.get('r2_score', 0)
            model_type = type(model_data['model']).__name__
            
            if vlc_payload == 0 and ble_payload == 0:
                dynamic_text = f"Baseline(VLC=0,BLE=0): S1:{dynamic_currents[0]:.1f}mA, S2:{dynamic_currents[1]:.1f}mA, S3:{dynamic_currents[2]:.1f}mA, S4:{dynamic_currents[3]:.1f}mA"
            else:
                dynamic_text = f"Dynamic(VLC={vlc_payload},BLE={ble_payload}): S1:{dynamic_currents[0]:.1f}mA, S2:{dynamic_currents[1]:.1f}mA, S3:{dynamic_currents[2]:.1f}mA, S4:{dynamic_currents[3]:.1f}mA"
            
            info_text = f"R²: {r2_score:.4f}, Type: {model_type}, {dynamic_text}"
            self.model_info_label.setText(info_text)
            
            print(f"📊 Updated model info: {dynamic_text}")
            
        except Exception as e:
            print(f"❌ Error in update_dynamic_prediction: {e}")
            import traceback
            traceback.print_exc()


def main():
    app = QApplication(sys.argv)
    window = PowerPredictionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()