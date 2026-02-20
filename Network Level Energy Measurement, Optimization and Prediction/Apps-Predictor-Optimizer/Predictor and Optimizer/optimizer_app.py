"""
Energy Consumption Optimizer App
Uses trained models to find optimal IoT configurations for minimum energy consumption
"""
import sys
import os
import joblib
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox, 
                             QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar,
                             QScrollArea, QGridLayout, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPalette, QColor, QPainter, QLinearGradient, QPixmap
import itertools
from scipy.optimize import differential_evolution, minimize
import glob


class OptimizationWorker(QThread):
    """Worker thread for optimization to keep UI responsive"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    status_update = pyqtSignal(str)
    
    def __init__(self, model_data, constraints, algorithm, state_durations, ble_connected_duration=0):
        super().__init__()
        self.model_data = model_data
        self.constraints = constraints
        self.algorithm = algorithm
        self.state_durations = state_durations
        self.ble_connected_duration = ble_connected_duration
        self.is_running = True
        
        # Transmission time parameters (from dataset analysis)
        self.time_per_vlc_byte = 0.017  # ms/byte
        self.time_per_ble_byte = 0.041  # ms/byte
        
    def run(self):
        """Run optimization in background thread"""
        try:
            if self.algorithm == "Grid Search":
                results = self.grid_search_optimization()
            elif self.algorithm == "Random Search":
                results = self.random_search_optimization()
            elif self.algorithm == "Bayesian Optimization":
                results = self.bayesian_optimization()
            else:
                results = self.grid_search_optimization()
            
            self.finished.emit(results)
        except Exception as e:
            print(f"Optimization error: {e}")
            import traceback
            traceback.print_exc()
            self.finished.emit([])
    
    def stop(self):
        """Stop optimization"""
        self.is_running = False
    
    def calculate_total_energy(self, config):
        """Calculate total energy for a configuration across all states"""
        total_energy = 0
        voltage = 3.3  # Default voltage
        
        for state in [1, 2, 3, 4]:
            if state not in self.state_durations or self.state_durations[state] == 0:
                continue
            
            duration = self.state_durations[state]
            
            # Special handling for State 3: 4-situation logic
            if state == 3:
                energy_mJ = self._calculate_state3_energy(config, duration, voltage)
                total_energy += energy_mJ
            else:
                # Simple prediction for other states
                input_data = pd.DataFrame({
                    'State': [state],
                    'VLC_Payload': [config.get('vlc_payload', 0)],
                    'BLE_Payload': [config.get('ble_payload', 0)],
                    'TxPower': [config['txpower']],
                    'AdvInterval': [config['adv_interval']],
                    'ConnInterval': [config['conn_interval']],
                    'Comm_Mode': [config.get('comm_mode', 1)]
                })
                
                current_uA = self._predict_current(input_data)
                current_mA = current_uA / 1000.0
                energy_mJ = current_mA * voltage * duration
                total_energy += energy_mJ
        
        return total_energy
    
    def _predict_current(self, input_data):
        """Helper to predict current from input data"""
        model = self.model_data['model']
        scaler = self.model_data.get('scaler', None)
        
        if scaler is not None:
            input_scaled = scaler.transform(input_data)
            current_uA = model.predict(input_scaled)[0]
        else:
            current_uA = model.predict(input_data)[0]
        
        return current_uA
    
    def _calculate_state3_energy(self, config, total_duration, voltage):
        """
        Calculate State 3 energy using 4-situation logic:
        1. BLE disconnected (no payload)
        2. BLE connected (no payload)
        3. BLE transmission only
        4. VLC transmission only
        """
        vlc_payload = config.get('vlc_payload', 0)
        ble_payload = config.get('ble_payload', 0)
        
        # Calculate transmission durations
        vlc_tx_duration = vlc_payload * self.time_per_vlc_byte / 1000.0  # Convert ms to seconds
        ble_tx_duration = ble_payload * self.time_per_ble_byte / 1000.0  # Convert ms to seconds
        
        # Get BLE connected duration (limited to total_duration)
        ble_connected_duration = min(self.ble_connected_duration, total_duration)
        
        # Calculate idle durations
        ble_disconnected_duration = total_duration - ble_connected_duration
        ble_connected_idle_duration = max(0, ble_connected_duration - vlc_tx_duration - ble_tx_duration)
        
        total_energy = 0
        
        # Situation 1: BLE Disconnected (no payload)
        if ble_disconnected_duration > 0:
            input_data = pd.DataFrame({
                'State': [3],
                'VLC_Payload': [0],
                'BLE_Payload': [0],
                'TxPower': [config['txpower']],
                'AdvInterval': [config['adv_interval']],
                'ConnInterval': [config['conn_interval']],
                'Comm_Mode': [1]  # Disconnected
            })
            current_uA = self._predict_current(input_data)
            current_mA = current_uA / 1000.0
            energy_mJ = current_mA * voltage * ble_disconnected_duration
            total_energy += energy_mJ
        
        # Situation 2: BLE Connected Idle (no payload)
        if ble_connected_idle_duration > 0:
            input_data = pd.DataFrame({
                'State': [3],
                'VLC_Payload': [0],
                'BLE_Payload': [0],
                'TxPower': [config['txpower']],
                'AdvInterval': [config['adv_interval']],
                'ConnInterval': [config['conn_interval']],
                'Comm_Mode': [2]  # Connected
            })
            current_uA = self._predict_current(input_data)
            current_mA = current_uA / 1000.0
            energy_mJ = current_mA * voltage * ble_connected_idle_duration
            total_energy += energy_mJ
        
        # Situation 3: BLE Transmission
        if ble_tx_duration > 0 and ble_payload > 0:
            input_data = pd.DataFrame({
                'State': [3],
                'VLC_Payload': [0],
                'BLE_Payload': [ble_payload],
                'TxPower': [config['txpower']],
                'AdvInterval': [config['adv_interval']],
                'ConnInterval': [config['conn_interval']],
                'Comm_Mode': [2]  # Connected
            })
            current_uA = self._predict_current(input_data)
            current_mA = current_uA / 1000.0
            energy_mJ = current_mA * voltage * ble_tx_duration
            total_energy += energy_mJ
        
        # Situation 4: VLC Transmission
        if vlc_tx_duration > 0 and vlc_payload > 0:
            input_data = pd.DataFrame({
                'State': [3],
                'VLC_Payload': [vlc_payload],
                'BLE_Payload': [0],
                'TxPower': [config['txpower']],
                'AdvInterval': [config['adv_interval']],
                'ConnInterval': [config['conn_interval']],
                'Comm_Mode': [2]  # Connected
            })
            current_uA = self._predict_current(input_data)
            current_mA = current_uA / 1000.0
            energy_mJ = current_mA * voltage * vlc_tx_duration
            total_energy += energy_mJ
        
        return total_energy
    
    def grid_search_optimization(self):
        """Grid search: exhaustive search over all combinations"""
        self.status_update.emit("Starting Grid Search...")
        
        # Generate parameter grids
        txpower_range = np.linspace(self.constraints['txpower'][0], 
                                    self.constraints['txpower'][1], 
                                    self.constraints['txpower_steps'])
        adv_range = np.linspace(self.constraints['adv_interval'][0], 
                               self.constraints['adv_interval'][1], 
                               self.constraints['adv_interval_steps'])
        conn_range = np.linspace(self.constraints['conn_interval'][0], 
                                self.constraints['conn_interval'][1], 
                                self.constraints['conn_interval_steps'])
        vlc_range = np.linspace(self.constraints['vlc_payload'][0], 
                               self.constraints['vlc_payload'][1], 
                               self.constraints['vlc_payload_steps'])
        ble_range = np.linspace(self.constraints['ble_payload'][0], 
                               self.constraints['ble_payload'][1], 
                               self.constraints['ble_payload_steps'])
        comm_modes = [0, 2] if self.constraints['comm_mode'][1] >= 2 else [0]
        
        # Total combinations
        total = (len(txpower_range) * len(adv_range) * len(conn_range) * 
                len(vlc_range) * len(ble_range) * len(comm_modes))
        
        self.status_update.emit(f"Evaluating {total} configurations...")
        
        results = []
        count = 0
        
        for txp in txpower_range:
            for adv in adv_range:
                for conn in conn_range:
                    for vlc in vlc_range:
                        for ble in ble_range:
                            for comm in comm_modes:
                                if not self.is_running:
                                    return results
                                
                                config = {
                                    'txpower': txp,
                                    'adv_interval': adv,
                                    'conn_interval': conn,
                                    'vlc_payload': vlc,
                                    'ble_payload': ble,
                                    'comm_mode': comm
                                }
                                
                                energy = self.calculate_total_energy(config)
                                results.append({'config': config, 'energy': energy})
                                
                                count += 1
                                if count % max(1, total // 100) == 0:
                                    progress = int((count / total) * 100)
                                    self.progress.emit(progress)
        
        # Sort by energy (ascending) - LOWEST energy first (best results)
        results.sort(key=lambda x: x['energy'])
        return results  # Return ALL configurations for accurate baseline
    
    def random_search_optimization(self):
        """Random search: sample random configurations"""
        self.status_update.emit("Starting Random Search...")
        
        n_samples = 1000  # Number of random samples
        results = []
        
        for i in range(n_samples):
            if not self.is_running:
                return results
            
            # Random sample within constraints
            config = {
                'txpower': np.random.uniform(self.constraints['txpower'][0], 
                                            self.constraints['txpower'][1]),
                'adv_interval': np.random.uniform(self.constraints['adv_interval'][0], 
                                                 self.constraints['adv_interval'][1]),
                'conn_interval': np.random.uniform(self.constraints['conn_interval'][0], 
                                                  self.constraints['conn_interval'][1]),
                'vlc_payload': np.random.uniform(self.constraints['vlc_payload'][0], 
                                                self.constraints['vlc_payload'][1]),
                'ble_payload': np.random.uniform(self.constraints['ble_payload'][0], 
                                                self.constraints['ble_payload'][1]),
                'comm_mode': np.random.choice([0, 2] if self.constraints['comm_mode'][1] >= 2 else [0])
            }
            
            energy = self.calculate_total_energy(config)
            results.append({'config': config, 'energy': energy})
            
            if i % 10 == 0:
                progress = int((i / n_samples) * 100)
                self.progress.emit(progress)
                self.status_update.emit(f"Evaluated {i}/{n_samples} configurations...")
        
        # Sort by energy (ascending) - LOWEST energy first (best results)
        results.sort(key=lambda x: x['energy'])
        return results  # Return ALL configurations for accurate baseline
    
    def bayesian_optimization(self):
        """Bayesian optimization using differential evolution"""
        self.status_update.emit("Starting Bayesian Optimization...")
        
        # Define bounds for differential evolution
        bounds = [
            self.constraints['txpower'],
            self.constraints['adv_interval'],
            self.constraints['conn_interval'],
            self.constraints['vlc_payload'],
            self.constraints['ble_payload'],
            (0, 2)  # comm_mode
        ]
        
        # Objective function to minimize
        def objective(x):
            config = {
                'txpower': x[0],
                'adv_interval': x[1],
                'conn_interval': x[2],
                'vlc_payload': x[3],
                'ble_payload': x[4],
                'comm_mode': int(round(x[5] / 2) * 2)  # Round to 0 or 2
            }
            return self.calculate_total_energy(config)
        
        # Run optimization
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100,
            popsize=15,
            callback=lambda xk, convergence: self.progress.emit(int((1 - convergence) * 100))
        )
        
        # Get best result
        best_config = {
            'txpower': result.x[0],
            'adv_interval': result.x[1],
            'conn_interval': result.x[2],
            'vlc_payload': result.x[3],
            'ble_payload': result.x[4],
            'comm_mode': int(round(result.x[5] / 2) * 2)
        }
        
        # Sample around best solution for top 3
        results = [{'config': best_config, 'energy': result.fun}]
        
        # Add some variations
        for _ in range(50):
            varied_config = best_config.copy()
            # Add small variations
            varied_config['txpower'] += np.random.normal(0, 0.5)
            varied_config['adv_interval'] += np.random.normal(0, 5)
            varied_config['conn_interval'] += np.random.normal(0, 10)
            
            # Clip to bounds
            varied_config['txpower'] = np.clip(varied_config['txpower'], *bounds[0])
            varied_config['adv_interval'] = np.clip(varied_config['adv_interval'], *bounds[1])
            varied_config['conn_interval'] = np.clip(varied_config['conn_interval'], *bounds[2])
            
            energy = self.calculate_total_energy(varied_config)
            results.append({'config': varied_config, 'energy': energy})
        
        # Sort by energy (ascending) - LOWEST energy first (best results)
        results.sort(key=lambda x: x['energy'])
        return results  # Return ALL configurations for accurate baseline


class MedalWidget(QWidget):
    """Custom widget to display optimization results with medals"""
    def __init__(self, rank, config, energy, baseline_energy=None):
        super().__init__()
        self.rank = rank
        self.config = config
        self.energy = energy
        self.baseline_energy = baseline_energy
        self.init_ui()
    
    def init_ui(self):
        """Initialize the medal display UI - PLAIN TEXT"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)  # Minimal margins
        
        # Medal info
        medals = {
            1: '🥇',
            2: '🥈',
            3: '🥉'
        }
        
        medal_icon = medals.get(self.rank, '🏅')
        
        # Build configuration summary string
        ble_status = 'BLE Connected' if self.config['comm_mode'] == 2 else 'BLE Disconnected'
        config_str = (f"TxPower={self.config['txpower']:.1f}dBm, "
                     f"AdvInt={self.config['adv_interval']:.0f}ms, "
                     f"ConnInt={self.config['conn_interval']:.0f}ms, "
                     f"VLC={self.config['vlc_payload']:.0f}B, "
                     f"BLE={self.config['ble_payload']:.0f}B, "
                     f"{ble_status}")
        
        # Calculate savings
        saves_str = ""
        if self.baseline_energy and self.baseline_energy > 0:
            savings = ((self.baseline_energy - self.energy) / self.baseline_energy) * 100
            saves_str = f", Saves {savings:.1f}%"
        
        # Plain text lines
        result_text = f"Top {self.rank}: {medal_icon}\n{config_str}\nEnergy {self.energy:.2f}mJ{saves_str}"
        
        # Create plain label - no background, no border
        label = QLabel(result_text)
        label.setWordWrap(True)
        label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #333;
                padding: 2px;
            }
        """)
        
        layout.addWidget(label)
        self.setLayout(layout)


class EnergyOptimizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡ Energy Consumption Optimizer")
        self.setGeometry(100, 100, 1200, 750)  # Slightly larger to show all content
        
        # Load models
        self.models = {}
        self.current_model = None
        self.load_available_models()
        
        # Optimization worker
        self.optimization_worker = None
        self.is_optimizing = False
        
        # Results
        self.optimization_results = []
        self.baseline_energy = None
        
        # Initialize UI
        self.init_ui()
    
    def load_available_models(self):
        """Load all available trained models"""
        model_patterns = [
            '*_enhanced_model.joblib',
            'current_prediction_enhanced_model.joblib',
            '*_model.joblib',
            'current_prediction_model.joblib'
        ]
        
        print("Loading trained models...")
        models_loaded = 0
        
        for pattern in model_patterns:
            model_files = glob.glob(pattern)
            for model_file in model_files:
                try:
                    model_data = joblib.load(model_file)
                    base_name = os.path.splitext(model_file)[0]
                    
                    if 'current_prediction' in base_name:
                        if 'enhanced' in base_name:
                            model_name = 'Best Model (7-Features)'
                        else:
                            model_name = 'Best Model (3-Features)'
                    else:
                        model_name = base_name.replace('_model', '').replace('_', ' ').title()
                    
                    self.models[model_name] = model_data
                    models_loaded += 1
                    
                    r2_score = model_data.get('r2_score', 'N/A')
                    print(f"✓ Loaded: {model_name} (R² = {r2_score:.4f})")
                    
                    if self.current_model is None:
                        self.current_model = model_data
                        print(f"  -> Set as default model")
                
                except Exception as e:
                    print(f"✗ Failed to load {model_file}: {e}")
        
        if models_loaded == 0:
            QMessageBox.critical(self, "Error",
                               "No trained models found!\n"
                               "Please run train_model.py first.")
            sys.exit(1)
        
        print(f"Successfully loaded {models_loaded} models!")
    
    def create_custom_title_bar(self, layout):
        """Create custom title bar with logo, centered title, matching predict_app.py style"""
        title_bar = QWidget()
        title_bar.setFixedHeight(50)
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
        title_label = QLabel("Energy Consumption Optimizer")
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
        
        # Empty spacer on right to balance the logo
        spacer_label = QLabel()
        spacer_label.setFixedSize(34, 34)
        title_layout.addWidget(spacer_label)
        
        layout.addWidget(title_bar)
    
    def init_ui(self):
        """Initialize the main UI"""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create custom title bar (matching predict_app.py)
        self.create_custom_title_bar(main_layout)
        
        # Create content area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(5)
        content_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(content_widget)
        
        # Left panel: Constraints and controls
        left_panel = self.create_left_panel()
        content_layout.addWidget(left_panel, 60)  # More space for left
        
        # Right panel: Results display (narrower)
        right_panel = self.create_right_panel()
        content_layout.addWidget(right_panel, 40)  # Narrower right panel
        
        # Set style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 5px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
        """)
    
    def create_left_panel(self):
        """Create left panel with constraints and controls"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Scroll area for constraints
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Model selection
        scroll_layout.addWidget(self.create_model_selection_group())
        
        # State durations
        scroll_layout.addWidget(self.create_state_durations_group())
        
        # Parameter constraints
        scroll_layout.addWidget(self.create_constraints_group())
        
        # Algorithm selection
        scroll_layout.addWidget(self.create_algorithm_group())
        
        # Optimization control
        scroll_layout.addWidget(self.create_control_group())
        
        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        
        layout.addWidget(scroll)
        panel.setLayout(layout)
        
        return panel
    
    def create_model_selection_group(self):
        """Create model selection group"""
        group = QGroupBox("🤖 Model Selection")
        layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        for model_name in self.models.keys():
            model_data = self.models[model_name]
            r2_score = model_data.get('r2_score', 0)
            self.model_combo.addItem(f"{model_name} (R²={r2_score:.4f})")
        
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        layout.addWidget(self.model_combo)
        
        group.setLayout(layout)
        return group
    
    def create_state_durations_group(self):
        """Create state durations input group"""
        group = QGroupBox("⏱️ State Durations (seconds)")
        layout = QGridLayout()
        
        self.state_duration_spins = {}
        states = [
            (1, "State 1 (Initialization)", 1.0),
            (2, "State 2 (Advertising)", 1.0),
            (3, "State 3 (Operation)", 1.0),
            (4, "State 4 (Sleep)", 1.0)
        ]
        
        row = 0
        for state_num, state_name, default_val in states:
            label = QLabel(state_name)
            spin = QDoubleSpinBox()
            spin.setRange(0, 1000)
            spin.setValue(default_val)
            spin.setSuffix(" s")
            
            self.state_duration_spins[state_num] = spin
            
            layout.addWidget(label, row, 0)
            layout.addWidget(spin, row, 1)
            row += 1
            
            # Add BLE Connected Duration right after State 3
            if state_num == 3:
                ble_connected_label = QLabel("  └─ State 3 BLE Connected Duration")
                ble_connected_label.setStyleSheet("color: #666; font-size: 11px;")
                self.ble_connected_duration_spin = QDoubleSpinBox()
                self.ble_connected_duration_spin.setRange(0, 1000)
                self.ble_connected_duration_spin.setValue(0.5)  # Default: 0.5 seconds
                self.ble_connected_duration_spin.setSuffix(" s")
                self.ble_connected_duration_spin.setToolTip("Duration of BLE connected time within State 3 (must be ≤ State 3 duration)")
                
                layout.addWidget(ble_connected_label, row, 0)
                layout.addWidget(self.ble_connected_duration_spin, row, 1)
                row += 1
        
        group.setLayout(layout)
        return group
    
    def create_constraints_group(self):
        """Create parameter constraints group"""
        group = QGroupBox("📊 Parameter Constraints")
        main_layout = QVBoxLayout()
        
        # Add explanation for "Steps"
        steps_explanation = QLabel("💡 <b>Steps</b>: Number of values to test between Min and Max. Example: Min=0, Max=100, Steps=5 → tests [0, 25, 50, 75, 100]")
        steps_explanation.setStyleSheet("color: #0066CC; font-size: 10px; padding: 5px;")
        steps_explanation.setWordWrap(True)
        main_layout.addWidget(steps_explanation)
        
        layout = QGridLayout()
        
        self.constraint_inputs = {}
        
        # Define parameters with (label, min_default, max_default, step, suffix)
        # DEMO MODE: Narrow ranges + low steps for FAST results (3×3×3×2×2×2 = 216 combinations ≈ 2-3 seconds)
        parameters = [
            ('txpower', 'Tx Power', -4, 4, 3, ' dBm'),              # Tests: -4, 0, 4 (3 values)
            ('adv_interval', 'Advertising Interval', 40, 150, 3, ' ms'),    # Tests: 40, 95, 150 (3 values)
            ('conn_interval', 'Connection Interval', 50, 500, 3, ' ms'),    # Tests: 50, 275, 500 (3 values)
            ('vlc_payload', 'VLC Payload', 0, 100, 2, ' bytes'),            # Tests: 0, 100 (2 values)
            ('ble_payload', 'BLE Payload', 0, 100, 2, ' bytes'),            # Tests: 0, 100 (2 values)
            ('comm_mode', 'BLE Disconnected/Connected', 1, 2, 2, '')        # Tests: 1, 2 (2 values) ✅ Fixed!
        ]
        
        row = 0
        for param_key, param_label, min_default, max_default, step_default, suffix in parameters:
            # Parameter label
            label = QLabel(f"{param_label}:")
            label.setStyleSheet("font-weight: bold;")
            layout.addWidget(label, row, 0)
            
            # Min value
            min_label = QLabel("Min:")
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-100, 10000)
            min_spin.setValue(min_default)
            min_spin.setSuffix(suffix)
            min_spin.setMaximumWidth(100)  # Original width
            
            layout.addWidget(min_label, row, 1)
            layout.addWidget(min_spin, row, 2)
            
            # Max value
            max_label = QLabel("Max:")
            max_spin = QDoubleSpinBox()
            max_spin.setRange(-100, 10000)
            max_spin.setValue(max_default)
            max_spin.setSuffix(suffix)
            max_spin.setMaximumWidth(100)  # Original width
            
            layout.addWidget(max_label, row, 3)
            layout.addWidget(max_spin, row, 4)
            
            # Steps (for grid search) - with tooltip
            steps_label = QLabel("Steps:")
            steps_label.setToolTip("Number of values to test between Min and Max\n(For Grid Search only)")
            steps_spin = QSpinBox()
            steps_spin.setRange(1, 100)
            steps_spin.setValue(step_default)
            steps_spin.setMaximumWidth(60)  # Original width
            steps_spin.setToolTip("Grid Search samples this many points between Min and Max.\nExample: Min=0, Max=100, Steps=5 → tests [0, 25, 50, 75, 100]")
            
            layout.addWidget(steps_label, row, 5)
            layout.addWidget(steps_spin, row, 6)
            
            self.constraint_inputs[param_key] = {
                'min': min_spin,
                'max': max_spin,
                'steps': steps_spin
            }
            
            row += 1
        
        main_layout.addLayout(layout)  # Add grid to main layout
        group.setLayout(main_layout)
        return group
    
    def create_algorithm_group(self):
        """Create algorithm selection group"""
        group = QGroupBox("🔬 Optimization Algorithm")
        layout = QVBoxLayout()
        
        self.algorithm_combo = QComboBox()
        algorithms = [
            "Grid Search (Exhaustive)",
            "Random Search (Fast)",
            "Bayesian Optimization (Smart)"
        ]
        self.algorithm_combo.addItems(algorithms)
        
        layout.addWidget(self.algorithm_combo)
        
        # Algorithm description
        self.algo_desc = QLabel()
        self.algo_desc.setWordWrap(True)
        self.algo_desc.setStyleSheet("color: #666; font-style: italic;")
        self.update_algorithm_description()
        self.algorithm_combo.currentIndexChanged.connect(self.update_algorithm_description)
        
        layout.addWidget(self.algo_desc)
        
        group.setLayout(layout)
        return group
    
    def update_algorithm_description(self):
        """Update algorithm description based on selection"""
        descriptions = {
            0: "Tries all possible combinations within constraints. Most thorough but slowest.",
            1: "Randomly samples configurations. Faster, good balance of speed and quality.",
            2: "Uses intelligent sampling to find optimal solutions quickly. Best for complex spaces."
        }
        idx = self.algorithm_combo.currentIndex()
        self.algo_desc.setText(descriptions.get(idx, ""))
    
    def create_control_group(self):
        """Create optimization control group"""
        group = QGroupBox("🚀 Optimization Control")
        layout = QVBoxLayout()
        
        # Start/Stop button - smaller and centered
        button_container = QHBoxLayout()
        button_container.addStretch()
        
        self.optimize_button = QPushButton("▶️ Start Optimization")
        self.optimize_button.setFixedWidth(180)  # Even smaller, fixed width
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 12px;
                padding: 8px 15px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.optimize_button.clicked.connect(self.toggle_optimization)
        
        button_container.addWidget(self.optimize_button)
        button_container.addStretch()
        layout.addLayout(button_container)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #CCC;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to optimize")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
        group.setLayout(layout)
        return group
    
    def create_right_panel(self):
        """Create right panel for results display"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Results scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                background-color: #FAFAFA;
            }
        """)
        
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_layout.setSpacing(15)
        self.results_layout.setContentsMargins(10, 10, 10, 10)
        
        # Initial message - compact
        initial_msg = QLabel("🎯 Start optimization to see results")
        initial_msg.setAlignment(Qt.AlignCenter)
        initial_msg.setStyleSheet("""
            font-size: 12px;
            color: #999;
            padding: 10px;
        """)
        initial_msg.setWordWrap(True)
        self.results_layout.addWidget(initial_msg)
        self.results_layout.addStretch()
        
        self.results_widget.setLayout(self.results_layout)
        scroll.setWidget(self.results_widget)
        
        layout.addWidget(scroll)
        panel.setLayout(layout)
        
        return panel
    
    def on_model_changed(self):
        """Handle model selection change"""
        model_text = self.model_combo.currentText()
        model_name = model_text.split(" (R²=")[0]
        self.current_model = self.models.get(model_name)
        print(f"Selected model: {model_name}")
    
    def toggle_optimization(self):
        """Start or stop optimization"""
        if self.is_optimizing:
            self.stop_optimization()
        else:
            self.start_optimization()
    
    def start_optimization(self):
        """Start the optimization process"""
        # Collect constraints
        constraints = {}
        for param_key, inputs in self.constraint_inputs.items():
            constraints[param_key] = (inputs['min'].value(), inputs['max'].value())
            constraints[f'{param_key}_steps'] = inputs['steps'].value()
        
        # Collect state durations
        state_durations = {}
        for state_num, spin in self.state_duration_spins.items():
            state_durations[state_num] = spin.value()
        
        # Get algorithm
        algo_text = self.algorithm_combo.currentText()
        algorithm = algo_text.split(" (")[0]
        
        # Validate inputs
        if not self.current_model:
            QMessageBox.warning(self, "Warning", "No model selected!")
            return
        
        if sum(state_durations.values()) == 0:
            QMessageBox.warning(self, "Warning", "Total duration is zero!")
            return
        
        # Update UI
        self.is_optimizing = True
        self.optimize_button.setText("⏹️ Stop Optimization")
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                font-size: 16px;
                padding: 15px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.progress_bar.setValue(0)
        self.status_label.setText("Optimizing...")
        
        # Calculate baseline energy (worst case)
        baseline_config = {
            'txpower': constraints['txpower'][1],  # Max
            'adv_interval': constraints['adv_interval'][0],  # Min (more frequent)
            'conn_interval': constraints['conn_interval'][0],  # Min (more frequent)
            'vlc_payload': constraints['vlc_payload'][1],  # Max
            'ble_payload': constraints['ble_payload'][1],  # Max
            'comm_mode': 2  # On
        }
        
        # Get BLE connected duration
        ble_connected_duration = self.ble_connected_duration_spin.value()
        
        # Start optimization worker thread
        self.optimization_worker = OptimizationWorker(
            self.current_model,
            constraints,
            algorithm,
            state_durations,
            ble_connected_duration
        )
        
        self.optimization_worker.progress.connect(self.update_progress)
        self.optimization_worker.status_update.connect(self.update_status)
        self.optimization_worker.finished.connect(self.on_optimization_finished)
        
        self.optimization_worker.start()
        
        print(f"Starting {algorithm} optimization...")
        print(f"Constraints: {constraints}")
        print(f"State durations: {state_durations}")
    
    def stop_optimization(self):
        """Stop the optimization process"""
        if self.optimization_worker:
            self.optimization_worker.stop()
            self.optimization_worker.wait()
        
        self.is_optimizing = False
        self.optimize_button.setText("▶️ Start Optimization")
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 15px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.status_label.setText("Optimization stopped")
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
    
    def on_optimization_finished(self, results):
        """Handle optimization completion"""
        self.is_optimizing = False
        self.optimize_button.setText("▶️ Start Optimization")
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 15px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        if not results:
            self.status_label.setText("Optimization failed or was cancelled")
            QMessageBox.warning(self, "Warning", "Optimization returned no results!")
            return
        
        self.optimization_results = results
        self.progress_bar.setValue(100)
        self.status_label.setText(f"✅ Optimization complete! Found {len(results)} configurations")
        
        # Calculate baseline for comparison
        if len(results) > 0:
            self.baseline_energy = results[-1]['energy']  # Worst case
        
        # Display top 3 results
        self.display_top_results()
        
        print(f"\n{'='*80}")
        print("OPTIMIZATION RESULTS")
        print(f"{'='*80}")
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. Energy: {result['energy']:.2f} mJ")
            print(f"   Config: {result['config']}")
        
        # Print baseline (worst configuration) for comparison
        if len(results) > 0:
            baseline_energy = results[-1]['energy']
            baseline_config = results[-1]['config']
            savings = ((baseline_energy - results[0]['energy']) / baseline_energy) * 100
            print(f"\n{'='*80}")
            print(f"BASELINE (Worst Configuration - Rank #{len(results)}):")
            print(f"{'='*80}")
            print(f"Energy: {baseline_energy:.2f} mJ")
            print(f"Config: {baseline_config}")
            print(f"\n✅ Maximum Savings: {savings:.2f}% (Best vs Worst)")
    
    def create_baseline_widget(self, config, energy, rank):
        """Create baseline (worst configuration) display widget - same format as top results"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)  # Same as MedalWidget
        
        # Build configuration string (same format as top results)
        ble_status = 'BLE Connected' if config['comm_mode'] == 2 else 'BLE Disconnected'
        config_str = (f"TxPower={config['txpower']:.1f}dBm, "
                     f"AdvInt={config['adv_interval']:.0f}ms, "
                     f"ConnInt={config['conn_interval']:.0f}ms, "
                     f"VLC={config['vlc_payload']:.0f}B, "
                     f"BLE={config['ble_payload']:.0f}B, "
                     f"{ble_status}")
        
        # Use exhausted face emoji for baseline (😫 = worst/exhausted)
        result_text = f"Baseline: 😫\n{config_str}\nEnergy {energy:.2f}mJ"
        
        # Create plain label - same style as top results
        label = QLabel(result_text)
        label.setWordWrap(True)
        label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #333;
                padding: 8px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(label)
        
        widget.setLayout(layout)
        return widget
    
    def display_top_results(self):
        """Display top 3 optimization results and baseline"""
        # Clear previous results
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add caption for Top 3
        top3_caption = QLabel("🏆 Top 3 Optimal Configurations")
        top3_caption.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: #0066CC;
            padding: 8px;
            margin-bottom: 5px;
        """)
        top3_caption.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(top3_caption)
        
        # Display top 3
        for i, result in enumerate(self.optimization_results[:3], 1):
            medal_widget = MedalWidget(
                rank=i,
                config=result['config'],
                energy=result['energy'],
                baseline_energy=self.baseline_energy
            )
            self.results_layout.addWidget(medal_widget)
            
            # Add grey line separator between Top 1, 2, 3
            if i < 3:
                separator_mini = QFrame()
                separator_mini.setFrameShape(QFrame.HLine)
                separator_mini.setFrameShadow(QFrame.Sunken)
                separator_mini.setStyleSheet("background-color: #DDDDDD; max-height: 1px; margin: 5px 0px;")
                self.results_layout.addWidget(separator_mini)
        
        # Add separator line (thicker, to separate Top 3 from Baseline)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #CCCCCC; max-height: 2px; margin: 10px 0px;")
        self.results_layout.addWidget(separator)
        
        # Add caption for Baseline
        baseline_caption = QLabel("😫 Baseline (Worst Configuration)")
        baseline_caption.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: #CC6600;
            padding: 8px;
            margin-top: 5px;
        """)
        baseline_caption.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(baseline_caption)
        
        # Display baseline (worst configuration)
        if len(self.optimization_results) > 0:
            worst_result = self.optimization_results[-1]
            baseline_widget = self.create_baseline_widget(
                config=worst_result['config'],
                energy=worst_result['energy'],
                rank=len(self.optimization_results)
            )
            self.results_layout.addWidget(baseline_widget)
        
        self.results_layout.addStretch()


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = EnergyOptimizerApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

