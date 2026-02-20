"""
Python State Manager - Centralized state management for PyQt5
Based on Nordic PPK2 architecture patterns but adapted for Python/PyQt5
"""

import time
import logging
from typing import Dict, Any, Callable, List, Optional
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from collections import deque


class StateManager(QObject):
    """
    Centralized state management with worker synchronization
    Python equivalent of the JavaScript StateManager
    """
    
    # Signals
    stateChanged = pyqtSignal(dict)
    dataUpdate = pyqtSignal(dict)
    networkEvent = pyqtSignal(dict)
    serialEvent = pyqtSignal(dict)
    workerError = pyqtSignal(dict)
    error = pyqtSignal(dict)
    
    def __init__(self, worker_manager=None):
        super().__init__()
        self.worker_manager = worker_manager
        self.state = self.create_initial_state()
        self.subscribers: List[Callable] = []
        self.update_queue = deque()
        self.is_processing_queue = False
        
        if self.worker_manager:
            self.setup_worker_listeners()
        
        # Setup update processor timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.process_update_queue)
        self.update_timer.start(16)  # ~60 FPS
    
    def create_initial_state(self) -> Dict[str, Any]:
        """Create initial application state"""
        return {
            # PPK2 and measurement state
            'measurement': {
                'active': False,
                'startTime': None,
                'sampleRate': 500,
                'voltage': 5000,
                'totalSamples': 0
            },
            
            # Real-time data
            'data': {
                'current': 0,
                'voltage': 5000,
                'power': 0,
                'timestamp': 0,
                'samples': [],  # Recent samples for plotting
                'buffer': {
                    'raw': [],
                    'processed': [],
                    'display': []
                }
            },
            
            # Node state
            'node': {
                'guestId': 1,
                'runningMode': 0,
                'runningState': 0,
                'txPower': 0,
                'communicationVolume': 0,
                'informationVolume': 0,
                'perUnitTime': 1,
                'communicationMode': 0,
                'protocolMode': 0,
                'powerSupplyMode': 0,
                'advertisingInterval': 150,
                'connectionInterval': 60,
                'fastTime': 0
            },
            
            # Filtering configuration
            'filters': {
                'enabled': False,
                'type': 'DelayedMedian',
                'windowSize': 10,
                'extremeSpikeLimit': 25000,
                'transmissionMode': 'ReducedWindow',
                'transmissionWindowSize': 3
            },
            
            # Baseline configuration
            'baseline': {
                'enabled': True,
                'value': 0,
                'mode': 'LowerThanBaseline',
                'calculating': False,
                'collectionDelay': 2.0,
                'collectionDuration': 2.0
            },
            
            # Calibration configuration
            'calibration': {
                'enabled': True,
                'normalOffset': 0,
                'sendingOffset': 0
            },
            
            # Network state
            'network': {
                'hostConnected': False,
                'hostAddress': '',
                'guestConnections': [],
                'lastDataSent': None
            },
            
            # Serial communication
            'serial': {
                'connected': False,
                'port': '',
                'commands': {
                    'cmd1': 'AA 01 10 0C 01 01 EE',
                    'cmd2': 'AA 02 20 01 04 D6 D6 70 4D 29 6B EE',
                    'cmd3': 'AA 01 04 01 EE'
                },
                'lastCommand': '',
                'status': ''
            },
            
            # UI state
            'ui': {
                'plotTimeWindow': 5.0,
                'plotResolution': 'All',
                'showLogs': True,
                'showMeasurements': True,
                'performanceMode': False,
                'maxTableRows': 1
            },
            
            # Performance metrics
            'performance': {
                'workerHealth': {},
                'dataRate': 0,
                'processingLatency': 0,
                'memoryUsage': {},
                'lastUpdate': 0
            }
        }
    
    def setup_worker_listeners(self):
        """Setup listeners for worker events"""
        if not self.worker_manager:
            return
        
        # Connect worker manager signals
        self.worker_manager.dataUpdate.connect(self.handle_data_update)
        self.worker_manager.networkEvent.connect(self.handle_network_event)
        self.worker_manager.serialEvent.connect(self.handle_serial_event)
        self.worker_manager.workerError.connect(self.handle_worker_error)
    
    def queue_update(self, update_type: str, data: Any):
        """Queue an update for processing"""
        self.update_queue.append({
            'type': update_type,
            'data': data,
            'timestamp': time.time()
        })
    
    def process_update_queue(self):
        """Process queued updates"""
        # Guard against early calls before initialization
        try:
            if self.is_processing_queue:
                return
            if not getattr(self, 'update_queue', None):
                return
        except Exception as e:
            logging.error(f"StateManager not ready to process updates: {e}")
            return
        
        self.is_processing_queue = True
        
        try:
            # Process a limited number of updates per cycle
            max_updates = 10
            processed = 0
            
            while self.update_queue and processed < max_updates:
                update = self.update_queue.popleft()
                if not isinstance(update, dict):
                    logging.warning(f"Skipping non-dict update: {type(update)}")
                    continue
                self.process_update(update)
                processed += 1
        
        except Exception as e:
            logging.error(f"Error processing update queue: {e}")
        
        finally:
            self.is_processing_queue = False
    
    def process_update(self, update: Dict[str, Any]):
        """Process a single update"""
        try:
            update_type = update['type']
            data = update['data']
            
            if update_type == 'DATA_UPDATE':
                self.handle_data_update(data)
            elif update_type == 'NETWORK_EVENT':
                self.handle_network_event(data)
            elif update_type == 'SERIAL_EVENT':
                self.handle_serial_event(data)
            elif update_type == 'WORKER_ERROR':
                self.handle_worker_error(data)
            elif update_type == 'CONFIG_UPDATE':
                self.handle_config_update(data)
            else:
                logging.warning(f"Unknown update type: {update_type}")
                
        except Exception as e:
            logging.error(f"Error processing update {update.get('type', 'unknown')}: {e}")
    
    def handle_data_update(self, data: Dict[str, Any]):
        """Handle data updates from data processor"""
        try:
            samples = data.get('samples', [])
            metadata = data.get('metadata', {})
            
            if samples:
                latest_sample = samples[-1] if samples else 0
                current_time = time.time()
                
                # Update data state
                self.update_state('data', {
                    'current': latest_sample,
                    'timestamp': current_time,
                    'samples': self.state['data']['samples'] + samples,
                    'power': (self.state['data']['voltage'] / 1000) * (latest_sample / 1000)
                })
                
                # Keep only recent samples
                if len(self.state['data']['samples']) > 1000:
                    self.state['data']['samples'] = self.state['data']['samples'][-1000:]
                
                # Update measurement totals
                total_samples = metadata.get('bufferIndices', {}).get('processed', 0)
                if total_samples:
                    self.update_state('measurement', {'totalSamples': total_samples})
            
            # Update performance metrics
            self.update_state('performance', {
                'dataRate': len(samples) * 10 if samples else 0,  # samples per second estimate
                'processingLatency': time.time() - metadata.get('timestamp', time.time()),
                'lastUpdate': time.time()
            })
            
            # Emit signal
            self.dataUpdate.emit(data)
            
        except Exception as e:
            logging.error(f"Error handling data update: {e}")
    
    def handle_network_event(self, data: Dict[str, Any]):
        """Handle network events"""
        try:
            event_type = data.get('type', '')
            
            if event_type == 'HOST_CONNECTED':
                self.update_state('network', {
                    'hostConnected': True,
                    'hostAddress': data.get('address', '')
                })
            elif event_type == 'HOST_DISCONNECTED':
                self.update_state('network', {
                    'hostConnected': False,
                    'hostAddress': ''
                })
            elif event_type == 'DATA_SENT':
                self.update_state('network', {
                    'lastDataSent': time.time()
                })
            
            self.networkEvent.emit(data)
            
        except Exception as e:
            logging.error(f"Error handling network event: {e}")
    
    def handle_serial_event(self, data: Dict[str, Any]):
        """Handle serial events"""
        try:
            event_type = data.get('type', '')
            
            if event_type == 'CONNECTED':
                self.update_state('serial', {
                    'connected': True,
                    'port': data.get('port', ''),
                    'status': 'Connected'
                })
            elif event_type == 'DISCONNECTED':
                self.update_state('serial', {
                    'connected': False,
                    'port': '',
                    'status': 'Disconnected'
                })
            elif event_type == 'COMMAND_SENT':
                self.update_state('serial', {
                    'lastCommand': data.get('command', ''),
                    'status': f"Sent: {data.get('command', '')}"
                })
            elif event_type == 'DATA_RECEIVED':
                self.update_node_state_from_serial(data.get('data', {}))
            elif event_type == 'ERROR':
                self.update_state('serial', {
                    'status': f"Error: {data.get('error', '')}"
                })
            
            self.serialEvent.emit(data)
            
        except Exception as e:
            logging.error(f"Error handling serial event: {e}")
    
    def handle_worker_error(self, data: Dict[str, Any]):
        """Handle worker errors"""
        worker_type = data.get('workerType', 'unknown')
        error = data.get('error', 'Unknown error')
        
        logging.error(f"Worker error from {worker_type}: {error}")
        
        self.update_state('performance.workerHealth', {
            worker_type: {
                'status': 'error',
                'error': error,
                'timestamp': time.time()
            }
        })
        
        self.workerError.emit(data)
    
    def handle_config_update(self, data: Dict[str, Any]):
        """Handle configuration updates"""
        for key, value in data.items():
            if key in self.state:
                self.update_state(key, value)
    
    def update_node_state_from_serial(self, serial_data: Dict[str, Any]):
        """Update node state from serial data"""
        try:
            # Parse serial data and update node state
            node_updates = self.parse_serial_data(serial_data)
            if node_updates:
                self.update_state('node', node_updates)
                
                # Notify workers about node state changes
                if self.worker_manager:
                    self.worker_manager.update_node_state(self.state['node'])
        except Exception as e:
            logging.error(f"Error parsing serial data: {e}")
    
    def parse_serial_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse serial data - placeholder implementation"""
        # This would contain the actual serial protocol parsing logic
        return None
    
    def update_state(self, path: str, value: Any):
        """Update state at given path"""
        try:
            keys = path.split('.')
            target = self.state
            
            # Navigate to parent object
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            
            # Update final property
            final_key = keys[-1]
            if isinstance(value, dict) and isinstance(target.get(final_key), dict):
                target[final_key].update(value)
            else:
                target[final_key] = value
            
            # Emit state change signal
            self.stateChanged.emit({'path': path, 'value': value})
            
            # Sync to workers if needed
            if self.worker_manager:
                self.sync_state_to_workers(path, value)
                
        except Exception as e:
            logging.error(f"Error updating state at path '{path}': {e}")
    
    def sync_state_to_workers(self, path: str, value: Any):
        """Sync state changes to workers"""
        try:
            root_key = path.split('.')[0]
            
            if root_key == 'filters':
                self.worker_manager.update_filters(self.state['filters'])
            elif root_key == 'baseline':
                self.worker_manager.update_baseline(self.state['baseline'])
            elif root_key == 'calibration':
                self.worker_manager.update_calibration(self.state['calibration'])
            elif root_key == 'measurement' and 'sampleRate' in path:
                self.worker_manager.set_sample_rate(self.state['measurement']['sampleRate'])
            elif root_key == 'node':
                self.worker_manager.update_node_state(self.state['node'])
                
        except Exception as e:
            logging.error(f"Error syncing state to workers: {e}")
    
    # Public API methods
    def get_state(self, path: Optional[str] = None) -> Any:
        """Get state value at path"""
        if path is None:
            return dict(self.state)  # Return copy
        
        try:
            keys = path.split('.')
            result = self.state
            for key in keys:
                if isinstance(result, dict) and key in result:
                    result = result[key]
                else:
                    return None
            return result
        except Exception:
            return None
    
    def set_state(self, path: str, value: Any):
        """Set state value at path"""
        self.update_state(path, value)
    
    def subscribe(self, callback: Callable):
        """Subscribe to state changes"""
        self.subscribers.append(callback)
        
        def unsubscribe():
            if callback in self.subscribers:
                self.subscribers.remove(callback)
        
        return unsubscribe
    
    def notify_subscribers(self, event_type: str, data: Any):
        """Notify all subscribers"""
        for callback in self.subscribers:
            try:
                callback({
                    'type': event_type,
                    'data': data,
                    'state': self.get_state()
                })
            except Exception as e:
                logging.error(f"Error in state subscriber: {e}")
    
    # High-level operations
    def start_measurement(self, config: Optional[Dict[str, Any]] = None):
        """Start measurement"""
        measurement_config = config or {}
        measurement_config.update({
            'voltage': self.state['measurement']['voltage'],
            'sampleRate': self.state['measurement']['sampleRate']
        })
        
        self.update_state('measurement', {
            'active': True,
            'startTime': time.time()
        })
        
        if self.worker_manager:
            self.worker_manager.start_measurement(measurement_config)
    
    def stop_measurement(self):
        """Stop measurement"""
        self.update_state('measurement', {
            'active': False,
            'startTime': None
        })
        
        if self.worker_manager:
            self.worker_manager.stop_measurement()