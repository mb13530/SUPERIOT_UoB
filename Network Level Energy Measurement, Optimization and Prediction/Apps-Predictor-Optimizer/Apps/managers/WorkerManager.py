"""
Python Worker Manager - Simplified threading management for PyQt5
Based on Nordic PPK2 architecture patterns but adapted for Python/PyQt5
"""

import threading
import queue
import time
import logging
from typing import Dict, Any, Callable, Optional
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QThread


class WorkerThread(QThread):
    """
    Worker thread for handling specific tasks
    """
    messageReceived = pyqtSignal(dict)
    errorOccurred = pyqtSignal(str)
    
    def __init__(self, worker_type: str, task_function: Callable):
        super().__init__()
        self.worker_type = worker_type
        self.task_function = task_function
        self.message_queue = queue.Queue()
        self.is_running = True
        
    def run(self):
        """Main worker thread loop"""
        try:
            while self.is_running:
                try:
                    # Check for messages
                    try:
                        message = self.message_queue.get(timeout=0.1)
                        if message.get('type') == 'SHUTDOWN':
                            break
                        
                        # Process message with task function
                        result = self.task_function(message)
                        if result:
                            self.messageReceived.emit(result)
                            
                    except queue.Empty:
                        # No message, continue
                        pass
                    
                    # Give other threads a chance
                    self.msleep(1)
                    
                except Exception as e:
                    self.errorOccurred.emit(f"Worker {self.worker_type} error: {str(e)}")
                    
        except Exception as e:
            self.errorOccurred.emit(f"Fatal error in worker {self.worker_type}: {str(e)}")
    
    def send_message(self, message: Dict[str, Any]):
        """Send message to worker"""
        self.message_queue.put(message)
    
    def shutdown(self):
        """Graceful shutdown"""
        self.is_running = False
        self.send_message({'type': 'SHUTDOWN'})
        self.wait(3000)  # Wait up to 3 seconds
        if self.isRunning():
            self.terminate()


class WorkerManager(QObject):
    """
    Manages multiple worker threads - Python equivalent of the JavaScript version
    """
    
    # Signals
    workerMessage = pyqtSignal(dict)
    dataUpdate = pyqtSignal(dict)
    networkEvent = pyqtSignal(dict)
    serialEvent = pyqtSignal(dict)
    workerError = pyqtSignal(dict)
    initialized = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.workers: Dict[str, WorkerThread] = {}
        self.task_functions: Dict[str, Callable] = {}
        self.is_initialized = False
        
        # Register default task functions
        self.register_task_functions()
        
    def register_task_functions(self):
        """Register task functions for different worker types"""
        self.task_functions = {
            'dataProcessor': self.data_processor_task,
            'networkManager': self.network_manager_task,
            'serialManager': self.serial_manager_task
        }
    
    def initialize(self):
        """Initialize all worker threads"""
        if self.is_initialized:
            return
        
        try:
            # Create worker threads
            for worker_type, task_function in self.task_functions.items():
                self.create_worker(worker_type, task_function)
            
            self.is_initialized = True
            self.initialized.emit()
            logging.info("✅ WorkerManager initialized successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize WorkerManager: {e}")
            raise
    
    def create_worker(self, worker_type: str, task_function: Callable):
        """Create a new worker thread"""
        try:
            worker = WorkerThread(worker_type, task_function)
            
            # Connect signals
            worker.messageReceived.connect(
                lambda msg, wt=worker_type: self.handle_worker_message(wt, msg)
            )
            worker.errorOccurred.connect(
                lambda error, wt=worker_type: self.handle_worker_error(wt, error)
            )
            
            # Start the worker
            worker.start()
            self.workers[worker_type] = worker
            
            logging.info(f"✅ Worker {worker_type} created and started")
            
        except Exception as e:
            logging.error(f"❌ Failed to create worker {worker_type}: {e}")
            raise
    
    def handle_worker_message(self, worker_type: str, message: Dict[str, Any]):
        """Handle messages from worker threads"""
        try:
            # Add worker type to message
            enriched_message = {
                **message,
                'workerType': worker_type,
                'receivedAt': time.time()
            }
            
            # Emit general worker message signal
            self.workerMessage.emit(enriched_message)
            
            # Emit specific signals based on message type
            msg_type = message.get('type', '')
            
            if msg_type == 'DATA_UPDATE':
                self.dataUpdate.emit(message.get('data', {}))
            elif msg_type == 'NETWORK_EVENT':
                self.networkEvent.emit(message.get('data', {}))
            elif msg_type == 'SERIAL_EVENT':
                self.serialEvent.emit(message.get('data', {}))
            elif msg_type == 'ERROR':
                self.workerError.emit({
                    'workerType': worker_type,
                    'error': message.get('data', {}).get('error', 'Unknown error')
                })
                
        except Exception as e:
            logging.error(f"Error handling worker message: {e}")
    
    def handle_worker_error(self, worker_type: str, error: str):
        """Handle worker errors"""
        logging.error(f"❌ Worker {worker_type} error: {error}")
        self.workerError.emit({
            'workerType': worker_type,
            'error': error
        })
        
        # Optionally restart the worker
        self.restart_worker(worker_type)
    
    def restart_worker(self, worker_type: str):
        """Restart a failed worker"""
        try:
            logging.info(f"🔄 Restarting worker {worker_type}...")
            
            # Stop existing worker
            if worker_type in self.workers:
                self.workers[worker_type].shutdown()
                del self.workers[worker_type]
            
            # Create new worker
            if worker_type in self.task_functions:
                self.create_worker(worker_type, self.task_functions[worker_type])
                logging.info(f"✅ Worker {worker_type} restarted successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to restart worker {worker_type}: {e}")
    
    def send_to_worker(self, worker_type: str, message: Dict[str, Any]):
        """Send message to specific worker"""
        if worker_type in self.workers:
            try:
                enhanced_message = {
                    **message,
                    'sentAt': time.time()
                }
                self.workers[worker_type].send_message(enhanced_message)
            except Exception as e:
                logging.error(f"Failed to send message to worker {worker_type}: {e}")
        else:
            logging.warning(f"Worker {worker_type} not found")
    
    def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all workers"""
        for worker_type in self.workers:
            self.send_to_worker(worker_type, message)
    
    # High-level API methods
    def start_measurement(self, config: Dict[str, Any] = None):
        """Start measurement"""
        self.send_to_worker('dataProcessor', {
            'type': 'START_MEASUREMENT',
            'data': config or {}
        })
    
    def stop_measurement(self):
        """Stop measurement"""
        self.send_to_worker('dataProcessor', {
            'type': 'STOP_MEASUREMENT'
        })
    
    def set_sample_rate(self, rate: int):
        """Set sample rate"""
        self.send_to_worker('dataProcessor', {
            'type': 'SET_SAMPLE_RATE',
            'data': {'rate': rate}
        })
    
    def update_filters(self, filter_config: Dict[str, Any]):
        """Update filter configuration"""
        self.send_to_worker('dataProcessor', {
            'type': 'UPDATE_FILTERS',
            'data': filter_config
        })
    
    def update_baseline(self, baseline_config: Dict[str, Any]):
        """Update baseline configuration"""
        self.send_to_worker('dataProcessor', {
            'type': 'UPDATE_BASELINE',
            'data': baseline_config
        })
    
    def update_calibration(self, calibration_config: Dict[str, Any]):
        """Update calibration configuration"""
        self.send_to_worker('dataProcessor', {
            'type': 'UPDATE_CALIBRATION',
            'data': calibration_config
        })
    
    def update_node_state(self, node_state: Dict[str, Any]):
        """Update node state"""
        self.send_to_worker('dataProcessor', {
            'type': 'UPDATE_NODE_STATE',
            'data': node_state
        })
    
    def send_serial_command(self, command: str):
        """Send serial command"""
        self.send_to_worker('serialManager', {
            'type': 'SEND_COMMAND',
            'data': {'command': command}
        })
    
    def shutdown(self):
        """Shutdown all workers"""
        logging.info("🔄 Shutting down all workers...")
        
        for worker_type, worker in self.workers.items():
            try:
                worker.shutdown()
                logging.info(f"✅ Worker {worker_type} shut down")
            except Exception as e:
                logging.error(f"❌ Error shutting down worker {worker_type}: {e}")
        
        self.workers.clear()
        self.is_initialized = False
    
    # Task functions (simplified implementations)
    def data_processor_task(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Data processor task function"""
        msg_type = message.get('type')
        
        if msg_type == 'START_MEASUREMENT':
            # Simulate starting measurement
            return {
                'type': 'MEASUREMENT_STARTED',
                'data': {
                    'status': 'started',
                    'config': message.get('data', {})
                }
            }
        elif msg_type == 'SET_SAMPLE_RATE':
            rate = message.get('data', {}).get('rate', 500)
            return {
                'type': 'SAMPLE_RATE_UPDATED',
                'data': {'rate': rate}
            }
        
        # Simulate periodic data updates
        import random
        if random.random() < 0.1:  # 10% chance of sending data update
            return {
                'type': 'DATA_UPDATE',
                'data': {
                    'samples': [random.uniform(-100, 5000) for _ in range(10)],
                    'metadata': {
                        'timestamp': time.time(),
                        'sampleRate': 500
                    }
                }
            }
        
        return None
    
    def network_manager_task(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Network manager task function"""
        msg_type = message.get('type')
        
        if msg_type == 'START_HOST_CONNECTION':
            return {
                'type': 'NETWORK_EVENT',
                'data': {
                    'type': 'HOST_CONNECTED',
                    'address': '192.168.1.100'
                }
            }
        
        return None
    
    def serial_manager_task(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Serial manager task function"""
        msg_type = message.get('type')
        
        if msg_type == 'SEND_COMMAND':
            command = message.get('data', {}).get('command', '')
            return {
                'type': 'SERIAL_EVENT',
                'data': {
                    'type': 'COMMAND_SENT',
                    'command': command
                }
            }
        
        return None