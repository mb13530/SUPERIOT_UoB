/**
 * Worker Manager - Coordinates all worker threads
 * Based on Nordic PPK2 app architecture patterns
 */

const { Worker } = require('worker_threads');
const EventEmitter = require('events');
const path = require('path');

class WorkerManager extends EventEmitter {
    constructor() {
        super();
        this.workers = new Map();
        this.messageQueue = new Map();
        this.isInitialized = false;
        this.workerPaths = {
            dataProcessor: path.join(__dirname, '../../workers/data-processor.worker.js'),
            networkManager: path.join(__dirname, '../../workers/network-manager.worker.js'),
            serialManager: path.join(__dirname, '../../workers/serial-manager.worker.js')
        };
    }

    async initialize() {
        if (this.isInitialized) return;

        try {
            // Initialize all worker threads
            await this.createWorker('dataProcessor');
            await this.createWorker('networkManager');
            await this.createWorker('serialManager');

            this.isInitialized = true;
            this.emit('initialized');
            console.log('✅ All workers initialized successfully');
        } catch (error) {
            this.emit('error', new Error(`Failed to initialize workers: ${error.message}`));
            throw error;
        }
    }

    async createWorker(type) {
        const workerPath = this.workerPaths[type];
        if (!workerPath) {
            throw new Error(`Unknown worker type: ${type}`);
        }

        try {
            const worker = new Worker(workerPath);
            
            // Set up worker event handlers
            worker.on('message', (message) => this.handleWorkerMessage(type, message));
            worker.on('error', (error) => this.handleWorkerError(type, error));
            worker.on('exit', (code) => this.handleWorkerExit(type, code));

            this.workers.set(type, worker);
            this.messageQueue.set(type, []);

            // Wait for worker to be ready
            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error(`Worker ${type} initialization timeout`));
                }, 5000);

                const messageHandler = (message) => {
                    if (message.type === 'WORKER_READY' || message.type === 'PPK2_INITIALIZED') {
                        clearTimeout(timeout);
                        worker.off('message', messageHandler);
                        console.log(`✅ Worker ${type} ready`);
                        resolve(worker);
                    }
                };

                worker.on('message', messageHandler);
            });
        } catch (error) {
            throw new Error(`Failed to create worker ${type}: ${error.message}`);
        }
    }

    handleWorkerMessage(workerType, message) {
        const { type, data, timestamp } = message;
        
        // Add worker source to message
        const enrichedMessage = {
            ...message,
            workerType,
            receivedAt: performance.now()
        };

        // Emit to main application
        this.emit('workerMessage', enrichedMessage);
        
        // Handle specific message types
        switch (type) {
            case 'DATA_UPDATE':
                this.emit('dataUpdate', data);
                break;
            case 'NETWORK_EVENT':
                this.emit('networkEvent', data);
                break;
            case 'SERIAL_EVENT':
                this.emit('serialEvent', data);
                break;
            case 'ERROR':
                this.emit('workerError', { workerType, error: data.error });
                break;
            default:
                // Forward all other messages
                this.emit(type.toLowerCase(), data);
        }
    }

    handleWorkerError(workerType, error) {
        console.error(`❌ Worker ${workerType} error:`, error);
        this.emit('workerError', { workerType, error });

        // Attempt to restart the worker
        this.restartWorker(workerType);
    }

    handleWorkerExit(workerType, code) {
        if (code !== 0) {
            console.error(`❌ Worker ${workerType} exited with code ${code}`);
            this.emit('workerError', { 
                workerType, 
                error: new Error(`Worker exited with code ${code}`) 
            });
            
            // Attempt to restart the worker
            this.restartWorker(workerType);
        }
    }

    async restartWorker(workerType) {
        console.log(`🔄 Restarting worker ${workerType}...`);
        
        try {
            // Clean up existing worker
            const existingWorker = this.workers.get(workerType);
            if (existingWorker) {
                await existingWorker.terminate();
                this.workers.delete(workerType);
            }

            // Create new worker
            await this.createWorker(workerType);
            
            console.log(`✅ Worker ${workerType} restarted successfully`);
            this.emit('workerRestarted', { workerType });
        } catch (error) {
            console.error(`❌ Failed to restart worker ${workerType}:`, error);
            this.emit('workerError', { workerType, error });
        }
    }

    // Send message to specific worker
    sendToWorker(workerType, message) {
        const worker = this.workers.get(workerType);
        if (!worker) {
            throw new Error(`Worker ${workerType} not found`);
        }

        try {
            worker.postMessage({
                ...message,
                sentAt: performance.now()
            });
        } catch (error) {
            this.emit('workerError', { 
                workerType, 
                error: new Error(`Failed to send message: ${error.message}`) 
            });
        }
    }

    // Broadcast message to all workers
    broadcast(message) {
        for (const [workerType, worker] of this.workers) {
            try {
                worker.postMessage({
                    ...message,
                    sentAt: performance.now()
                });
            } catch (error) {
                this.emit('workerError', { 
                    workerType, 
                    error: new Error(`Failed to broadcast message: ${error.message}`) 
                });
            }
        }
    }

    // High-level API methods
    async startMeasurement(config = {}) {
        this.sendToWorker('dataProcessor', {
            type: 'START_MEASUREMENT',
            data: config
        });
    }

    async stopMeasurement() {
        this.sendToWorker('dataProcessor', {
            type: 'STOP_MEASUREMENT'
        });
    }

    setSampleRate(rate) {
        this.sendToWorker('dataProcessor', {
            type: 'SET_SAMPLE_RATE',
            data: { rate }
        });
    }

    updateFilters(filterConfig) {
        this.sendToWorker('dataProcessor', {
            type: 'UPDATE_FILTERS',
            data: filterConfig
        });
    }

    updateBaseline(baselineConfig) {
        this.sendToWorker('dataProcessor', {
            type: 'UPDATE_BASELINE',
            data: baselineConfig
        });
    }

    updateCalibration(calibrationConfig) {
        this.sendToWorker('dataProcessor', {
            type: 'UPDATE_CALIBRATION',
            data: calibrationConfig
        });
    }

    updateNodeState(nodeState) {
        this.sendToWorker('dataProcessor', {
            type: 'UPDATE_NODE_STATE',
            data: nodeState
        });
    }

    // Network operations
    startHostConnection(config) {
        this.sendToWorker('networkManager', {
            type: 'START_HOST_CONNECTION',
            data: config
        });
    }

    stopHostConnection() {
        this.sendToWorker('networkManager', {
            type: 'STOP_HOST_CONNECTION'
        });
    }

    sendToHost(data) {
        this.sendToWorker('networkManager', {
            type: 'SEND_TO_HOST',
            data
        });
    }

    // Serial operations
    connectSerial(port) {
        this.sendToWorker('serialManager', {
            type: 'CONNECT_SERIAL',
            data: { port }
        });
    }

    disconnectSerial() {
        this.sendToWorker('serialManager', {
            type: 'DISCONNECT_SERIAL'
        });
    }

    sendSerialCommand(command) {
        this.sendToWorker('serialManager', {
            type: 'SEND_COMMAND',
            data: { command }
        });
    }

    // Health monitoring
    getWorkerHealth() {
        const health = {};
        for (const [workerType, worker] of this.workers) {
            health[workerType] = {
                active: worker && !worker.killed,
                threadId: worker?.threadId,
                resourceUsage: worker?.resourceLimits
            };
        }
        return health;
    }

    // Graceful shutdown
    async shutdown() {
        console.log('🔄 Shutting down all workers...');
        
        const shutdownPromises = [];
        
        for (const [workerType, worker] of this.workers) {
            shutdownPromises.push(
                worker.terminate().then(() => {
                    console.log(`✅ Worker ${workerType} terminated`);
                }).catch((error) => {
                    console.error(`❌ Error terminating worker ${workerType}:`, error);
                })
            );
        }

        try {
            await Promise.all(shutdownPromises);
            this.workers.clear();
            this.messageQueue.clear();
            this.isInitialized = false;
            console.log('✅ All workers shut down successfully');
        } catch (error) {
            console.error('❌ Error during worker shutdown:', error);
            throw error;
        }
    }
}

module.exports = WorkerManager;