/**
 * State Manager - Centralized state management with worker synchronization
 * Inspired by PPK2 app's state management patterns
 */

const EventEmitter = require('events');

class StateManager extends EventEmitter {
    constructor(workerManager) {
        super();
        this.workerManager = workerManager;
        this.state = this.createInitialState();
        this.subscribers = new Set();
        this.updateQueue = [];
        this.isProcessingQueue = false;
        
        this.setupWorkerListeners();
        this.startUpdateProcessor();
    }

    createInitialState() {
        return {
            // PPK2 and measurement state
            measurement: {
                active: false,
                startTime: null,
                sampleRate: 500,
                voltage: 5000,
                totalSamples: 0
            },
            
            // Real-time data
            data: {
                current: 0,
                voltage: 5000,
                power: 0,
                timestamp: 0,
                samples: [], // Recent samples for plotting
                buffer: {
                    raw: [],
                    processed: [],
                    display: []
                }
            },
            
            // Node state
            node: {
                guestId: 1,
                runningMode: 0,
                runningState: 0,
                txPower: 0,
                communicationVolume: 0,
                informationVolume: 0,
                perUnitTime: 1,
                communicationMode: 0,
                protocolMode: 0,
                powerSupplyMode: 0,
                advertisingInterval: 150,
                connectionInterval: 60,
                fastTime: 0
            },
            
            // Filtering configuration
            filters: {
                enabled: false,
                type: 'DelayedMedian',
                windowSize: 10,
                extremeSpikeLimit: 25000,
                transmissionMode: 'ReducedWindow',
                transmissionWindowSize: 3
            },
            
            // Baseline configuration
            baseline: {
                enabled: true,
                value: 0,
                mode: 'LowerThanBaseline',
                calculating: false,
                collectionDelay: 2.0,
                collectionDuration: 2.0
            },
            
            // Calibration configuration
            calibration: {
                enabled: true,
                normalOffset: 0,
                sendingOffset: 0
            },
            
            // Network state
            network: {
                hostConnected: false,
                hostAddress: '',
                guestConnections: [],
                lastDataSent: null
            },
            
            // Serial communication
            serial: {
                connected: false,
                port: '',
                commands: {
                    cmd1: 'AA 01 10 0C 01 01 EE',
                    cmd2: 'AA 02 20 01 04 D6 D6 70 4D 29 6B EE',
                    cmd3: 'AA 01 04 01 EE'
                },
                lastCommand: '',
                status: ''
            },
            
            // UI state
            ui: {
                plotTimeWindow: 5.0,
                plotResolution: 'All',
                showLogs: true,
                showMeasurements: true,
                performanceMode: false,
                maxTableRows: 1
            },
            
            // Performance metrics
            performance: {
                workerHealth: {},
                dataRate: 0,
                processingLatency: 0,
                memoryUsage: {},
                lastUpdate: 0
            }
        };
    }

    setupWorkerListeners() {
        if (!this.workerManager) return;

        // Data updates from data processor
        this.workerManager.on('dataUpdate', (data) => {
            this.queueUpdate('DATA_UPDATE', data);
        });

        // Network events
        this.workerManager.on('networkEvent', (data) => {
            this.queueUpdate('NETWORK_EVENT', data);
        });

        // Serial events
        this.workerManager.on('serialEvent', (data) => {
            this.queueUpdate('SERIAL_EVENT', data);
        });

        // Worker health updates
        this.workerManager.on('workerError', (data) => {
            this.queueUpdate('WORKER_ERROR', data);
        });

        // Configuration updates
        this.workerManager.on('configurationUpdated', (data) => {
            this.queueUpdate('CONFIG_UPDATE', data);
        });
    }

    queueUpdate(type, data) {
        this.updateQueue.push({
            type,
            data,
            timestamp: performance.now()
        });
        
        if (!this.isProcessingQueue) {
            this.processUpdateQueue();
        }
    }

    async processUpdateQueue() {
        this.isProcessingQueue = true;
        
        while (this.updateQueue.length > 0) {
            const update = this.updateQueue.shift();
            await this.processUpdate(update);
        }
        
        this.isProcessingQueue = false;
    }

    async processUpdate(update) {
        const { type, data } = update;
        
        try {
            switch (type) {
                case 'DATA_UPDATE':
                    this.handleDataUpdate(data);
                    break;
                case 'NETWORK_EVENT':
                    this.handleNetworkEvent(data);
                    break;
                case 'SERIAL_EVENT':
                    this.handleSerialEvent(data);
                    break;
                case 'WORKER_ERROR':
                    this.handleWorkerError(data);
                    break;
                case 'CONFIG_UPDATE':
                    this.handleConfigUpdate(data);
                    break;
                default:
                    console.warn(`Unknown update type: ${type}`);
            }
        } catch (error) {
            console.error(`Error processing update ${type}:`, error);
            this.emit('error', { type: 'STATE_UPDATE_ERROR', error, update });
        }
    }

    handleDataUpdate(data) {
        const { samples, metadata } = data;
        const now = performance.now();
        
        // Update measurement data
        if (samples && samples.length > 0) {
            const latestSample = samples[samples.length - 1];
            
            this.updateState('data', {
                current: latestSample,
                timestamp: now,
                samples: [...this.state.data.samples, ...samples].slice(-1000), // Keep last 1000 samples
                power: (this.state.data.voltage / 1000) * (latestSample / 1000)
            });
            
            this.updateState('measurement', {
                totalSamples: metadata?.bufferIndices?.processed || this.state.measurement.totalSamples
            });
        }
        
        // Update performance metrics
        this.updateState('performance', {
            dataRate: samples ? samples.length / 0.1 : 0, // samples per 100ms
            processingLatency: now - (metadata?.timestamp || now),
            lastUpdate: now
        });
        
        this.notifySubscribers('DATA_UPDATE', { samples, metadata });
    }

    handleNetworkEvent(data) {
        const { type: eventType, ...eventData } = data;
        
        switch (eventType) {
            case 'HOST_CONNECTED':
                this.updateState('network', {
                    hostConnected: true,
                    hostAddress: eventData.address
                });
                break;
            case 'HOST_DISCONNECTED':
                this.updateState('network', {
                    hostConnected: false,
                    hostAddress: ''
                });
                break;
            case 'GUEST_CONNECTED':
                this.addGuestConnection(eventData);
                break;
            case 'GUEST_DISCONNECTED':
                this.removeGuestConnection(eventData.guestId);
                break;
            case 'DATA_SENT':
                this.updateState('network', {
                    lastDataSent: performance.now()
                });
                break;
        }
        
        this.notifySubscribers('NETWORK_EVENT', data);
    }

    handleSerialEvent(data) {
        const { type: eventType, ...eventData } = data;
        
        switch (eventType) {
            case 'CONNECTED':
                this.updateState('serial', {
                    connected: true,
                    port: eventData.port,
                    status: 'Connected'
                });
                break;
            case 'DISCONNECTED':
                this.updateState('serial', {
                    connected: false,
                    port: '',
                    status: 'Disconnected'
                });
                break;
            case 'COMMAND_SENT':
                this.updateState('serial', {
                    lastCommand: eventData.command,
                    status: `Sent: ${eventData.command}`
                });
                break;
            case 'DATA_RECEIVED':
                this.updateNodeStateFromSerial(eventData.data);
                break;
            case 'ERROR':
                this.updateState('serial', {
                    status: `Error: ${eventData.error}`
                });
                break;
        }
        
        this.notifySubscribers('SERIAL_EVENT', data);
    }

    handleWorkerError(data) {
        const { workerType, error } = data;
        console.error(`Worker error from ${workerType}:`, error);
        
        this.updateState('performance.workerHealth', {
            [workerType]: {
                status: 'error',
                error: error.message,
                timestamp: performance.now()
            }
        });
        
        this.notifySubscribers('WORKER_ERROR', data);
    }

    handleConfigUpdate(data) {
        // Update relevant state sections based on config type
        Object.keys(data).forEach(key => {
            if (this.state.hasOwnProperty(key)) {
                this.updateState(key, data[key]);
            }
        });
        
        this.notifySubscribers('CONFIG_UPDATE', data);
    }

    updateNodeStateFromSerial(serialData) {
        // Parse serial data and update node state
        try {
            const nodeUpdates = this.parseSerialData(serialData);
            if (nodeUpdates) {
                this.updateState('node', nodeUpdates);
                
                // Notify workers about node state changes
                if (this.workerManager) {
                    this.workerManager.updateNodeState(this.state.node);
                }
            }
        } catch (error) {
            console.error('Error parsing serial data:', error);
        }
    }

    parseSerialData(data) {
        // Implement serial data parsing logic
        // This would parse the actual serial protocol data
        // Return null if data is invalid
        return null; // Placeholder
    }

    // State update methods
    updateState(path, value) {
        const keys = path.split('.');
        let target = this.state;
        
        // Navigate to the parent object
        for (let i = 0; i < keys.length - 1; i++) {
            if (!target[keys[i]]) {
                target[keys[i]] = {};
            }
            target = target[keys[i]];
        }
        
        // Update the final property
        const finalKey = keys[keys.length - 1];
        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
            target[finalKey] = { ...target[finalKey], ...value };
        } else {
            target[finalKey] = value;
        }
        
        this.emit('stateChanged', { path, value });
    }

    // Public API methods
    getState(path = null) {
        if (!path) return { ...this.state };
        
        const keys = path.split('.');
        let result = this.state;
        for (const key of keys) {
            if (result && typeof result === 'object') {
                result = result[key];
            } else {
                return undefined;
            }
        }
        return result;
    }

    setState(path, value) {
        this.updateState(path, value);
        
        // Notify workers if relevant
        if (this.workerManager) {
            this.syncStateToWorkers(path, value);
        }
    }

    syncStateToWorkers(path, value) {
        const rootKey = path.split('.')[0];
        
        switch (rootKey) {
            case 'filters':
                this.workerManager.updateFilters(this.state.filters);
                break;
            case 'baseline':
                this.workerManager.updateBaseline(this.state.baseline);
                break;
            case 'calibration':
                this.workerManager.updateCalibration(this.state.calibration);
                break;
            case 'measurement':
                if (path.includes('sampleRate')) {
                    this.workerManager.setSampleRate(this.state.measurement.sampleRate);
                }
                break;
            case 'node':
                this.workerManager.updateNodeState(this.state.node);
                break;
        }
    }

    // Subscription management
    subscribe(callback) {
        this.subscribers.add(callback);
        return () => this.subscribers.delete(callback);
    }

    notifySubscribers(type, data) {
        for (const callback of this.subscribers) {
            try {
                callback({ type, data, state: this.getState() });
            } catch (error) {
                console.error('Error in state subscriber:', error);
            }
        }
    }

    // Helper methods
    addGuestConnection(guestData) {
        const currentConnections = [...this.state.network.guestConnections];
        const existingIndex = currentConnections.findIndex(g => g.id === guestData.id);
        
        if (existingIndex >= 0) {
            currentConnections[existingIndex] = { ...currentConnections[existingIndex], ...guestData };
        } else {
            currentConnections.push(guestData);
        }
        
        this.updateState('network', { guestConnections: currentConnections });
    }

    removeGuestConnection(guestId) {
        const currentConnections = this.state.network.guestConnections.filter(g => g.id !== guestId);
        this.updateState('network', { guestConnections: currentConnections });
    }

    startMeasurement(config = {}) {
        const measurementConfig = {
            ...config,
            voltage: this.state.measurement.voltage,
            sampleRate: this.state.measurement.sampleRate
        };
        
        this.updateState('measurement', {
            active: true,
            startTime: performance.now()
        });
        
        if (this.workerManager) {
            this.workerManager.startMeasurement(measurementConfig);
        }
    }

    stopMeasurement() {
        this.updateState('measurement', {
            active: false,
            startTime: null
        });
        
        if (this.workerManager) {
            this.workerManager.stopMeasurement();
        }
    }

    startUpdateProcessor() {
        // Process updates at 60 FPS max
        setInterval(() => {
            if (this.updateQueue.length > 0 && !this.isProcessingQueue) {
                this.processUpdateQueue();
            }
        }, 16); // ~60 FPS
    }
}

module.exports = StateManager;