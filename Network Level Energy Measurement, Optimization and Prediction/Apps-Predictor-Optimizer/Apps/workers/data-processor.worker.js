/**
 * Data Processing Worker - Handles PPK2 communication and data processing
 * Based on Nordic PPK2 official app architecture
 */

const { parentPort } = require('worker_threads');

class DataProcessor {
    constructor() {
        this.ppk2 = null;
        this.isActive = false;
        this.buffers = {
            raw: new Float32Array(100000),      // Raw PPK2 data
            processed: new Float32Array(50000), // Processed data
            display: new Float32Array(10000)    // Display-ready data
        };
        this.bufferIndices = {
            raw: 0,
            processed: 0,
            display: 0
        };
        this.sampleRate = 500; // Hz
        this.filters = {
            enabled: false,
            type: 'DelayedMedian',
            windowSize: 10,
            extremeSpikeLimit: 25000
        };
        this.baseline = {
            value: 0,
            enabled: false,
            calculating: false,
            buffer: []
        };
        this.calibration = {
            enabled: false,
            normalOffset: 0,
            sendingOffset: 0
        };
        this.nodeState = {
            runningMode: 0,
            runningState: 0
        };
        
        // High-performance timers
        this.lastProcessTime = performance.now();
        this.processingInterval = 1000 / this.sampleRate; // ms between samples
        
        this.setupMessageHandling();
        this.initializePPK2();
    }

    setupMessageHandling() {
        if (parentPort) {
            parentPort.on('message', this.handleMessage.bind(this));
        }
    }

    async handleMessage(message) {
        const { type, data } = message;
        
        try {
            switch (type) {
                case 'START_MEASUREMENT':
                    await this.startMeasurement(data);
                    break;
                case 'STOP_MEASUREMENT':
                    await this.stopMeasurement();
                    break;
                case 'UPDATE_CONFIG':
                    this.updateConfig(data);
                    break;
                case 'SET_SAMPLE_RATE':
                    this.setSampleRate(data.rate);
                    break;
                case 'UPDATE_FILTERS':
                    this.updateFilters(data);
                    break;
                case 'UPDATE_BASELINE':
                    this.updateBaseline(data);
                    break;
                case 'UPDATE_CALIBRATION':
                    this.updateCalibration(data);
                    break;
                case 'UPDATE_NODE_STATE':
                    this.updateNodeState(data);
                    break;
                default:
                    this.sendMessage('ERROR', { error: `Unknown message type: ${type}` });
            }
        } catch (error) {
            this.sendMessage('ERROR', { error: error.message, type });
        }
    }

    async initializePPK2() {
        try {
            // Import PPK2 API dynamically to avoid issues
            const PPK2API = require('../src/ppk2_api/ppk2_api.py');
            this.ppk2 = new PPK2API();
            this.sendMessage('PPK2_INITIALIZED', { status: 'ready' });
        } catch (error) {
            this.sendMessage('PPK2_ERROR', { error: 'Failed to initialize PPK2' });
        }
    }

    async startMeasurement(config = {}) {
        if (!this.ppk2) {
            throw new Error('PPK2 not initialized');
        }

        this.isActive = true;
        this.clearBuffers();
        
        // Configure PPK2 for high-frequency sampling
        await this.ppk2.setup_dut_voltage(config.voltage || 5000);
        
        this.sendMessage('MEASUREMENT_STARTED', { 
            sampleRate: this.sampleRate,
            bufferSizes: {
                raw: this.buffers.raw.length,
                processed: this.buffers.processed.length,
                display: this.buffers.display.length
            }
        });
        
        // Start high-frequency data collection
        this.startDataCollection();
    }

    async stopMeasurement() {
        this.isActive = false;
        this.sendMessage('MEASUREMENT_STOPPED', {
            finalCounts: {
                raw: this.bufferIndices.raw,
                processed: this.bufferIndices.processed,
                display: this.bufferIndices.display
            }
        });
    }

    startDataCollection() {
        // Use high-precision timer for data collection
        const collectData = () => {
            if (!this.isActive) return;

            const now = performance.now();
            const deltaTime = now - this.lastProcessTime;
            
            // Only process if enough time has passed
            if (deltaTime >= this.processingInterval) {
                this.processSamplesFromPPK2();
                this.lastProcessTime = now;
            }

            // Schedule next collection
            setImmediate(collectData);
        };

        collectData();
    }

    processSamplesFromPPK2() {
        if (!this.ppk2 || !this.isActive) return;

        try {
            // Get raw samples from PPK2
            const data = this.ppk2.get_data();
            if (!data) return;

            const samples = this.ppk2.get_samples(data);
            if (!samples || samples.length === 0) return;

            // Process each sample
            for (const sample of samples) {
                this.processSingleSample(sample);
            }

            // Send batched updates to main thread
            this.sendBatchedUpdate();

        } catch (error) {
            this.sendMessage('PROCESSING_ERROR', { error: error.message });
        }
    }

    processSingleSample(rawCurrent) {
        const timestamp = performance.now();
        
        // Store raw sample
        if (this.bufferIndices.raw < this.buffers.raw.length) {
            this.buffers.raw[this.bufferIndices.raw] = rawCurrent;
            this.bufferIndices.raw++;
        }

        // Apply filtering
        let processedCurrent = this.applyFiltering(rawCurrent);
        
        // Apply calibration
        if (this.calibration.enabled) {
            const offset = this.getCalibrationOffset();
            processedCurrent += offset;
        }

        // Apply baseline replacement
        if (this.baseline.enabled && this.baseline.value > 0) {
            processedCurrent = this.applyBaselineReplacement(processedCurrent);
        }

        // Store processed sample
        if (this.bufferIndices.processed < this.buffers.processed.length) {
            this.buffers.processed[this.bufferIndices.processed] = processedCurrent;
            this.bufferIndices.processed++;
        }

        // Decimate for display buffer
        if (this.bufferIndices.processed % this.getDecimationFactor() === 0) {
            if (this.bufferIndices.display < this.buffers.display.length) {
                this.buffers.display[this.bufferIndices.display] = processedCurrent;
                this.bufferIndices.display++;
            }
        }
    }

    applyFiltering(rawCurrent) {
        if (!this.filters.enabled) return rawCurrent;

        // Apply extreme spike filtering first
        if (Math.abs(rawCurrent) > this.filters.extremeSpikeLimit) {
            return this.baseline.value || 0;
        }

        // Apply mode-specific filtering
        if (this.nodeState.runningMode === 0) {
            // Normal operation - full filtering
            return this.applyMainFilter(rawCurrent);
        } else {
            // Transmission mode - reduced or bypassed filtering
            return this.applyTransmissionFilter(rawCurrent);
        }
    }

    applyMainFilter(current) {
        // Implement the selected filter type
        switch (this.filters.type) {
            case 'DelayedMedian':
                return this.applyDelayedMedianFilter(current);
            case 'MovingAverage':
                return this.applyMovingAverageFilter(current);
            default:
                return current;
        }
    }

    applyTransmissionFilter(current) {
        // During transmission, use reduced window or bypass
        const reducedWindowSize = Math.max(1, this.filters.windowSize / 3);
        return this.applyDelayedMedianFilter(current, reducedWindowSize);
    }

    getCalibrationOffset() {
        return this.nodeState.runningState <= 2 ? 
            this.calibration.sendingOffset : 
            this.calibration.normalOffset;
    }

    applyBaselineReplacement(current) {
        if (this.nodeState.runningMode >= 1 && current < this.baseline.value) {
            return this.baseline.value;
        }
        return current;
    }

    getDecimationFactor() {
        // Smart decimation based on sample rate
        if (this.sampleRate >= 10000) return 50;
        if (this.sampleRate >= 5000) return 25;
        if (this.sampleRate >= 1000) return 10;
        return 5;
    }

    sendBatchedUpdate() {
        // Send update every 100ms or when buffers are getting full
        const bufferUsage = this.bufferIndices.display / this.buffers.display.length;
        
        if (bufferUsage > 0.1 || Date.now() - this.lastUpdateTime > 100) {
            this.sendMessage('DATA_UPDATE', {
                samples: this.buffers.display.slice(0, this.bufferIndices.display),
                metadata: {
                    sampleRate: this.sampleRate,
                    bufferIndices: { ...this.bufferIndices },
                    nodeState: { ...this.nodeState },
                    timestamp: performance.now()
                }
            });
            
            // Reset display buffer
            this.bufferIndices.display = 0;
            this.lastUpdateTime = Date.now();
        }
    }

    clearBuffers() {
        this.bufferIndices = { raw: 0, processed: 0, display: 0 };
        this.buffers.raw.fill(0);
        this.buffers.processed.fill(0);
        this.buffers.display.fill(0);
    }

    setSampleRate(rate) {
        this.sampleRate = Math.max(10, Math.min(25000, rate));
        this.processingInterval = 1000 / this.sampleRate;
        this.sendMessage('SAMPLE_RATE_UPDATED', { rate: this.sampleRate });
    }

    updateFilters(filterConfig) {
        Object.assign(this.filters, filterConfig);
        this.sendMessage('FILTERS_UPDATED', this.filters);
    }

    updateBaseline(baselineConfig) {
        Object.assign(this.baseline, baselineConfig);
        this.sendMessage('BASELINE_UPDATED', this.baseline);
    }

    updateCalibration(calibrationConfig) {
        Object.assign(this.calibration, calibrationConfig);
        this.sendMessage('CALIBRATION_UPDATED', this.calibration);
    }

    updateNodeState(nodeState) {
        this.nodeState = { ...this.nodeState, ...nodeState };
        this.sendMessage('NODE_STATE_UPDATED', this.nodeState);
    }

    updateConfig(config) {
        if (config.sampleRate) this.setSampleRate(config.sampleRate);
        if (config.filters) this.updateFilters(config.filters);
        if (config.baseline) this.updateBaseline(config.baseline);
        if (config.calibration) this.updateCalibration(config.calibration);
    }

    sendMessage(type, data = {}) {
        if (parentPort) {
            parentPort.postMessage({ type, data, timestamp: performance.now() });
        }
    }

    // Filter implementations
    applyDelayedMedianFilter(current, windowSize = this.filters.windowSize) {
        // Simplified median filter implementation
        // In real implementation, maintain a sliding window
        return current; // Placeholder
    }

    applyMovingAverageFilter(current, windowSize = this.filters.windowSize) {
        // Simplified moving average implementation
        // In real implementation, maintain a sliding window
        return current; // Placeholder
    }
}

// Initialize the worker
const dataProcessor = new DataProcessor();

// Handle worker termination
process.on('SIGTERM', () => {
    if (dataProcessor) {
        dataProcessor.stopMeasurement();
    }
    process.exit(0);
});